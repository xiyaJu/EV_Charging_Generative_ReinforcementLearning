import os
import time
import numpy as np
import pickle
import yaml
import csv
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import shutil
import gzip

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.utilities.arg_parser import arg_parser
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives, profit_maximization, SimpleReward
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads
from ev2gym.baselines.heuristics import RandomAgent, RoundRobin_GF, ChargeAsFastAsPossible
from utils import PST_V2G_ProfitMax_reward, PST_V2G_ProfitMax_state, PST_V2G_ProfitMaxGNN_state, graph_data_to_dict

import pandas as pd


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_tag(tag):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag.strip())


def load_day_schedule(day_list_csv):
    day_df = pd.read_csv(day_list_csv)
    if 'sim_datetime' in day_df.columns:
        raw_values = day_df['sim_datetime']
    elif {'year', 'month', 'day', 'hour', 'minute'}.issubset(day_df.columns):
        raw_values = day_df.apply(
            lambda row: f"{int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d} "
                        f"{int(row['hour']):02d}:{int(row['minute']):02d}:00",
            axis=1,
        )
    else:
        raw_values = day_df.iloc[:, 0]

    day_values = []
    for value in raw_values:
        value = str(value).strip()
        if len(value) == 10:
            value = value + " 00:00:00"
        day_values.append(datetime.fromisoformat(value))
    return day_values


def set_env_start_datetime(env, dt):
    env.config['random_day'] = False
    env.config['year'] = dt.year
    env.config['month'] = dt.month
    env.config['day'] = dt.day
    env.config['hour'] = dt.hour
    env.config['minute'] = dt.minute
    env.sim_date = dt
    env.sim_starting_date = dt


def append_manifest_row(manifest_path, row):
    ensure_dir(os.path.dirname(manifest_path))
    file_exists = os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_action_mask(env):
    action_mask = np.zeros(env.number_of_ports, dtype=np.float32)
    for i, cs in enumerate(env.charging_stations):
        for j in range(cs.n_ports):
            if cs.evs_connected[j] is not None:
                action_mask[i * cs.n_ports + j] = 1.0
    return action_mask

if __name__ == "__main__":

    args = arg_parser()
    SAVE_EVAL_REPLAYS = args.save_eval_replays
    SAVE_TRAJECTORIES = (not SAVE_EVAL_REPLAYS) or args.save_trajectory_file

    if args.generated_csv:
        os.environ["EV2GYM_GENERATED_DATA_PATH"] = str(Path(args.generated_csv).expanduser().resolve())

    day_schedule = load_day_schedule(args.day_list_csv) if args.day_list_csv else None

    reward_function = PST_V2G_ProfitMax_reward
    state_function = PST_V2G_ProfitMax_state
    problem = args.config_file.split("/")[-1].split(".")[0]
    tag_suffix = f"_{normalize_tag(args.tag)}" if args.tag else ""


    env = EV2Gym(config_file=args.config_file,
                 state_function=state_function,
                 reward_function=reward_function,
                 save_replay=SAVE_EVAL_REPLAYS,
                 use_generated=args.use_generated
                 )

    temp_env = EV2Gym(config_file=args.config_file,
                      save_replay=True,
                      reward_function=reward_function,
                      state_function=state_function,
                      use_generated=args.use_generated
                      )

    n_trajectories = args.n_trajectories

    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    steps = config["simulation_length"]
    timescale = config["timescale"]

    trajectories = []
    if args.dataset not in ["random", "optimal", "bau",
                            "mixed_bau_50", "mixed_bau_25", "mixed_bau_75"
                            ]:
        raise ValueError(
            f"Trajectories type {args.dataset} not supported")

    trajecotries_type = args.dataset  # random, optimal, bau

    artifact_stem = f"{problem}_{trajecotries_type}_{number_of_charging_stations}_{n_trajectories}{tag_suffix}"
    file_name = f"{artifact_stem}.pkl"
    save_folder_path = f"./trajectories/"
    ensure_dir(save_folder_path)
    episode_rows = []
    generated_csv_path = os.environ.get("EV2GYM_GENERATED_DATA_PATH", "")
    manifest_path = "./artifacts/trajectory_generation_manifest.csv"

    # make eval replay folder
    if SAVE_EVAL_REPLAYS:
        ensure_dir("eval_replays")

        replay_folder_name = artifact_stem
        save_folder_path = f"./eval_replays/{replay_folder_name}"
        ensure_dir(save_folder_path)

        print(f"Saving evaluation replays to {save_folder_path}")

    epoch = 0
    # use tqdm with a fancy bar
    for i in tqdm(range(n_trajectories)):
        episode_seed = args.seed + i
        sim_dt = None
        if day_schedule:
            sim_dt = day_schedule[i % len(day_schedule)]
            set_env_start_datetime(env, sim_dt)
            set_env_start_datetime(temp_env, sim_dt)

        trajectory_i = {"observations": [],
                        "graph_observations": [],
                        "actions": [],
                        "rewards": [],
                        "dones": [],
                        "action_mask": [],
                        }

        epoch_return = 0

        if trajecotries_type == "random":
            agent = RandomAgent(env)
        elif trajecotries_type == "bau":
            agent = RoundRobin_GF(env)

        elif trajecotries_type == "mixed_bau_50":
            if i % 2 == 0:
                agent = RoundRobin_GF(env)
            else:
                agent = RandomAgent(env)

        elif trajecotries_type == "mixed_bau_25":
            if i % 4 == 0:
                agent = RoundRobin_GF(env)
            else:
                agent = RandomAgent(env)

        elif trajecotries_type == "mixed_bau_75":
            if i % 4 == 0:
                agent = RandomAgent(env)
            else:
                agent = RoundRobin_GF(env)

        elif trajecotries_type == "optimal":
            from ev2gym.baselines.gurobi_models.PST_V2G_profit_max_mo import mo_PST_V2GProfitMaxOracleGB
            _, _ = temp_env.reset(seed=episode_seed)
            agent = ChargeAsFastAsPossible()

            for _ in range(temp_env.simulation_length):
                actions = agent.get_action(temp_env)
                new_state, reward, done, truncated, stats = temp_env.step(
                    actions)  # takes action
                if done:
                    break

            new_replay_path = f"./replay/replay_{temp_env.sim_name}.pkl"

            timelimit = 180

            agent = mo_PST_V2GProfitMaxOracleGB(new_replay_path,
                                                timelimit=timelimit,
                                                MIPGap=None,
                                                )

        elif trajecotries_type == "mpc":
            from ev2gym.baselines.mpc.eMPC_v2 import eMPC_V2G_v2
            agent = eMPC_V2G_v2(env,
                                control_horizon=10,
                                MIPGap=0.1,
                                time_limit=30,
                                verbose=False)
        else:
            raise ValueError(
                f"Trajectories type {trajecotries_type} not supported")

        if trajecotries_type == "optimal":
            env = EV2Gym(config_file=args.config_file,
                         load_from_replay_path=new_replay_path,
                         state_function=state_function,
                         reward_function=reward_function,
                         save_replay=SAVE_EVAL_REPLAYS,
                         )
            os.remove(new_replay_path)

        if SAVE_EVAL_REPLAYS:
            env.eval_mode = "optimal" if trajecotries_type == "optimal" else "Normal"
        state, _ = env.reset(seed=episode_seed)

        if day_schedule and trajecotries_type == "optimal":
            sim_dt = temp_env.sim_starting_date

        while True:
            current_action_mask = build_action_mask(env)
            current_graph_state = graph_data_to_dict(PST_V2G_ProfitMaxGNN_state(env))

            actions = agent.get_action(env)

            new_state, reward, done, truncated, stats = env.step(actions)

            trajectory_i["observations"].append(state)
            trajectory_i["graph_observations"].append(current_graph_state)
            trajectory_i["actions"].append(actions)
            trajectory_i["rewards"].append(reward)
            trajectory_i["dones"].append(done)
            trajectory_i["action_mask"].append(current_action_mask)

            state = new_state

            if done:
                # move the replay file to the eval replay folder
                if SAVE_EVAL_REPLAYS:
                    replay_path = env.replay_path + 'replay_' + env.sim_name + '.pkl'
                    new_replay_path = f"{save_folder_path}/replay_{env.sim_name}_{i}.pkl"
                    shutil.move(replay_path, new_replay_path)

                break
        print(f'Stats: {env.stats["total_reward"]}')
        episode_rows.append({
            "artifact_stem": artifact_stem,
            "episode_idx": i,
            "episode_seed": episode_seed,
            "trajectory_policy": trajecotries_type,
            "use_generated": args.use_generated,
            "generated_csv": generated_csv_path,
            "sim_datetime": env.sim_starting_date.isoformat(sep=" "),
            "n_charging_stations": number_of_charging_stations,
            "simulation_length": steps,
            "total_reward": env.stats.get("total_reward"),
            "total_profits": env.stats.get("total_profits"),
            "average_user_satisfaction": env.stats.get("average_user_satisfaction"),
            "power_tracker_violation": env.stats.get("power_tracker_violation"),
            "total_transformer_overload": env.stats.get("total_transformer_overload"),
            "trajectory_length": len(trajectory_i["rewards"]),
        })
        trajectory_i["observations"] = np.array(trajectory_i["observations"])
        trajectory_i["actions"] = np.array(trajectory_i["actions"])
        trajectory_i["rewards"] = np.array(trajectory_i["rewards"])
        trajectory_i["dones"] = np.array(trajectory_i["dones"])
        trajectory_i["action_mask"] = np.array(trajectory_i["action_mask"])

        trajectories.append(trajectory_i)

        if trajecotries_type == "optimal":
            divident = 100
        else:
            divident = 1000

        if i % divident == 0 and SAVE_TRAJECTORIES and i > 0:
            print(f'Saving trajectories to ./trajectories/{file_name}')

            with gzip.open(f"./trajectories/{file_name}.gz", 'wb') as f:
                pickle.dump(trajectories, f)

    env.close()

    episode_stats_path = f"./trajectories/{artifact_stem}.episode_stats.csv"
    pd.DataFrame(episode_rows).to_csv(episode_stats_path, index=False)

    if SAVE_TRAJECTORIES:
        print(f'Saving trajectories to ./trajectories/{file_name}')

        with gzip.open(f"./trajectories/{file_name}.gz", 'wb') as f:
            pickle.dump(trajectories, f)

        # To read the compressed pickle file
        with gzip.open(f"./trajectories/{file_name}.gz", 'rb') as f:
            loaded_data = pickle.load(f)

        print(loaded_data[0]["observations"].shape)
        print(loaded_data[0]["actions"].shape)
        print(loaded_data[0]["rewards"].shape)
        print(loaded_data[0]["dones"].shape)
        print(loaded_data[0]["action_mask"].shape)

    append_manifest_row(
        manifest_path,
        {
            "artifact_stem": artifact_stem,
            "trajectory_file": f"./trajectories/{file_name}.gz" if SAVE_TRAJECTORIES else "",
            "episode_stats_file": episode_stats_path,
            "replay_dir": save_folder_path if SAVE_EVAL_REPLAYS else "",
            "config_file": args.config_file,
            "trajectory_policy": trajecotries_type,
            "use_generated": args.use_generated,
            "generated_csv": generated_csv_path,
            "day_list_csv": args.day_list_csv or "",
            "n_trajectories": n_trajectories,
            "n_charging_stations": number_of_charging_stations,
            "seed_start": args.seed,
            "tag": args.tag,
        },
    )

    if SAVE_EVAL_REPLAYS:
        print(
            f'Generated {n_trajectories} replay files and saved them in {save_folder_path}')
