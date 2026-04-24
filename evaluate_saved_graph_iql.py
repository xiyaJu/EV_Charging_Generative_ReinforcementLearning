import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from DT.models.graph_iql import GraphIQLPolicy
from ev2gym.models.ev2gym_env import EV2Gym
from utils import PST_V2G_ProfitMax_reward, PST_V2G_ProfitMax_state, PST_V2G_ProfitMaxGNN_state, graph_data_to_dict


STATE_DIM = 1500
ACT_DIM = 250


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def build_policy(run_vars, device):
    return GraphIQLPolicy(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        fx_node_sizes={"ev": 5, "cs": 4, "tr": 2, "env": 6},
        feature_dim=run_vars["feature_dim"],
        hidden_size=run_vars["embed_dim"],
        gnn_hidden_dim=run_vars["GNN_hidden_dim"],
        num_gcn_layers=run_vars["num_gcn_layers"],
        device=device,
    ).to(device=device)


def summarize_results(df, suite_name):
    metric_columns = [
        col for col in df.columns
        if col not in {
            "exp_id",
            "suite_name",
            "checkpoint",
            "replay_file",
            "sim_datetime",
            "n_cs",
            "n_transformers",
            "scenario",
        }
    ]
    metric_columns = [col for col in metric_columns if pd.api.types.is_numeric_dtype(df[col])]

    rows = []
    for metric in metric_columns:
        series = df[metric].dropna()
        if series.empty:
            continue
        mean = series.mean()
        std = series.std(ddof=1) if len(series) > 1 else 0.0
        sem = std / np.sqrt(len(series)) if len(series) > 1 else 0.0
        ci95 = 1.96 * sem
        rows.append(
            {
                "suite_name": suite_name,
                "metric": metric,
                "n_replays": len(series),
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95_low": mean - ci95,
                "ci95_high": mean + ci95,
            }
        )
    return pd.DataFrame(rows)


def run_single_replay(env, policy, device):
    state, _ = env.reset()
    graph_states = [graph_data_to_dict(PST_V2G_ProfitMaxGNN_state(env))]
    actions = torch.zeros((0, ACT_DIM), device=device, dtype=torch.float32)
    action_mask = np.zeros(ACT_DIM, dtype=np.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    timesteps = torch.zeros((1, 1), device=device, dtype=torch.long)

    for _ in range(env.simulation_length):
        actions = torch.cat([actions, torch.zeros((1, ACT_DIM), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        action_mask[:] = 0.0
        for i, cs in enumerate(env.charging_stations):
            for j in range(cs.n_ports):
                if cs.evs_connected[j] is not None:
                    action_mask[i * cs.n_ports + j] = 1.0

        action = policy.get_action(
            states=torch.zeros((1, STATE_DIM), device=device),
            actions=actions,
            rewards=rewards,
            action_mask=torch.from_numpy(action_mask).to(device=device).reshape(1, ACT_DIM),
            graph_states=graph_states,
            config=env.config,
        )
        action_to_env = action.detach().cpu().numpy()[: env.action_space.shape[0]]
        _, _, done, _, stats = env.step(action_to_env)
        graph_states.append(graph_data_to_dict(PST_V2G_ProfitMaxGNN_state(env)))
        if done:
            total_reward = float(stats.get("total_reward", 0.0))
            stats = dict(stats)
            stats["reward_per_cs"] = total_reward / max(int(env.cs), 1)
            stats["reward_per_cs_step"] = total_reward / max(int(env.cs * env.simulation_length), 1)
            return stats
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vars_yaml", default=None)
    parser.add_argument("--config_file", default=None)
    parser.add_argument("--replay_dir", required=True)
    parser.add_argument("--suite_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    run_dir = checkpoint_path.parent
    vars_path = Path(args.vars_yaml).resolve() if args.vars_yaml else run_dir / "vars.yaml"
    run_vars = load_yaml(vars_path)
    config_path = Path(args.config_file).resolve() if args.config_file else Path("./config_files") / run_vars["config_file"]

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    policy = build_policy(run_vars, device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()

    replay_dir = Path(args.replay_dir).resolve()
    replay_files = sorted([path for path in replay_dir.iterdir() if path.is_file() and path.suffix == ".pkl"])
    rows = []
    for replay_file in replay_files:
        env = EV2Gym(
            config_file=str(config_path),
            load_from_replay_path=str(replay_file),
            state_function=PST_V2G_ProfitMax_state,
            reward_function=PST_V2G_ProfitMax_reward,
        )
        stats = run_single_replay(env, policy, device)
        row = {
            "exp_id": run_dir.name,
            "suite_name": args.suite_name,
            "checkpoint": str(checkpoint_path),
            "replay_file": str(replay_file),
            "sim_datetime": getattr(env.replay, "sim_date", ""),
            "n_cs": getattr(env.replay, "n_cs", np.nan),
            "n_transformers": getattr(env.replay, "n_transformers", np.nan),
            "scenario": getattr(env.replay, "scenario", ""),
        }
        row.update(stats)
        rows.append(row)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_replay_df = pd.DataFrame(rows)
    per_replay_df.to_csv(output_dir / f"{args.suite_name}.per_replay.csv", index=False)
    summarize_results(per_replay_df, args.suite_name).to_csv(
        output_dir / f"{args.suite_name}.summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
