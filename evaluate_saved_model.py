import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from DT.models.gnn_decision_transformer import GNN_DecisionTransformer
from ev2gym.models.ev2gym_env import EV2Gym
from utils import PST_V2G_ProfitMax_reward, PST_V2G_ProfitMax_state, PST_V2G_ProfitMaxGNN_state, graph_data_to_dict


STATE_DIM = 1500
ACT_DIM = 250


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def build_model(run_vars, env_config, device):
    model = GNN_DecisionTransformer(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        max_length=run_vars["K"],
        max_ep_len=env_config["simulation_length"],
        hidden_size=run_vars["embed_dim"],
        n_layer=run_vars["n_layer"],
        n_head=run_vars["n_head"],
        n_inner=4 * run_vars["embed_dim"],
        activation_function=run_vars["activation_function"],
        n_positions=1024,
        resid_pdrop=run_vars["dropout"],
        attn_pdrop=run_vars["dropout"],
        action_tanh=True,
        action_masking=run_vars["action_masking"],
        fx_node_sizes={"ev": 5, "cs": 4, "tr": 2, "env": 6},
        feature_dim=run_vars["feature_dim"],
        GNN_hidden_dim=run_vars["GNN_hidden_dim"],
        num_gcn_layers=run_vars["num_gcn_layers"],
        config=env_config,
        device=device,
    )
    return model.to(device=device)


def get_state_norm(run_dir, state_normalization, device):
    if not state_normalization:
        return (
            torch.zeros(STATE_DIM, device=device, dtype=torch.float32),
            torch.ones(STATE_DIM, device=device, dtype=torch.float32),
        )

    state_mean = np.load(run_dir / "state_mean.npy")
    state_std = np.load(run_dir / "state_std.npy")
    return (
        torch.from_numpy(state_mean).to(device=device, dtype=torch.float32),
        torch.from_numpy(state_std).to(device=device, dtype=torch.float32),
    )


def pad_state(state):
    padded_state = np.zeros(STATE_DIM, dtype=np.float32)
    clipped = state[:STATE_DIM]
    padded_state[: len(clipped)] = clipped
    return padded_state


def build_current_action_mask(env):
    action_mask = np.zeros(ACT_DIM, dtype=np.float32)
    for i, cs in enumerate(env.charging_stations):
        for j in range(cs.n_ports):
            if cs.evs_connected[j] is not None:
                action_mask[i * cs.n_ports + j] = 1.0
    return action_mask


def run_single_replay(env, model, max_ep_len, scale, state_mean, state_std, device):
    model.eval()

    state, _ = env.reset()
    states = torch.from_numpy(pad_state(state)).reshape(1, STATE_DIM).to(device=device, dtype=torch.float32)
    graph_states = [graph_data_to_dict(PST_V2G_ProfitMaxGNN_state(env))]
    actions = torch.zeros((0, ACT_DIM), device=device, dtype=torch.float32)
    initial_action_mask = build_current_action_mask(env)
    actions_mask = torch.from_numpy(initial_action_mask).to(device=device, dtype=torch.float32).reshape(1, ACT_DIM)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(0, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, ACT_DIM), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            actions_mask.to(dtype=torch.float32),
            config=env.config,
            graph_states=graph_states,
        )

        actions[-1] = action
        action_to_env = action.detach().cpu().numpy()[: env.action_space.shape[0]]
        next_state, reward, done, truncated, stats = env.step(action_to_env)

        action_mask = np.zeros(ACT_DIM, dtype=np.float32)
        orig_mask = stats["action_mask"]
        action_mask[: len(orig_mask)] = orig_mask
        action_mask_torch = torch.from_numpy(action_mask).to(device=device).reshape(1, ACT_DIM)
        actions_mask = torch.cat([actions_mask, action_mask_torch], dim=0)

        states = torch.cat(
            [states, torch.from_numpy(pad_state(next_state)).reshape(1, STATE_DIM).to(device=device, dtype=torch.float32)],
            dim=0,
        )
        graph_states.append(graph_data_to_dict(PST_V2G_ProfitMaxGNN_state(env)))
        rewards[-1] = reward
        pred_return = target_return[0, -1] - (reward / scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )

        if done:
            total_reward = float(stats.get("total_reward", 0.0))
            n_cs = max(int(getattr(env, "cs", 1)), 1)
            sim_steps = max(int(getattr(env, "simulation_length", 1)), 1)
            stats = dict(stats)
            stats["reward_per_cs"] = total_reward / n_cs
            stats["reward_per_cs_step"] = total_reward / (n_cs * sim_steps)
            return stats

    return stats


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
        sem = std / math.sqrt(len(series)) if len(series) > 1 else 0.0
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved GNN-DT checkpoint on a replay directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (e.g. model.best).")
    parser.add_argument("--replay_dir", required=True, help="Directory containing replay_*.pkl files.")
    parser.add_argument("--suite_name", required=True, help="Short name for this replay suite, used in output CSVs.")
    parser.add_argument("--output_dir", required=True, help="Directory for per-replay and summary CSVs.")
    parser.add_argument("--config_file", default=None, help="Optional config file. Defaults to vars.yaml's config_file.")
    parser.add_argument("--vars_yaml", default=None, help="Optional vars.yaml path. Defaults to checkpoint sibling vars.yaml.")
    parser.add_argument("--device", default=None, help="cuda / cpu. Defaults to auto.")
    parser.add_argument("--max_replays", type=int, default=None, help="Optional cap on number of replay files.")
    parser.add_argument("--exp_id", default=None, help="Optional experiment id for output tables.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    run_dir = checkpoint_path.parent
    vars_path = Path(args.vars_yaml).resolve() if args.vars_yaml else run_dir / "vars.yaml"
    run_vars = load_yaml(vars_path)
    config_path = Path(args.config_file).resolve() if args.config_file else Path("./config_files") / run_vars["config_file"]
    env_config = load_yaml(config_path)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(run_vars, env_config, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    state_mean, state_std = get_state_norm(
        run_dir=run_dir,
        state_normalization=run_vars.get("state_normalization", False),
        device=device,
    )

    replay_dir = Path(args.replay_dir).resolve()
    replay_files = sorted(replay_dir.iterdir())
    replay_files = [path for path in replay_files if path.is_file() and path.suffix == ".pkl"]
    if args.max_replays is not None:
        replay_files = replay_files[: args.max_replays]
    if not replay_files:
        raise FileNotFoundError(f"No replay .pkl files found in {replay_dir}")

    rows = []
    for replay_file in replay_files:
        env = EV2Gym(
            config_file=str(config_path),
            load_from_replay_path=str(replay_file),
            state_function=PST_V2G_ProfitMax_state,
            reward_function=PST_V2G_ProfitMax_reward,
        )
        stats = run_single_replay(
            env=env,
            model=model,
            max_ep_len=env_config["simulation_length"],
            scale=1,
            state_mean=state_mean,
            state_std=state_std,
            device=device,
        )
        row = {
            "exp_id": args.exp_id or run_dir.name,
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
    per_replay_path = output_dir / f"{args.suite_name}.per_replay.csv"
    per_replay_df.to_csv(per_replay_path, index=False)

    summary_df = summarize_results(per_replay_df, args.suite_name)
    summary_path = output_dir / f"{args.suite_name}.summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Per-replay metrics saved to {per_replay_path}")
    print(f"Summary metrics saved to {summary_path}")


if __name__ == "__main__":
    main()
