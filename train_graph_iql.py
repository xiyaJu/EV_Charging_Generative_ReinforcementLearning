import argparse
import csv
import gzip
import json
import os
import pickle
import random

import numpy as np
import torch
import yaml

from DT.evaluation.evaluate_episodes import evaluate_episode_rtg
from DT.models.graph_iql import GraphIQLPolicy
from ev2gym.models.ev2gym_env import EV2Gym
from utils import PST_V2G_ProfitMax_reward, PST_V2G_ProfitMax_state


def load_trajectories(dataset_path):
    if dataset_path.endswith(".gz"):
        with gzip.open(dataset_path, "rb") as f:
            return pickle.load(f)
    with open(dataset_path, "rb") as f:
        return pickle.load(f)


def build_transition_bank(trajectories, act_dim, normalize_reward_by_ncs=False):
    transitions = []
    for traj in trajectories:
        rewards = traj["rewards"]
        dones = traj.get("dones", traj.get("terminals"))
        graph_obs = traj.get("graph_observations", traj["observations"])
        n_cs = max(int(traj["actions"].shape[1]), 1)
        reward_scale = float(n_cs) if normalize_reward_by_ncs else 1.0

        for t in range(len(rewards)):
            action = np.zeros(act_dim, dtype=np.float32)
            action[: traj["actions"].shape[1]] = traj["actions"][t]

            action_mask = np.zeros(act_dim, dtype=np.float32)
            action_mask[: traj["action_mask"].shape[1]] = traj["action_mask"][t]

            next_graph = graph_obs[t + 1] if t + 1 < len(graph_obs) else None
            done_value = float(dones[t]) if dones is not None else float(t == len(rewards) - 1)

            transitions.append(
                {
                    "graph_state": graph_obs[t],
                    "next_graph_state": next_graph,
                    "action": action,
                    "action_mask": action_mask,
                    "reward": float(rewards[t]) / reward_scale,
                    "done": done_value,
                }
            )
    return transitions


def sample_transition_batch(transitions, batch_size, device):
    batch = random.sample(transitions, batch_size)
    graph_states = [item["graph_state"] for item in batch]
    next_graph_states = [item["next_graph_state"] for item in batch]
    actions = torch.from_numpy(np.stack([item["action"] for item in batch], axis=0)).to(device=device, dtype=torch.float32)
    action_masks = torch.from_numpy(np.stack([item["action_mask"] for item in batch], axis=0)).to(device=device, dtype=torch.float32)
    rewards = torch.from_numpy(np.asarray([item["reward"] for item in batch], dtype=np.float32)).to(device=device)
    dones = torch.from_numpy(np.asarray([item["done"] for item in batch], dtype=np.float32)).to(device=device)
    return graph_states, next_graph_states, actions, action_masks, rewards, dones


def asymmetric_l2_loss(diff, tau):
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="graph_iql")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="PST_V2G_ProfixMax_100.yaml")
    parser.add_argument("--eval_replay_path", type=str, default="")
    parser.add_argument("--num_eval_episodes", type=int, default=0)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--feature_dim", type=int, default=8)
    parser.add_argument("--GNN_hidden_dim", type=int, default=32)
    parser.add_argument("--num_gcn_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_iters", type=int, default=150)
    parser.add_argument("--num_updates_per_iter", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--adv_clip", type=float, default=20.0)
    parser.add_argument("--best_metric", type=str, default="test/reward_per_cs_step")
    parser.add_argument("--plot_metric", type=str, default=None)
    parser.add_argument("--normalize_reward_by_ncs", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join("./config_files", args.config_file)
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    steps = int(config["simulation_length"])
    act_dim = 250
    state_dim = 1500

    trajectories = load_trajectories(os.path.join("./trajectories", args.dataset_path))
    transitions = build_transition_bank(
        trajectories=trajectories,
        act_dim=act_dim,
        normalize_reward_by_ncs=args.normalize_reward_by_ncs,
    )
    print(f"Loaded {len(transitions)} transitions from {len(trajectories)} trajectories")

    save_path = os.path.join("./saved_models", f"graph_iql-{args.name}")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "vars.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    csv_log_path = os.path.join(save_path, "metrics.csv")
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "iter",
                    "critic_loss",
                    "value_loss",
                    "actor_loss",
                    "test_total_reward",
                    "test_reward_per_cs",
                    "test_reward_per_cs_step",
                    "best_metric_value",
                ]
            )

    eval_envs = []
    if args.eval_replay_path and os.path.exists(args.eval_replay_path):
        replay_files = sorted(os.listdir(args.eval_replay_path))
        for replay in replay_files:
            eval_env = EV2Gym(
                config_file=config_path,
                load_from_replay_path=os.path.join(args.eval_replay_path, replay),
                state_function=PST_V2G_ProfitMax_state,
                reward_function=PST_V2G_ProfitMax_reward,
            )
            eval_envs.append(eval_env)
            if len(eval_envs) >= args.num_eval_episodes:
                break
        print(f"Loaded {len(eval_envs)} eval replays")

    policy = GraphIQLPolicy(
        state_dim=state_dim,
        act_dim=act_dim,
        fx_node_sizes={"ev": 5, "cs": 4, "tr": 2, "env": 6},
        feature_dim=args.feature_dim,
        hidden_size=args.embed_dim,
        gnn_hidden_dim=args.GNN_hidden_dim,
        num_gcn_layers=args.num_gcn_layers,
        device=device,
    ).to(device=device)

    actor_optimizer = torch.optim.AdamW(policy.actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    critic_optimizer = torch.optim.AdamW(
        list(policy.q1.parameters()) + list(policy.q2.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    value_optimizer = torch.optim.AdamW(policy.value.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def evaluate_policy():
        if not eval_envs:
            return {}
        with torch.no_grad():
            return evaluate_episode_rtg(
                eval_envs,
                f"graph_iql-{args.name}",
                state_dim,
                act_dim,
                policy,
                max_ep_len=steps,
                scale=1,
                target_return=0,
                device=device,
                state_mean=np.zeros(state_dim, dtype=np.float32),
                state_std=np.ones(state_dim, dtype=np.float32),
                n_test_episodes=len(eval_envs),
                config_file=config_path,
            )

    best_metric_value = -np.inf
    for iter_idx in range(1, args.max_iters + 1):
        critic_losses, value_losses, actor_losses = [], [], []
        policy.train()

        for _ in range(args.num_updates_per_iter):
            graph_states, next_graph_states, actions, action_masks, rewards, dones = sample_transition_batch(
                transitions=transitions,
                batch_size=args.batch_size,
                device=device,
            )

            with torch.no_grad():
                next_v = policy.value(next_graph_states, config=config)
                q_target = rewards + args.gamma * (1.0 - dones) * next_v

            q1 = policy.q1(graph_states, actions, config=config)
            q2 = policy.q2(graph_states, actions, config=config)
            critic_loss = ((q1 - q_target) ** 2 + (q2 - q_target) ** 2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            with torch.no_grad():
                q_detached = torch.minimum(
                    policy.q1(graph_states, actions, config=config),
                    policy.q2(graph_states, actions, config=config),
                )
            v = policy.value(graph_states, config=config)
            value_loss = asymmetric_l2_loss(q_detached - v, args.expectile)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            with torch.no_grad():
                advantages = q_detached - policy.value(graph_states, config=config)
                weights = torch.exp(args.temperature * advantages).clamp(max=args.adv_clip)

            pred_actions = policy.actor(graph_states, config=config)
            per_sample_bc = (((pred_actions - actions) ** 2) * action_masks).mean(dim=-1)
            actor_loss = (weights * per_sample_bc).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_losses.append(float(critic_loss.detach().cpu().item()))
            value_losses.append(float(value_loss.detach().cpu().item()))
            actor_losses.append(float(actor_loss.detach().cpu().item()))

        eval_outputs = evaluate_policy()
        metric_key = args.best_metric
        metric_value = eval_outputs.get(metric_key, -np.inf)
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            torch.save(policy.state_dict(), os.path.join(save_path, "model.best"))

        with open(csv_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    iter_idx,
                    np.mean(critic_losses),
                    np.mean(value_losses),
                    np.mean(actor_losses),
                    eval_outputs.get("test/total_reward"),
                    eval_outputs.get("test/reward_per_cs"),
                    eval_outputs.get("test/reward_per_cs_step"),
                    best_metric_value,
                ]
            )

        print("- - " * 20)
        print(f"Iteration {iter_idx}")
        print(f"training/critic_loss_mean: {np.mean(critic_losses):.6f}")
        print(f"training/value_loss_mean: {np.mean(value_losses):.6f}")
        print(f"training/actor_loss_mean: {np.mean(actor_losses):.6f}")
        for key, value in eval_outputs.items():
            print(f"{key}: {value}")
        print(f"best_metric_value ({metric_key}): {best_metric_value}")
        print("=" * 80)

    torch.save(policy.state_dict(), os.path.join(save_path, "model.last"))


if __name__ == "__main__":
    main()
