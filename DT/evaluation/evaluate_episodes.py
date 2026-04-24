import numpy as np
import torch
from ev2gym.models import ev2gym_env
import os
import tqdm

from ev2gym.models.ev2gym_env import EV2Gym
from utils import PST_V2G_ProfitMax_reward, PST_V2G_ProfitMaxGNN_state, PST_V2G_ProfitMax_state
from utils import graph_data_to_dict

from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives, profit_maximization, SimpleReward


def _get_state_normalization_tensors(state_mean, state_std, state_dim, device, use_state_norm):
    if use_state_norm and isinstance(state_mean, np.ndarray):
        mean_tensor = torch.from_numpy(state_mean).to(device=device, dtype=torch.float32)
        std_tensor = torch.from_numpy(state_std).to(device=device, dtype=torch.float32)
        return mean_tensor, std_tensor

    if use_state_norm and torch.is_tensor(state_mean):
        return (
            state_mean.to(device=device, dtype=torch.float32),
            state_std.to(device=device, dtype=torch.float32),
        )

    return (
        torch.zeros(state_dim, device=device, dtype=torch.float32),
        torch.ones(state_dim, device=device, dtype=torch.float32),
    )


def _build_stats_summary(test_stats):
    keys_to_keep = [
        'total_reward',
        'reward_per_cs',
        'reward_per_cs_step',
        'total_profits',
        'total_ev_served',
        'total_energy_charged',
        'total_energy_discharged',
        'average_user_satisfaction',
        'min_user_satisfaction',
        'power_tracker_violation',
        'tracking_error',
        'energy_tracking_error',
        'energy_user_satisfaction',
        'min_energy_user_satisfaction',
        'total_transformer_overload',
        'battery_degradation',
        'battery_degradation_calendar',
        'battery_degradation_cycling',
        'total_steps_min_emergency_battery_capacity_violation',
    ]

    stats = {}
    for key in test_stats[0].keys():
        if "opt" in key:
            key_name = "opt/" + key.split("opt_")[1]
            if key.split("opt_")[1] not in keys_to_keep:
                continue
        else:
            if key not in keys_to_keep:
                continue
            key_name = "test/" + key
        stats[key_name] = np.mean([test_stats[i][key]
                                   for i in range(len(test_stats))])
    return stats


def _augment_stats_with_scale_normalization(stats, env):
    augmented = dict(stats)
    n_cs = max(int(getattr(env, "cs", 1)), 1)
    sim_steps = max(int(getattr(env, "simulation_length", 1)), 1)
    total_reward = float(augmented.get("total_reward", 0.0))
    augmented["reward_per_cs"] = total_reward / n_cs
    augmented["reward_per_cs_step"] = total_reward / (n_cs * sim_steps)
    return augmented


def _build_current_action_mask(env, act_dim):
    action_mask = np.zeros(act_dim, dtype=np.float32)
    for i, cs in enumerate(env.charging_stations):
        for j in range(cs.n_ports):
            if cs.evs_connected[j] is not None:
                action_mask[i * cs.n_ports + j] = 1.0
    return action_mask


def evaluate_episode(
        test_env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
        use_state_norm=False,
):

    model.eval()
    model.to(device=device)

    state_mean, state_std = _get_state_normalization_tensors(
        state_mean=state_mean,
        state_std=state_std,
        state_dim=state_dim,
        device=device,
        use_state_norm=use_state_norm,
    )

    test_rewards = []
    test_stats = []

    for test_cycle in tqdm.tqdm(range(len(test_env))):
        env = test_env[test_cycle]
        state, _ = env.reset()

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(
            1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        target_return = torch.tensor(
            target_return, device=device, dtype=torch.float32)

        episode_return, episode_length = 0, 0
        for t in range(max_ep_len):

            # add padding
            actions = torch.cat([actions, torch.zeros(
                (1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return=target_return,
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, truncated, stats = env.step(action)

            cur_state = torch.from_numpy(state).to(
                device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            episode_return += reward
            episode_length += 1

            if done:
                test_stats.append(_augment_stats_with_scale_normalization(stats, env))
                test_rewards.append(episode_return)
                break

    return _build_stats_summary(test_stats)  # , episode_length


def evaluate_episode_rtg(
    test_env,
    exp_prefix,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    scale=1000.,
    state_mean=0.,
    state_std=1.,
    device='cuda',
    target_return=None,
    mode='normal',
    n_test_episodes=10,
    use_state_norm=False,
    **kwargs
):
    model.eval()
    model.to(device=device)

    state_mean, state_std = _get_state_normalization_tensors(
        state_mean=state_mean,
        state_std=state_std,
        state_dim=state_dim,
        device=device,
        use_state_norm=use_state_norm,
    )

    test_rewards = []
    test_stats = []

    # env = test_env

    global_target_return = 0

    for test_cycle in tqdm.tqdm(range(len(test_env))):
        env = test_env[test_cycle]
        state, _ = env.reset()

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        
        # 修改：Padding 到固定的 state_dim，增加安全检查
        padded_state = np.zeros(state_dim)
        if len(state) > state_dim:
             print(f"Warning: state dimension {len(state)} exceeds padded state_dim {state_dim}!")
             state = state[:state_dim]
        padded_state[:len(state)] = state
        states = torch.from_numpy(padded_state).reshape(
            1, state_dim).to(device=device, dtype=torch.float32)
        graph_states = [graph_data_to_dict(PST_V2G_ProfitMaxGNN_state(env))]
            
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        initial_action_mask = _build_current_action_mask(env, act_dim)
        actions_mask = torch.from_numpy(initial_action_mask).to(
            device=device, dtype=torch.float32).reshape(1, act_dim)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        ep_return = global_target_return
        target_return = torch.tensor(
            ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(
            0, device=device, dtype=torch.long).reshape(1, 1)

        episode_return, episode_length = 0, 0
        for t in range(max_ep_len):

            # add padding
            actions = torch.cat([actions, torch.zeros(
                (1, act_dim), device=device)], dim=0)
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
            action_to_env = action.detach().cpu().numpy()[:env.action_space.shape[0]]

            state, reward, done, truncated, stats = env.step(action_to_env)

            # 填充 action_mask
            action_mask = np.zeros(act_dim)
            orig_am = stats['action_mask']
            action_mask[:len(orig_am)] = orig_am
            action_mask_torch = torch.from_numpy(action_mask).to(
                device=device).reshape(1, act_dim)
            actions_mask = torch.cat([actions_mask, action_mask_torch], dim=0)

            # 填充 state
            padded_state = np.zeros(state_dim)
            if len(state) > state_dim:
                 state = state[:state_dim]
            padded_state[:len(state)] = state
            cur_state = torch.from_numpy(padded_state).to(
                device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            graph_states.append(graph_data_to_dict(PST_V2G_ProfitMaxGNN_state(env)))
            rewards[-1] = reward

            if mode != 'delayed':
                pred_return = target_return[0, -1] - (reward/scale)
            else:
                pred_return = target_return[0, -1]
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                test_stats.append(_augment_stats_with_scale_normalization(stats, env))
                test_rewards.append(episode_return)
                break

    return _build_stats_summary(test_stats)  # , episode_length


def evaluate_episode_rtg_from_replays(
        env,
        model,
        max_ep_len=1000,
        scale=1.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
):

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model.eval()
    model.to(device=device)

    # state_mean = torch.from_numpy(state_mean).to(device=device)
    # state_std = torch.from_numpy(state_std).to(device=device)

    state_mean = torch.zeros(state_dim, device=device)
    state_std = torch.ones(state_dim, device=device)

    test_rewards = []
    test_stats = []

    global_target_return = 0

    state, _ = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(
        1, state_dim).to(device=device, dtype=torch.float32)
    graph_states = [graph_data_to_dict(PST_V2G_ProfitMaxGNN_state(env))]
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    initial_action_mask = _build_current_action_mask(env, act_dim)
    actions_mask = torch.from_numpy(initial_action_mask).to(
        device=device, dtype=torch.float32).reshape(1, act_dim)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = global_target_return
    target_return = torch.tensor(
        ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(
        0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros(
            (1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            actions_mask.to(dtype=torch.float32),
            graph_states=graph_states,
        )

        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, truncated, stats = env.step(action)

        action_mask = torch.from_numpy(stats['action_mask']).to(
            device=device).reshape(1, act_dim)
        actions_mask = torch.cat([actions_mask, action_mask], dim=0)

        cur_state = torch.from_numpy(state).to(
            device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        graph_states.append(graph_data_to_dict(PST_V2G_ProfitMaxGNN_state(env)))
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0, -1] - (reward/scale)
        else:
            pred_return = target_return[0, -1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            test_stats.append(stats)
            test_rewards.append(episode_return)

    return stats, episode_return
