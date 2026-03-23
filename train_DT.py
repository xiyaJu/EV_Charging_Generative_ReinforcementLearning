# import gymnasium as gym
import numpy as np
import torch

import csv
import json

import argparse
import pickle
import random
import sys
import os
import yaml

from DT.evaluation.evaluate_episodes import evaluate_episode_rtg
from DT.training.act_trainer import ActTrainer
from DT.training.seq_trainer import SequenceTrainer

from DT.models.gnn_decision_transformer import GNN_DecisionTransformer
from ev2gym.models.ev2gym_env import EV2Gym
from utils import PST_V2G_ProfitMax_reward, PST_V2G_ProfitMaxGNN_state, PST_V2G_ProfitMax_state


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(vars):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env_name = vars['env']
    dataset_path = f'./trajectories/{vars["dataset_path"]}'

    run_name = vars['name']
    exp_prefix = f'gnn_dt-{run_name}'

    # seed everything
    seed = vars['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    scale = 1   # 奖励缩放因子（后续RTG归一化用）
    env_targets = [0]  # evaluation conditioning targets


    config_path = f'./config_files/{vars["config_file"]}'
    config = yaml.load(open(config_path, 'r'),
                       Loader=yaml.FullLoader)

    number_of_charging_stations = config["number_of_charging_stations"]
    steps = config["simulation_length"] 

    reward_function = PST_V2G_ProfitMax_reward
    state_function = PST_V2G_ProfitMax_state

    env = EV2Gym(config_file=config_path,
                 state_function=state_function,
                 reward_function=reward_function,
                 )

    # 修改：统一维度，以便适配不同规模的环境
    state_dim = 1500 # 增加到 1500，以适配 250 个充电桩（约 1257 维）
    act_dim = 250    # 保持 250
    print(
        f'Fixed dimensions - Observation space: {state_dim}, action space: {act_dim}')

    max_ep_len = steps

    save_path = f'./saved_models/{exp_prefix}/'
    # create folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save the vars to the save path as yaml
    with open(f'{save_path}/vars.yaml', 'w') as f:
        yaml.dump(vars, f)

    # ---- local eval/train log files ----
    csv_log_path = os.path.join(save_path, "metrics.csv")
    jsonl_log_path = os.path.join(save_path, "metrics.jsonl")

    # write csv header once
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iter", "train_loss_mean", "action_error", "test_total_reward", "best_total_reward"])

    if "gz" in dataset_path:
        import gzip
        with gzip.open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    else:
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    
    # save all path information into separate lists
    mode = vars.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
            
        # 修改：在计算 state_mean 前对 observations 进行 Padding
        orig_obs = path['observations']
        padded_obs = np.zeros((orig_obs.shape[0], state_dim))
        padded_obs[:, :orig_obs.shape[1]] = orig_obs
        states.append(padded_obs)
        
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens = np.array(traj_lens)
    all_rewards = np.concatenate([traj['rewards'] for traj in trajectories])
    print(f"奖励范围: [{all_rewards.min()}, {all_rewards.max()}]")
    print(f"奖励均值: {all_rewards.mean()}, 标准差: {all_rewards.std()}")
    print(f"单条轨迹总回报范围: [{min(returns)}, {max(returns)}]")
    # print(trajectories[0])
    # exit()
# ====== 新增：状态和动作的范围 ======
    all_states = np.concatenate([traj['observations'] for traj in trajectories])
    all_actions = np.concatenate([traj['actions'] for traj in trajectories])

    print(f"状态范围: [{all_states.min()}, {all_states.max()}]")
    print(f"状态均值: {all_states.mean()}, 标准差: {all_states.std()}")

    print(f"动作范围: [{all_actions.min()}, {all_actions.max()}]")
    print(f"动作均值: {all_actions.mean()}, 标准差: {all_actions.std()}")

    # 非零状态值的范围（排除padding的0）
    nonzero_states = all_states[all_states != 0]
    if len(nonzero_states) > 0:
        print(f"非零状态范围: [{nonzero_states.min()}, {nonzero_states.max()}]")
        print(f"非零状态均值: {nonzero_states.mean()}, 标准差: {nonzero_states.std()}")

    # 非零动作值的范围
    nonzero_actions = all_actions[all_actions != 0]
    if len(nonzero_actions) > 0:
        print(f"非零动作范围: [{nonzero_actions.min()}, {nonzero_actions.max()}]")
        print(f"非零动作均值: {nonzero_actions.mean()}, 标准差: {nonzero_actions.std()}")
    # Initialize eval_envs from replays
    # eval_replay_path = vars['eval_replay_path']
    # eval_replays = os.listdir(eval_replay_path)
    # eval_envs = []
    # print(f'Loading evaluation replays from {eval_replay_path}')
    # for replay in eval_replays:
    #     eval_env = EV2Gym(config_file=config_path,
    #                       load_from_replay_path=eval_replay_path + replay,
    #                       state_function=state_function,
    #                       reward_function=reward_function,
    #                       )
                
    #     eval_envs.append(eval_env)
        
    #     if len(eval_envs) >= vars['num_eval_episodes']:
    #         break

    # print(f'Loaded {len(eval_envs)} evaluation replays')
    eval_replay_path = vars['eval_replay_path']
    eval_envs = []
    if os.path.exists(eval_replay_path):
        eval_replays = os.listdir(eval_replay_path)
        print(f'Loading evaluation replays from {eval_replay_path}')
        for replay in eval_replays:
            eval_env = EV2Gym(config_file=config_path,
                              load_from_replay_path=eval_replay_path + replay,
                              state_function=state_function,
                              reward_function=reward_function,
                              )
            eval_envs.append(eval_env)
            if len(eval_envs) >= vars['num_eval_episodes']:
                break
        print(f'Loaded {len(eval_envs)} evaluation replays')
    else:
        print(f'No eval replay path found at {eval_replay_path}, skipping evaluation.')


    # used for input normalization
    states = np.concatenate(states, axis=0)

    state_mean, state_std = np.mean(
        states, axis=0), np.std(states, axis=0) + 1e-6

    # save state mean and std
    np.save(f'{save_path}/state_mean.npy', state_mean)
    np.save(f'{save_path}/state_std.npy', state_std)
    np.savetxt(f'{save_path}/state_mean.csv', state_mean.reshape(1, -1), 
           delimiter=',', header='mean_values', comments='')
    np.savetxt(f'{save_path}/state_std.csv', state_std.reshape(1, -1), 
           delimiter=',', header='std_values', comments='')

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(
        f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = vars['K']
    batch_size = vars['batch_size']
    num_eval_episodes = vars['num_eval_episodes']
    pct_traj = vars.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        # 从筛选后的轨迹中（共 num_trajectories 条）随机选 batch_size 条的索引。按 p_sample 加权采样，长轨迹被选中的概率更高。replace=True 允许重复选取
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )
        

        s, a, r, d, rtg, timesteps, mask, action_mask = [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # 获取当前轨迹的真实维度
            orig_state_dim = traj['observations'].shape[1]
            orig_act_dim = traj['actions'].shape[1]

            # 1. 提取原始切片
            curr_s = traj['observations'][si:si + max_len]
            curr_a = traj['actions'][si:si + max_len]
            curr_am = traj['action_mask'][si:si + max_len]

            # 2. 对特征维度进行 Padding (适配固定 state_dim 和 act_dim)
            # 状态维度填充
            si_state = np.zeros((curr_s.shape[0], state_dim))
            si_state[:, :orig_state_dim] = curr_s
            s.append(si_state.reshape(1, -1, state_dim))

            # 动作维度填充
            si_action = np.zeros((curr_a.shape[0], act_dim))
            si_action[:, :orig_act_dim] = curr_a
            a.append(si_action.reshape(1, -1, act_dim))

            # 奖励
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            
            # Action Mask 维度填充
            si_am = np.zeros((curr_am.shape[0], act_dim))
            si_am[:, :orig_act_dim] = curr_am
            action_mask.append(si_am.reshape(1, -1, act_dim))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >=
                          max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[
                       :s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1],
                                         np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len -
                                   tlen, state_dim)), s[-1]], axis=1)
            # s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len -
                                   tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len -
                                   tlen, 1)), r[-1]], axis=1)
            action_mask[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, act_dim)), action_mask[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen))
                                   * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)),
                                     rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate(
                [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        action_mask = torch.from_numpy(np.concatenate(action_mask, axis=0)).to(
            dtype=torch.float32, device=device)

        return s, a, r, d, rtg, timesteps, mask, action_mask

    def eval_episodes(target_rew):
        def fn(model):
            with torch.no_grad():
                stats = evaluate_episode_rtg(
                    eval_envs,
                    exp_prefix,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew/scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    n_test_episodes=num_eval_episodes,
                    config_file=config_path,
                )
                
            return stats

        return fn

    model = GNN_DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=vars['embed_dim'],
        n_layer=vars['n_layer'],
        n_head=vars['n_head'],
        n_inner=4*vars['embed_dim'],
        activation_function=vars['activation_function'],
        n_positions=1024,
        resid_pdrop=vars['dropout'],
        attn_pdrop=vars['dropout'],
        action_tanh=True,
        action_masking=vars['action_masking'],
        fx_node_sizes={'ev': 5, 'cs': 4, 'tr': 2, 'env': 6},
        feature_dim=vars['feature_dim'],
        GNN_hidden_dim=vars['GNN_hidden_dim'],
        num_gcn_layers=vars['num_gcn_layers'],
        config=config,
        device=device,
    )

    model = model.to(device=device)
    
    if vars.get('resume_model') is not None:
        model.load_state_dict(torch.load(vars['resume_model'], map_location=device))
        print(f'Loaded pretrained model from {vars["resume_model"]}')

    warmup_steps = vars['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=vars['learning_rate'],
        weight_decay=vars['weight_decay'],
    )

    max_iters = vars['max_iters']
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda max_iters: min((max_iters+1)/warmup_steps, 1)
    )

    if vars['action_masking']:
        def loss_fn(s_hat, a_hat, r_hat, s, a, r, a_masks): return torch.mean(
            ((a_hat - a)**2) * a_masks
        )

    else:
        def loss_fn(s_hat, a_hat, r_hat, s, a, r, _): return torch.mean(
            (a_hat - a)**2),

    if num_eval_episodes > 0 and len(eval_envs) > 0:
        eval_fns = [eval_episodes(tar) for tar in env_targets]
    else:
        eval_fns = []

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=loss_fn,
        eval_fns=eval_fns,
    )


    num_steps_per_iter = vars['num_steps_per_iter']
    # num_steps_per_iter = int(1000/batch_size)
    # if num_steps_per_iter == 0:
    #     num_steps_per_iter = vars['batch_size']

    best_reward = -np.inf  # 顺手也修掉 np.Inf
    best_iter = None
    for iter in range(vars['max_iters']):
        outputs = trainer.train_iteration(
            num_steps=num_steps_per_iter, iter_num=iter+1, print_logs=True
        )

        # 先把 best 写好（有 eval 才更新）
        if 'test/total_reward' in outputs:
            if outputs['test/total_reward'] > best_reward:
                best_reward = outputs['test/total_reward']
                best_iter = iter + 1
                torch.save(model.state_dict(), f'{save_path}/model.best')
                print(f' Saving best model with reward {best_reward} at path {save_path}/model.best')

        # 不管有没有 eval，都写 best 字段，方便统一记录
        outputs['best'] = best_reward

        # ---- save metrics locally (ALWAYS) ----
        iter_num = iter + 1

        # 用你这份代码真实存在的 key
        train_loss = outputs.get("training/train_loss_mean", None)
        test_total_reward = outputs.get("test/total_reward", None)  # 没有 eval 就是 None

        with open(csv_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            action_error = outputs.get("training/action_error", None)
            writer.writerow([iter_num, train_loss, action_error, test_total_reward, best_reward])


        with open(jsonl_log_path, "a") as f:
            f.write(json.dumps({"iter": iter_num, **outputs}) + "\n")



    torch.save(model.state_dict(), f'{save_path}/model.last')
    import matplotlib.pyplot as plt
    # ---- plot curves ----
    try:
        iters, rewards = [], []
        with open(csv_log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["test_total_reward"] not in [None, "", "None"]:
                    iters.append(int(row["iter"]))
                    rewards.append(float(row["test_total_reward"]))

        if len(iters) > 0:
            plt.figure()
            plt.plot(iters, rewards)
            plt.xlabel("Iteration")
            plt.ylabel("Test Total Reward")
            plt.title("Evaluation Reward vs Iteration")
            plt.grid(True)
            plt.savefig(os.path.join(save_path, "eval_reward_curve.png"), dpi=200)
            plt.close()
            print(f"Saved eval curve to {os.path.join(save_path, 'eval_reward_curve.png')}")
        else:
            print("No evaluation rewards found to plot (maybe num_eval_episodes=0 or trainer didn't return test metrics).")
    except Exception as e:
        print(f"Plotting failed: {e}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PST_V2G_ProfixMax')
    parser.add_argument('--name', type=str, default='1')
    parser.add_argument('--seed', type=int, default=42)
    # trajectory path
    parser.add_argument('--dataset_path', type=str, default='PST_V2G_ProfixMax_25_random_25_10.pkl.gz')
    parser.add_argument('--config_file', type=str, default="PST_V2G_ProfixMax_25.yaml")
    # normal for standard setting, delayed for sparse
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_iters', type=int, default=5)
    parser.add_argument('--num_steps_per_iter', type=int, default=10)  # 1000


    parser.add_argument('--num_eval_episodes', type=int, default=3)
    parser.add_argument('--eval_replay_path', type=str, default="./eval_replays/PST_V2G_ProfixMax_25_optimal_25_50/")
    # New parameters
    parser.add_argument('--action_masking',
                        type=bool,
                        default=True)

    # GNN_DT parameters
    parser.add_argument('--feature_dim', type=int, default=8)
    parser.add_argument('--GNN_hidden_dim', type=int, default=32)
    parser.add_argument('--num_gcn_layers', type=int, default=3)
    parser.add_argument('--act_GNN_hidden_dim', type=int, default=32)
    parser.add_argument('--num_act_gcn_layers', type=int, default=3)
    parser.add_argument('--gnn_type', type=str, default='GCN')
    parser.add_argument('--resume_model', type=str, default=None,help='Path to a pretrained model to resume training from')
    args = parser.parse_args()

    experiment(vars=vars(args))
