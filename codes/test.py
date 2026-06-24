# diagnose_ensemble_agreement.py
# 放在 codes/ 目录下运行
# 目的：在 LunarLander-v2 上找出5组良性策略预测不一致的状态区域
# （即 Ensemble 决策边界，是 Sequential Backdoor 的潜在触发位置）

import numpy as np
import torch
import gymnasium as gym
from collections import Counter
from options import get_options
from agent import Agent
from utils import get_inner_model, env_wrapper
from worker import Worker, _reset_env, _step_env


def run_diagnosis():
    import sys
    sys.argv = [
        'diagnose',
        '--env_name', 'LunarLander-v2',
        '--num_worker', '30',
        '--num_Byzantine', '0',
        '--num_groups', '5',
        '--ensemble',
        '--no_saving',
        '--no_tb',
    ]
    opts = get_options()
    opts.device = torch.device('cpu')
    opts.eval_only = False
    opts.use_cuda = False
    opts.val_size = 5
    opts.val_max_steps = 1000
    opts.render = False
    opts.multiple_run = 1
    opts.seeds = [0]
    opts.save_dir = None
    opts.log_dir = None
    # 快速诊断
    opts.max_trajectories = 5000

    print("=" * 60)
    print("Step 1: 训练5组独立的良性策略")
    print("=" * 60)

    agent = Agent(opts)
    agent.start_training(tb_logger=None, run_id=0)

    print("\n" + "=" * 60)
    print("Step 2: 采集状态样本")
    print("=" * 60)

    env = gym.make('LunarLander-v2')
    collected_states = []
    num_episodes = 30

    for ep in range(num_episodes):
        obs = _reset_env(env)
        done = False
        steps = 0
        while not done and steps < 500:
            collected_states.append(obs.copy())
            obs_wrapped = env_wrapper('LunarLander-v2', obs)
            with torch.no_grad():
                act, _ = agent.group_masters[0].logits_net(
                    torch.as_tensor(obs_wrapped, dtype=torch.float32),
                    sample=False
                )
            obs, _, done, _ = _step_env(env, act)
            steps += 1

    collected_states = np.array(collected_states)
    print(f"收集到 {len(collected_states)} 个状态样本")

    print("\n" + "=" * 60)
    print("Step 3: 统计各状态上K组策略的预测分歧")
    print("=" * 60)

    disagreement_counts = []  # 每个状态上不同动作的个数
    group_actions_list = []   # 每组在各状态上的预测动作

    for obs in collected_states:
        obs_wrapped = env_wrapper('LunarLander-v2', obs)
        obs_tensor = torch.as_tensor(obs_wrapped, dtype=torch.float32)

        actions = []
        with torch.no_grad():
            for k in range(agent.num_groups):
                act, _ = agent.group_masters[k].logits_net(
                    obs_tensor, sample=False
                )
                actions.append(int(act))

        # 不同动作的数量：1=所有组一致，>1=存在分歧
        n_unique = len(set(actions))
        disagreement_counts.append(n_unique)
        group_actions_list.append(actions)

    disagreement_counts = np.array(disagreement_counts)
    group_actions_list = np.array(group_actions_list)

    # 统计
    n_total = len(disagreement_counts)
    n_agree = (disagreement_counts == 1).sum()
    n_disagree = n_total - n_agree
    n_major_disagree = (disagreement_counts >= 3).sum()

    print(f"\n分歧统计：")
    print(f"  总状态数:        {n_total}")
    print(f"  5组完全一致:     {n_agree} ({100*n_agree/n_total:.1f}%)")
    print(f"  存在分歧:        {n_disagree} ({100*n_disagree/n_total:.1f}%)")
    print(f"  分歧≥3种动作:    {n_major_disagree} ({100*n_major_disagree/n_total:.1f}%)")

    # 分歧状态的动作分布
    disagree_mask = disagreement_counts > 1
    disagree_actions = group_actions_list[disagree_mask]
    disagree_states = collected_states[disagree_mask]

    print(f"\n分歧状态上的投票分布（前20个）：")
    print(f"{'状态':>6} {'组0':>4} {'组1':>4} {'组2':>4} {'组3':>4} {'组4':>4} {'多数票':>6} {'共识?':>6}")
    print("-" * 65)
    for i in range(min(20, len(disagree_actions))):
        acts = disagree_actions[i]
        counter = Counter(acts)
        majority_count = counter.most_common(1)[0][1]
        consensus = "✓" if majority_count >= 4 else ("△" if majority_count >= 3 else "✗")
        print(f"{i:>6} {acts[0]:>4} {acts[1]:>4} {acts[2]:>4} {acts[3]:>4} {acts[4]:>4} "
              f"{majority_count:>4}/5  {consensus:>6}")

    # 分歧状态的特征分布
    if len(disagree_states) > 0:
        print(f"\n分歧状态的特征分布 (共 {len(disagree_states)} 个)：")
        state_names = ['x_pos', 'y_pos', 'x_vel', 'y_vel', 'angle', 'ang_vel',
                       'legL_contact', 'legR_contact']
        for i in range(min(disagree_states.shape[1], 8)):
            name = state_names[i] if i < len(state_names) else f'dim_{i}'
            print(f"  {name:>14s}: mean={disagree_states[:, i].mean():.4f}, "
                  f"std={disagree_states[:, i].std():.4f}, "
                  f"range=[{disagree_states[:, i].min():.4f}, {disagree_states[:, i].max():.4f}]")

    print("\n" + "=" * 60)
    print("Step 4: 结论")
    print("=" * 60)

    if n_disagree > 0.05 * n_total:
        print(f"✅ 决策边界存在！{n_disagree}/{n_total} = {100*n_disagree/n_total:.1f}% 的状态上存在分歧")
        print(f"   Sequential Backdoor 在 LunarLander 上有可行性")
        print(f"   建议触发条件：收集分歧状态的特征范围，定义 trigger 区域")
    elif n_disagree > 0:
        print(f"⚠️  分歧稀少 ({n_disagree} 个)，攻击理论上可行但需要精确定位 trigger")
        print(f"   建议：增加训练轮数，或在更有挑战性的状态区域采样")
    else:
        print("❌ 各组策略完全一致，LunarLander 上的 Ensemble 过于稳定")
        print("   建议：尝试不同的随机种子或增加 K 值")

    env.close()
    return disagreement_counts, disagree_states, disagree_actions


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    disagreement_counts, disagree_states, disagree_actions = run_diagnosis()
