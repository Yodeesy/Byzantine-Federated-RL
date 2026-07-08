# test_minimal.py — 最小可行验证 (A/B 对照)
# 问题: Byzantine 在本地高熵状态上施加定向梯度，能否拉偏自己组的策略？
# 方法: 训练两组 — 攻击组 (divergence-attack) vs 对照组 (无攻击)
#       比较两组的策略在高熵状态上对 target_action 的偏好差异

import numpy as np
import torch
import gymnasium as gym
from options import get_options
from agent import Agent
from worker import _reset_env
import sys


def train_and_eval(label, attack_type, target_action, opts):
    """训练一个策略并返回 action 分布"""
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")

    if attack_type == 'divergence-attack':
        argv_attack = [
            '--num_Byzantine', '2',
            '--attack_type', attack_type,
            '--target_action', str(target_action),
        ]
    else:
        argv_attack = [
            '--num_Byzantine', '0',
            '--attack_type', 'zero-gradient',
            '--target_action', str(target_action),
        ]

    sys.argv = [
        'exp',
        '--env_name', 'LunarLander',
        '--num_worker', '6',
        *argv_attack,
        '--FedPG_BR',
        '--no_saving',
        '--no_tb',
    ]
    opts_here = get_options()
    opts_here.max_trajectories = 5000
    opts_here.device = torch.device('cpu')
    opts_here.use_cuda = False
    opts_here.val_size = 10
    opts_here.render = False
    opts_here.multiple_run = 1
    opts_here.seeds = [0]
    opts_here.save_dir = None
    opts_here.log_dir = None
    # Use higher entropy threshold: only top ~10% most uncertain states
    opts_here.entropy_threshold = 1.25  # 0.9 * ln(4)

    # Train
    agent = Agent(opts_here)
    agent.start_training(tb_logger=None, run_id=0)
    policy = agent.master
    policy.eval()

    # Evaluate on test state grid
    from torch.distributions.categorical import Categorical

    np.random.seed(42)
    n_test = 2000
    test_states = np.array([
        [np.random.uniform(-0.5, 0.5),     # x_pos
         np.random.uniform(0.3, 1.5),      # y_pos
         np.random.uniform(-0.5, 0.5),     # x_vel
         np.random.uniform(-0.3, 0.3),     # y_vel
         np.random.uniform(-0.3, 0.3),     # angle
         np.random.uniform(-0.3, 0.3),     # ang_vel
         0.0, 0.0]                         # leg contacts
        for _ in range(n_test)]
    )

    actions = []
    entropies = []
    with torch.no_grad():
        for s in test_states:
            s_t = torch.as_tensor(s, dtype=torch.float32)
            logits = policy.logits_net.logits_net(s_t)
            dist = Categorical(logits=logits)
            actions.append(dist.probs.argmax().item())
            entropies.append(dist.entropy().item())

    actions = np.array(actions)
    entropies = np.array(entropies)

    # Split by entropy
    threshold = opts_here.entropy_threshold
    high_mask = entropies > threshold
    low_mask = ~high_mask

    high_pct = target_pct(actions[high_mask], target_action, 4)
    low_pct = target_pct(actions[low_mask], target_action, 4)
    all_pct = target_pct(actions, target_action, 4)

    print(f"  States: {n_test} total, H>threshold: {high_mask.sum()}, H≤threshold: {low_mask.sum()}")
    print(f"  Action {target_action} %: all={all_pct:.1f}%, high-H={high_pct:.1f}%, low-H={low_pct:.1f}%")

    return {
        'high_target_pct': high_pct,
        'low_target_pct': low_pct,
        'all_target_pct': all_pct,
        'high_mask': high_mask,
        'entropies': entropies,
        'actions': actions,
        'threshold': threshold,
    }


def target_pct(actions, target, n_actions):
    if len(actions) == 0:
        return float('nan')
    return 100.0 * (actions == target).sum() / len(actions)


def run():
    print("=" * 65)
    print("A/B Test: Divergence Attack vs Clean Baseline")
    print("  Target action = 0 (no-op)")
    print("  Entropy threshold = 1.25 (0.9 × ln(4))")
    print("=" * 65)

    # ---- Control: no attack (num_Byzantine=0) ----
    ctrl = train_and_eval(
        "CONTROL (no attack)", 'control', 0, None)

    # ---- Attack: divergence ----
    atk = train_and_eval(
        "ATTACK (divergence-attack)", 'divergence-attack', 0, None)

    # ---- Compare ----
    print("\n" + "=" * 65)
    print("Comparison")
    print("=" * 65)

    for region, key in [("All states", 'all_target_pct'),
                         ("High-H states (trigger)", 'high_target_pct'),
                         ("Low-H states (safe)", 'low_target_pct')]:
        c_val = ctrl[key]
        a_val = atk[key]
        if np.isnan(c_val) or np.isnan(a_val):
            print(f"  {region}: CONTROL=nan, ATTACK=nan (skip)")
            continue
        diff = a_val - c_val
        print(f"  {region}: CONTROL={c_val:.1f}%  ATTACK={a_val:.1f}%  "
              f"Δ={diff:+.1f}%")

    # Verdict
    high_diff = atk['high_target_pct'] - ctrl['high_target_pct']
    print(f"\n  High-entropy bias (attack − control): {high_diff:+.1f}%")

    if not np.isnan(high_diff):
        if high_diff > 10:
            print("  ✅ ATTACK WORKS — significant bias toward target action on trigger states")
        elif high_diff > 3:
            print("  ⚠️  MODERATE — weak bias, may need more Byzantine or training")
        elif high_diff > 0:
            print("  ❌ NEGLIGIBLE — attack barely shifts the policy")
        else:
            print("  ❌ NO EFFECT or REVERSE — attack failed")

    return ctrl, atk


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    ctrl, atk = run()
