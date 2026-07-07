# test_exp1_correlation.py
# Experiment 1: Validate that Δ_local(s) is a proxy for global voting weakness.
#
# Hypothesis: When a single group's Top-1 / Top-2 action probability gap
# Δ_local(s) = π(a₁|s) − π(a₂|s) is small, the global Ensemble voting margin
# n'(s) = ⌊(v(s,x)−v(s,y)−1{y<x})/2⌋ is also small — meaning the state is
# "flippable" by a few malicious groups.
#
# If the correlation holds, the attacker can reliably find vulnerable states
# using only LOCAL information (no cross-group communication needed).

import numpy as np
import torch
import gymnasium as gym
from collections import Counter
from options import get_options
from agent import Agent
from worker import _reset_env, _step_env
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_global_nprime(all_group_actions):
    """Compute n'(s) from Theorem 1.

    n'(s) = ⌊(v(s,x) − v(s,y) − 1{y<x}) / 2⌋

    where x = winning action, y = runner-up action.
    If the vote is unanimous (only one action), n' is effectively infinite
    (no number of malicious groups can flip the result).
    """
    counts = Counter(all_group_actions)
    sorted_acts = counts.most_common()
    if len(sorted_acts) < 2:
        return len(all_group_actions) + 1  # unanimous → unflippable

    x, vx = sorted_acts[0]
    y, vy = sorted_acts[1]
    indicator = 1 if y < x else 0
    n_prime = (vx - vy - indicator) // 2
    return n_prime


def run():
    import sys
    sys.argv = [
        'exp1',
        '--env_name', 'LunarLander-v3',
        '--num_worker', '30',
        '--num_Byzantine', '0',
        '--num_groups', '5',
        '--ensemble',
        '--no_saving',
        '--no_tb',
    ]
    opts = get_options()
    opts.device = torch.device('cpu')
    opts.use_cuda = False
    opts.render = False
    opts.multiple_run = 1
    opts.seeds = [0]
    opts.save_dir = None
    opts.log_dir = None
    opts.max_trajectories = 5000  # shorter for correlation study

    print("=" * 60)
    print("Experiment 1: Δ_local(s) vs n'(s) Correlation")
    print("=" * 60)

    # ---- Train clean ensemble baseline ----
    print("\n[1/3] Training clean Ensemble baseline...")
    agent = Agent(opts)
    agent.start_training(tb_logger=None, run_id=0)

    # ---- Collect states from rollouts ----
    print("\n[2/3] Collecting states and computing Δ_local + n'(s)...")
    env = gym.make('LunarLander-v3')
    all_data = []  # list of (Δ_local, n_prime, state_summary)

    num_episodes = 50
    for ep in range(num_episodes):
        obs = _reset_env(env)
        done = False
        steps = 0
        while not done and steps < 500:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)

            # ---- Compute Δ_local from Group 0's policy ----
            with torch.no_grad():
                logits = agent.group_masters[0].logits_net.logits_net(obs_t)
                probs = torch.softmax(logits, dim=-1)
                top2 = torch.topk(probs, 2)
                delta_local = (top2.values[0] - top2.values[1]).item()

            # ---- Compute n'(s) from all 5 groups ----
            all_actions = []
            with torch.no_grad():
                for k in range(agent.num_groups):
                    logits_k = agent.group_masters[k].logits_net.logits_net(obs_t)
                    act = logits_k.argmax().item()
                    all_actions.append(act)
            n_prime = compute_global_nprime(all_actions)

            all_data.append({
                'delta_local': delta_local,
                'n_prime': n_prime,
                'actions': all_actions,
            })

            # Step forward with Group 0's action
            act0 = all_actions[0]
            obs, _, done, _ = _step_env(env, act0)
            steps += 1

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}, "
                  f"collected {len(all_data)} states so far")

    env.close()
    print(f"  Total states: {len(all_data)}")

    # ---- Analyze correlation ----
    print("\n[3/3] Analyzing correlation...")
    deltas = np.array([d['delta_local'] for d in all_data])
    nprimes = np.array([d['n_prime'] for d in all_data])

    # Bin by Δ_local
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    bin_labels = [f'[{bins[i]:.2f},{bins[i+1]:.2f})' for i in range(len(bins)-1)]

    bin_means = []
    bin_stds = []
    bin_flippable = []  # fraction with n' <= 2 (flippable by ≤2 malicious groups)
    bin_flippable_1 = []  # fraction with n' <= 1 (flippable by 1 group)
    bin_counts = []

    for i in range(len(bins)-1):
        mask = (deltas >= bins[i]) & (deltas < bins[i+1])
        count = mask.sum()
        bin_counts.append(count)
        if count > 0:
            bin_means.append(nprimes[mask].mean())
            bin_stds.append(nprimes[mask].std())
            bin_flippable.append(100 * (nprimes[mask] <= 2).sum() / count)
            bin_flippable_1.append(100 * (nprimes[mask] <= 1).sum() / count)
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            bin_flippable.append(np.nan)
            bin_flippable_1.append(np.nan)

    # ---- Print results table ----
    print(f"\n{'Δ_local bin':<14s} {'Count':>6s}  {'mean n':>7s}  "
          f"{'n≤2%':>7s}  {'n≤1%':>7s}  {'Interpretation':>30s}")
    print("-" * 80)
    for i, label in enumerate(bin_labels):
        n2_str = f"{bin_flippable[i]:.1f}%" if not np.isnan(bin_flippable[i]) else "N/A"
        n1_str = f"{bin_flippable_1[i]:.1f}%" if not np.isnan(bin_flippable_1[i]) else "N/A"
        m_str = f"{bin_means[i]:.2f}" if not np.isnan(bin_means[i]) else "N/A"
        interp = ""
        if not np.isnan(bin_flippable[i]):
            if bin_flippable[i] > 50:
                interp = "← VULNERABLE (most flippable)"
            elif bin_flippable[i] > 20:
                interp = "← borderline"
            else:
                interp = "← robust"
        print(f"{label:<14s} {bin_counts[i]:>6d}  {m_str:>7s}  "
              f"{n2_str:>7s}  {n1_str:>7s}  {interp:>30s}")

    # Overall correlation
    valid = deltas < 1.0  # exclude degenerate cases
    corr = np.corrcoef(deltas[valid], nprimes[valid])[0, 1]
    print(f"\nPearson correlation: r = {corr:.3f}")
    print(f"Spearman rank corr : ρ = {np.corrcoef(np.argsort(deltas[valid]), np.argsort(nprimes[valid]))[0,1]:.3f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Δ_local vs n' scatter
    ax1 = axes[0]
    jitter = np.random.uniform(-0.02, 0.02, len(deltas))
    ax1.scatter(deltas + jitter * 0.01, nprimes + jitter, alpha=0.15, s=8,
                c='#3498db', edgecolors='none')
    # Bin means line
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    valid_bins = ~np.isnan(bin_means)
    ax1.plot(np.array(bin_centers)[valid_bins],
             np.array(bin_means)[valid_bins],
             'o-', color='#e74c3c', linewidth=2.5, markersize=8,
             label='Bin mean')
    ax1.set_xlabel('Δ_local(s) = π(a₁|s) − π(a₂|s)', fontsize=12)
    ax1.set_ylabel("n'(s) — groups needed to flip vote", fontsize=12)
    ax1.set_title(f'Δ_local vs Global Voting Margin\n(r={corr:.3f})',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(-0.02, 1.02)

    # Right: flippable fraction per bin
    ax2 = axes[1]
    x_pos = np.arange(len(bin_labels))
    w = 0.35
    bars1 = ax2.bar(x_pos - w/2, bin_flippable, w,
                     color='#e74c3c', edgecolor='white',
                     label="n'(s) ≤ 2 (flippable)")
    bars2 = ax2.bar(x_pos + w/2, bin_flippable_1, w,
                     color='#c0392b', edgecolor='white',
                     label="n'(s) ≤ 1 (very flippable)")
    for bar, val in zip(bars1, bin_flippable):
        if not np.isnan(val) and val > 1:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.0f}%', ha='center', fontsize=8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bin_labels, fontsize=8)
    ax2.set_ylabel('% of States', fontsize=12)
    ax2.set_xlabel('Δ_local(s) Bin', fontsize=12)
    ax2.set_title('Fraction of Vulnerable States per Δ_local Bin',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/exp1_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nFigure saved: outputs/exp1_correlation.png")

    # ---- Verdict ----
    print("\n" + "=" * 60)
    print("Verdict")
    print("=" * 60)
    if corr > 0.3:
        print(f"✅ Δ_local(s) IS a valid proxy (r={corr:.3f}).")
        print("   States with small Δ_local are significantly more flippable.")
        print("   → Stage I of BSA is theoretically justified.")
    elif corr > 0.1:
        print(f"⚠️  Weak correlation (r={corr:.3f}).")
        print("   Δ_local has some predictive power but is noisy.")
    else:
        print(f"❌ No meaningful correlation (r={corr:.3f}).")
        print("   Δ_local is NOT a reliable proxy for voting weakness.")
        print("   → BSA Stage I trigger selection needs redesign.")

    return all_data, deltas, nprimes, bin_flippable


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    run()
