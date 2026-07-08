# test_exp1_correlation.py
# Experiment 1: Validate that Δ_local(s) is a proxy for global voting weakness.
#
# Computes Δ_local(s) = π(a₁|s) − π(a₂|s) from a SINGLE group's policy
# and n'(s) from Theorem 1 using all K=5 group master votes.
#
# Bug fixes (2026-07-08):
#   1. compute_global_nprime: unanimous → None (not 6); votes from K=5 groups
#   2. Spearman: use scipy.stats.spearmanr instead of broken manual rank

import numpy as np
import torch
import gymnasium as gym
from collections import Counter
from scipy.stats import spearmanr, pointbiserialr
from options import get_options
from agent import Agent
from worker import _reset_env, _step_env
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_global_nprime(all_group_actions):
    """Compute n'(s) from Theorem 1 (Fang et al. WWW 2025).

    n'(s) = ⌊(v(s,x) − v(s,y) − 1{y<x}) / 2⌋

    Args:
        all_group_actions: list of K actions, one from each group master.
    Returns:
        n_prime in {0, 1, 2}, or None if unanimous (infinitely robust).
    """
    K = len(all_group_actions)
    counts = Counter(all_group_actions)
    # Sort by vote count descending; break ties by action index ascending
    sorted_acts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(sorted_acts) < 2:
        return None  # unanimous — no runner-up exists

    x, vx = sorted_acts[0]  # winner: highest votes, smallest index on tie
    y, vy = sorted_acts[1]  # runner-up
    indicator = 1 if y < x else 0
    n_prime = (vx - vy - indicator) // 2
    return max(0, n_prime)  # clamp (negative can happen with 2:2 ties)


def run(load_base_path=None):
    sys.argv = [
        'exp1',
        '--env_name', 'LunarLander',
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
    opts.max_trajectories = 10000

    print("=" * 60)
    print("Experiment 1: Δ_local(s) vs n'(s) Correlation (FIXED)")
    print("=" * 60)

    # ---- Load or train ----
    print("\n[1/3] Loading clean Ensemble baseline...")
    agent = Agent(opts)

    import glob as _glob
    import os as _os
    loaded = False

    # Auto-scan for clean baseline checkpoints in outputs/
    search_dirs = []
    # Search both new (LunarLander/) and old (LunarLander-v2/) directories
    for pattern in ['outputs/LunarLander*/worker30_byzantine0_*/Clean_Base*',
                    'outputs/LunarLander/worker30_byzantine0_*/Clean_Base*',
                    'outputs/LunarLander-v2/worker30_byzantine0_*/Clean_Base*']:
        search_dirs.extend(sorted(_glob.glob(pattern), reverse=True))
    if not search_dirs:
        for pattern in ['outputs/LunarLander*/worker30_byzantine0_*/*',
                        'outputs/LunarLander/worker30_byzantine0_*/*',
                        'outputs/LunarLander-v2/worker30_byzantine0_*/*']:
            search_dirs.extend(sorted(_glob.glob(pattern), reverse=True))

    for run_dir in search_dirs:
        if not _os.path.isdir(run_dir):
            continue
        # Find highest epoch checkpoint
        ckpt_files = sorted(_glob.glob(_os.path.join(run_dir, 'r0-epoch-*-group0.pt')))
        if not ckpt_files:
            continue
        # Extract epoch number from filename
        latest = max(ckpt_files, key=lambda f: int(
            f.split('epoch-')[1].split('-group')[0]))
        epoch_str = latest.split('epoch-')[1].split('-group')[0]
        epoch = int(epoch_str)

        # Load all K groups
        all_ok = True
        for k in range(opts.num_groups):
            fname = _os.path.join(run_dir, f'r0-epoch-{epoch}-group{k}.pt')
            if not _os.path.exists(fname):
                all_ok = False
                break
        if not all_ok:
            continue

        print(f"  Found: {run_dir}")
        print(f"  Loading K={opts.num_groups} groups from epoch {epoch}...")
        for k in range(opts.num_groups):
            ckpt = torch.load(
                _os.path.join(run_dir, f'r0-epoch-{epoch}-group{k}.pt'),
                map_location='cpu')
            agent.group_masters[k].logits_net.load_state_dict(ckpt['master'])
        agent.master = agent.group_masters[0]
        loaded = True
        break

    if loaded:
        print("  Checkpoint loaded — skipping training.")
    else:
        print("  No checkpoint found. Training from scratch (max_traj=10000)...")
        agent.start_training(tb_logger=None, run_id=0)

    # ---- Collect states from rollouts ----
    print("\n[2/3] Collecting states and computing Δ_local + n'(s)...")
    from utils import resolve_env_name
    env = gym.make(resolve_env_name('LunarLander'))
    data = []

    num_episodes = 50
    for ep in range(num_episodes):
        obs = _reset_env(env)
        done = False
        steps = 0
        while not done and steps < 500:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)

            # Δ_local, H(π), and mean Δ across ALL 5 groups
            deltas_per_group = []
            entropies_per_group = []
            with torch.no_grad():
                for k in range(agent.num_groups):
                    lk = agent.group_masters[k].logits_net.logits_net(obs_t)
                    pk = torch.softmax(lk, dim=-1)
                    top2k = torch.topk(pk, 2)
                    deltas_per_group.append(
                        (top2k.values[0] - top2k.values[1]).item())
                    entropies_per_group.append(
                        -(pk * torch.log(pk + 1e-10)).sum().item())

            delta_local = deltas_per_group[0]          # Group 0 only
            delta_mean = np.mean(deltas_per_group)      # all 5 groups averaged
            delta_std = np.std(deltas_per_group)         # cross-group spread
            entropy = entropies_per_group[0]             # Group 0 entropy

            # Physical state features (zero-cost triggers)
            y_pos = obs[1]    # height
            y_vel = obs[3]    # vertical velocity
            angle = obs[4]    # lander tilt

            # n'(s) from all K=5 group votes
            group_votes = []
            with torch.no_grad():
                for k in range(agent.num_groups):
                    lk = agent.group_masters[k].logits_net.logits_net(obs_t)
                    group_votes.append(lk.argmax().item())
            n_prime = compute_global_nprime(group_votes)

            data.append({
                'delta_local': delta_local,
                'delta_mean': delta_mean,
                'delta_std': delta_std,
                'entropy': entropy,
                'n_prime': n_prime,
                'votes': group_votes,
                'y_pos': y_pos,
                'y_vel': y_vel,
                'angle': angle,
            })

            # Step with true ensemble majority vote (consistent with real attack eval)
            ensemble_act = agent.ensemble_predict(obs, opts.device, sample=False)
            obs, _, done, _ = _step_env(env, ensemble_act)
            steps += 1

        if (ep + 1) % 10 == 0:
            print(f"  Ep {ep+1}/{num_episodes}, states={len(data)}")

    env.close()
    print(f"  Total states: {len(data)}")

    # ---- Analysis ----
    print("\n[3/3] Analyzing correlation...")

    is_unanimous = np.array([d['n_prime'] is None for d in data])
    n_unanimous = is_unanimous.sum()
    non_unanimous = len(data) - n_unanimous
    is_unanimous_int = is_unanimous.astype(int)

    all_delta_local = np.array([d['delta_local'] for d in data])
    all_delta_mean = np.array([d['delta_mean'] for d in data])
    all_entropy = np.array([d['entropy'] for d in data])
    all_y_pos = np.array([d['y_pos'] for d in data])
    all_y_vel = np.array([d['y_vel'] for d in data])
    all_angle = np.array([d['angle'] for d in data])

    # ==================================================================
    # Test 1: Conflicting Confidence — when votes split, are groups
    #         individually CONFIDENT (high mean Δ across groups)?
    # ==================================================================
    print(f"\n{'='*60}")
    print("Test 1: Conflicting Confidence (mean Δ across all 5 groups)")
    print(f"{'='*60}")

    split_mask = ~is_unanimous
    unan_mask = is_unanimous

    mean_delta_split = all_delta_mean[split_mask].mean()
    mean_delta_unan = all_delta_mean[unan_mask].mean()
    mean_delta_local_split = all_delta_local[split_mask].mean()
    mean_delta_local_unan = all_delta_local[unan_mask].mean()

    print(f"  Mean Δ (5-group avg) on SPLIT  states: {mean_delta_split:.4f}")
    print(f"  Mean Δ (5-group avg) on UNAN. states: {mean_delta_unan:.4f}")
    print(f"  Difference: {mean_delta_split - mean_delta_unan:+.4f} "
          f"{'(split > unan = conflicting confidence!)' if mean_delta_split > mean_delta_unan else '(split < unan = genuine uncertainty)'}")
    print(f"\n  Mean Δ (Group 0 only) on SPLIT  states: {mean_delta_local_split:.4f}")
    print(f"  Mean Δ (Group 0 only) on UNAN. states: {mean_delta_local_unan:.4f}")

    # Point-biserial: mean Δ vs unanimous?
    r_pb_dmean, p_pb_dmean = pointbiserialr(is_unanimous_int, all_delta_mean)
    print(f"\n  Point-biserial (mean Δ vs is_unanimous): "
          f"r={r_pb_dmean:.4f}, p={p_pb_dmean:.4f}")

    # ==================================================================
    # Test 2: Physical State as Zero-Cost Trigger
    # ==================================================================
    print(f"\n{'='*60}")
    print("Test 2: Physical State as Zero-Cost Trigger")
    print(f"{'='*60}")

    # Physical state binning
    phys_bins = {
        'y_pos (height)': {
            'data': all_y_pos,
            'bins': [-1.0, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5],
            'label': 'Height'
        },
        'y_vel (vert. speed)': {
            'data': all_y_vel,
            'bins': [-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0],
            'label': 'Vertical Velocity'
        },
        'angle (tilt)': {
            'data': all_angle,
            'bins': [-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0],
            'label': 'Angle'
        },
    }

    best_phys_r = 0.0
    best_phys_name = ''

    for name, spec in phys_bins.items():
        arr = spec['data']
        bins = spec['bins']
        bin_labels = [f'[{bins[i]:.1f},{bins[i+1]:.1f})'
                      for i in range(len(bins)-1)]

        print(f"\n  --- {name} ---")
        print(f"  {'Bin':<18s} {'Count':>6s}  {'Unanim%':>8s}  {'Note':>25s}")
        print(f"  {'-'*60}")
        for i in range(len(bins)-1):
            mask = (arr >= bins[i]) & (arr < bins[i+1])
            if mask.sum() > 0:
                unan_pct = 100 * is_unanimous[mask].sum() / mask.sum()
                note = ''
                if unan_pct < 15:
                    note = '← HIGH DISAGREEMENT'
                elif unan_pct > 35:
                    note = '← high consensus'
                print(f"  {bin_labels[i]:<18s} {mask.sum():>6d}  "
                      f"{unan_pct:>7.1f}%  {note:>25s}")

        # Point-biserial vs is_unanimous
        r_pb, p_pb = pointbiserialr(is_unanimous_int, arr)
        print(f"  Point-biserial r = {r_pb:.4f}  (p = {p_pb:.4f})")
        if abs(r_pb) > abs(best_phys_r):
            best_phys_r = abs(r_pb)
            best_phys_name = name

    # ==================================================================
    # Test 3: Which predictor is strongest? Head-to-head comparison
    # ==================================================================
    print(f"\n{'='*60}")
    print("Test 3: Predictor Showdown")
    print(f"{'='*60}")

    predictors = {
        'Δ_local (Group 0 only)': all_delta_local,
        'Δ_mean (5-group avg)': all_delta_mean,
        'H(π) entropy (Group 0)': all_entropy,
    }
    for pname in phys_bins:
        predictors[pname] = phys_bins[pname]['data']

    results = []
    for pname, parr in predictors.items():
        r_pb, p_pb = pointbiserialr(is_unanimous_int, parr)
        results.append((abs(r_pb), pname, r_pb, p_pb))

    results.sort(key=lambda x: -x[0])
    print(f"\n  {'Predictor':<35s} {'|r|':>6s}  {'raw r':>7s}  {'p':>10s}")
    print(f"  {'-'*65}")
    for abs_r, pname, r_pb, p_pb in results:
        marker = ' ← BEST' if abs_r == results[0][0] else ''
        print(f"  {pname:<35s} {abs_r:>6.4f}  {r_pb:>+7.4f}  "
              f"{p_pb:>10.4f}{marker}")

    if results[0][1].startswith('y_pos') or results[0][1].startswith('y_vel') or results[0][1].startswith('angle'):
        print(f"\n  ✅ PHYSICAL STATE is the strongest predictor!")
        print(f"     Attacker needs ZERO policy knowledge — just physics.")
    else:
        print(f"\n  Policy-based predictor ({results[0][1]}) is strongest.")
        print(f"     Physics alone is insufficient.")

    # ---- Plot: physical state vs unanimous rate ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Height vs unanimous rate
    ax1 = axes[0]
    for name, spec in [('y_pos (height)', phys_bins['y_pos (height)']),
                        ('y_vel (vert. speed)', phys_bins['y_vel (vert. speed)'])]:
        arr = spec['data']
        bins = spec['bins']
        bl = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        upcts = []
        for i in range(len(bins)-1):
            m = (arr >= bins[i]) & (arr < bins[i+1])
            upcts.append(100 * is_unanimous[m].sum() / max(m.sum(), 1))
        ax1.plot(bl, upcts, 'o-', lw=2, ms=7, label=spec['label'])

    ax1.set_xlabel('Physical State Value', fontsize=12)
    ax1.set_ylabel('5/5 Unanimous Rate (%)', fontsize=12)
    ax1.set_title('Physical State vs. Ensemble Agreement', fontsize=13,
                  fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.axhline(y=20.3, color='gray', ls='--', alpha=0.5,
                label='Overall mean (20.3%)')
    ax1.legend(fontsize=9)

    # Right: predictor showdown bar chart
    ax2 = axes[1]
    pred_names_short = ['Δ_local\n(G0)', 'Δ_mean\n(5G)', 'H(π)\n(G0)',
                         'Height', 'V.Vel', 'Angle']
    pred_abs_r = [abs(pointbiserialr(is_unanimous_int, all_delta_local)[0]),
                  abs(pointbiserialr(is_unanimous_int, all_delta_mean)[0]),
                  abs(pointbiserialr(is_unanimous_int, all_entropy)[0]),
                  abs(pointbiserialr(is_unanimous_int, all_y_pos)[0]),
                  abs(pointbiserialr(is_unanimous_int, all_y_vel)[0]),
                  abs(pointbiserialr(is_unanimous_int, all_angle)[0])]
    colors_pred = ['#3498db']*3 + ['#e74c3c']*3
    bars = ax2.bar(range(6), pred_abs_r, color=colors_pred, edgecolor='white')
    for bar, val in zip(bars, pred_abs_r):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax2.set_xticks(range(6))
    ax2.set_xticklabels(pred_names_short, fontsize=9)
    ax2.set_ylabel('|Point-biserial r| vs is_unanimous', fontsize=12)
    ax2.set_title('Predictor Showdown: Policy vs. Physics', fontsize=13,
                  fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0.1, color='gray', ls='--', alpha=0.5, label='Weak threshold')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('outputs/exp1_correlation_fixed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nFigure: outputs/exp1_correlation_fixed.png")

    # ---- Verdict ----
    print("\n" + "=" * 60)
    print("Verdict")
    print("=" * 60)

    best_name = results[0][1]
    best_r = results[0][2]

    print(f"\n  Strongest predictor: {best_name} (r={best_r:.4f})")
    print(f"  Unanimous rate: {100*n_unanimous/len(data):.1f}% overall")

    if 'Conflicting Confidence' in locals():
        if mean_delta_split > mean_delta_unan:
            print(f"\n  ✅ CONFLICTING CONFIDENCE confirmed: "
                  f"mean Δ on split states ({mean_delta_split:.3f}) > "
                  f"unan states ({mean_delta_unan:.3f})")
        else:
            print(f"\n  ❌ Conflicting confidence NOT confirmed: "
                  f"groups are genuinely uncertain on split states")

    if abs(best_r) < 0.1:
        print(f"\n  OVERALL: No single zero-cost local signal reliably "
              f"predicts voting weakness (all |r| < 0.1).")
        print(f"  → BSA Stage I trigger selection needs physical-domain "
              f"informed design, not pure statistical proxy.")

    return data, is_unanimous, results


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    run()
