# plot_results.py — 生成报告所需全部图表
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.unicode_minus'] = False

# 尝试设置中文字体
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
except:
    pass

# ============================================================
# Figure 1: 主实验对比柱状图
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))

experiments = [
    'No Attack\nNo Ensemble\n(CartPole)',
    'Normalized Attack\nNo Ensemble\n(CartPole)',
    'Normalized Attack\n+ Ensemble\n(CartPole)',
    'Cross-Group\nDivergence\n+ Ensemble\n(LunarLander)',
]
rewards = [500, 432.5, 500, 151.9]
colors = ['#2ecc71', '#e74c3c', '#3498db', '#c0392b']

bars = ax.bar(range(len(experiments)), rewards, color=colors, edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, rewards):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.set_xticks(range(len(experiments)))
ax.set_xticklabels(experiments, fontsize=10)
ax.set_ylabel('Test Reward', fontsize=13)
ax.set_title('Cross-Group Divergence Attack Breaks Ensemble Defense', fontsize=15, fontweight='bold')
ax.set_ylim(0, 600)
ax.axhline(y=500, color='green', linestyle='--', alpha=0.4, label='Optimal (500)')
ax.legend(fontsize=10)

# Add annotations
ax.annotate('Defense\nWorks', xy=(2, 500), xytext=(2.5, 560),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            fontsize=10, color='green', fontweight='bold')
ax.annotate('Defense\nBroken', xy=(3, 151.9), xytext=(3.5, 280),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, color='red', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fig1_main_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 1 saved.")

# ============================================================
# Figure 2: Ensemble 决策边界分歧分布
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: vote agreement pie
ax1 = axes[0]
sizes = [25.2, 74.8 - 21.5, 21.5]
labels = ['All 5 groups agree\n(25.2%)',
          '2 different actions\n(53.3%)',
          '≥3 different actions\n(21.5%)']
colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']
explode = (0.02, 0.02, 0.08)

wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels,
                                     colors=colors_pie, autopct='',
                                     startangle=140,
                                     textprops={'fontsize': 10})
ax1.set_title('Ensemble Voting Agreement\n(LunarLander, 5 groups, 15000 states)',
              fontsize=12, fontweight='bold')

# Right: vote distribution histogram
ax2 = axes[1]
vote_counts = [25.2, 53.3, 14.0, 7.5]  # 5/5, 4/5, 3/5, ≤2/5 agreement
vote_labels = ['5/5\nConsensus', '4/5\nStrong', '3/5\nWeak', '≤2/5\nNo Majority']
vote_colors = ['#2ecc71', '#27ae60', '#e67e22', '#e74c3c']
bars2 = ax2.bar(vote_labels, vote_counts, color=vote_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars2, vote_counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax2.set_ylabel('% of States', fontsize=12)
ax2.set_title('Vote Strength Distribution', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 60)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/fig2_ensemble_divergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 saved.")

# ============================================================
# Figure 3: A/B Test — Divergence Attack Effectiveness
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(3)
width = 0.35

control = [19.5, 0.0, 19.9]
attack = [98.8, 40.0, 98.9]
regions = ['All States', 'High-Entropy\n(Trigger)', 'Low-Entropy\n(Safe)']

bars1 = ax.bar(x - width/2, control, width, label='Control (No Attack)',
               color='#3498db', edgecolor='white', linewidth=1.2)
bars2 = ax.bar(x + width/2, attack, width, label='Divergence Attack',
               color='#e74c3c', edgecolor='white', linewidth=1.2)

for bar, val in zip(bars1, control):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
for bar, val in zip(bars2, attack):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

# Add delta annotations
for i, (c, a) in enumerate(zip(control, attack)):
    delta = a - c
    ax.annotate(f'Δ=+{delta:.1f}%', xy=(i, max(c, a) + 12),
                ha='center', fontsize=10, fontweight='bold',
                color='#c0392b')

ax.set_xticks(x)
ax.set_xticklabels(regions, fontsize=11)
ax.set_ylabel('Target Action (0) Frequency (%)', fontsize=12)
ax.set_title('A/B Test: Divergence Attack Policy Bias\n(6 workers, 2 Byzantine, Target=Action 0)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.set_ylim(0, 115)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/fig3_ab_test.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved.")

# ============================================================
# Figure 4: Attack success comparison heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(10, 3.5))

data = np.array([
    [500, 432.5, 500, 151.9],
])
norm = matplotlib.colors.Normalize(vmin=100, vmax=500)

im = ax.imshow(data, cmap='RdYlGn', aspect='auto', norm=norm)

ax.set_xticks(range(4))
ax.set_xticklabels(['Baseline\n(No Attack)', 'Normalized\nAttack', 'Normalized\n+ Ensemble',
                     'Cross-Group\nDivergence\n+ Ensemble'],
                    fontsize=9)
ax.set_yticks([])
for i in range(4):
    color = 'white' if data[0, i] < 300 else 'black'
    ax.text(i, 0, f'{data[0, i]}', ha='center', va='center',
            fontweight='bold', fontsize=16, color=color)

ax.set_title('Test Reward Heatmap: Attack vs Defense', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Test Reward', fontsize=11)
plt.tight_layout()
plt.savefig('outputs/fig4_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 saved.")

print("\nAll figures saved to outputs/")
