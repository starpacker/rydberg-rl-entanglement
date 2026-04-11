#!/usr/bin/env python
"""Fig.10 -- Scenario B algorithm comparison bar chart.

Loads results from results/*.json and shows mean fidelity with error bars
for STIRAP, GRAPE, and PPO+DR.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
base = os.path.join(os.path.dirname(__file__), '../..')

with open(os.path.join(base, 'results', 'stirap_B.json')) as f:
    stirap = json.load(f)
with open(os.path.join(base, 'results', 'grape_B.json')) as f:
    grape = json.load(f)
with open(os.path.join(base, 'results', 'ppo_B.json')) as f:
    ppo = json.load(f)

methods = ['STIRAP', 'GRAPE', 'PPO+DR']
mean_F = [stirap['mean_F'], grape['mean_F'], ppo['mean_F']]
F_05   = [stirap['F_05'],   grape['F_05'],   ppo['F_05']]
std_F  = [stirap['std_F'],  grape['std_F'],  ppo['std_F']]

# Asymmetric error bars: lower = mean - F_05, upper = std
yerr_lo = [m - f5 for m, f5 in zip(mean_F, F_05)]
yerr_hi = std_F
yerr = [yerr_lo, yerr_hi]

bar_colors = [COLORS['blue'], COLORS['orange'], COLORS['green']]

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=SINGLE_COL)

x = np.arange(len(methods))
bars = ax.bar(x, mean_F, width=0.5, color=bar_colors, edgecolor='black',
              linewidth=0.5, yerr=yerr, capsize=5, error_kw={'linewidth': 1.2},
              zorder=3)

# Value labels
for i, (bar, val) in enumerate(zip(bars, mean_F)):
    y_top = bar.get_height() + std_F[i] + 0.003
    ax.text(bar.get_x() + bar.get_width() / 2, y_top,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel(r'Fidelity $F$')
ax.set_title('Scenario B: Algorithm Comparison')
ax.set_ylim(0.80, 1.01)
ax.grid(axis='y', alpha=0.3, zorder=0)

outpath = os.path.join(os.path.dirname(__file__), '../../', FIGURE_DIR, 'fig10_scenario_B_comparison.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
fig.savefig(outpath.replace('.pdf', '.png'))
plt.close(fig)
print(f"Saved {os.path.abspath(outpath)}")
