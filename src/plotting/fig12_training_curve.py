#!/usr/bin/env python
"""Fig.12 -- PPO training learning curve across 3 seeds.

Loads training_logs.json and plots per-episode fidelity with rolling average.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
base = os.path.join(os.path.dirname(__file__), '../..')
with open(os.path.join(base, 'results', 'training_logs.json')) as f:
    logs = json.load(f)

seeds_data = logs['seeds']

# ---------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------
def rolling_mean(x, window=50):
    """Compute rolling mean with same-length output."""
    cumsum = np.cumsum(np.insert(x, 0, 0))
    result = np.zeros(len(x))
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        result[i] = (cumsum[i + 1] - cumsum[lo]) / (i - lo + 1)
    return result

# Colors for seeds
seed_colors = [COLORS['blue'], COLORS['orange'], COLORS['red']]
seed_labels = []

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=DOUBLE_COL)

for idx, sd in enumerate(seeds_data):
    fids = np.array(sd['fidelities'])
    episodes = np.arange(len(fids))
    smoothed = rolling_mean(fids, window=50)

    label_base = f"Seed {sd['seed']}"
    if sd['seed'] == 264:
        label_base += ' (best)'

    # Raw data as faint line
    ax.plot(episodes, fids, color=seed_colors[idx], alpha=0.08, linewidth=0.3)

    # Smoothed curve
    ax.plot(episodes, smoothed, color=seed_colors[idx], linewidth=1.5,
            label=label_base)

# Target line
ax.axhline(0.99, color='gray', linestyle='--', linewidth=1.0, zorder=1)
ax.text(50, 0.995, r'$F = 0.99$ target', fontsize=9, color='gray', va='bottom')

ax.set_xlabel('Episode')
ax.set_ylabel('Episode fidelity (reward)')
ax.set_title('PPO training curves')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(0, max(len(sd['fidelities']) for sd in seeds_data))
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

outpath = os.path.join(os.path.dirname(__file__), '../../', FIGURE_DIR, 'fig12_training_curve.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
fig.savefig(outpath.replace('.pdf', '.png'))
plt.close(fig)
print(f"Saved {os.path.abspath(outpath)}")
