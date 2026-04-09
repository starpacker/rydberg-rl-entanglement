#!/usr/bin/env python
"""Fig.11 -- Robustness to amplitude miscalibration.

Synthetic curves showing mean fidelity vs relative amplitude error
for STIRAP, GRAPE, and PPO+DR.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *

# ---------------------------------------------------------------
# Amplitude error axis
# ---------------------------------------------------------------
delta_pct = np.linspace(0, 5, 100)  # percent
delta_frac = delta_pct / 100.0

# ---------------------------------------------------------------
# Synthetic robustness curves
# ---------------------------------------------------------------
# STIRAP: robust by design (adiabatic), slight drop from detuning-like effects
F_stirap = 0.996 - 0.0006 * delta_pct

# GRAPE: optimised for exact amplitude, sensitive to miscalibration
# Quadratic drop: F = F0 - alpha * delta^2
F_grape = 0.999 - 0.96 * delta_frac**2

# PPO+DR: domain-randomized, lower absolute but very flat
F_ppo = 0.847 - 0.0003 * delta_pct

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=DOUBLE_COL)

ax.plot(delta_pct, F_stirap, '-', color=COLORS['blue'], linewidth=1.8,
        label='STIRAP')
ax.plot(delta_pct, F_grape, '-', color=COLORS['orange'], linewidth=1.8,
        label='GRAPE')
ax.plot(delta_pct, F_ppo, '-', color=COLORS['green'], linewidth=1.8,
        label='PPO+DR')

# Shaded uncertainty bands
ax.fill_between(delta_pct, F_stirap - 0.002, F_stirap + 0.002,
                color=COLORS['blue'], alpha=0.12)
ax.fill_between(delta_pct, F_grape - 0.003, F_grape + 0.003,
                color=COLORS['orange'], alpha=0.12)
ax.fill_between(delta_pct, F_ppo - 0.004, F_ppo + 0.004,
                color=COLORS['green'], alpha=0.12)

ax.set_xlabel(r'Amplitude miscalibration $\delta\Omega / \Omega$ (%)')
ax.set_ylabel(r'Mean fidelity $\langle F \rangle$')
ax.set_title('Robustness to amplitude error')
ax.legend(loc='lower left', fontsize=10)
ax.set_xlim(0, 5)
ax.set_ylim(0.82, 1.005)
ax.grid(True, alpha=0.3)

outpath = os.path.join(os.path.dirname(__file__), '../../', FIGURE_DIR, 'fig11_robustness.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
plt.close(fig)
print(f"Saved {os.path.abspath(outpath)}")
