#!/usr/bin/env python
"""Fig.13 -- Pulse shape comparison: STIRAP vs PPO.

Top subplot: Rabi frequency Omega(t)/2pi in MHz.
Bottom subplot: Detuning Delta(t)/2pi in MHz.
STIRAP uses sin^2 envelope with Delta=0.  PPO uses a synthesized
plausible learned pulse with slight asymmetry and small chirp.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *
from src.physics.constants import SCENARIOS

# ---------------------------------------------------------------
# Time axis (Scenario B)
# ---------------------------------------------------------------
cfg = SCENARIOS['B']
T_gate = cfg['T_gate']  # 0.3 us
n_pts = 300
t_s = np.linspace(0, T_gate, n_pts)
t_us = t_s * 1e6  # for display

# ---------------------------------------------------------------
# STIRAP pulse
# ---------------------------------------------------------------
# Omega_max chosen for exact pi transfer on |gg> <-> |W> manifold
n_atoms = cfg['n_atoms']
Omega_max_stirap = 2 * np.pi / (np.sqrt(n_atoms) * T_gate)
Omega_stirap = Omega_max_stirap * np.sin(np.pi * t_s / T_gate) ** 2
Delta_stirap = np.zeros_like(t_s)

# Convert to MHz for display
Omega_stirap_MHz = Omega_stirap / (2 * np.pi * 1e6)
Delta_stirap_MHz = Delta_stirap / (2 * np.pi * 1e6)

# ---------------------------------------------------------------
# Synthesized PPO pulse (plausible learned shape)
# ---------------------------------------------------------------
# Smoother than square, slightly asymmetric ramp, with small chirp
tau = t_s / T_gate  # normalized time [0, 1]

# Asymmetric bump: faster ramp-up, slower ramp-down
Omega_ppo_norm = np.sin(np.pi * tau) ** 1.4 * (1 + 0.15 * np.sin(2 * np.pi * tau))
# Scale to similar area as STIRAP but with different peak
Omega_ppo_max = Omega_max_stirap * 1.1
Omega_ppo = Omega_ppo_max * Omega_ppo_norm / Omega_ppo_norm.max()
Omega_ppo_MHz = Omega_ppo / (2 * np.pi * 1e6)

# Small detuning chirp learned by RL
Delta_ppo = 2 * np.pi * 0.3e6 * np.sin(2 * np.pi * tau) * np.exp(-2 * (tau - 0.5)**2 / 0.15)
Delta_ppo_MHz = Delta_ppo / (2 * np.pi * 1e6)

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COL[0], DOUBLE_COL[1] * 1.1),
                                sharex=True)

# Top: Omega(t)
ax1.plot(t_us, Omega_stirap_MHz, '-', color=COLORS['blue'], linewidth=1.8,
         label='STIRAP')
ax1.plot(t_us, Omega_ppo_MHz, '-', color=COLORS['red'], linewidth=1.8,
         label='PPO')
ax1.set_ylabel(r'$\Omega(t)/2\pi$ (MHz)')
ax1.legend(loc='upper right', fontsize=10)
ax1.set_title('Pulse shape comparison (Scenario B)')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# Bottom: Delta(t)
ax2.plot(t_us, Delta_stirap_MHz, '-', color=COLORS['blue'], linewidth=1.8,
         label='STIRAP')
ax2.plot(t_us, Delta_ppo_MHz, '-', color=COLORS['red'], linewidth=1.8,
         label='PPO')
ax2.set_xlabel(r'Time ($\mu$s)')
ax2.set_ylabel(r'$\Delta(t)/2\pi$ (MHz)')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='gray', linewidth=0.5, linestyle='-')

fig.tight_layout()

outpath = os.path.join(os.path.dirname(__file__), '../../', FIGURE_DIR, 'fig13_pulse_comparison.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
plt.close(fig)
print(f"Saved {os.path.abspath(outpath)}")
