#!/usr/bin/env python
"""Fig.8 -- Traditional algorithm performance ceiling vs gate time.

Shows STIRAP and GRAPE fidelity as a function of T_gate, illustrating the
adiabatic-violation / decoherence trade-off for STIRAP and the decoherence
limit for GRAPE.  Uses analytic/synthetic curves.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *

# ---------------------------------------------------------------
# Gate-time axis (microseconds for display, seconds for computation)
# ---------------------------------------------------------------
T_us = np.logspace(-1, 1, 300)  # 0.1 to 10 us
T_s = T_us * 1e-6

# ---------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------
tau_eff = 80.6e-6  # effective Rydberg lifetime (s)

# STIRAP: adiabatic transfer requires slow pulse; decoherence kills at long times
# Model: F_stirap ~ F_adiabatic * F_decay
# F_adiabatic ~ 1 - (T_ad / T)^2 where T_ad ~ 0.15 us sets adiabatic scale
# F_decay ~ exp(-T / tau_eff)
T_ad = 0.15e-6  # adiabatic timescale
F_stirap = (1 - (T_ad / T_s)**2) * np.exp(-T_s / tau_eff)
F_stirap = np.clip(F_stirap, 0, 1)

# GRAPE: near-optimal at short times (unitary limit), then decoherence at long times
# Model: F_grape ~ F_unitary * exp(-T / tau_eff)
# F_unitary saturates at ~0.9999 for T > 0.05 us
F_unitary = 1 - 0.0001 * np.exp(-T_us / 0.05)
F_grape = F_unitary * np.exp(-T_s / tau_eff)

# ---------------------------------------------------------------
# Scenario markers
# ---------------------------------------------------------------
T_A = 5.0   # us
T_B = 0.3   # us

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=DOUBLE_COL)

ax.semilogx(T_us, F_stirap, '-', color=COLORS['blue'], linewidth=1.8,
            label='STIRAP')
ax.semilogx(T_us, F_grape, '-', color=COLORS['orange'], linewidth=1.8,
            label='GRAPE')

# Fault-tolerant threshold
ax.axhline(0.999, color='gray', linestyle='--', linewidth=1.0,
           label=r'$F = 0.999$ (fault-tolerant)')

# Scenario markers
ax.axvline(T_B, color=COLORS['green'], linestyle=':', linewidth=1.2, alpha=0.7)
ax.text(T_B * 1.15, 0.50, 'Scenario B\n(0.3 $\\mu$s)', fontsize=9,
        color=COLORS['green'], va='center')
ax.axvline(T_A, color=COLORS['red'], linestyle=':', linewidth=1.2, alpha=0.7)
ax.text(T_A * 1.15, 0.50, 'Scenario A\n(5 $\\mu$s)', fontsize=9,
        color=COLORS['red'], va='center')

ax.set_xlabel(r'Gate time $T_{\mathrm{gate}}$ ($\mu$s)')
ax.set_ylabel(r'Fidelity $F$')
ax.set_title('Traditional algorithm performance vs gate time')
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0.0, 1.02)
ax.set_xlim(0.1, 10)
ax.grid(True, alpha=0.3, which='both')

outpath = os.path.join(os.path.dirname(__file__), '../../', FIGURE_DIR, 'fig08_traditional_ceiling.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
plt.close(fig)
print(f"Saved {os.path.abspath(outpath)}")
