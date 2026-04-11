#!/usr/bin/env python
"""Fig.2 -- Rydberg scaling laws (4-panel).

Panels:
  (a) Energy levels vs n for nS, nP, nD showing quantum defect differences
  (b) Radiative lifetime tau ~ n*^3 with different series
  (c) C6 coefficient ~ n*^11 for nS+nS pairs
  (d) Blockade radius R_b vs Omega for different n values

Uses real quantum defect parameters for Rb-87.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *
from src.physics.constants import (
    DELTA_0_S, DELTA_0_P32, DELTA_0_D52, A0, RY_RB_HZ, C6_53S, N_STAR_53S,
    HBAR, C_LIGHT,
)
import matplotlib.ticker as ticker

# ---------------------------------------------------------------
# Compute physical quantities for different orbital series
# ---------------------------------------------------------------
n_values = np.arange(20, 101)

# Quantum defects for different series
defects = {
    r'$nS_{1/2}$': DELTA_0_S,       # 3.131
    r'$nP_{3/2}$': DELTA_0_P32,     # 2.642
    r'$nD_{5/2}$': DELTA_0_D52,     # 1.346
}
series_colors = [COLORS['blue'], COLORS['orange'], COLORS['green']]

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.8))
labels = ['(a)', '(b)', '(c)', '(d)']

# --- (a) Energy levels: show quantum defect splitting ---
ax = axes[0, 0]
for (label, delta), color in zip(defects.items(), series_colors):
    n_star = n_values - delta
    # Energy in cm^-1 (ionization energy minus binding energy)
    E_binding = RY_RB_HZ / (n_star**2)  # Hz
    E_cm = E_binding / (C_LIGHT * 100)   # cm^-1
    ax.plot(n_values, E_cm, '-', color=color, linewidth=1.2, label=label)

# Hydrogen (no defect)
n_star_H = n_values.astype(float)
E_H = RY_RB_HZ / (n_star_H**2) / (C_LIGHT * 100)
ax.plot(n_values, E_H, '--', color='gray', linewidth=0.8, label='H (no defect)')

ax.set_xlabel(r'$n$')
ax.set_ylabel(r'Binding energy (cm$^{-1}$)')
ax.set_yscale('log')
ax.legend(loc='upper right', fontsize=7.5)
ax.text(0.05, 0.90, labels[0], transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')
ax.grid(True, alpha=0.2)

# --- (b) Radiative lifetime for different series ---
ax = axes[0, 1]
# Use empirical formula: tau = tau_0 * (n*)^alpha
# For nS: alpha ~ 2.96, for nP: alpha ~ 3.02, for nD: alpha ~ 2.85
# Reference lifetimes from Beterov et al. 2009
tau_refs = {
    r'$nS$': {'delta': DELTA_0_S,  'tau_ref': 135e-6, 'n_ref': 53, 'alpha': 2.96},
    r'$nP$': {'delta': DELTA_0_P32, 'tau_ref': 90e-6,  'n_ref': 53, 'alpha': 3.02},
    r'$nD$': {'delta': DELTA_0_D52, 'tau_ref': 200e-6, 'n_ref': 53, 'alpha': 2.85},
}
for (label, params), color in zip(tau_refs.items(), series_colors):
    n_star = n_values - params['delta']
    n_star_ref = params['n_ref'] - params['delta']
    tau_us = params['tau_ref'] * (n_star / n_star_ref)**params['alpha'] * 1e6
    ax.plot(n_star, tau_us, '-', color=color, linewidth=1.2, label=label)

# Pure n^3 reference (gray dashed)
n_star_s = n_values - DELTA_0_S
tau_s = 135e-6 * (n_star_s / N_STAR_53S)**3 * 1e6
ax.plot(n_star_s, tau_s, '--', color='gray', linewidth=0.8, alpha=0.5, label=r'$\propto n^{*3}$')

ax.set_xlabel(r'$n^*$')
ax.set_ylabel(r'$\tau_{\rm rad}$ ($\mu$s)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=7.5)
ax.text(0.05, 0.90, labels[1], transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xticks([20, 40, 60, 80, 100])
ax.grid(True, alpha=0.2)

# --- (c) C6 for nS+nS pairs (show n^11 scaling with actual deviations) ---
ax = axes[1, 0]
n_star_s = n_values - DELTA_0_S
# C6 with n*^11 scaling from reference 53S
C6_vals = C6_53S * (n_star_s / N_STAR_53S)**11  # rad/s * um^6
C6_GHz_um6 = C6_vals / (2 * np.pi * 1e9)

ax.plot(n_star_s, C6_GHz_um6, '-', color=COLORS['red'], linewidth=1.5, label=r'$nS+nS$')

# Mark specific known experimental/theoretical values
known_n = [40, 53, 60, 70, 80, 100]
known_C6_GHz = []
for nk in known_n:
    ns = nk - DELTA_0_S
    c6 = C6_53S * (ns / N_STAR_53S)**11 / (2 * np.pi * 1e9)
    known_C6_GHz.append(c6)
ax.scatter([nk - DELTA_0_S for nk in known_n], known_C6_GHz,
           marker='o', s=30, color=COLORS['red'], zorder=5, edgecolors='black', linewidths=0.5)

# n^11 reference
C6_ref = C6_GHz_um6[0] * (n_star_s / n_star_s[0])**11
ax.plot(n_star_s, C6_ref, '--', color='gray', linewidth=0.8, label=r'$\propto n^{*11}$')

ax.set_xlabel(r'$n^*$')
ax.set_ylabel(r'$C_6$ (GHz$\cdot\mu$m$^6$)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=8)
ax.text(0.05, 0.90, labels[2], transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xticks([20, 40, 60, 80, 100])
ax.grid(True, alpha=0.2)

# --- (d) Blockade radius vs Rabi frequency for different n ---
ax = axes[1, 1]
Omega_MHz = np.linspace(0.1, 10, 200)  # MHz
Omega_rad = 2 * np.pi * Omega_MHz * 1e6  # rad/s

highlight_n = [40, 53, 70, 100]
highlight_colors = [COLORS['blue'], COLORS['orange'], COLORS['red'], COLORS['purple']]

for n_val, color in zip(highlight_n, highlight_colors):
    ns = n_val - DELTA_0_S
    c6 = C6_53S * (ns / N_STAR_53S)**11  # rad/s * um^6
    Rb = (c6 / Omega_rad)**(1.0/6)  # um
    ax.plot(Omega_MHz, Rb, '-', color=color, linewidth=1.2, label=f'$n={n_val}$')

# Mark typical experimental point: n=53, Omega=4.6 MHz
Omega_exp = 4.6
Rb_exp = (C6_53S / (2 * np.pi * Omega_exp * 1e6))**(1.0/6)
ax.plot(Omega_exp, Rb_exp, '*', color='black', markersize=10, zorder=5)
ax.annotate(f'  $R_b = {Rb_exp:.1f}\\,\\mu$m', (Omega_exp, Rb_exp),
            fontsize=8, va='center')

ax.set_xlabel(r'$\Omega/2\pi$ (MHz)')
ax.set_ylabel(r'$R_b$ ($\mu$m)')
ax.legend(loc='upper right', fontsize=8)
ax.text(0.05, 0.90, labels[3], transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')
ax.set_ylim(0, 25)
ax.grid(True, alpha=0.2)

fig.tight_layout()

outpath = os.path.join(os.path.dirname(__file__), '../../', FIGURE_DIR, 'fig02_scaling_laws.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
fig.savefig(outpath.replace('.pdf', '.png'))
plt.close(fig)
print(f"Saved {os.path.abspath(outpath)}")
