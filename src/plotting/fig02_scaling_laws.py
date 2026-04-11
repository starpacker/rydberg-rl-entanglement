#!/usr/bin/env python
"""Fig.2 -- Rydberg scaling laws (4-panel log-log).

Panels:
  (a) Orbital radius <r> ~ n*^2 a0
  (b) Radiative lifetime tau ~ n*^3
  (c) C6 coefficient ~ n*^11
  (d) Blockade radius R_b ~ n*^(11/6)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *
from src.physics.constants import (
    DELTA_0_S, A0, RY_RB_HZ, C6_53S, N_STAR_53S,
)

# ---------------------------------------------------------------
# Compute scaling quantities for nS series, n = 20..100
# ---------------------------------------------------------------
n_values = np.arange(20, 101)
n_star = n_values - DELTA_0_S  # effective quantum number

# (a) Orbital radius: <r> ~ n*^2 a0 (in um)
r_orbital = n_star**2 * A0 * 1e6  # convert m -> um

# (b) Radiative lifetime: tau ~ n*^3 (normalise to known 53S value)
tau_0 = 135e-6  # s, 53S
tau = tau_0 * (n_star / N_STAR_53S)**3  # s -> us
tau_us = tau * 1e6

# (c) C6 ~ n*^11 (normalise to 53S value)
C6 = C6_53S * (n_star / N_STAR_53S)**11  # rad/s * um^6
C6_GHz_um6 = C6 / (2 * np.pi * 1e9)

# (d) Blockade radius: R_b = (C6 / Omega)^{1/6}, Omega = 2pi*1 MHz
Omega_ref = 2 * np.pi * 1e6  # rad/s
R_b = (C6 / Omega_ref)**(1.0 / 6)  # um

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

panels = [
    (axes[0, 0], n_star, r_orbital, r'$\langle r \rangle$ ($\mu$m)', 2,
     r'$\propto n^{*2}$', COLORS['blue']),
    (axes[0, 1], n_star, tau_us, r'$\tau_{\rm rad}$ ($\mu$s)', 3,
     r'$\propto n^{*3}$', COLORS['orange']),
    (axes[1, 0], n_star, C6_GHz_um6, r'$C_6$ (GHz$\cdot\mu$m$^6$)', 11,
     r'$\propto n^{*11}$', COLORS['green']),
    (axes[1, 1], n_star, R_b, r'$R_b$ ($\mu$m)', 11.0/6,
     r'$\propto n^{*11/6}$', COLORS['red']),
]

labels = ['(a)', '(b)', '(c)', '(d)']

for idx, (ax, x, y, ylabel, exponent, fit_label, color) in enumerate(panels):
    ax.loglog(x, y, '-', color=color, linewidth=1.5)

    # Power-law reference line
    y_fit = y[0] * (x / x[0])**exponent
    ax.loglog(x, y_fit, '--', color='gray', linewidth=0.8, label=fit_label)

    ax.set_xlabel(r'$n^*$')
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left', fontsize=9)
    ax.text(0.05, 0.90, labels[idx], transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')

    # Fix x-axis: use simple scalar ticks instead of garbled log formatter
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xticks([20, 40, 60, 80, 100])

fig.tight_layout()

outpath = os.path.join(os.path.dirname(__file__), '../../', FIGURE_DIR, 'fig02_scaling_laws.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
fig.savefig(outpath.replace('.pdf', '.png'))
plt.close(fig)
print(f"Saved {os.path.abspath(outpath)}")
