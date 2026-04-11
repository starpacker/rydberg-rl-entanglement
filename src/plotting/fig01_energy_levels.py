#!/usr/bin/env python
"""Fig.1 -- Rb-87 Rydberg energy levels vs hydrogen.

Plots S, P, D series using Rydberg-Ritz formula with quantum defects,
overlaid with hydrogen levels (dashed) to highlight defect shifts.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *
from src.physics.constants import (
    DELTA_0_S, DELTA_0_P32, DELTA_0_D32, RY_RB_HZ,
    DELTA_2_S,
)

# ---------------------------------------------------------------
# Compute energy levels
# ---------------------------------------------------------------
n_values = np.arange(5, 31)

def energy_rb(n, delta_0, delta_2=0.0):
    """E_n = -Ry_Rb / (n - delta)^2   in GHz."""
    n_star = n - delta_0 - delta_2 / (n - delta_0)**2
    return -RY_RB_HZ / n_star**2 / 1e9  # GHz

def energy_hydrogen(n):
    return -RY_RB_HZ / n**2 / 1e9  # GHz

E_S = energy_rb(n_values, DELTA_0_S, DELTA_2_S)
E_P = energy_rb(n_values, DELTA_0_P32)
E_D = energy_rb(n_values, DELTA_0_D32)
E_H = energy_hydrogen(n_values)

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=DOUBLE_COL)

ax.plot(n_values, E_H, '--', color='gray', linewidth=1.0, label='Hydrogen', zorder=1)
ax.plot(n_values, E_S, 'o-', color=COLORS['blue'], markersize=3, linewidth=1.2,
        label=r'Rb $nS_{1/2}$')
ax.plot(n_values, E_P, 's-', color=COLORS['orange'], markersize=3, linewidth=1.2,
        label=r'Rb $nP_{3/2}$')
ax.plot(n_values, E_D, '^-', color=COLORS['green'], markersize=3, linewidth=1.2,
        label=r'Rb $nD_{3/2}$')

# Annotate quantum defect shifts for n=10
n_ann = 10
y_S = energy_rb(n_ann, DELTA_0_S, DELTA_2_S)
y_H = energy_hydrogen(n_ann)
y_P = energy_rb(n_ann, DELTA_0_P32)
y_D = energy_rb(n_ann, DELTA_0_D32)

ax.annotate('', xy=(n_ann + 0.3, y_S), xytext=(n_ann + 0.3, y_H),
            arrowprops=dict(arrowstyle='<->', color=COLORS['red'], lw=1.0))
ax.text(n_ann + 0.7, (y_S + y_H) / 2, r'$\delta_S$', color=COLORS['red'],
        fontsize=10, va='center')

ax.annotate('', xy=(n_ann + 1.5, y_P), xytext=(n_ann + 1.5, y_H),
            arrowprops=dict(arrowstyle='<->', color=COLORS['red'], lw=1.0))
ax.text(n_ann + 1.9, (y_P + y_H) / 2, r'$\delta_P$', color=COLORS['red'],
        fontsize=10, va='center')

ax.set_xlabel('Principal quantum number $n$')
ax.set_ylabel('Energy (GHz)')
ax.set_title('Rb-87 Rydberg energy levels')
ax.legend(loc='lower right', framealpha=0.9)
ax.set_xlim(4, 31)

outpath = os.path.join(os.path.dirname(__file__), '../../', FIGURE_DIR, 'fig01_energy_levels.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
fig.savefig(outpath.replace('.pdf', '.png'))
plt.close(fig)
print(f"Saved {os.path.abspath(outpath)}")
