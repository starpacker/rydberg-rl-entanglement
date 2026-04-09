#!/usr/bin/env python
"""Fig.3 -- Two-photon excitation level diagram (schematic).

Energy level diagram: 5S_{1/2} -> 5P_{3/2} -> nS/nD ladder.
Uses matplotlib arrows and text annotations.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *

# ---------------------------------------------------------------
# Layout parameters
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5, 5.5))

# Level positions  (x_left, x_right, y)
levels = {
    '5S':   (0.2, 0.8, 0.0),
    '5P':   (0.2, 0.8, 0.45),
    '5P_v': (0.2, 0.8, 0.55),   # virtual / detuned level (dashed)
    'nS':   (0.2, 0.8, 1.0),
    'nD':   (0.55, 0.95, 0.92),  # slightly offset
}

# Draw energy levels
line_kw = dict(linewidth=2.0, solid_capstyle='round')
ax.plot([0.15, 0.85], [0.0, 0.0], color='k', **line_kw)
ax.plot([0.15, 0.85], [0.45, 0.45], color='gray', linestyle='--', linewidth=1.2)
ax.plot([0.15, 0.85], [1.0, 1.0], color='k', **line_kw)
ax.plot([0.55, 0.95], [0.92, 0.92], color='k', linewidth=1.5)

# Level labels
ax.text(0.08, 0.0, r'$5S_{1/2}$', fontsize=13, ha='right', va='center')
ax.text(0.08, 0.45, r'$5P_{3/2}$', fontsize=12, ha='right', va='center', color='gray')
ax.text(0.08, 1.0, r'$nS_{1/2}$', fontsize=13, ha='right', va='center')
ax.text(0.97, 0.92, r'$nD_{3/2}$', fontsize=12, ha='left', va='center')

# Arrow: first photon  5S -> (near 5P)
arrow_x = 0.40
ax.annotate('', xy=(arrow_x, 0.43), xytext=(arrow_x, 0.02),
            arrowprops=dict(arrowstyle='->', color=COLORS['blue'],
                            lw=2.5, connectionstyle='arc3,rad=0'))
ax.text(arrow_x - 0.08, 0.20, r'$\Omega_1$', fontsize=14, color=COLORS['blue'],
        ha='center', fontweight='bold')
ax.text(arrow_x + 0.08, 0.13, r'780 nm', fontsize=9, color=COLORS['blue'],
        ha='center', style='italic')

# Arrow: second photon  (near 5P) -> nS
ax.annotate('', xy=(arrow_x, 0.98), xytext=(arrow_x, 0.47),
            arrowprops=dict(arrowstyle='->', color=COLORS['red'],
                            lw=2.5, connectionstyle='arc3,rad=0'))
ax.text(arrow_x - 0.08, 0.73, r'$\Omega_2$', fontsize=14, color=COLORS['red'],
        ha='center', fontweight='bold')
ax.text(arrow_x + 0.08, 0.66, r'480 nm', fontsize=9, color=COLORS['red'],
        ha='center', style='italic')

# Detuning annotation Delta
ax.annotate('', xy=(0.70, 0.45), xytext=(0.70, 0.55),
            arrowprops=dict(arrowstyle='<->', color=COLORS['orange'], lw=1.5))
ax.text(0.73, 0.50, r'$\Delta$', fontsize=14, color=COLORS['orange'],
        ha='left', va='center')

# Virtual level indicator (dashed)
ax.plot([0.15, 0.85], [0.55, 0.55], color=COLORS['orange'], linestyle=':', linewidth=1.0)
ax.text(0.87, 0.55, 'virtual', fontsize=8, color=COLORS['orange'],
        ha='left', va='center', style='italic')

# Effective Rabi frequency label
ax.text(0.50, 1.08, r'$\Omega_{\rm eff} = \frac{\Omega_1 \Omega_2}{2\Delta}$',
        fontsize=13, ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray',
                  alpha=0.9))

ax.set_xlim(-0.05, 1.15)
ax.set_ylim(-0.15, 1.25)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Two-photon Rydberg excitation', fontsize=14, pad=10)

outpath = os.path.join(os.path.dirname(__file__), '../../', FIGURE_DIR, 'fig03_two_photon.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
plt.close(fig)
print(f"Saved {os.path.abspath(outpath)}")
