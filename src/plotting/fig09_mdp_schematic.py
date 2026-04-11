#!/usr/bin/env python
"""Fig.9 -- MDP mapping schematic for Rydberg RL control.

Draws a clean block diagram showing the RL-environment loop:
  Environment (Lindblad) -> state s_t -> Policy pi_theta -> action a_t -> back
  with reward at terminal step.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *
import matplotlib.patches as mpatches

# ---------------------------------------------------------------
# Layout parameters
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=DOUBLE_COL)
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.set_aspect('equal')
ax.axis('off')

# Box drawing helper
def draw_box(ax, xy, w, h, text, facecolor='#E8F0FE', edgecolor='black',
             fontsize=11, text_offset=(0, 0)):
    box = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.15",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5,
        zorder=2)
    ax.add_patch(box)
    cx = xy[0] + w / 2 + text_offset[0]
    cy = xy[1] + h / 2 + text_offset[1]
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', zorder=3)

# Arrow helper
def draw_arrow(ax, start, end, text='', text_offset=(0, 0.2), fontsize=9,
               color='black'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                zorder=1)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, ha='center', va='center', fontsize=fontsize,
                color=color, style='italic', zorder=3)

# ---------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------
# Environment box (left)
env_x, env_y, env_w, env_h = 0.8, 3.5, 3.0, 2.0
draw_box(ax, (env_x, env_y), env_w, env_h,
         'Environment\n(Lindblad)',
         facecolor='#DCEEFB')
# rho(t) label inside
ax.text(env_x + env_w / 2, env_y + 0.35,
        r'$\rho(t)$', ha='center', va='center', fontsize=12,
        color=COLORS['blue'], zorder=3)

# Policy box (right)
pol_x, pol_y, pol_w, pol_h = 6.2, 3.5, 3.0, 2.0
draw_box(ax, (pol_x, pol_y), pol_w, pol_h,
         r'Policy $\pi_\theta$' + '\n(MLP)',
         facecolor='#E8F8E0')

# Reward box (bottom)
rew_x, rew_y, rew_w, rew_h = 3.0, 0.5, 4.0, 1.2
draw_box(ax, (rew_x, rew_y), rew_w, rew_h,
         r'Reward $r = \mathrm{Tr}(\rho \cdot \rho_{\mathrm{tgt}})$',
         facecolor='#FFF3CD', fontsize=10)

# ---------------------------------------------------------------
# Arrows
# ---------------------------------------------------------------
# Environment -> state -> Policy (top arrow)
draw_arrow(ax, (env_x + env_w, env_y + env_h - 0.5),
           (pol_x, pol_y + pol_h - 0.5),
           text=r'state $s_t = \mathrm{vec}(\rho)$',
           text_offset=(0, 0.3), fontsize=10)

# Policy -> action -> Environment (bottom arrow, going left)
draw_arrow(ax, (pol_x, pol_y + 0.5),
           (env_x + env_w, env_y + 0.5),
           text=r'action $a_t = (\Omega,\,\Delta)$',
           text_offset=(0, -0.35), fontsize=10)

# Environment -> Reward (downward at terminal step)
draw_arrow(ax, (env_x + env_w / 2, env_y),
           (rew_x + 0.5, rew_y + rew_h),
           text=r'$t = T$',
           text_offset=(-0.45, 0.0), fontsize=10,
           color=COLORS['red'])

# ---------------------------------------------------------------
# Step label
# ---------------------------------------------------------------
ax.text(5.0, 6.5, 'MDP formulation for Rydberg quantum control',
        ha='center', va='center', fontsize=13, fontweight='bold')

outpath = os.path.join(os.path.dirname(__file__), '../../', FIGURE_DIR, 'fig09_mdp_schematic.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
fig.savefig(outpath.replace('.pdf', '.png'))
plt.close(fig)
print(f"Saved {os.path.abspath(outpath)}")
