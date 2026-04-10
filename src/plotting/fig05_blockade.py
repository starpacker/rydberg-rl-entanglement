#!/usr/bin/env python
"""Fig.5 -- Rydberg blockade mechanism.

Panel (a): Energy level diagram for 2-atom system.
Panel (b): Population dynamics: V=0 (free) vs V >> Omega (blockade).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *
import qutip
from src.physics.hamiltonian import build_two_atom_hamiltonian, get_ground_state, get_target_state

# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
Omega = 2 * np.pi * 1e6  # 1 MHz
T = 3.0 / (Omega / (2 * np.pi))  # 3 Rabi periods
npts = 500
tlist = np.linspace(0, T, npts)

psi0 = get_ground_state(2)  # |gg>

# Basis states for projectors
g = qutip.basis(2, 0)
r = qutip.basis(2, 1)
gg = qutip.tensor(g, g)
W_state = get_target_state(2)  # (|gr> + |rg>)/sqrt(2)
rr = qutip.tensor(r, r)

P_gg = gg * gg.dag()
P_W = W_state * W_state.dag()
P_rr = rr * rr.dag()

# ---------------------------------------------------------------
# Case 1: V = 0 (free oscillation)
# ---------------------------------------------------------------
H_free = build_two_atom_hamiltonian(Omega=Omega, Delta=0.0, V_vdW=0.0)
res_free = qutip.mesolve(H_free, psi0, tlist, c_ops=[], e_ops=[P_gg, P_W, P_rr])

# ---------------------------------------------------------------
# Case 2: V >> Omega (blockade)
# ---------------------------------------------------------------
V_blockade = 100 * Omega  # strong blockade
H_block = build_two_atom_hamiltonian(Omega=Omega, Delta=0.0, V_vdW=V_blockade)
res_block = qutip.mesolve(H_block, psi0, tlist, c_ops=[], e_ops=[P_gg, P_W, P_rr])

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))

# --- Panel (a): Energy level schematic ---
ax_level = axes[0]
ax_level.set_xlim(-0.5, 1.5)
ax_level.set_ylim(-0.3, 1.4)
ax_level.axis('off')
ax_level.set_title('(a) Two-atom energy levels', fontsize=13)

# Level positions
y_gg = 0.0
y_W = 0.55
y_rr_free = 1.1
y_rr_shift = 1.3

lw = 2.5
# |gg>
ax_level.plot([0.0, 1.0], [y_gg, y_gg], color='k', linewidth=lw)
ax_level.text(1.05, y_gg, r'$|gg\rangle$', fontsize=12, va='center')

# |W> = (|gr>+|rg>)/sqrt(2)
ax_level.plot([0.0, 1.0], [y_W, y_W], color=COLORS['blue'], linewidth=lw)
ax_level.text(1.05, y_W, r'$|W\rangle$', fontsize=12, va='center', color=COLORS['blue'])

# |rr> (unshifted, dashed)
ax_level.plot([0.0, 1.0], [y_rr_free, y_rr_free], color='gray', linewidth=1.5, linestyle='--')
ax_level.text(1.05, y_rr_free, r'$|rr\rangle_{V=0}$', fontsize=10, va='center', color='gray')

# |rr> (shifted)
ax_level.plot([0.0, 1.0], [y_rr_shift, y_rr_shift], color=COLORS['red'], linewidth=lw)
ax_level.text(1.05, y_rr_shift, r'$|rr\rangle + V$', fontsize=11, va='center', color=COLORS['red'])

# Coupling arrows
ax_level.annotate('', xy=(0.35, y_W - 0.03), xytext=(0.35, y_gg + 0.03),
                  arrowprops=dict(arrowstyle='<->', color=COLORS['blue'], lw=2))
ax_level.text(0.20, (y_gg + y_W) / 2, r'$\sqrt{2}\,\Omega$', fontsize=12,
              color=COLORS['blue'], ha='center', va='center')

# V shift annotation
ax_level.annotate('', xy=(0.65, y_rr_shift - 0.02), xytext=(0.65, y_rr_free + 0.02),
                  arrowprops=dict(arrowstyle='<->', color=COLORS['red'], lw=1.5))
ax_level.text(0.55, (y_rr_free + y_rr_shift) / 2, r'$V$', fontsize=13,
              color=COLORS['red'], ha='center', va='center')

# Blocked transition (X mark)
ax_level.annotate('', xy=(0.35, y_rr_shift - 0.03), xytext=(0.35, y_W + 0.03),
                  arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='dashed'))
ax_level.text(0.42, (y_W + y_rr_shift) / 2, r'$\times$', fontsize=16,
              color=COLORS['red'], ha='center', va='center', fontweight='bold')

# --- Panel (b): Population dynamics ---
ax_pop = axes[1]
t_us = tlist * 1e6

# Free case (thin dashed)
ax_pop.plot(t_us, res_free.expect[0], '--', color=COLORS['blue'], linewidth=1.0,
            label=r'$P_{|gg\rangle}$ ($V=0$)', alpha=0.7)
ax_pop.plot(t_us, res_free.expect[2], '--', color=COLORS['red'], linewidth=1.0,
            label=r'$P_{|rr\rangle}$ ($V=0$)', alpha=0.7)

# Blockade case (solid)
ax_pop.plot(t_us, res_block.expect[0], '-', color=COLORS['blue'], linewidth=1.5,
            label=r'$P_{|gg\rangle}$ (blockade)')
ax_pop.plot(t_us, res_block.expect[1], '-', color=COLORS['green'], linewidth=1.5,
            label=r'$P_{|W\rangle}$ (blockade)')
ax_pop.plot(t_us, res_block.expect[2], '-', color=COLORS['red'], linewidth=1.5,
            label=r'$P_{|rr\rangle}$ (blockade)')

ax_pop.set_xlabel(r'Time ($\mu$s)')
ax_pop.set_ylabel('Population')
ax_pop.set_title('(b) Population dynamics')
ax_pop.legend(loc='right', fontsize=7.5, framealpha=0.9)
ax_pop.set_ylim(-0.05, 1.1)

fig.tight_layout()

outpath = os.path.join(os.path.dirname(__file__), '../../', FIGURE_DIR, 'fig05_blockade.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
plt.close(fig)
print(f"Saved {os.path.abspath(outpath)}")
