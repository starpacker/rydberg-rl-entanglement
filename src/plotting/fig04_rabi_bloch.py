#!/usr/bin/env python
"""Fig.4 -- Rabi oscillations + Bloch sphere trajectory.

Panel (a): P_r(t) for resonant and two detuned cases (single 2-level atom).
Panel (b): Bloch-sphere trajectory for resonant case (2D projection if 3D fails).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *
import qutip

# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
Omega = 2 * np.pi * 1e6  # 1 MHz Rabi frequency
T = 4.0 / (Omega / (2 * np.pi))  # 4 Rabi periods
npts = 500
tlist = np.linspace(0, T, npts)

# Operators for single 2-level system
g = qutip.basis(2, 0)
r = qutip.basis(2, 1)
Pr = r * r.dag()  # |r><r| projector

# Detuning cases
cases = [
    (r'$\Delta = 0$', 0.0, COLORS['blue']),
    (r'$\Delta = \Omega$', Omega, COLORS['orange']),
    (r'$\Delta = 2\Omega$', 2 * Omega, COLORS['green']),
]

# ---------------------------------------------------------------
# Panel (a): Rabi oscillations
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
ax_rabi = axes[0]

for label, Delta, color in cases:
    H = (Omega / 2) * (r * g.dag() + g * r.dag()) - Delta * Pr
    result = qutip.mesolve(H, g, tlist, c_ops=[], e_ops=[Pr])
    P_r = result.expect[0]
    ax_rabi.plot(tlist * 1e6, P_r, color=color, linewidth=1.3, label=label)

ax_rabi.set_xlabel(r'Time ($\mu$s)')
ax_rabi.set_ylabel(r'$P_r(t)$')
ax_rabi.set_title('(a) Rabi oscillations')
ax_rabi.legend(loc='upper right', fontsize=9)
ax_rabi.set_ylim(-0.05, 1.1)

# ---------------------------------------------------------------
# Panel (b): Bloch sphere (2D projection for robustness)
# ---------------------------------------------------------------
ax_bloch = axes[1]

# Resonant Rabi: compute Bloch vector components
Delta_res = 0.0
H_res = (Omega / 2) * (r * g.dag() + g * r.dag()) - Delta_res * Pr
sx = qutip.sigmax()
sy = qutip.sigmay()
sz = qutip.sigmaz()
result_bloch = qutip.mesolve(H_res, g, tlist, c_ops=[], e_ops=[sx, sy, sz])
bx = result_bloch.expect[0]
by = result_bloch.expect[1]
bz = result_bloch.expect[2]

# 2D projection: plot (by, bz) — resonant Rabi with real coupling rotates in y-z plane
# Draw unit circle
theta_circ = np.linspace(0, 2 * np.pi, 200)
ax_bloch.plot(np.cos(theta_circ), np.sin(theta_circ), 'k-', linewidth=0.5, alpha=0.3)
ax_bloch.axhline(0, color='gray', linewidth=0.3)
ax_bloch.axvline(0, color='gray', linewidth=0.3)

# Trajectory: use y-z plane where the rotation actually happens
ax_bloch.plot(by, bz, color=COLORS['blue'], linewidth=1.3, label='Trajectory')
ax_bloch.plot(by[0], bz[0], 'o', color=COLORS['green'], markersize=7, zorder=5,
              label=r'$|g\rangle$')
# |r> state at t = pi/Omega = T/8 (half a Rabi period = pi-pulse)
idx_r = npts // 8
ax_bloch.plot(by[idx_r], bz[idx_r], 's', color=COLORS['red'], markersize=7,
              zorder=5, label=r'$|r\rangle$')

ax_bloch.set_xlabel(r'$\langle \sigma_y \rangle$')
ax_bloch.set_ylabel(r'$\langle \sigma_z \rangle$')
ax_bloch.set_title('(b) Bloch sphere (y-z)')
ax_bloch.set_aspect('equal')
ax_bloch.set_xlim(-1.3, 1.3)
ax_bloch.set_ylim(-1.3, 1.3)
ax_bloch.legend(loc='upper right', fontsize=8)

fig.tight_layout()

outpath = os.path.join(os.path.dirname(__file__), '../../', FIGURE_DIR, 'fig04_rabi_bloch.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
fig.savefig(outpath.replace('.pdf', '.png'))
plt.close(fig)
print(f"Saved {os.path.abspath(outpath)}")
