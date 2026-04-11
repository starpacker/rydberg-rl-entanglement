#!/usr/bin/env python
"""Fig.6 -- Fidelity of W-state preparation vs inter-atomic distance.

Sweeps R from 0.5 to 15 um, computes V_vdW = C6/R^6, runs mesolve for
t = pi/(sqrt(2)*Omega), and plots fidelity F(|W>) vs R.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *
import qutip
from src.physics.hamiltonian import build_two_atom_hamiltonian, get_ground_state, get_target_state
from src.physics.lindblad import compute_fidelity
from src.physics.constants import C6_53S

# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
Omega = 2 * np.pi * 1e6  # 1 MHz
T_pi = np.pi / (np.sqrt(2) * Omega)  # pi-time for collective oscillation
npts_t = 300
tlist = np.linspace(0, T_pi, npts_t)

psi0 = get_ground_state(2)
target = get_target_state(2)  # |W> = (|gr>+|rg>)/sqrt(2)

# Blockade radius for annotation
R_b = (C6_53S / Omega)**(1.0 / 6)  # um

# ---------------------------------------------------------------
# Sweep inter-atomic distance
# ---------------------------------------------------------------
R_values = np.linspace(1.0, 15.0, 80)  # um (start at 1 um to avoid extreme stiffness)
fidelities = np.zeros(len(R_values))

# Solver options: increase max steps for stiff Hamiltonians at small R
solver_opts = {"nsteps": 10000}

for i, R in enumerate(R_values):
    V = C6_53S / R**6
    H = build_two_atom_hamiltonian(Omega=Omega, Delta=0.0, V_vdW=V)
    # Use more time points for small-R (fast oscillation) cases
    n_t = max(npts_t, int(300 * max(1, V / Omega)))
    n_t = min(n_t, 5000)  # cap
    tl = np.linspace(0, T_pi, n_t)
    try:
        result = qutip.mesolve(H, psi0, tl, c_ops=[], options=solver_opts)
        rho_final = result.states[-1]
        fidelities[i] = compute_fidelity(rho_final, target)
    except Exception:
        # For extremely stiff cases, use analytical approximation:
        # In perfect blockade (V >> Omega), F ~ 1; in free regime F ~ sin^2(pi/2*sqrt(2)) ~ 0.5
        fidelities[i] = np.nan

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=DOUBLE_COL)

ax.plot(R_values, fidelities, '-', color=COLORS['blue'], linewidth=1.5)
ax.axvline(R_b, color=COLORS['red'], linestyle='--', linewidth=1.0,
           label=r'$R_b = %.1f\;\mu$m' % R_b)

ax.set_xlabel(r'Inter-atomic distance $R$ ($\mu$m)')
ax.set_ylabel(r'Fidelity $F(|W\rangle)$')
ax.set_title(r'$|W\rangle$-state fidelity vs distance ($\Omega = 2\pi \times 1$ MHz)')
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(0, 15.5)

# Annotate blockade regime
ax.annotate('Blockade regime', xy=(R_b / 2, 0.95), fontsize=10,
            ha='center', color=COLORS['red'], style='italic')
ax.annotate('Free regime', xy=((R_b + 15) / 2, 0.3), fontsize=10,
            ha='center', color='gray', style='italic')

outpath = os.path.join(os.path.dirname(__file__), '../../', FIGURE_DIR, 'fig06_fidelity_vs_distance.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
fig.savefig(outpath.replace('.pdf', '.png'))
plt.close(fig)
print(f"Saved {os.path.abspath(outpath)}")
