"""Two-atom and three-atom Rydberg Hamiltonian builders.

Basis ordering
--------------
- 2-atom: |gg>, |gr>, |rg>, |rr>  (binary: 00, 01, 10, 11)
- 3-atom: |ggg>, |ggr>, |grg>, |grr>, |rgg>, |rgr>, |rrg>, |rrr>

The single-atom qubit space is {|g>=|0>, |r>=|1>}.

Hamiltonian (resonant, rotating frame)
--------------------------------------
H = sum_i [ (Omega/2)(|r_i><g_i| + h.c.) - Delta |r_i><r_i| ]
    + sum_{i<j} V_{ij} |r_i r_j><r_i r_j|

where V_{ij} = C6 / |r_i - r_j|^6.
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

import numpy as np
import qutip


# ===================================================================
# Single-site operators
# ===================================================================

def _sigma_gr(site: int, n_atoms: int) -> qutip.Qobj:
    """Return |g_i><r_i| acting on site *site* in the n-atom Hilbert space."""
    # Single-site: |g><r| = |0><1|
    op = qutip.basis(2, 0) * qutip.basis(2, 1).dag()
    ops = [qutip.qeye(2)] * n_atoms
    ops[site] = op
    return qutip.tensor(ops)


def _sigma_rg(site: int, n_atoms: int) -> qutip.Qobj:
    """Return |r_i><g_i| acting on site *site* in the n-atom Hilbert space."""
    return _sigma_gr(site, n_atoms).dag()


def _n_r(site: int, n_atoms: int) -> qutip.Qobj:
    """Return |r_i><r_i| (Rydberg number operator) at site *site*."""
    op = qutip.basis(2, 1) * qutip.basis(2, 1).dag()
    ops = [qutip.qeye(2)] * n_atoms
    ops[site] = op
    return qutip.tensor(ops)


# ===================================================================
# Two-atom Hamiltonian
# ===================================================================

def build_two_atom_hamiltonian(
    Omega: float,
    Delta: float,
    V_vdW: float,
) -> qutip.Qobj:
    """Build the 2-atom Rydberg Hamiltonian (4x4).

    Parameters
    ----------
    Omega : float
        Rabi frequency (rad/s).
    Delta : float
        Single-photon detuning (rad/s).  Positive Delta = blue detuning.
    V_vdW : float
        van der Waals interaction energy (rad/s).

    Returns
    -------
    qutip.Qobj
        4x4 Hermitian operator in the {|gg>, |gr>, |rg>, |rr>} basis.
    """
    n = 2
    # Start from zero operator with correct tensor dimensions
    H = 0 * qutip.tensor([qutip.qeye(2)] * n)

    # Driving terms
    for i in range(n):
        H = H + (Omega / 2) * (_sigma_rg(i, n) + _sigma_gr(i, n))
        H = H - Delta * _n_r(i, n)

    # Interaction: V |rr><rr|
    H = H + V_vdW * _n_r(0, n) * _n_r(1, n)

    return H


# ===================================================================
# Three-atom Hamiltonian
# ===================================================================

def build_three_atom_hamiltonian(
    Omega: float,
    Delta: float,
    positions: List[float],
    C6: float,
) -> qutip.Qobj:
    """Build the 3-atom Rydberg Hamiltonian (8x8).

    Parameters
    ----------
    Omega : float
        Global Rabi frequency (rad/s).
    Delta : float
        Global detuning (rad/s).
    positions : list of float
        1-D positions of the 3 atoms in micrometres.
    C6 : float
        van der Waals C6 coefficient (rad/s * um^6).

    Returns
    -------
    qutip.Qobj
        8x8 Hermitian operator.
    """
    n = 3
    # Start from zero operator with correct tensor dimensions
    H = 0 * qutip.tensor([qutip.qeye(2)] * n)

    # Single-atom driving and detuning
    for i in range(n):
        H = H + (Omega / 2) * (_sigma_rg(i, n) + _sigma_gr(i, n))
        H = H - Delta * _n_r(i, n)

    # Pairwise van der Waals interactions
    for i, j in itertools.combinations(range(n), 2):
        R_ij = abs(positions[i] - positions[j])  # μm
        if R_ij < 1e-12:
            raise ValueError(f"Atoms {i} and {j} are at the same position.")
        V_ij = C6 / R_ij**6
        H = H + V_ij * _n_r(i, n) * _n_r(j, n)

    return H


# ===================================================================
# Target / ground states
# ===================================================================

def get_ground_state(n_atoms: int) -> qutip.Qobj:
    """Return |gg...g> (all atoms in ground state)."""
    g = qutip.basis(2, 0)
    return qutip.tensor([g] * n_atoms)


def get_target_state(n_atoms: int) -> qutip.Qobj:
    r"""Return the symmetric W-state for *n_atoms* atoms.

    |W_n> = (1/sqrt(n)) sum_i |g...r_i...g>

    For n=2: |W_2> = (|gr> + |rg>) / sqrt(2)
    For n=3: |W_3> = (|ggrr> + |grg> + |rgg>) / sqrt(3)  -- one excitation
    """
    g = qutip.basis(2, 0)
    r = qutip.basis(2, 1)
    states = []
    for i in range(n_atoms):
        kets = [g] * n_atoms
        kets[i] = r
        states.append(qutip.tensor(kets))
    W = sum(states)
    return W.unit()
