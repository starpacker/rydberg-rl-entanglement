"""Collapse operators and mesolve wrapper for Rydberg simulations.

Provides:
- Lindblad collapse operators for spontaneous decay |r> -> |g>
- A convenience wrapper around qutip.mesolve that incorporates
  Doppler shifts, position jitter, amplitude noise, and phase noise
  into a time-dependent Hamiltonian.
- Fidelity computation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import qutip

from src.physics.constants import C6_53S, TAU_EFF_53S
from src.physics.hamiltonian import (
    _n_r,
    _sigma_gr,
    _sigma_rg,
    build_two_atom_hamiltonian,
)


# ===================================================================
# Collapse operators
# ===================================================================

def get_collapse_operators(
    n_atoms: int,
    tau_eff: float = TAU_EFF_53S,
) -> List[qutip.Qobj]:
    r"""Return Lindblad collapse operators for spontaneous decay.

    Each atom decays independently: L_i = sqrt(gamma) |g_i><r_i|
    where gamma = 1 / tau_eff.

    Parameters
    ----------
    n_atoms : int
        Number of atoms (2 or 3).
    tau_eff : float
        Effective Rydberg lifetime (s), including BBR at 300 K.

    Returns
    -------
    list of qutip.Qobj
        One collapse operator per atom.
    """
    gamma = 1.0 / tau_eff
    c_ops = []
    for i in range(n_atoms):
        c_ops.append(np.sqrt(gamma) * _sigma_gr(i, n_atoms))
    return c_ops


# ===================================================================
# Time-dependent Hamiltonian builder (internal)
# ===================================================================

def _build_td_hamiltonian(
    n_atoms: int,
    Omega_base: float,
    Delta_base: float,
    R_base: float,
    C6: float,
    noise_params: Dict[str, Any],
    ou_series: Optional[np.ndarray],
    tlist: np.ndarray,
) -> qutip.QobjEvo:
    """Build a time-dependent QobjEvo incorporating all noise sources.

    The total Hamiltonian is:

        H(t) = H_drive(t) + H_detuning + H_interaction

    where:
        H_drive(t) = (Omega_base / 2) * (1 + eta(t)) * sum_i (|r_i><g_i| + h.c.)
        H_detuning = -sum_i (Delta_base + delta_doppler_i + phi_noise / T) |r_i><r_i|
        H_interaction = V_vdW(R_eff) |rr><rr| (or pairwise for 3 atoms)

    eta(t) is the Ornstein-Uhlenbeck amplitude noise.
    """
    delta_doppler = noise_params.get("delta_doppler", [0.0] * n_atoms)
    delta_R = noise_params.get("delta_R", [0.0] * n_atoms)
    phase_noise = noise_params.get("phase_noise", 0.0)
    ou_sigma = noise_params.get("ou_sigma", 0.0)

    # --- Static part: detuning + interaction ---
    # Detuning: each atom gets Delta_base + Doppler shift
    # Phase noise is modelled as an effective detuning: delta_phase = phase_noise / T_gate
    T_gate = tlist[-1] - tlist[0] if len(tlist) > 1 else 1.0
    delta_phase = phase_noise / T_gate if T_gate > 0 else 0.0

    H_static = 0 * qutip.tensor([qutip.qeye(2)] * n_atoms)

    for i in range(n_atoms):
        Delta_eff_i = Delta_base + delta_doppler[i] + delta_phase
        H_static = H_static - Delta_eff_i * _n_r(i, n_atoms)

    # Interaction (van der Waals)
    if n_atoms == 2:
        R_eff = R_base + (delta_R[1] - delta_R[0]) if len(delta_R) >= 2 else R_base
        R_eff = max(R_eff, 0.1)  # safety clamp
        V_vdW = C6 / R_eff**6
        H_static = H_static + V_vdW * _n_r(0, n_atoms) * _n_r(1, n_atoms)
    elif n_atoms == 3:
        # Assume linear chain with spacing R_base
        positions = [0.0, R_base, 2 * R_base]
        positions = [positions[k] + delta_R[k] for k in range(3)]
        import itertools
        for i, j in itertools.combinations(range(3), 2):
            R_ij = abs(positions[i] - positions[j])
            R_ij = max(R_ij, 0.1)
            V_ij = C6 / R_ij**6
            H_static = H_static + V_ij * _n_r(i, n_atoms) * _n_r(j, n_atoms)

    # --- Driving term ---
    # H_drive = (Omega_base / 2) * (1 + eta(t)) * sum_i (sigma_rg_i + sigma_gr_i)
    H_drive_op = 0 * qutip.tensor([qutip.qeye(2)] * n_atoms)
    for i in range(n_atoms):
        H_drive_op = H_drive_op + (Omega_base / 2) * (_sigma_rg(i, n_atoms) + _sigma_gr(i, n_atoms))

    if ou_sigma > 0 and ou_series is not None and len(ou_series) == len(tlist):
        # Time-dependent driving with amplitude noise
        # H = H_static + H_drive_op * (1 + eta(t))
        # = H_static + H_drive_op + H_drive_op * eta(t)
        # The constant part:
        H_const = H_static + H_drive_op

        # The time-dependent part: H_drive_op * eta(t)
        # Use cubic spline interpolation for the coefficient
        from scipy.interpolate import CubicSpline
        eta_interp = CubicSpline(tlist, ou_series)

        def eta_coeff(t, args=None):
            return float(eta_interp(t))

        return qutip.QobjEvo([H_const, [H_drive_op, eta_coeff]])
    else:
        # No amplitude noise: fully static Hamiltonian
        return qutip.QobjEvo(H_static + H_drive_op)


# ===================================================================
# Master equation solver wrapper
# ===================================================================

def mesolve_with_noise(
    H_base: Optional[qutip.Qobj],
    psi0: qutip.Qobj,
    tlist: np.ndarray,
    c_ops: List[qutip.Qobj],
    noise_params: Dict[str, Any],
    n_atoms: int,
    Omega_base: float,
    Delta_base: float,
    R_base: float,
    C6: float = C6_53S,
    ou_series: Optional[np.ndarray] = None,
    e_ops: Optional[List[qutip.Qobj]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> qutip.Result:
    """Run qutip.mesolve with noise-modified Hamiltonian.

    Parameters
    ----------
    H_base : qutip.Qobj or None
        If provided, used as-is (ignoring noise_params for H construction).
        If None, the Hamiltonian is built from parameters + noise.
    psi0 : qutip.Qobj
        Initial state (ket or density matrix).
    tlist : ndarray
        Time points.
    c_ops : list of qutip.Qobj
        Collapse operators (from get_collapse_operators).
    noise_params : dict
        Noise realisation from NoiseModel.sample().
    n_atoms : int
        Number of atoms.
    Omega_base : float
        Base Rabi frequency (rad/s).
    Delta_base : float
        Base detuning (rad/s).
    R_base : float
        Base atom separation (μm).
    C6 : float
        C6 coefficient.
    ou_series : ndarray or None
        Pre-generated OU amplitude noise series.
    e_ops : list or None
        Expectation value operators.
    options : dict or None
        Options passed to qutip.mesolve.

    Returns
    -------
    qutip.Result
    """
    if H_base is not None:
        H = H_base
    else:
        H = _build_td_hamiltonian(
            n_atoms=n_atoms,
            Omega_base=Omega_base,
            Delta_base=Delta_base,
            R_base=R_base,
            C6=C6,
            noise_params=noise_params,
            ou_series=ou_series,
            tlist=tlist,
        )

    kw = {}
    if options is not None:
        kw["options"] = options
    if e_ops is not None:
        kw["e_ops"] = e_ops

    return qutip.mesolve(H, psi0, tlist, c_ops=c_ops if c_ops else [], **kw)


# ===================================================================
# Fidelity
# ===================================================================

def compute_fidelity(rho: qutip.Qobj, target_ket: qutip.Qobj) -> float:
    """Compute state fidelity F = <target|rho|target>.

    Parameters
    ----------
    rho : qutip.Qobj
        Density matrix (or ket -- will be converted).
    target_ket : qutip.Qobj
        Target pure state (ket).

    Returns
    -------
    float
        Fidelity in [0, 1].
    """
    if rho.isket:
        rho = qutip.ket2dm(rho)
    # F = <target|rho|target>
    bra = target_ket.dag()
    fid = (bra * rho * target_ket)
    # In QuTiP 5, ket.dag() * oper * ket returns a scalar complex
    if isinstance(fid, complex):
        return float(np.real(fid))
    elif isinstance(fid, qutip.Qobj):
        return float(np.real(fid.full()[0, 0]))
    return float(np.real(fid))
