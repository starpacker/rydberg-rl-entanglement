"""GRAPE optimal control for Rydberg Bell / W-state preparation.

Gradient Ascent Pulse Engineering (GRAPE) uses piecewise-constant control
pulses and gradient-based optimisation to maximise state-transfer fidelity.

Implementation: unitary propagation (no noise during optimisation),
with optional noise evaluation of the optimised pulse via mesolve.

Speed optimisation: pre-build operator matrices as numpy arrays and use
scipy.linalg.expm for matrix exponential, avoiding QuTiP overhead in the
inner loop.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import qutip
from scipy.linalg import expm

# Allow running as a script
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.physics.constants import C6_53S, SCENARIOS
from src.physics.hamiltonian import (
    _n_r,
    _sigma_gr,
    _sigma_rg,
    get_ground_state,
    get_target_state,
)
from src.physics.lindblad import compute_fidelity, get_collapse_operators
from src.physics.noise_model import NoiseModel


# ===================================================================
# Helpers (numpy-based for speed)
# ===================================================================

def _build_operators_np(n_atoms: int, V_vdW: float):
    """Pre-build operator matrices as numpy arrays.

    Returns (H_vdW, H_drive, H_det) as dense complex128 arrays.
    H(Omega, Delta) = H_vdW + Omega * H_drive - Delta * H_det
    """
    dim = 2 ** n_atoms

    # Drive operator: sum_i 0.5 * (sigma_rg_i + sigma_gr_i)
    H_drive = np.zeros((dim, dim), dtype=complex)
    for i in range(n_atoms):
        op = 0.5 * (_sigma_rg(i, n_atoms) + _sigma_gr(i, n_atoms))
        H_drive += op.full()

    # Detuning operator: sum_i |r_i><r_i|
    H_det = np.zeros((dim, dim), dtype=complex)
    for i in range(n_atoms):
        H_det += _n_r(i, n_atoms).full()

    # vdW interaction
    H_vdW = np.zeros((dim, dim), dtype=complex)
    if n_atoms == 2:
        H_vdW += V_vdW * (_n_r(0, n_atoms) * _n_r(1, n_atoms)).full()
    elif n_atoms == 3:
        import itertools
        for i, j in itertools.combinations(range(n_atoms), 2):
            dist = abs(j - i)
            V_ij = V_vdW / dist ** 6
            H_vdW += V_ij * (_n_r(i, n_atoms) * _n_r(j, n_atoms)).full()

    return H_vdW, H_drive, H_det


def _propagate_np(
    H_vdW: np.ndarray,
    H_drive: np.ndarray,
    H_det: np.ndarray,
    omega_pulse: np.ndarray,
    delta_pulse: np.ndarray,
    dt: float,
    psi0: np.ndarray,
) -> np.ndarray:
    """Forward-propagate using numpy/scipy.  Returns final state vector."""
    psi = psi0.copy()
    for k in range(len(omega_pulse)):
        H_k = H_vdW + omega_pulse[k] * H_drive - delta_pulse[k] * H_det
        U_k = expm(-1j * H_k * dt)
        psi = U_k @ psi
    return psi


def _fidelity_np(psi: np.ndarray, target: np.ndarray) -> float:
    """State fidelity |<target|psi>|^2."""
    return float(np.abs(target.conj() @ psi) ** 2)


# ===================================================================
# QuTiP-based propagation (for external use)
# ===================================================================

def _build_H_k(
    n_atoms: int,
    Omega_k: float,
    Delta_k: float,
    V_vdW: float,
) -> qutip.Qobj:
    """Build Hamiltonian for one piecewise-constant segment."""
    H = 0 * qutip.tensor([qutip.qeye(2)] * n_atoms)
    for i in range(n_atoms):
        H = H + (Omega_k / 2) * (_sigma_rg(i, n_atoms) + _sigma_gr(i, n_atoms))
        H = H - Delta_k * _n_r(i, n_atoms)
    if n_atoms == 2:
        H = H + V_vdW * _n_r(0, n_atoms) * _n_r(1, n_atoms)
    elif n_atoms == 3:
        import itertools
        for i, j in itertools.combinations(range(n_atoms), 2):
            dist = abs(j - i)
            H = H + (V_vdW / dist ** 6) * _n_r(i, n_atoms) * _n_r(j, n_atoms)
    return H


def _propagate(
    n_atoms: int,
    omega_pulse: np.ndarray,
    delta_pulse: np.ndarray,
    V_vdW: float,
    dt: float,
    psi0: qutip.Qobj,
) -> qutip.Qobj:
    """Forward-propagate through all segments (QuTiP).  Returns final ket."""
    psi = psi0.copy()
    for k in range(len(omega_pulse)):
        H_k = _build_H_k(n_atoms, omega_pulse[k], delta_pulse[k], V_vdW)
        U_k = (-1j * H_k * dt).expm()
        psi = U_k * psi
    return psi


def _fidelity_ket(psi_final: qutip.Qobj, target: qutip.Qobj) -> float:
    """State fidelity |<target|psi>|^2 for kets."""
    overlap = target.dag() * psi_final
    if isinstance(overlap, qutip.Qobj):
        val = overlap.full()[0, 0]
    else:
        val = complex(overlap)
    return float(np.abs(val) ** 2)


# ===================================================================
# GRAPE optimisation
# ===================================================================

def run_grape(
    scenario: str,
    n_steps: int = 30,
    n_iter: int = 500,
    noise_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """GRAPE optimization.

    Returns (fidelity, omega_pulse, delta_pulse).

    Parameters
    ----------
    scenario : str
        "A", "B", or "D".
    n_steps : int
        Number of piecewise-constant segments.
    n_iter : int
        Number of gradient-ascent iterations.
    noise_params : dict or None
        If given, the optimised pulse is evaluated under noise via mesolve.
        The optimisation itself is always noiseless (unitary).
    verbose : bool
        Print progress every 50 iterations.
    """
    cfg = SCENARIOS[scenario]
    T_gate = cfg["T_gate"]
    n_atoms = cfg["n_atoms"]
    R = cfg["R"]
    Omega_cfg = cfg["Omega"]

    dt = T_gate / n_steps
    V_vdW = C6_53S / R ** 6

    psi0_q = get_ground_state(n_atoms)
    target_q = get_target_state(n_atoms)

    # Numpy vectors for fast inner loop
    psi0_np = psi0_q.full().flatten()
    target_np = target_q.full().flatten()

    Omega_max = Omega_cfg  # reference amplitude

    # Pre-build operator matrices
    H_vdW, H_drive, H_det = _build_operators_np(n_atoms, V_vdW)

    # --- Initialise with constant near pi-pulse amplitude ---
    # For the |gg> <-> |W> manifold, coupling is Omega*sqrt(n)/2.
    # Full transfer: Omega*sqrt(n)/2 * T = pi/2 => Omega = pi / (sqrt(n)*T)
    Omega_init = np.pi / (np.sqrt(n_atoms) * T_gate)
    omega_pulse = np.full(n_steps, Omega_init)
    delta_pulse = np.zeros(n_steps)

    # Clip bounds
    omega_lo, omega_hi = 0.0, 2 * Omega_max
    delta_lo, delta_hi = -Omega_max, Omega_max

    # Gradient ascent parameters
    eps = Omega_max * 1e-4
    lr = Omega_max * 0.01

    best_fid = 0.0
    best_omega = omega_pulse.copy()
    best_delta = delta_pulse.copy()

    for iteration in range(n_iter):
        # Current fidelity
        psi_f = _propagate_np(H_vdW, H_drive, H_det, omega_pulse, delta_pulse, dt, psi0_np)
        fid = _fidelity_np(psi_f, target_np)

        if fid > best_fid:
            best_fid = fid
            best_omega = omega_pulse.copy()
            best_delta = delta_pulse.copy()

        if verbose and (iteration % 50 == 0 or iteration == n_iter - 1):
            print(f"  GRAPE iter {iteration:4d}:  F = {fid:.6f}")

        if fid > 0.9999:
            break

        # --- Numerical gradient for omega ---
        grad_omega = np.zeros(n_steps)
        for k in range(n_steps):
            omega_pulse[k] += eps
            psi_plus = _propagate_np(H_vdW, H_drive, H_det, omega_pulse, delta_pulse, dt, psi0_np)
            f_plus = _fidelity_np(psi_plus, target_np)
            omega_pulse[k] -= 2 * eps
            psi_minus = _propagate_np(H_vdW, H_drive, H_det, omega_pulse, delta_pulse, dt, psi0_np)
            f_minus = _fidelity_np(psi_minus, target_np)
            omega_pulse[k] += eps  # restore
            grad_omega[k] = (f_plus - f_minus) / (2 * eps)

        # --- Numerical gradient for delta ---
        grad_delta = np.zeros(n_steps)
        for k in range(n_steps):
            delta_pulse[k] += eps
            psi_plus = _propagate_np(H_vdW, H_drive, H_det, omega_pulse, delta_pulse, dt, psi0_np)
            f_plus = _fidelity_np(psi_plus, target_np)
            delta_pulse[k] -= 2 * eps
            psi_minus = _propagate_np(H_vdW, H_drive, H_det, omega_pulse, delta_pulse, dt, psi0_np)
            f_minus = _fidelity_np(psi_minus, target_np)
            delta_pulse[k] += eps
            grad_delta[k] = (f_plus - f_minus) / (2 * eps)

        # --- Gradient ascent step ---
        omega_pulse = omega_pulse + lr * grad_omega
        delta_pulse = delta_pulse + lr * grad_delta

        # Clip to physical bounds
        omega_pulse = np.clip(omega_pulse, omega_lo, omega_hi)
        delta_pulse = np.clip(delta_pulse, delta_lo, delta_hi)

    # Use best found
    omega_pulse = best_omega
    delta_pulse = best_delta

    # --- Final fidelity ---
    if noise_params is not None:
        # Evaluate the optimised pulse under noise via mesolve
        fid = _evaluate_with_noise(
            scenario, omega_pulse, delta_pulse, noise_params
        )
    else:
        psi_f = _propagate_np(H_vdW, H_drive, H_det, omega_pulse, delta_pulse, dt, psi0_np)
        fid = _fidelity_np(psi_f, target_np)

    return fid, omega_pulse, delta_pulse


# ===================================================================
# Noise evaluation for an optimised pulse
# ===================================================================

def _evaluate_with_noise(
    scenario: str,
    omega_pulse: np.ndarray,
    delta_pulse: np.ndarray,
    noise_params: Dict[str, Any],
) -> float:
    """Evaluate a piecewise-constant pulse under noise via mesolve."""
    cfg = SCENARIOS[scenario]
    T_gate = cfg["T_gate"]
    n_atoms = cfg["n_atoms"]
    R = cfg["R"]

    n_steps = len(omega_pulse)
    dt = T_gate / n_steps

    # Build time-dependent Hamiltonian
    tlist = np.linspace(0, T_gate, n_steps * 10 + 1)

    delta_doppler = noise_params.get("delta_doppler", [0.0] * n_atoms)
    delta_R = noise_params.get("delta_R", [0.0] * n_atoms)
    phase_noise = noise_params.get("phase_noise", 0.0)
    delta_phase = phase_noise / T_gate if T_gate > 0 else 0.0

    # Static detuning + interaction
    H_static = 0 * qutip.tensor([qutip.qeye(2)] * n_atoms)
    for i in range(n_atoms):
        Delta_eff = delta_doppler[i] + delta_phase
        H_static = H_static - Delta_eff * _n_r(i, n_atoms)

    if n_atoms == 2:
        R_eff = R + (delta_R[1] - delta_R[0]) if len(delta_R) >= 2 else R
        R_eff = max(R_eff, 0.1)
        V_vdW = C6_53S / R_eff ** 6
        H_static = H_static + V_vdW * _n_r(0, n_atoms) * _n_r(1, n_atoms)

    # Drive operator
    H_drive = 0 * qutip.tensor([qutip.qeye(2)] * n_atoms)
    for i in range(n_atoms):
        H_drive = H_drive + 0.5 * (_sigma_rg(i, n_atoms) + _sigma_gr(i, n_atoms))

    # Detuning operator
    H_det = 0 * qutip.tensor([qutip.qeye(2)] * n_atoms)
    for i in range(n_atoms):
        H_det = H_det - _n_r(i, n_atoms)

    # OU amplitude noise
    ou_sigma = noise_params.get("ou_sigma", 0.0)
    amp_bias = noise_params.get("amplitude_bias", 0.0)
    if ou_sigma > 0:
        rng = np.random.default_rng()
        nm = NoiseModel(scenario)
        ou_series = nm.generate_ou_series(rng, tlist)
    else:
        ou_series = np.zeros(len(tlist))

    from scipy.interpolate import CubicSpline
    ou_interp = CubicSpline(tlist, ou_series)

    def drive_coeff(t, args=None):
        k = min(int(t / dt), n_steps - 1)
        k = max(k, 0)
        Om = omega_pulse[k]
        eta = float(ou_interp(t)) if ou_sigma > 0 else 0.0
        return Om * (1.0 + eta) * (1.0 + amp_bias)

    def det_coeff(t, args=None):
        k = min(int(t / dt), n_steps - 1)
        k = max(k, 0)
        return delta_pulse[k]

    H = qutip.QobjEvo([H_static, [H_drive, drive_coeff], [H_det, det_coeff]])

    c_ops = []
    include_decay = noise_params.get("include_decay", True)
    if include_decay:
        c_ops = get_collapse_operators(n_atoms)
    psi0 = get_ground_state(n_atoms)

    result = qutip.mesolve(H, psi0, tlist, c_ops=c_ops)

    target = get_target_state(n_atoms)
    return compute_fidelity(result.states[-1], target)


# ===================================================================
# Standalone evaluation helper (for evaluate.py)
# ===================================================================

def run_grape_eval(
    scenario: str,
    omega_pulse: np.ndarray,
    delta_pulse: np.ndarray,
    noise_params: Optional[Dict[str, Any]] = None,
) -> float:
    """Evaluate a pre-optimised GRAPE pulse, optionally under noise.

    Returns fidelity.
    """
    cfg = SCENARIOS[scenario]
    n_atoms = cfg["n_atoms"]
    T_gate = cfg["T_gate"]
    R = cfg["R"]
    n_steps = len(omega_pulse)
    dt = T_gate / n_steps
    V_vdW = C6_53S / R ** 6

    psi0 = get_ground_state(n_atoms)
    target = get_target_state(n_atoms)

    if noise_params is not None:
        return _evaluate_with_noise(scenario, omega_pulse, delta_pulse, noise_params)
    else:
        psi_f = _propagate(n_atoms, omega_pulse, delta_pulse, V_vdW, dt, psi0)
        return _fidelity_ket(psi_f, target)


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print("=== GRAPE on scenario B (noiseless, 200 iters) ===")
    fid, omega, delta = run_grape("B", n_steps=30, n_iter=200, verbose=True)
    print(f"\nFinal fidelity: {fid:.6f}")
    print(f"Omega range: [{omega.min():.2e}, {omega.max():.2e}]")
    print(f"Delta range: [{delta.min():.2e}, {delta.max():.2e}]")
