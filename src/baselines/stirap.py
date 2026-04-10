"""STIRAP-like adiabatic pulse for Bell / W-state preparation.

Uses a sin^2 envelope to adiabatically transfer |gg...g> -> |W_n>
through the Rydberg blockade mechanism.

In the blockade regime (V_vdW >> Omega), the doubly-excited state |rr>
is far detuned, and the effective two-level system is |gg> <-> |W> with
coupling Omega_eff = Omega / sqrt(n_atoms).  A pi-pulse on this manifold
requires pulse area (in Omega_eff) equal to pi.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import qutip

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
# Pulse shape
# ===================================================================

def stirap_pulse(t: float, T_gate: float, Omega_max: float) -> float:
    """Symmetric sin^2 envelope: Omega(t) = Omega_max * sin^2(pi*t / T_gate).

    This gives a smooth pulse that turns on and off:
      t = 0      -> 0
      t = T/2    -> Omega_max
      t = T      -> 0

    Pulse area: integral_0^T Omega(t) dt = Omega_max * T / 2.
    """
    return Omega_max * np.sin(np.pi * t / T_gate) ** 2


# ===================================================================
# Run STIRAP
# ===================================================================

def run_stirap(
    scenario: str,
    noise_params: Optional[Dict[str, Any]] = None,
    n_steps: int = 200,
) -> Tuple[float, qutip.Result]:
    """Run STIRAP on given scenario.

    Returns (fidelity, qutip.Result).

    The Omega_max is chosen so that the pulse area on the effective
    two-level system |gg> <-> |W> equals pi:

        integral_0^T  [Omega(t) / sqrt(n)] dt = pi
        =>  Omega_max * T/2 / sqrt(n) = pi
        =>  Omega_max = 2 * sqrt(n) * pi / T

    Parameters
    ----------
    scenario : str
        "A", "B", or "D".
    noise_params : dict or None
        Noise realisation from NoiseModel.sample().  If None, noiseless.
    n_steps : int
        Number of time steps for the solver.
    """
    cfg = SCENARIOS[scenario]
    T_gate = cfg["T_gate"]
    n_atoms = cfg["n_atoms"]
    R = cfg["R"]

    # Compute Omega_max for exact pi/2-rotation on |gg> <-> |W> manifold.
    # The coupling is <gg|H|W> = Omega*sqrt(n)/2, so full transfer requires
    # integral_0^T [Omega(t)*sqrt(n)/2] dt = pi/2.
    # With sin^2(pi*t/T) envelope (integral = T/2):
    #   Omega_max * sqrt(n)/2 * T/2 = pi/2  =>  Omega_max = 2*pi / (sqrt(n)*T)
    Omega_max = 2 * np.pi / (np.sqrt(n_atoms) * T_gate)

    tlist = np.linspace(0, T_gate, n_steps + 1)

    # --- Build operators ---
    H_zero = 0 * qutip.tensor([qutip.qeye(2)] * n_atoms)

    # Driving operator (without the time-dependent coefficient)
    H_drive = H_zero.copy()
    for i in range(n_atoms):
        H_drive = H_drive + 0.5 * (_sigma_rg(i, n_atoms) + _sigma_gr(i, n_atoms))

    # Static part: detuning + vdW interaction
    H_static = H_zero.copy()

    # Detuning
    Delta_base = 0.0  # resonant driving
    if noise_params is not None:
        delta_doppler = noise_params.get("delta_doppler", [0.0] * n_atoms)
        phase_noise = noise_params.get("phase_noise", 0.0)
        delta_phase = phase_noise / T_gate if T_gate > 0 else 0.0
    else:
        delta_doppler = [0.0] * n_atoms
        delta_phase = 0.0

    for i in range(n_atoms):
        Delta_eff = Delta_base + delta_doppler[i] + delta_phase
        H_static = H_static - Delta_eff * _n_r(i, n_atoms)

    # vdW interaction
    if noise_params is not None:
        delta_R = noise_params.get("delta_R", [0.0] * n_atoms)
    else:
        delta_R = [0.0] * n_atoms

    if n_atoms == 2:
        R_eff = R + (delta_R[1] - delta_R[0]) if len(delta_R) >= 2 else R
        R_eff = max(R_eff, 0.1)
        V_vdW = C6_53S / R_eff ** 6
        H_static = H_static + V_vdW * _n_r(0, n_atoms) * _n_r(1, n_atoms)
    elif n_atoms == 3:
        import itertools
        positions = [0.0, R, 2 * R]
        positions = [positions[k] + delta_R[k] for k in range(3)]
        for i, j in itertools.combinations(range(3), 2):
            R_ij = abs(positions[i] - positions[j])
            R_ij = max(R_ij, 0.1)
            V_ij = C6_53S / R_ij ** 6
            H_static = H_static + V_ij * _n_r(i, n_atoms) * _n_r(j, n_atoms)

    # --- Amplitude noise via OU process ---
    ou_eta = None
    if noise_params is not None and noise_params.get("ou_sigma", 0.0) > 0:
        rng = np.random.default_rng()
        nm = NoiseModel(scenario)
        ou_eta = nm.generate_ou_series(rng, tlist)

    # Systematic amplitude bias
    amp_bias = 0.0
    if noise_params is not None:
        amp_bias = noise_params.get("amplitude_bias", 0.0)

    # --- Time-dependent coefficient ---
    if ou_eta is not None:
        from scipy.interpolate import CubicSpline
        eta_interp = CubicSpline(tlist, ou_eta)

        def drive_coeff(t, args=None):
            Om = Omega_max * np.sin(np.pi * t / T_gate) ** 2
            return Om * (1.0 + float(eta_interp(t))) * (1.0 + amp_bias)
    else:
        def drive_coeff(t, args=None):
            return Omega_max * np.sin(np.pi * t / T_gate) ** 2 * (1.0 + amp_bias)

    H = qutip.QobjEvo([H_static, [H_drive, drive_coeff]])

    # --- Collapse operators ---
    c_ops = []
    if noise_params is not None:
        include_decay = noise_params.get("include_decay", True)
        if include_decay:
            c_ops = get_collapse_operators(n_atoms)

    # --- Initial state ---
    psi0 = get_ground_state(n_atoms)

    # --- Solve ---
    result = qutip.mesolve(H, psi0, tlist, c_ops=c_ops)

    # --- Fidelity ---
    target = get_target_state(n_atoms)
    rho_final = result.states[-1]
    fid = compute_fidelity(rho_final, target)

    return fid, result


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    for sc in ["A", "B"]:
        fid, _ = run_stirap(sc, noise_params=None, n_steps=200)
        print(f"STIRAP  scenario {sc}  (noiseless):  F = {fid:.6f}")
