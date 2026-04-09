"""Gymnasium environment for 2-atom Rydberg Bell state preparation.

Observation : rho(t) flattened as [real, imag] -> 32-dim float32 vector
Action      : (Omega_norm, Delta_norm) each in [-1, 1]
Reward      : sparse terminal fidelity F = <target|rho|target>
Dynamics    : constant-H Lindblad propagation per time slice dt = T_gate / n_steps
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from scipy.linalg import expm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.physics.constants import (
    C6_53S,
    OU_SIGMA,
    SCENARIOS,
    SIGMA_DOPPLER,
    SIGMA_POSITION,
    TAU_EFF_53S,
)
from src.physics.hamiltonian import (
    build_two_atom_hamiltonian,
    get_ground_state,
    get_target_state,
)
from src.physics.lindblad import compute_fidelity, get_collapse_operators
from src.physics.noise_model import NoiseModel


class RydbergBellEnv(gymnasium.Env):
    """RL environment for 2-atom Bell state preparation.

    At each step the agent chooses (Omega, Delta), which are held constant
    for one time slice dt.  The density matrix is propagated via the
    Lindblad master equation (using matrix exponentiation of the
    superoperator for speed).  At the terminal step the agent receives
    the fidelity with the target W-state as reward.

    Parameters
    ----------
    scenario : str
        Scenario key ("A", "B", "D").  Only 2-atom scenarios supported.
    n_steps : int
        Number of control steps per episode (T_gate is divided into n_steps).
    use_noise : bool
        If True, domain-randomised noise is applied (resampled every reset).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "B",
        n_steps: int = 30,
        use_noise: bool = True,
    ) -> None:
        super().__init__()

        # ---- scenario config ------------------------------------------------
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'")
        self.scenario = scenario
        self.cfg = SCENARIOS[scenario]
        if self.cfg["n_atoms"] != 2:
            raise ValueError("RydbergBellEnv supports 2-atom scenarios only.")

        self.n_steps = n_steps
        self.use_noise = use_noise
        self.T_gate: float = self.cfg["T_gate"]
        self.dt: float = self.T_gate / self.n_steps
        self.Omega_max: float = self.cfg["Omega"]  # rad/s
        self.R_base: float = self.cfg["R"]  # um
        self.C6: float = C6_53S
        self.dim = 4  # 2-qubit Hilbert space dimension

        # ---- noise model ----------------------------------------------------
        self.noise_model = NoiseModel(scenario) if use_noise else None

        # ---- target state ---------------------------------------------------
        self._target_ket = get_target_state(2)
        self._target_dm_np = (self._target_ket * self._target_ket.dag()).full()

        # ---- spaces ---------------------------------------------------------
        # observation: real + imag parts of 4x4 density matrix = 2 * 16 = 32
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(32,), dtype=np.float32
        )
        # action: normalised (Omega, Delta) in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ---- internal state -------------------------------------------------
        self._rho_np: Optional[np.ndarray] = None  # 4x4 complex
        self._step_count: int = 0
        self._noise_params: Dict[str, Any] = {}
        self._V_vdW: float = C6_53S / self.R_base**6
        self._c_ops_np: list = []  # numpy arrays of collapse ops
        self._rng: np.random.Generator = np.random.default_rng()

        # Pre-compute identity for superoperator building
        self._I_d = np.eye(self.dim, dtype=complex)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to |gg> and resample noise."""
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # Initial state: |gg><gg|
        gg = get_ground_state(2)
        self._rho_np = (gg * gg.dag()).full()
        self._step_count = 0

        # Domain randomisation: resample noise once per episode
        if self.use_noise and self.noise_model is not None:
            self._noise_params = self.noise_model.sample(self._rng)
            # Update V_vdW with position jitter
            delta_R = self._noise_params.get("delta_R", [0.0, 0.0])
            self._V_vdW = NoiseModel.compute_V_vdW(self.R_base, delta_R, self.C6)
        else:
            self._noise_params = {
                "delta_doppler": [0.0, 0.0],
                "delta_R": [0.0, 0.0],
                "ou_sigma": 0.0,
                "ou_tau": 1.0,
                "phase_noise": 0.0,
            }
            self._V_vdW = self.C6 / self.R_base**6

        # Collapse operators (numpy arrays)
        if self.use_noise and "decay" in self.cfg.get("noise_sources", []):
            c_ops_qt = get_collapse_operators(2)
            self._c_ops_np = [c.full() for c in c_ops_qt]
        else:
            self._c_ops_np = []

        obs = self._rho_to_obs(self._rho_np)
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take one control step.

        Parameters
        ----------
        action : ndarray, shape (2,)
            action[0] -> Omega in [0, 2*Omega_max]
            action[1] -> Delta in [-Omega_max, Omega_max]
        """
        # Map action from [-1, 1] to physical parameters
        a = np.clip(action, -1.0, 1.0)
        Omega = float((a[0] + 1.0) / 2.0 * 2.0 * self.Omega_max)  # [0, 2*Omega_max]
        Delta = float(a[1] * self.Omega_max)  # [-Omega_max, Omega_max]

        # Apply amplitude noise (simplified OU: per-step static perturbation)
        ou_sigma = self._noise_params.get("ou_sigma", 0.0)
        if ou_sigma > 0:
            xi = float(self._rng.normal(0, ou_sigma))
            Omega = Omega * (1.0 + xi)

        # Apply Doppler shift to detuning
        doppler = self._noise_params.get("delta_doppler", [0.0, 0.0])
        # Average Doppler shift affects global detuning
        Delta_eff = Delta + 0.5 * (doppler[0] + doppler[1])

        # Build Hamiltonian (static for this time slice)
        H_qt = build_two_atom_hamiltonian(Omega, Delta_eff, self._V_vdW)
        H_np = H_qt.full()

        # Propagate via superoperator expm
        self._rho_np = self._propagate_lindblad(H_np, self._rho_np, self.dt)

        self._step_count += 1
        terminated = self._step_count >= self.n_steps
        truncated = False

        # Reward: sparse terminal fidelity
        reward = 0.0
        info: Dict[str, Any] = {}
        if terminated:
            fid = self._compute_fidelity_np(self._rho_np)
            reward = float(fid)
            info["fidelity"] = float(fid)

        obs = self._rho_to_obs(self._rho_np)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _propagate_lindblad(
        self, H: np.ndarray, rho: np.ndarray, dt: float
    ) -> np.ndarray:
        """Propagate density matrix by dt under Lindblad master equation.

        Uses matrix exponentiation of the 16x16 Liouvillian superoperator
        for a constant Hamiltonian, which is faster than qutip.mesolve for
        small systems.
        """
        d = self.dim
        I = self._I_d

        # Unitary part: -i(H x I - I x H^T)
        L = -1j * (np.kron(H, I) - np.kron(I, H.T))

        # Dissipative part
        for Lk in self._c_ops_np:
            Lk_dag = Lk.conj().T
            LdL = Lk_dag @ Lk
            L += (
                np.kron(Lk, Lk.conj())
                - 0.5 * (np.kron(LdL, I) + np.kron(I, LdL.T))
            )

        # Propagate: rho_vec(dt) = expm(L * dt) @ rho_vec(0)
        prop = expm(L * dt)
        rho_vec = rho.flatten(order="C")  # row-major vectorisation
        rho_new_vec = prop @ rho_vec
        rho_new = rho_new_vec.reshape(d, d)

        # Enforce Hermiticity and trace normalisation
        rho_new = 0.5 * (rho_new + rho_new.conj().T)
        rho_new /= np.trace(rho_new).real

        return rho_new

    def _compute_fidelity_np(self, rho: np.ndarray) -> float:
        """Compute F = Tr(rho * rho_target) using numpy."""
        return np.trace(rho @ self._target_dm_np).real

    @staticmethod
    def _rho_to_obs(rho: np.ndarray) -> np.ndarray:
        """Flatten density matrix to 32-dim float32 observation.

        Concatenates real and imaginary parts of the 4x4 matrix.
        Values are clipped to [-1, 1].
        """
        real_part = rho.real.flatten()
        imag_part = rho.imag.flatten()
        obs = np.concatenate([real_part, imag_part]).astype(np.float32)
        obs = np.clip(obs, -1.0, 1.0)
        return obs
