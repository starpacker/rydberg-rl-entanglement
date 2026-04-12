"""Gymnasium environment for 2-atom Rydberg Bell state preparation.

Observation : rho(t) flattened as [real, imag] -> 32-dim float32 vector
Action      : (Omega_norm, Delta_norm) each in [-1, 1]
Reward      : terminal fidelity + optional per-step shaping
Dynamics    : constant-H Lindblad propagation per time slice dt = T_gate / n_steps

Noise model aligned with baselines (lindblad.py):
  - Amplitude: Ornstein-Uhlenbeck process (pre-generated per episode)
  - Doppler: per-atom independent shifts applied to per-atom detuning
  - Position: modifies V_vdW via R_eff
  - Phase: static offset modelled as effective detuning
  - Decay: Lindblad collapse operators
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from scipy.linalg import expm

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.physics.constants import (
    C6_53S,
    SCENARIOS,
    TAU_EFF_53S,
)
from src.physics.hamiltonian import (
    _n_r,
    _sigma_gr,
    _sigma_rg,
    get_ground_state,
    get_target_state,
)
from src.physics.lindblad import compute_fidelity, get_collapse_operators
from src.physics.noise_model import NoiseModel


class RydbergBellEnv(gymnasium.Env):
    """RL environment for 2-atom Bell state preparation.

    Noise model is now aligned with the baseline evaluation (lindblad.py):
    - OU amplitude noise is pre-generated as a time series at reset()
    - Doppler shifts are applied per-atom (not averaged)
    - Position jitter modifies V_vdW

    Parameters
    ----------
    scenario : str
        Scenario key ("A", "B", "C").  Only 2-atom scenarios supported.
    n_steps : int
        Number of control steps per episode.
    use_noise : bool
        If True, domain-randomised noise is applied (resampled every reset).
    reward_shaping_alpha : float
        Weight for per-step fidelity improvement reward. 0 = sparse only.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "B",
        n_steps: int = 30,
        use_noise: bool = True,
        reward_shaping_alpha: float = 0.1,
        obs_include_time: bool = False,
        obs_mode: str = "full",
        noise_scale: float = None,
    ) -> None:
        super().__init__()

        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'")
        if obs_mode not in ("full", "time_only"):
            raise ValueError(f"obs_mode must be 'full' or 'time_only', got '{obs_mode}'")
        self.scenario = scenario
        self.cfg = SCENARIOS[scenario]
        if self.cfg["n_atoms"] != 2:
            raise ValueError("RydbergBellEnv supports 2-atom scenarios only.")

        self.n_steps = n_steps
        self.use_noise = use_noise
        self.reward_shaping_alpha = reward_shaping_alpha
        self.obs_include_time = obs_include_time
        self.obs_mode = obs_mode
        self.noise_scale = noise_scale
        self.T_gate: float = self.cfg["T_gate"]
        self.dt: float = self.T_gate / self.n_steps
        self.Omega_max: float = self.cfg["Omega"]
        self.R_base: float = self.cfg["R"]
        self.C6: float = C6_53S
        self.dim = 4  # 2-qubit Hilbert space dimension

        self.noise_model = NoiseModel(scenario, noise_scale=noise_scale) if use_noise else None

        self._target_ket = get_target_state(2)
        self._target_dm_np = (self._target_ket * self._target_ket.dag()).full()

        # Observation space depends on obs_mode:
        #   "full": rho flattened [real, imag] (32) + optional t/T (1) -> 32 or 33
        #   "time_only": just (t/T,) -> 1  (open-loop: policy is pi(a|t))
        if self.obs_mode == "time_only":
            obs_dim = 1
        else:
            obs_dim = 33 if self.obs_include_time else 32
        self.observation_space = spaces.Box(
            low=-1.0 if self.obs_mode == "full" else 0.0,
            high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        # Action: normalised (Omega, Delta) in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Internal state
        self._rho_np: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._noise_params: Dict[str, Any] = {}
        self._V_vdW: float = C6_53S / self.R_base**6
        self._c_ops_np: list = []
        self._rng: np.random.Generator = np.random.default_rng()
        self._prev_fidelity: float = 0.0

        # Pre-generated OU amplitude noise series (aligned with baselines)
        self._ou_series: Optional[np.ndarray] = None

        # Per-atom Doppler shifts
        self._delta_doppler: list = [0.0, 0.0]
        # Phase noise as effective detuning
        self._delta_phase: float = 0.0
        # Systematic amplitude bias
        self._amplitude_bias: float = 0.0

        # Pre-compute operators for per-atom Hamiltonian building
        self._I_d = np.eye(self.dim, dtype=complex)
        self._precompute_operators()

    def _precompute_operators(self) -> None:
        """Pre-compute numpy operator matrices for fast Hamiltonian assembly."""
        n = 2
        # Drive operator: sum_i 0.5 * (sigma_rg_i + sigma_gr_i)
        self._H_drive_np = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(n):
            op = 0.5 * (_sigma_rg(i, n) + _sigma_gr(i, n))
            self._H_drive_np += op.full()

        # Per-atom number operators for detuning
        self._n_r_np = []
        for i in range(n):
            self._n_r_np.append(_n_r(i, n).full())

        # Interaction operator: |rr><rr|
        self._n_rr_np = (_n_r(0, n) * _n_r(1, n)).full()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to |gg> and resample noise."""
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # Initial state: |gg><gg|
        gg = get_ground_state(2)
        self._rho_np = (gg * gg.dag()).full()
        self._step_count = 0
        self._prev_fidelity = self._compute_fidelity_np(self._rho_np)

        if self.use_noise and self.noise_model is not None:
            self._noise_params = self.noise_model.sample(self._rng)

            # Per-atom Doppler shifts (NOT averaged)
            self._delta_doppler = self._noise_params.get("delta_doppler", [0.0, 0.0])

            # Phase noise as effective detuning
            phase_noise = self._noise_params.get("phase_noise", 0.0)
            self._delta_phase = phase_noise / self.T_gate if self.T_gate > 0 else 0.0

            # Position jitter -> V_vdW
            delta_R = self._noise_params.get("delta_R", [0.0, 0.0])
            self._V_vdW = NoiseModel.compute_V_vdW(self.R_base, delta_R, self.C6)

            # Systematic amplitude bias
            self._amplitude_bias = self._noise_params.get("amplitude_bias", 0.0)

            # Pre-generate OU amplitude noise series for full episode
            ou_sigma = self._noise_params.get("ou_sigma", 0.0)
            if ou_sigma > 0:
                tlist = np.linspace(0, self.T_gate, self.n_steps + 1)
                self._ou_series = self.noise_model.generate_ou_series(self._rng, tlist)
            else:
                self._ou_series = None
        else:
            self._noise_params = {}
            self._delta_doppler = [0.0, 0.0]
            self._delta_phase = 0.0
            self._V_vdW = self.C6 / self.R_base**6
            self._amplitude_bias = 0.0
            self._ou_series = None

        # Collapse operators
        if self.use_noise and "decay" in self.cfg.get("noise_sources", []):
            c_ops_qt = get_collapse_operators(2)
            self._c_ops_np = [c.full() for c in c_ops_qt]
        else:
            self._c_ops_np = []

        obs = self._get_obs(time_frac=0.0)
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take one control step."""
        a = np.clip(action, -1.0, 1.0)
        Omega = float((a[0] + 1.0) / 2.0 * 2.0 * self.Omega_max)
        Delta = float(a[1] * self.Omega_max)

        # Apply OU amplitude noise (time-correlated, from pre-generated series)
        if self._ou_series is not None:
            eta = float(self._ou_series[self._step_count])
            Omega = Omega * (1.0 + eta)

        # Apply systematic amplitude bias
        Omega = Omega * (1.0 + self._amplitude_bias)

        # Build Hamiltonian with per-atom Doppler shifts
        H_np = self._build_hamiltonian_np(Omega, Delta)

        # Propagate via superoperator expm
        self._rho_np = self._propagate_lindblad(H_np, self._rho_np, self.dt)

        self._step_count += 1
        terminated = self._step_count >= self.n_steps
        truncated = False

        # Compute current fidelity
        current_fid = self._compute_fidelity_np(self._rho_np)

        # Reward
        reward = 0.0
        info: Dict[str, Any] = {}

        # Per-step shaping reward: encourage fidelity improvement
        if self.reward_shaping_alpha > 0:
            reward += self.reward_shaping_alpha * (current_fid - self._prev_fidelity)

        if terminated:
            reward += current_fid  # terminal reward
            info["fidelity"] = float(current_fid)

        self._prev_fidelity = current_fid

        time_frac = self._step_count / self.n_steps
        obs = self._get_obs(time_frac=time_frac)
        return obs, reward, terminated, truncated, info

    def _get_obs(self, time_frac: float) -> np.ndarray:
        """Build observation vector based on obs_mode."""
        if self.obs_mode == "time_only":
            return np.array([time_frac], dtype=np.float32)
        else:
            tf = time_frac if self.obs_include_time else None
            return self._rho_to_obs(self._rho_np, time_frac=tf)

    def _build_hamiltonian_np(self, Omega: float, Delta: float) -> np.ndarray:
        """Build Hamiltonian with per-atom Doppler shifts (aligned with lindblad.py)."""
        H = Omega * self._H_drive_np.copy()

        # Per-atom detuning: Delta + doppler_i + phase_noise
        for i in range(2):
            Delta_eff_i = Delta + self._delta_doppler[i] + self._delta_phase
            H -= Delta_eff_i * self._n_r_np[i]

        # vdW interaction
        H += self._V_vdW * self._n_rr_np

        return H

    def _propagate_lindblad(
        self, H: np.ndarray, rho: np.ndarray, dt: float
    ) -> np.ndarray:
        """Propagate density matrix by dt under Lindblad master equation."""
        d = self.dim
        I = self._I_d

        L = -1j * (np.kron(H, I) - np.kron(I, H.T))

        for Lk in self._c_ops_np:
            Lk_dag = Lk.conj().T
            LdL = Lk_dag @ Lk
            L += (
                np.kron(Lk, Lk.conj())
                - 0.5 * (np.kron(LdL, I) + np.kron(I, LdL.T))
            )

        prop = expm(L * dt)
        rho_vec = rho.flatten(order="C")
        rho_new_vec = prop @ rho_vec
        rho_new = rho_new_vec.reshape(d, d)

        rho_new = 0.5 * (rho_new + rho_new.conj().T)
        rho_new /= np.trace(rho_new).real

        return rho_new

    def _compute_fidelity_np(self, rho: np.ndarray) -> float:
        """Compute F = Tr(rho * rho_target)."""
        return np.trace(rho @ self._target_dm_np).real

    @staticmethod
    def _rho_to_obs(rho: np.ndarray, time_frac: float = None) -> np.ndarray:
        """Flatten density matrix to observation vector.

        If time_frac is not None, appends it as the last element (33-dim).
        Otherwise returns 32-dim.
        """
        real_part = rho.real.flatten()
        imag_part = rho.imag.flatten()
        parts = [real_part, imag_part]
        if time_frac is not None:
            parts.append([time_frac])
        obs = np.concatenate(parts).astype(np.float32)
        obs = np.clip(obs, -1.0, 1.0)
        return obs
