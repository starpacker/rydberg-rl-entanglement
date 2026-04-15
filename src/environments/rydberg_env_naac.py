"""Extended Rydberg environment for NAAC training.

Modifications from base RydbergBellEnv:
1. Expose ground-truth noise parameters for training
2. Support calibration mode (return ρ trajectory)
3. Batch processing for efficient training
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.environments.rydberg_env import RydbergBellEnv


class RydbergBellEnvNAAC(RydbergBellEnv):
    """Extended environment for NAAC training.

    Additional features:
    - get_noise_params(): Returns ground-truth noise for supervised training
    - Trajectory recording: Stores ρ(t) sequence for estimator training
    """

    def __init__(
        self,
        scenario: str = "C",
        n_steps: int = 60,
        use_noise: bool = True,
        reward_shaping_alpha: float = 0.1,
        obs_include_time: bool = True,
        obs_mode: str = "full",
        noise_scale: float = None,
        record_trajectory: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        record_trajectory : bool
            If True, store ρ(t) trajectory for NAAC training
        """
        super().__init__(
            scenario=scenario,
            n_steps=n_steps,
            use_noise=use_noise,
            reward_shaping_alpha=reward_shaping_alpha,
            obs_include_time=obs_include_time,
            obs_mode=obs_mode,
            noise_scale=noise_scale,
        )

        self.record_trajectory = record_trajectory
        self._rho_trajectory: List[np.ndarray] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset and clear trajectory."""
        obs, info = super().reset(seed=seed, options=options)

        if self.record_trajectory:
            self._rho_trajectory = [self._rho_np.copy()]

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step and record ρ(t)."""
        obs, reward, terminated, truncated, info = super().step(action)

        if self.record_trajectory:
            self._rho_trajectory.append(self._rho_np.copy())

        return obs, reward, terminated, truncated, info

    def get_noise_params(self) -> Dict[str, Any]:
        """Return ground-truth noise parameters for supervised training.

        Returns
        -------
        noise_params : dict
            Keys: delta_doppler, delta_R, phase_noise, ou_sigma, amplitude_bias
        """
        if not self.use_noise:
            return {
                "delta_doppler": [0.0, 0.0],
                "delta_R": [0.0, 0.0],
                "phase_noise": 0.0,
                "ou_sigma": 0.0,
                "amplitude_bias": 0.0,
            }

        return self._noise_params.copy()

    def get_trajectory(self) -> np.ndarray:
        """Return recorded ρ(t) trajectory.

        Returns
        -------
        trajectory : np.ndarray
            Shape (n_recorded, 4, 4) complex
        """
        if not self.record_trajectory or not self._rho_trajectory:
            raise ValueError("No trajectory recorded. Set record_trajectory=True.")

        return np.array(self._rho_trajectory)

    def get_noise_vector(self) -> np.ndarray:
        """Return noise parameters as a vector for training.

        Returns
        -------
        noise_vec : np.ndarray
            Shape (6,): [δ_doppler_1, δ_doppler_2, δ_R_1, δ_R_2, δ_phase, η_OU]
        """
        params = self.get_noise_params()

        # Convert phase noise to effective detuning
        delta_phase = params.get("phase_noise", 0.0) / self.T_gate if self.T_gate > 0 else 0.0

        noise_vec = np.array([
            params["delta_doppler"][0],
            params["delta_doppler"][1],
            params["delta_R"][0],
            params["delta_R"][1],
            delta_phase,
            params.get("ou_sigma", 0.0),
        ], dtype=np.float32)

        return noise_vec


# ===================================================================
# Batch Environment Wrapper
# ===================================================================

class BatchRydbergEnvNAAC:
    """Vectorized environment for efficient NAAC training.

    Runs multiple environments in parallel (sequentially, but with batch interface).
    """

    def __init__(
        self,
        n_envs: int,
        scenario: str = "C",
        n_steps: int = 60,
        use_noise: bool = True,
        noise_scale: float = None,
        record_trajectory: bool = True,
    ):
        """
        Parameters
        ----------
        n_envs : int
            Number of parallel environments
        """
        self.n_envs = n_envs
        self.envs = [
            RydbergBellEnvNAAC(
                scenario=scenario,
                n_steps=n_steps,
                use_noise=use_noise,
                reward_shaping_alpha=0.0,  # Sparse reward for NAAC
                obs_include_time=True,
                obs_mode="full",
                noise_scale=noise_scale,
                record_trajectory=record_trajectory,
            )
            for _ in range(n_envs)
        ]

    def reset(self, seeds: Optional[List[int]] = None) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments.

        Returns
        -------
        obs : np.ndarray
            Shape (n_envs, obs_dim)
        infos : list of dict
        """
        if seeds is None:
            seeds = [None] * self.n_envs

        obs_list = []
        infos = []
        for env, seed in zip(self.envs, seeds):
            obs, info = env.reset(seed=seed)
            obs_list.append(obs)
            infos.append(info)

        return np.array(obs_list), infos

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments.

        Parameters
        ----------
        actions : np.ndarray
            Shape (n_envs, 2)

        Returns
        -------
        obs : np.ndarray
            Shape (n_envs, obs_dim)
        rewards : np.ndarray
            Shape (n_envs,)
        terminated : np.ndarray
            Shape (n_envs,)
        truncated : np.ndarray
            Shape (n_envs,)
        infos : list of dict
        """
        obs_list = []
        rewards = []
        terminated_list = []
        truncated_list = []
        infos = []

        for env, action in zip(self.envs, actions):
            obs, reward, term, trunc, info = env.step(action)
            obs_list.append(obs)
            rewards.append(reward)
            terminated_list.append(term)
            truncated_list.append(trunc)
            infos.append(info)

        return (
            np.array(obs_list),
            np.array(rewards),
            np.array(terminated_list),
            np.array(truncated_list),
            infos,
        )

    def get_noise_vectors(self) -> np.ndarray:
        """Get noise parameters from all environments.

        Returns
        -------
        noise_vecs : np.ndarray
            Shape (n_envs, 6)
        """
        return np.array([env.get_noise_vector() for env in self.envs])

    def get_trajectories(self) -> np.ndarray:
        """Get ρ(t) trajectories from all environments.

        Returns
        -------
        trajectories : np.ndarray
            Shape (n_envs, n_recorded, 4, 4) complex
        """
        trajs = [env.get_trajectory() for env in self.envs]
        # Ensure all have same length (should be true if all stepped same number of times)
        return np.array(trajs)
