"""Rydberg environment with Fourier feature encoding for open-loop control.

For open-loop PPO with obs_mode='time_only', the 1-dim input t/T is insufficient
for MLPs to learn smooth time-varying pulses. This wrapper adds Fourier features:

    obs = [sin(2πk·t/T), cos(2πk·t/T)]  for k = 0, 1, 2, ..., K

This provides a rich basis for representing periodic/oscillatory control pulses.
"""
import numpy as np
import gymnasium
from typing import Optional, Dict, Any, Tuple


class FourierFeatureWrapper(gymnasium.ObservationWrapper):
    """Wraps RydbergBellEnv to add Fourier features when obs_mode='time_only'.

    Parameters
    ----------
    env : RydbergBellEnv
        Base environment (must have obs_mode='time_only')
    n_fourier : int
        Number of Fourier frequency components (k = 0, 1, ..., n_fourier-1)
        Output dim = 2 * n_fourier (sin and cos for each k)
    """

    def __init__(self, env, n_fourier: int = 8):
        super().__init__(env)

        if not hasattr(env, 'obs_mode') or env.obs_mode != 'time_only':
            raise ValueError("FourierFeatureWrapper requires obs_mode='time_only'")

        self.n_fourier = n_fourier
        self.freqs = np.arange(n_fourier, dtype=np.float32)  # k = 0, 1, 2, ..., K-1

        # New observation space: [sin(2πk·t), cos(2πk·t)] for k=0..K-1
        obs_dim = 2 * n_fourier
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Transform [t/T] -> [sin(2πk·t/T), cos(2πk·t/T)]."""
        t_frac = obs[0]  # obs is [t/T] from base env

        # Compute 2π k t/T for all k
        angles = 2 * np.pi * self.freqs * t_frac

        # Stack [sin, cos] for each frequency
        features = np.empty(2 * self.n_fourier, dtype=np.float32)
        features[0::2] = np.sin(angles)  # sin components at even indices
        features[1::2] = np.cos(angles)  # cos components at odd indices

        return features
