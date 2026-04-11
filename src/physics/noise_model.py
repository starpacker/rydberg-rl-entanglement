"""Noise sampler for Rydberg atom simulations.

Supports five noise sources:
1. Doppler shift  -- static Gaussian per atom
2. Position jitter -- static Gaussian per atom
3. Amplitude noise -- Ornstein-Uhlenbeck process
4. Phase noise     -- static Gaussian (laser linewidth)
5. Spontaneous decay -- handled via Lindblad operators (see lindblad.py)

Each scenario ("A", "B", "D") declares which sources are active.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.physics.constants import (
    C6_53S,
    OU_CORRELATION_TIME,
    OU_SIGMA,
    SCENARIOS,
    SIGMA_DOPPLER,
    SIGMA_POSITION,
)


class NoiseModel:
    """Generate noise realisations for a given scenario.

    Parameters
    ----------
    scenario : str
        One of "A", "B", "D".
    """

    def __init__(self, scenario: str) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Must be one of {list(SCENARIOS)}")
        self.scenario = scenario
        self.cfg = SCENARIOS[scenario]
        self.n_atoms: int = self.cfg["n_atoms"]
        self.active: List[str] = list(self.cfg["noise_sources"])
        self.noise_amp: float = self.cfg.get("noise_amplification", 1.0)
        self.amp_bias_range: float = self.cfg.get("amplitude_bias_range", 0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Draw a single noise realisation.

        Returns
        -------
        dict with keys:
            delta_doppler : list[float]  -- per-atom Doppler shifts (rad/s)
            delta_R       : list[float]  -- per-atom position offsets (μm)
            ou_sigma      : float        -- OU amplitude noise strength
            ou_tau        : float        -- OU correlation time (s)
            phase_noise   : float        -- static phase offset (rad)
        """
        result: Dict[str, Any] = {}

        # 1. Doppler (static Gaussian per atom)
        if "doppler" in self.active:
            sigma_d = SIGMA_DOPPLER * self.noise_amp
            result["delta_doppler"] = [
                float(rng.normal(0, sigma_d)) for _ in range(self.n_atoms)
            ]
        else:
            result["delta_doppler"] = [0.0] * self.n_atoms

        # 2. Position jitter (static Gaussian per atom)
        if "position" in self.active:
            sigma_p = SIGMA_POSITION * self.noise_amp
            result["delta_R"] = [
                float(rng.normal(0, sigma_p)) for _ in range(self.n_atoms)
            ]
        else:
            result["delta_R"] = [0.0] * self.n_atoms

        # 3. Amplitude noise (OU parameters)
        if "amplitude" in self.active:
            result["ou_sigma"] = OU_SIGMA * self.noise_amp
            result["ou_tau"] = OU_CORRELATION_TIME
        else:
            result["ou_sigma"] = 0.0
            result["ou_tau"] = OU_CORRELATION_TIME

        # 3b. Systematic amplitude bias (for Scenario C)
        if self.amp_bias_range > 0 and "amplitude" in self.active:
            result["amplitude_bias"] = float(
                rng.uniform(-self.amp_bias_range, self.amp_bias_range)
            )
        else:
            result["amplitude_bias"] = 0.0

        # 4. Phase noise (static Gaussian)
        if "phase" in self.active:
            sigma_phase = 2 * np.pi * 1e3 * self.cfg["T_gate"] * self.noise_amp
            result["phase_noise"] = float(rng.normal(0, sigma_phase))
        else:
            result["phase_noise"] = 0.0

        return result

    def generate_ou_series(
        self,
        rng: np.random.Generator,
        tlist: np.ndarray,
    ) -> np.ndarray:
        """Generate an Ornstein-Uhlenbeck amplitude noise time series.

        The OU process satisfies:
            dx = -(x / tau) dt + sigma * sqrt(2/tau) dW

        with stationary variance sigma^2.

        Parameters
        ----------
        rng : numpy Generator
        tlist : 1-D array of time points (s).

        Returns
        -------
        noise : 1-D array, same length as tlist.
            Relative amplitude fluctuation (dimensionless).
        """
        sigma = OU_SIGMA * self.noise_amp if "amplitude" in self.active else 0.0
        tau = OU_CORRELATION_TIME
        n = len(tlist)
        x = np.zeros(n)
        # Start from stationary distribution
        x[0] = rng.normal(0, sigma) if sigma > 0 else 0.0

        for k in range(1, n):
            dt = tlist[k] - tlist[k - 1]
            # Exact update for OU process
            decay = np.exp(-dt / tau)
            x[k] = x[k - 1] * decay + sigma * np.sqrt(1 - decay**2) * rng.normal()

        return x

    @staticmethod
    def compute_V_vdW(
        R_base: float,
        delta_R: List[float],
        C6: float = C6_53S,
    ) -> float:
        """Compute van der Waals interaction with position jitter.

        For 2 atoms the effective separation is R_base + delta_R[1] - delta_R[0].
        This returns C6 / R_eff^6.

        Parameters
        ----------
        R_base : float
            Nominal inter-atom distance (μm).
        delta_R : list[float]
            Per-atom position offsets (μm). For 2-atom, expects length 2.
        C6 : float
            C6 coefficient (rad/s · μm^6).

        Returns
        -------
        float
            V_vdW in rad/s.
        """
        if len(delta_R) < 2:
            return C6 / R_base**6
        R_eff = R_base + (delta_R[1] - delta_R[0])
        if R_eff <= 0:
            # Extreme noise realization: clamp to a small positive distance
            R_eff = 0.01 * R_base
        return C6 / R_eff**6
