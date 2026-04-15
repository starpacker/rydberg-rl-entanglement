"""Noise-Aware Adaptive Control (NAAC) for Rydberg quantum gates.

Core innovation: Explicitly estimate noise parameters from early-time dynamics,
then adapt the control pulse to compensate for the specific noise realization.

Architecture:
    1. NoiseEstimator: ρ(0:k_calib) → noise parameters
    2. AdaptivePulseGenerator: (t, noise_est, ρ) → (Ω, Δ)
    3. NAAC: Combines both for end-to-end training

Key advantages over baselines:
    - vs GRAPE: Explicitly models noise structure
    - vs CMA-ES+DR: Adapts to specific noise realization (not averaged)
    - vs PPO closed-loop: More sample-efficient, physically interpretable
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
# Noise Estimator Network
# ===================================================================

class NoiseEstimator(nn.Module):
    """Estimate static noise parameters from early-time density matrix evolution.

    Physical intuition:
        - Doppler shifts → differential phase accumulation in |gr⟩ vs |rg⟩
        - Position jitter → modified V_vdW → changes in |rr⟩ population
        - OU amplitude noise → Rabi frequency variations
        - Phase noise → global phase drift

    Input: Sequence of density matrices ρ(t_0), ..., ρ(t_k)
    Output: Estimated noise parameters [δ_doppler_1, δ_doppler_2, δ_R_1, δ_R_2, δ_phase, η_OU_mean]
    """

    def __init__(
        self,
        k_calib: int = 10,
        hidden_dims: list = [256, 128],
        n_noise_params: int = 6,
    ):
        """
        Parameters
        ----------
        k_calib : int
            Number of calibration timesteps
        hidden_dims : list
            Hidden layer dimensions
        n_noise_params : int
            Number of noise parameters to estimate (default 6)
        """
        super().__init__()

        self.k_calib = k_calib
        self.n_noise_params = n_noise_params

        # Input: k_calib density matrices, each 4x4 complex → flatten to real/imag
        # ρ is 4x4 complex → 16 complex → 32 real values
        # Total input: k_calib * 32
        input_dim = k_calib * 32

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_noise_params))

        self.network = nn.Sequential(*layers)

        # Physical scales for noise parameters (used to convert normalized → physical)
        # These match the normalization in train_naac.py
        self.register_buffer('noise_scales', torch.tensor([
            2 * np.pi * 50e3,  # δ_doppler_1 (rad/s)
            2 * np.pi * 50e3,  # δ_doppler_2
            0.1,                # δ_R_1 (μm)
            0.1,                # δ_R_2
            2 * np.pi * 1e3,   # δ_phase (rad)
            0.02,               # η_OU_mean (relative)
        ]))

    def forward(self, rho_sequence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        rho_sequence : torch.Tensor
            Shape (batch, k_calib, 4, 4, 2) where last dim is [real, imag]

        Returns
        -------
        noise_est : torch.Tensor
            Shape (batch, n_noise_params) in physical units
        """
        batch_size = rho_sequence.shape[0]

        # Flatten: (batch, k_calib, 4, 4, 2) → (batch, k_calib * 32)
        x = rho_sequence.reshape(batch_size, -1)

        # Forward pass - network outputs normalized values (roughly [-3, 3])
        noise_norm = self.network(x)

        # Scale to physical units
        noise_est = noise_norm * self.noise_scales

        return noise_est


# ===================================================================
# Adaptive Pulse Generator Network
# ===================================================================

class AdaptivePulseGenerator(nn.Module):
    """Generate noise-compensated control pulse.

    Strategy:
        - Base pulse: Fourier parameterization (smooth, low-dimensional)
        - Correction: MLP that takes (t, noise_est, ρ_current) → correction
        - Final pulse: base + correction

    Input: (t/T, noise_est, ρ_current)
    Output: (Ω_norm, Δ_norm) ∈ [-1, 1]²
    """

    def __init__(
        self,
        n_fourier: int = 5,
        hidden_dims: list = [128, 64],
        n_noise_params: int = 6,
    ):
        """
        Parameters
        ----------
        n_fourier : int
            Number of Fourier components for base pulse
        hidden_dims : list
            Hidden layer dimensions for correction network
        n_noise_params : int
            Number of noise parameters
        """
        super().__init__()

        self.n_fourier = n_fourier
        self.n_noise_params = n_noise_params

        # Fourier base pulse parameters
        # For Ω: [a_0, b_0, a_1, b_1, ..., a_{K-1}, b_{K-1}]
        # For Δ: same structure
        self.fourier_omega = nn.Parameter(torch.randn(2 * n_fourier) * 0.1)
        self.fourier_delta = nn.Parameter(torch.randn(2 * n_fourier) * 0.1)

        # Correction network
        # Input: [t/T (1), noise_est (6), ρ_flattened (32)] = 39 dims
        correction_input_dim = 1 + n_noise_params + 32

        layers = []
        prev_dim = correction_input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))  # Output: (Ω_correction, Δ_correction)

        self.correction_network = nn.Sequential(*layers)

    def fourier_basis(self, t: torch.Tensor) -> torch.Tensor:
        """Compute Fourier basis functions.

        Parameters
        ----------
        t : torch.Tensor
            Time values in [0, 1], shape (batch,) or (batch, 1)

        Returns
        -------
        basis : torch.Tensor
            Shape (batch, 2*n_fourier)
            [sin(2πt), cos(2πt), sin(4πt), cos(4πt), ...]
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)  # (batch, 1)

        basis_list = []
        for k in range(self.n_fourier):
            basis_list.append(torch.sin(2 * np.pi * k * t))
            basis_list.append(torch.cos(2 * np.pi * k * t))

        return torch.cat(basis_list, dim=1)  # (batch, 2*n_fourier)

    def forward(
        self,
        t: torch.Tensor,
        noise_est: torch.Tensor,
        rho_current: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        t : torch.Tensor
            Normalized time t/T, shape (batch,) or (batch, 1)
        noise_est : torch.Tensor
            Estimated noise parameters, shape (batch, n_noise_params)
        rho_current : torch.Tensor
            Current density matrix, shape (batch, 4, 4, 2)

        Returns
        -------
        actions : torch.Tensor
            Shape (batch, 2) with (Ω_norm, Δ_norm) ∈ [-1, 1]²
        """
        batch_size = rho_current.shape[0]

        # Fourier base pulse
        basis = self.fourier_basis(t)  # (batch, 2*n_fourier)
        omega_base = torch.matmul(basis, self.fourier_omega)  # (batch,)
        delta_base = torch.matmul(basis, self.fourier_delta)  # (batch,)

        # Flatten ρ_current
        rho_flat = rho_current.reshape(batch_size, -1)  # (batch, 32)

        # Correction network input
        if t.dim() == 1:
            t = t.unsqueeze(1)
        correction_input = torch.cat([t, noise_est, rho_flat], dim=1)  # (batch, 39)

        # Correction
        correction = self.correction_network(correction_input)  # (batch, 2)
        omega_corr, delta_corr = correction[:, 0], correction[:, 1]

        # Final pulse
        omega_norm = torch.tanh(omega_base + omega_corr)
        delta_norm = torch.tanh(delta_base + delta_corr)

        return torch.stack([omega_norm, delta_norm], dim=1)  # (batch, 2)


# ===================================================================
# NAAC: Combined System
# ===================================================================

class NAAC(nn.Module):
    """Noise-Aware Adaptive Control: End-to-end trainable system.

    Workflow:
        1. Calibration phase (steps 0 to k_calib):
           - Execute calibration pulse
           - Collect ρ(t) trajectory
        2. Estimation:
           - Feed ρ trajectory to NoiseEstimator
        3. Adaptive execution (steps k_calib+1 to n_steps):
           - Use AdaptivePulseGenerator with estimated noise
    """

    def __init__(
        self,
        k_calib: int = 10,
        n_fourier: int = 5,
        estimator_hidden: list = [256, 128],
        generator_hidden: list = [128, 64],
        n_noise_params: int = 6,
    ):
        """
        Parameters
        ----------
        k_calib : int
            Number of calibration steps
        n_fourier : int
            Fourier components for base pulse
        estimator_hidden : list
            Hidden dims for noise estimator
        generator_hidden : list
            Hidden dims for pulse generator
        n_noise_params : int
            Number of noise parameters to estimate
        """
        super().__init__()

        self.k_calib = k_calib
        self.n_noise_params = n_noise_params

        self.estimator = NoiseEstimator(
            k_calib=k_calib,
            hidden_dims=estimator_hidden,
            n_noise_params=n_noise_params,
        )

        self.generator = AdaptivePulseGenerator(
            n_fourier=n_fourier,
            hidden_dims=generator_hidden,
            n_noise_params=n_noise_params,
        )

        # Learnable calibration pulse
        self.calib_omega = nn.Parameter(torch.ones(k_calib) * 0.5)
        self.calib_delta = nn.Parameter(torch.zeros(k_calib))

    def get_calibration_pulse(self, batch_size: int = 1) -> torch.Tensor:
        """Get calibration pulse for first k_calib steps.

        Returns
        -------
        calib_actions : torch.Tensor
            Shape (batch_size, k_calib, 2)
        """
        omega = torch.tanh(self.calib_omega)  # Normalize to [-1, 1]
        delta = torch.tanh(self.calib_delta)

        actions = torch.stack([omega, delta], dim=1)  # (k_calib, 2)
        return actions.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, k_calib, 2)

    def estimate_noise(self, rho_sequence: torch.Tensor) -> torch.Tensor:
        """Estimate noise from calibration trajectory.

        Parameters
        ----------
        rho_sequence : torch.Tensor
            Shape (batch, k_calib, 4, 4, 2)

        Returns
        -------
        noise_est : torch.Tensor
            Shape (batch, n_noise_params)
        """
        return self.estimator(rho_sequence)

    def generate_action(
        self,
        t: torch.Tensor,
        noise_est: torch.Tensor,
        rho_current: torch.Tensor,
    ) -> torch.Tensor:
        """Generate adaptive action for current timestep.

        Parameters
        ----------
        t : torch.Tensor
            Normalized time, shape (batch,)
        noise_est : torch.Tensor
            Estimated noise, shape (batch, n_noise_params)
        rho_current : torch.Tensor
            Current state, shape (batch, 4, 4, 2)

        Returns
        -------
        action : torch.Tensor
            Shape (batch, 2)
        """
        return self.generator(t, noise_est, rho_current)

    def forward(
        self,
        rho_calib_sequence: torch.Tensor,
        t_adaptive: torch.Tensor,
        rho_adaptive_sequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass (for training).

        Parameters
        ----------
        rho_calib_sequence : torch.Tensor
            Calibration trajectory, shape (batch, k_calib, 4, 4, 2)
        t_adaptive : torch.Tensor
            Time points for adaptive phase, shape (n_adaptive,)
        rho_adaptive_sequence : torch.Tensor
            States during adaptive phase, shape (batch, n_adaptive, 4, 4, 2)

        Returns
        -------
        noise_est : torch.Tensor
            Estimated noise, shape (batch, n_noise_params)
        actions_adaptive : torch.Tensor
            Generated actions, shape (batch, n_adaptive, 2)
        """
        batch_size = rho_calib_sequence.shape[0]
        n_adaptive = t_adaptive.shape[0]

        # Estimate noise from calibration
        noise_est = self.estimate_noise(rho_calib_sequence)

        # Generate adaptive actions
        actions_list = []
        for i in range(n_adaptive):
            t_i = t_adaptive[i].expand(batch_size)
            rho_i = rho_adaptive_sequence[:, i]
            action_i = self.generate_action(t_i, noise_est, rho_i)
            actions_list.append(action_i)

        actions_adaptive = torch.stack(actions_list, dim=1)  # (batch, n_adaptive, 2)

        return noise_est, actions_adaptive


# ===================================================================
# Utility Functions
# ===================================================================

def design_calibration_pulse(
    k_calib: int,
    omega_max: float = 1.0,
    strategy: str = "rabi_sweep",
) -> Tuple[np.ndarray, np.ndarray]:
    """Design a calibration pulse to maximize noise observability.

    Parameters
    ----------
    k_calib : int
        Number of calibration steps
    omega_max : float
        Maximum Ω (normalized)
    strategy : str
        "rabi_sweep": Rabi oscillation with frequency sweep
        "blockade_probe": Probe blockade shift

    Returns
    -------
    omega_calib : np.ndarray
        Shape (k_calib,), values in [-1, 1]
    delta_calib : np.ndarray
        Shape (k_calib,), values in [-1, 1]
    """
    if strategy == "rabi_sweep":
        # Rabi oscillation with Δ sweep to probe Doppler and position
        omega = np.ones(k_calib) * omega_max
        delta = np.linspace(-0.5, 0.5, k_calib)
    elif strategy == "blockade_probe":
        # Probe blockade shift (sensitive to V_vdW)
        omega = np.ones(k_calib) * omega_max
        delta = np.linspace(-1.0, 0.0, k_calib)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return omega, delta


def numpy_to_torch_rho(rho_np: np.ndarray) -> torch.Tensor:
    """Convert numpy density matrix to torch tensor.

    Parameters
    ----------
    rho_np : np.ndarray
        Shape (..., 4, 4) complex

    Returns
    -------
    rho_torch : torch.Tensor
        Shape (..., 4, 4, 2) with last dim [real, imag]
    """
    real = np.real(rho_np)
    imag = np.imag(rho_np)
    rho_stacked = np.stack([real, imag], axis=-1)
    return torch.from_numpy(rho_stacked).float()


def torch_to_numpy_rho(rho_torch: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy density matrix.

    Parameters
    ----------
    rho_torch : torch.Tensor
        Shape (..., 4, 4, 2) with last dim [real, imag]

    Returns
    -------
    rho_np : np.ndarray
        Shape (..., 4, 4) complex
    """
    rho_np = rho_torch.detach().cpu().numpy()
    return rho_np[..., 0] + 1j * rho_np[..., 1]
