"""Differentiable Lindblad simulator in PyTorch.

This module implements the Lindblad master equation dynamics with full
automatic differentiation support, enabling gradient-based optimization
of control pulses through the quantum dynamics.

Key insight: The Lindblad propagation is linear algebra:
    L = -i(H⊗I - I⊗H^T) + Σ_k [L_k⊗L_k* - 0.5(L_k†L_k⊗I + I⊗(L_k†L_k)^T)]
    ρ(t+dt) = expm(L*dt) @ vec(ρ(t))

All operations are differentiable via torch.linalg.matrix_exp.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.physics.constants import (
    C6_53S,
    OU_CORRELATION_TIME,
    OU_SIGMA,
    SCENARIOS,
    SIGMA_DOPPLER,
    SIGMA_POSITION,
    TAU_EFF_53S,
)

# Shared DNAAC noise normalizer — used by all noise-conditioned methods.
# Order: [doppler_1, doppler_2, R_1, R_2, phase_rate, ou_mean]
# Values are 3× the 1-sigma scale of each noise source (covers alpha≈3).
DNAAC_NOISE_NORMALIZER = np.array([
    2 * np.pi * 50e3 * 3,    # delta_doppler_1 (rad/s)
    2 * np.pi * 50e3 * 3,    # delta_doppler_2
    0.1 * 3,                  # delta_R_1 (μm)
    0.1 * 3,                  # delta_R_2
    2 * np.pi * 1e3 * 3,     # delta_phase (rad/s)
    0.02 * 3,                 # ou_mean
])


class DifferentiableLindblad(nn.Module):
    """GPU-accelerated differentiable Lindblad dynamics.

    Implements the 2-atom Rydberg Bell state preparation with full
    noise model support and automatic differentiation.

    Parameters
    ----------
    scenario : str
        Scenario key ("A", "B", "C")
    device : torch.device
        Device for computation
    use_decay : bool
        Whether to include spontaneous decay
    """

    def __init__(
        self,
        scenario: str = "C",
        device: torch.device = None,
        use_decay: bool = True,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.cfg = SCENARIOS[scenario]
        self.n_atoms = 2
        self.dim = 4  # 2-qubit Hilbert space
        self.superop_dim = 16  # Liouvillian dimension

        self.T_gate = self.cfg["T_gate"]
        self.Omega_max = self.cfg["Omega"]
        self.R_base = self.cfg["R"]
        self.C6 = C6_53S
        self.V_vdW_base = self.C6 / self.R_base**6

        self.use_decay = use_decay
        self.gamma = 1.0 / TAU_EFF_53S if use_decay else 0.0

        # Precompute operators
        self._build_operators()

        # Move to device
        self.to(device)

    def _build_operators(self):
        """Build and register operators as buffers."""
        # Basis states: |gg>=0, |gr>=1, |rg>=2, |rr>=3
        # Single-site operators
        g = np.array([1, 0], dtype=complex)
        r = np.array([0, 1], dtype=complex)

        # |g><r| = σ_gr (lowering) - single site
        sigma_gr_single = np.outer(g, r)
        # |r><g| = σ_rg (raising) - single site
        sigma_rg_single = np.outer(r, g)
        # |r><r| = n_r (number operator) - single site
        n_r_single = np.outer(r, r)

        # Two-atom operators via tensor product
        # Atom 1: operator ⊗ I
        sigma_gr_1 = np.kron(sigma_gr_single, np.eye(2))
        sigma_rg_1 = np.kron(sigma_rg_single, np.eye(2))
        n_r_1 = np.kron(n_r_single, np.eye(2))

        # Atom 2: I ⊗ operator
        sigma_gr_2 = np.kron(np.eye(2), sigma_gr_single)
        sigma_rg_2 = np.kron(np.eye(2), sigma_rg_single)
        n_r_2 = np.kron(np.eye(2), n_r_single)

        # |rr><rr| = n_rr (two-excitation projector)
        n_rr = n_r_1 @ n_r_2

        # Drive operator: sum_i 0.5 * (σ_rg_i + σ_gr_i)
        H_drive = 0.5 * (sigma_rg_1 + sigma_gr_1 + sigma_rg_2 + sigma_gr_2)

        # Use complex128 (double precision) for numerical stability during backprop
        # through 60 chained matrix_exp operations
        cdtype = torch.complex128

        # Convert to torch and register as buffers
        self.register_buffer('H_drive', torch.tensor(H_drive, dtype=cdtype))
        self.register_buffer('n_r_1', torch.tensor(n_r_1, dtype=cdtype))
        self.register_buffer('n_r_2', torch.tensor(n_r_2, dtype=cdtype))
        self.register_buffer('n_rr', torch.tensor(n_rr, dtype=cdtype))

        # Collapse operators: L_i = sqrt(gamma) * |g_i><r_i|
        self.register_buffer('c_op_1', torch.tensor(
            np.sqrt(self.gamma) * sigma_gr_1, dtype=cdtype
        ))
        self.register_buffer('c_op_2', torch.tensor(
            np.sqrt(self.gamma) * sigma_gr_2, dtype=cdtype
        ))

        # Identity matrices
        self.register_buffer('I_d', torch.eye(self.dim, dtype=cdtype))
        self.register_buffer('I_superop', torch.eye(self.superop_dim, dtype=cdtype))

        # Initial state: |gg><gg|
        gg = np.zeros(self.dim, dtype=complex)
        gg[0] = 1.0
        rho_init = np.outer(gg, gg)
        self.register_buffer('rho_init', torch.tensor(rho_init, dtype=cdtype))

        # Target state: (|gr> + |rg>) / sqrt(2)
        target_ket = np.zeros(self.dim, dtype=complex)
        target_ket[1] = 1.0 / np.sqrt(2)  # |gr>
        target_ket[2] = 1.0 / np.sqrt(2)  # |rg>
        target_dm = np.outer(target_ket, target_ket.conj())
        self.register_buffer('target_dm', torch.tensor(target_dm, dtype=cdtype))

    def build_hamiltonian(
        self,
        Omega: torch.Tensor,
        Delta: torch.Tensor,
        delta_doppler_1: torch.Tensor,
        delta_doppler_2: torch.Tensor,
        delta_phase: torch.Tensor,
        V_vdW: torch.Tensor,
    ) -> torch.Tensor:
        """Build batched Hamiltonian.

        Parameters
        ----------
        Omega : torch.Tensor
            Rabi frequency, shape (batch,)
        Delta : torch.Tensor
            Base detuning, shape (batch,)
        delta_doppler_1, delta_doppler_2 : torch.Tensor
            Per-atom Doppler shifts, shape (batch,)
        delta_phase : torch.Tensor
            Phase noise as detuning, shape (batch,)
        V_vdW : torch.Tensor
            van der Waals interaction, shape (batch,)

        Returns
        -------
        H : torch.Tensor
            Hamiltonian, shape (batch, 4, 4)
        """
        batch_size = Omega.shape[0]

        # Cast to double precision for stability
        Omega = Omega.double()
        Delta = Delta.double()
        delta_doppler_1 = delta_doppler_1.double()
        delta_doppler_2 = delta_doppler_2.double()
        delta_phase = delta_phase.double()
        V_vdW = V_vdW.double()

        # H = Ω * H_drive
        H = Omega.view(-1, 1, 1) * self.H_drive.unsqueeze(0)

        # - (Δ + δ_doppler_1 + δ_phase) * n_r_1
        Delta_eff_1 = Delta + delta_doppler_1 + delta_phase
        H = H - Delta_eff_1.view(-1, 1, 1) * self.n_r_1.unsqueeze(0)

        # - (Δ + δ_doppler_2 + δ_phase) * n_r_2
        Delta_eff_2 = Delta + delta_doppler_2 + delta_phase
        H = H - Delta_eff_2.view(-1, 1, 1) * self.n_r_2.unsqueeze(0)

        # + V_vdW * n_rr
        H = H + V_vdW.view(-1, 1, 1) * self.n_rr.unsqueeze(0)

        return H

    def build_liouvillian(self, H: torch.Tensor) -> torch.Tensor:
        """Build Lindblad superoperator.

        L = -i(H⊗I - I⊗H^T) + Σ_k [L_k⊗L_k* - 0.5(L_k†L_k⊗I + I⊗(L_k†L_k)^T)]

        Parameters
        ----------
        H : torch.Tensor
            Hamiltonian, shape (batch, 4, 4)

        Returns
        -------
        L : torch.Tensor
            Liouvillian superoperator, shape (batch, 16, 16)
        """
        batch_size = H.shape[0]
        d = self.dim
        I = self.I_d

        # Batched Kronecker product via einsum
        # For A⊗B where A and B are (batch, m, n) and (batch, p, q):
        # Result is (batch, m*p, n*q)
        def batched_kron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            """Compute batched Kronecker product."""
            # A: (batch, m, n), B: (batch, p, q)
            # Result: (batch, m*p, n*q)
            b, m, n = A.shape
            _, p, q = B.shape
            # (batch, m, 1, n, 1) * (batch, 1, p, 1, q) -> (batch, m, p, n, q)
            outer = A[:, :, None, :, None] * B[:, None, :, None, :]
            return outer.reshape(b, m * p, n * q)

        # Identity expanded for batch
        I_batch = I.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        H_contig = H.contiguous()
        HT = H.transpose(-1, -2).contiguous()

        # Coherent part: -i(H⊗I - I⊗H^T)
        H_kron_I = batched_kron(H_contig, I_batch)
        I_kron_HT = batched_kron(I_batch, HT)
        L = -1j * (H_kron_I - I_kron_HT)

        # Dissipator for each collapse operator
        if self.use_decay and self.gamma > 0:
            for c_op in [self.c_op_1, self.c_op_2]:
                c_dag = c_op.conj().T
                c_dag_c = c_dag @ c_op

                c_op_batch = c_op.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
                c_op_conj_batch = c_op.conj().unsqueeze(0).expand(batch_size, -1, -1).contiguous()
                c_dag_c_batch = c_dag_c.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
                c_dag_c_T_batch = c_dag_c.T.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

                # L_k ⊗ L_k*
                L_kron_Lconj = batched_kron(c_op_batch, c_op_conj_batch)

                # -0.5 * (L_k†L_k ⊗ I)
                LdL_kron_I = batched_kron(c_dag_c_batch, I_batch)

                # -0.5 * (I ⊗ (L_k†L_k)^T)
                I_kron_LdLT = batched_kron(I_batch, c_dag_c_T_batch)

                L = L + L_kron_Lconj - 0.5 * (LdL_kron_I + I_kron_LdLT)

        return L

    def propagate_step(
        self,
        rho_vec: torch.Tensor,
        L: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Propagate density matrix by one timestep.

        Parameters
        ----------
        rho_vec : torch.Tensor
            Vectorized density matrix, shape (batch, 16)
        L : torch.Tensor
            Liouvillian, shape (batch, 16, 16)
        dt : float
            Timestep

        Returns
        -------
        rho_vec_new : torch.Tensor
            New vectorized density matrix, shape (batch, 16)
        """
        # expm(L * dt)
        prop = torch.linalg.matrix_exp(L * dt)

        # rho_new = prop @ rho
        rho_vec_new = torch.bmm(prop, rho_vec.unsqueeze(-1)).squeeze(-1)

        return rho_vec_new

    def vec_to_rho(self, rho_vec: torch.Tensor) -> torch.Tensor:
        """Convert vectorized density matrix to matrix form.

        Parameters
        ----------
        rho_vec : torch.Tensor
            Shape (batch, 16)

        Returns
        -------
        rho : torch.Tensor
            Shape (batch, 4, 4)
        """
        return rho_vec.reshape(-1, self.dim, self.dim)

    def rho_to_vec(self, rho: torch.Tensor) -> torch.Tensor:
        """Convert density matrix to vectorized form.

        Parameters
        ----------
        rho : torch.Tensor
            Shape (batch, 4, 4)

        Returns
        -------
        rho_vec : torch.Tensor
            Shape (batch, 16)
        """
        return rho.reshape(-1, self.superop_dim)

    def compute_fidelity(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute fidelity F = Tr(ρ · ρ_target).

        Parameters
        ----------
        rho : torch.Tensor
            Density matrix, shape (batch, 4, 4)

        Returns
        -------
        fidelity : torch.Tensor
            Shape (batch,)
        """
        # F = Tr(ρ · ρ_target) = Σ_ij ρ_ij * (ρ_target)_ji
        # Since ρ_target is Hermitian: = Σ_ij ρ_ij * (ρ_target)_ij^*
        return torch.real(torch.sum(rho * self.target_dm.conj(), dim=(-2, -1)))

    def normalize_rho(self, rho: torch.Tensor) -> torch.Tensor:
        """Normalize and Hermitianize density matrix.

        Parameters
        ----------
        rho : torch.Tensor
            Shape (batch, 4, 4)

        Returns
        -------
        rho_normalized : torch.Tensor
            Shape (batch, 4, 4)
        """
        # Hermitianize
        rho = 0.5 * (rho + rho.conj().transpose(-1, -2))

        # Normalize trace to 1
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        trace = trace.unsqueeze(-1)  # (batch, 1, 1)
        rho = rho / trace.real.clamp(min=1e-10)

        return rho

    def simulate(
        self,
        actions: torch.Tensor,
        noise_params: Dict[str, torch.Tensor],
        n_steps: int = 60,
        return_trajectory: bool = False,
        use_checkpointing: bool = False,
        checkpoint_every: int = 10,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Simulate full trajectory with given actions and noise.

        Parameters
        ----------
        actions : torch.Tensor
            Normalized actions, shape (batch, n_steps, 2)
            actions[:, :, 0] = Omega_norm in [-1, 1]
            actions[:, :, 1] = Delta_norm in [-1, 1]
        noise_params : dict
            Dictionary with keys:
            - delta_doppler_1, delta_doppler_2: (batch,) Doppler shifts
            - delta_R_1, delta_R_2: (batch,) position offsets
            - delta_phase: (batch,) phase noise
            - ou_series: (batch, n_steps) OU amplitude noise
            - amplitude_bias: (batch,) systematic amplitude bias
        n_steps : int
            Number of timesteps
        return_trajectory : bool
            If True, return full ρ(t) trajectory
        use_checkpointing : bool
            If True, use gradient checkpointing to reduce memory and improve stability
        checkpoint_every : int
            Checkpoint frequency (recompute forward pass every N steps during backward)

        Returns
        -------
        fidelity : torch.Tensor
            Final fidelity, shape (batch,)
        trajectory : torch.Tensor, optional
            ρ(t) trajectory, shape (batch, n_steps+1, 4, 4)
        """
        from torch.utils.checkpoint import checkpoint

        batch_size = actions.shape[0]
        dt = self.T_gate / n_steps

        # Extract noise
        delta_doppler_1 = noise_params.get(
            'delta_doppler_1', torch.zeros(batch_size, device=self.device)
        )
        delta_doppler_2 = noise_params.get(
            'delta_doppler_2', torch.zeros(batch_size, device=self.device)
        )
        delta_R_1 = noise_params.get(
            'delta_R_1', torch.zeros(batch_size, device=self.device)
        )
        delta_R_2 = noise_params.get(
            'delta_R_2', torch.zeros(batch_size, device=self.device)
        )
        delta_phase = noise_params.get(
            'delta_phase', torch.zeros(batch_size, device=self.device)
        )
        ou_series = noise_params.get(
            'ou_series', torch.zeros(batch_size, n_steps, device=self.device)
        )
        amplitude_bias = noise_params.get(
            'amplitude_bias', torch.zeros(batch_size, device=self.device)
        )

        # Compute V_vdW with position jitter (use double)
        R_eff = (self.R_base + (delta_R_2 - delta_R_1)).double()
        R_eff = torch.clamp(R_eff, min=0.01 * self.R_base)
        V_vdW = self.C6 / (R_eff ** 6)

        # Initialize state
        rho = self.rho_init.unsqueeze(0).expand(batch_size, -1, -1).clone()
        rho_vec = self.rho_to_vec(rho)

        trajectory = [rho.clone()] if return_trajectory else None

        def single_step(rho_vec_in, step_actions, eta, step_idx):
            """Single propagation step - separable for checkpointing."""
            Omega_norm = step_actions[:, 0]
            Delta_norm = step_actions[:, 1]

            # Convert to physical units
            Omega = (Omega_norm + 1.0) / 2.0 * 2.0 * self.Omega_max
            Delta = Delta_norm * self.Omega_max

            # Apply OU amplitude noise
            Omega = Omega * (1.0 + eta)

            # Apply amplitude bias
            Omega = Omega * (1.0 + amplitude_bias)

            # Build Hamiltonian
            H = self.build_hamiltonian(
                Omega, Delta, delta_doppler_1, delta_doppler_2, delta_phase, V_vdW
            )

            # Build Liouvillian
            L = self.build_liouvillian(H)

            # Propagate
            rho_vec_out = self.propagate_step(rho_vec_in, L, dt)

            # Convert back to matrix and normalize
            rho_out = self.vec_to_rho(rho_vec_out)
            rho_out = self.normalize_rho(rho_out)
            rho_vec_out = self.rho_to_vec(rho_out)

            return rho_vec_out

        def chunk_steps(rho_vec_in, start_step, end_step):
            """Process a chunk of steps."""
            rho_vec_curr = rho_vec_in
            for step in range(start_step, end_step):
                a = actions[:, step, :]
                eta = ou_series[:, step]
                rho_vec_curr = single_step(rho_vec_curr, a, eta, step)
            return rho_vec_curr

        if use_checkpointing and self.training:
            # Use gradient checkpointing: process in chunks
            for chunk_start in range(0, n_steps, checkpoint_every):
                chunk_end = min(chunk_start + checkpoint_every, n_steps)
                rho_vec = checkpoint(
                    chunk_steps,
                    rho_vec,
                    chunk_start,
                    chunk_end,
                    use_reentrant=False,
                )
                if return_trajectory:
                    rho = self.vec_to_rho(rho_vec)
                    trajectory.append(rho.clone())
        else:
            # No checkpointing
            for step in range(n_steps):
                a = actions[:, step, :]
                eta = ou_series[:, step]
                rho_vec = single_step(rho_vec, a, eta, step)
                if return_trajectory:
                    rho = self.vec_to_rho(rho_vec)
                    trajectory.append(rho.clone())

        # Final fidelity
        rho = self.vec_to_rho(rho_vec)
        fidelity = self.compute_fidelity(rho)

        if return_trajectory:
            trajectory = torch.stack(trajectory, dim=1)
            return fidelity, trajectory
        else:
            return fidelity, None

    def simulate_partial(
        self,
        actions: torch.Tensor,
        noise_params: Dict[str, torch.Tensor],
        n_steps: int,
        start_step: int = 0,
        rho_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate partial trajectory (for calibration phase).

        Parameters
        ----------
        actions : torch.Tensor
            Shape (batch, n_steps, 2)
        noise_params : dict
            Noise parameters
        n_steps : int
            Number of steps to simulate
        start_step : int
            Starting step index (for OU noise indexing)
        rho_init : torch.Tensor, optional
            Initial state, shape (batch, 4, 4)

        Returns
        -------
        rho_final : torch.Tensor
            Final state, shape (batch, 4, 4)
        trajectory : torch.Tensor
            ρ(t) trajectory, shape (batch, n_steps+1, 4, 4)
        """
        batch_size = actions.shape[0]
        actual_n_steps = actions.shape[1]
        dt = self.T_gate / 60  # Use full gate time for dt

        # Extract noise
        delta_doppler_1 = noise_params.get(
            'delta_doppler_1', torch.zeros(batch_size, device=self.device)
        )
        delta_doppler_2 = noise_params.get(
            'delta_doppler_2', torch.zeros(batch_size, device=self.device)
        )
        delta_R_1 = noise_params.get(
            'delta_R_1', torch.zeros(batch_size, device=self.device)
        )
        delta_R_2 = noise_params.get(
            'delta_R_2', torch.zeros(batch_size, device=self.device)
        )
        delta_phase = noise_params.get(
            'delta_phase', torch.zeros(batch_size, device=self.device)
        )
        ou_series = noise_params.get(
            'ou_series', torch.zeros(batch_size, 60, device=self.device)
        )
        amplitude_bias = noise_params.get(
            'amplitude_bias', torch.zeros(batch_size, device=self.device)
        )

        # V_vdW with position jitter (double precision)
        R_eff = (self.R_base + (delta_R_2 - delta_R_1)).double()
        R_eff = torch.clamp(R_eff, min=0.01 * self.R_base)
        V_vdW = self.C6 / (R_eff ** 6)

        # Initialize
        if rho_init is None:
            rho = self.rho_init.unsqueeze(0).expand(batch_size, -1, -1).clone()
        else:
            rho = rho_init.clone()
        rho_vec = self.rho_to_vec(rho)

        trajectory = [rho.clone()]

        for step in range(actual_n_steps):
            a = actions[:, step, :]
            Omega_norm = a[:, 0]
            Delta_norm = a[:, 1]

            Omega = (Omega_norm + 1.0) / 2.0 * 2.0 * self.Omega_max
            Delta = Delta_norm * self.Omega_max

            # OU noise at correct index
            global_step = start_step + step
            if global_step < ou_series.shape[1]:
                eta = ou_series[:, global_step]
            else:
                eta = torch.zeros(batch_size, device=self.device)
            Omega = Omega * (1.0 + eta)
            Omega = Omega * (1.0 + amplitude_bias)

            H = self.build_hamiltonian(
                Omega, Delta, delta_doppler_1, delta_doppler_2, delta_phase, V_vdW
            )
            L = self.build_liouvillian(H)
            rho_vec = self.propagate_step(rho_vec, L, dt)
            rho = self.vec_to_rho(rho_vec)
            rho = self.normalize_rho(rho)
            rho_vec = self.rho_to_vec(rho)

            trajectory.append(rho.clone())

        trajectory = torch.stack(trajectory, dim=1)
        return rho, trajectory


class FourierPulseDecoder(nn.Module):
    """Decode Fourier parameters to pulse sequence (PyTorch version)."""

    def __init__(self, n_steps: int = 60, n_fourier: int = 5, device=None):
        super().__init__()
        self.n_steps = n_steps
        self.n_fourier = n_fourier
        self.n_params = 4 * n_fourier  # [a_Ω, b_Ω, a_Δ, b_Δ] per k

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Time points — endpoint=False to match CMA-ES convention
        t = torch.linspace(0, 1, n_steps + 1, device=device)[:-1]  # endpoint=False

        # Fourier basis: [sin(2πkt), cos(2πkt)] for k=0..K-1
        basis = torch.zeros(n_steps, 2 * n_fourier, device=device)
        for k in range(n_fourier):
            basis[:, 2*k] = torch.sin(2 * np.pi * k * t)
            basis[:, 2*k+1] = torch.cos(2 * np.pi * k * t)

        self.register_buffer('basis', basis)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Decode Fourier parameters to actions.

        Parameters
        ----------
        params : torch.Tensor
            Shape (batch, 4*n_fourier) or (4*n_fourier,)

        Returns
        -------
        actions : torch.Tensor
            Shape (batch, n_steps, 2) with values in [-1, 1]
        """
        if params.dim() == 1:
            params = params.unsqueeze(0)

        batch_size = params.shape[0]
        half = 2 * self.n_fourier

        omega_coeffs = params[:, :half]  # (batch, 2*n_fourier)
        delta_coeffs = params[:, half:]  # (batch, 2*n_fourier)

        # (batch, 2*n_fourier) @ (2*n_fourier, n_steps)^T = (batch, n_steps)
        omega_raw = torch.matmul(omega_coeffs, self.basis.T)
        delta_raw = torch.matmul(delta_coeffs, self.basis.T)

        # Clip to [-1, 1]
        omega_norm = torch.clamp(omega_raw, -1, 1)
        delta_norm = torch.clamp(delta_raw, -1, 1)

        # Stack: (batch, n_steps, 2)
        actions = torch.stack([omega_norm, delta_norm], dim=-1)

        return actions


def sample_noise_batch(
    batch_size: int,
    noise_scale: float,
    n_steps: int = 60,
    T_gate: float = 1e-6,
    device: torch.device = None,
    rng: np.random.Generator = None,
) -> Dict[str, torch.Tensor]:
    """Sample batch of noise realizations.

    Parameters
    ----------
    batch_size : int
        Number of noise samples
    noise_scale : float
        Noise amplification factor (alpha)
    n_steps : int
        Number of timesteps
    T_gate : float
        Gate time in seconds
    device : torch.device
        Device for tensors
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    noise_params : dict
        Dictionary of noise tensors
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rng is None:
        rng = np.random.default_rng()

    # Doppler shifts
    sigma_doppler = SIGMA_DOPPLER * noise_scale
    delta_doppler_1 = torch.tensor(
        rng.normal(0, sigma_doppler, batch_size), dtype=torch.float32, device=device
    )
    delta_doppler_2 = torch.tensor(
        rng.normal(0, sigma_doppler, batch_size), dtype=torch.float32, device=device
    )

    # Position jitter
    sigma_position = SIGMA_POSITION * noise_scale
    delta_R_1 = torch.tensor(
        rng.normal(0, sigma_position, batch_size), dtype=torch.float32, device=device
    )
    delta_R_2 = torch.tensor(
        rng.normal(0, sigma_position, batch_size), dtype=torch.float32, device=device
    )

    # Phase noise
    sigma_phase = 2 * np.pi * 1e3 * T_gate * noise_scale
    phase_noise = rng.normal(0, sigma_phase, batch_size)
    delta_phase = torch.tensor(
        phase_noise / T_gate, dtype=torch.float32, device=device
    )

    # OU amplitude noise
    ou_sigma = OU_SIGMA * noise_scale
    tau = OU_CORRELATION_TIME
    tlist = np.linspace(0, T_gate, n_steps + 1)

    ou_series = np.zeros((batch_size, n_steps))
    for b in range(batch_size):
        x = np.zeros(n_steps + 1)
        x[0] = rng.normal(0, ou_sigma) if ou_sigma > 0 else 0.0
        for k in range(1, n_steps + 1):
            dt = tlist[k] - tlist[k - 1]
            decay = np.exp(-dt / tau)
            x[k] = x[k - 1] * decay + ou_sigma * np.sqrt(1 - decay**2) * rng.normal()
        ou_series[b] = x[:-1]

    ou_series = torch.tensor(ou_series, dtype=torch.float32, device=device)

    # Amplitude bias (for Scenario C) — NOT scaled by noise_scale, matching numpy env
    amplitude_bias = torch.tensor(
        rng.uniform(-0.05, 0.05, batch_size),
        dtype=torch.float32, device=device
    )

    return {
        'delta_doppler_1': delta_doppler_1,
        'delta_doppler_2': delta_doppler_2,
        'delta_R_1': delta_R_1,
        'delta_R_2': delta_R_2,
        'delta_phase': delta_phase,
        'ou_series': ou_series,
        'amplitude_bias': amplitude_bias,
    }


def noise_to_vector(noise_params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Convert noise dict to vector for network input.

    Parameters
    ----------
    noise_params : dict
        Noise parameter dictionary

    Returns
    -------
    noise_vec : torch.Tensor
        Shape (batch, 6): [δ_doppler_1, δ_doppler_2, δ_R_1, δ_R_2, δ_phase, η_OU_mean]
    """
    delta_doppler_1 = noise_params['delta_doppler_1']
    delta_doppler_2 = noise_params['delta_doppler_2']
    delta_R_1 = noise_params['delta_R_1']
    delta_R_2 = noise_params['delta_R_2']
    delta_phase = noise_params['delta_phase']
    ou_mean = noise_params['ou_series'].mean(dim=-1)

    return torch.stack([
        delta_doppler_1,
        delta_doppler_2,
        delta_R_1,
        delta_R_2,
        delta_phase,
        ou_mean,
    ], dim=-1)


# ===================================================================
# Verification: Compare to numpy implementation
# ===================================================================

def verify_against_numpy(
    scenario: str = "C",
    n_steps: int = 60,
    seed: int = 42,
    verbose: bool = True,
) -> float:
    """Verify differentiable sim matches numpy sim.

    Parameters
    ----------
    scenario : str
        Scenario key
    n_steps : int
        Number of steps
    seed : int
        Random seed
    verbose : bool
        Print comparison

    Returns
    -------
    fidelity_diff : float
        Absolute difference in fidelity
    """
    from src.environments.rydberg_env import RydbergBellEnv

    device = torch.device("cpu")  # For exact comparison
    rng = np.random.default_rng(seed)

    # Create both simulators
    diff_sim = DifferentiableLindblad(scenario=scenario, device=device)
    env = RydbergBellEnv(scenario=scenario, n_steps=n_steps, use_noise=True)

    # Reset numpy env
    env.reset(seed=seed)

    # Get noise from numpy env
    noise_params_np = env._noise_params
    delta_doppler = noise_params_np.get("delta_doppler", [0.0, 0.0])
    delta_R = noise_params_np.get("delta_R", [0.0, 0.0])
    phase_noise = noise_params_np.get("phase_noise", 0.0)

    # Create noise dict for torch
    noise_params = {
        'delta_doppler_1': torch.tensor([delta_doppler[0]], dtype=torch.float32),
        'delta_doppler_2': torch.tensor([delta_doppler[1]], dtype=torch.float32),
        'delta_R_1': torch.tensor([delta_R[0]], dtype=torch.float32),
        'delta_R_2': torch.tensor([delta_R[1]], dtype=torch.float32),
        'delta_phase': torch.tensor([phase_noise / env.T_gate], dtype=torch.float32),
        'ou_series': torch.tensor(
            env._ou_series[:n_steps].reshape(1, -1) if env._ou_series is not None
            else np.zeros((1, n_steps)),
            dtype=torch.float32
        ),
        'amplitude_bias': torch.tensor([noise_params_np.get("amplitude_bias", 0.0)], dtype=torch.float32),
    }

    # Generate random actions
    actions_np = rng.uniform(-1, 1, (n_steps, 2)).astype(np.float32)
    actions = torch.tensor(actions_np, dtype=torch.float32).unsqueeze(0)  # (1, n_steps, 2)

    # Run numpy sim
    env.reset(seed=seed)  # Reset again to same state
    for step in range(n_steps):
        _, _, _, _, info = env.step(actions_np[step])
    fid_numpy = info.get("fidelity", 0.0)

    # Run torch sim
    fid_torch, _ = diff_sim.simulate(actions, noise_params, n_steps=n_steps)
    fid_torch = fid_torch.item()

    diff = abs(fid_numpy - fid_torch)

    if verbose:
        print(f"Verification (seed={seed}):")
        print(f"  Numpy fidelity:  {fid_numpy:.6f}")
        print(f"  Torch fidelity:  {fid_torch:.6f}")
        print(f"  Difference:      {diff:.6e}")
        if diff < 1e-4:
            print("  ✓ PASSED")
        else:
            print("  ✗ FAILED - difference too large")

    return diff


if __name__ == "__main__":
    print("Testing differentiable Lindblad simulator...")
    print()

    # Verify against numpy
    for seed in [42, 123, 456]:
        verify_against_numpy(seed=seed)
        print()

    # Test gradient flow
    print("Testing gradient flow...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    diff_sim = DifferentiableLindblad(scenario="C", device=device)
    decoder = FourierPulseDecoder(n_steps=60, n_fourier=5, device=device)

    # Random params
    params = torch.randn(1, 20, device=device, requires_grad=True)
    noise = sample_noise_batch(1, noise_scale=1.0, device=device)

    # Move noise tensors to device
    for k, v in noise.items():
        noise[k] = v.to(device)

    # Forward
    actions = decoder(params)
    fidelity, _ = diff_sim.simulate(actions, noise)

    print(f"  Fidelity: {fidelity.item():.4f}")

    # Backward
    (-fidelity).backward()

    print(f"  Gradient norm: {params.grad.norm().item():.4f}")
    print("  ✓ Gradients flow correctly!")
