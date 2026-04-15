"""Differentiable NAAC (DNAAC) training.

This script implements the 4-phase training strategy:
    Phase A: Oracle test - with perfect noise knowledge, optimize per-noise pulses
    Phase B: Noise-conditioned generator - train network: noise_params → pulse_correction
    Phase C: End-to-end estimator - train ρ(t) → noise_est for fidelity (not MSE)
    Phase D: Evaluation - compare all methods against CMA-ES

Key insight: exact gradients through differentiable Lindblad dynamics enable
direct optimization, avoiding the noise of REINFORCE policy gradients.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.physics.differentiable_lindblad import (
    DNAAC_NOISE_NORMALIZER,
    DifferentiableLindblad,
    FourierPulseDecoder,
    sample_noise_batch,
    noise_to_vector,
)


# ===================================================================
# Networks
# ===================================================================

class NoiseConditionedCorrector(nn.Module):
    """Network that predicts pulse correction from noise parameters.

    Input: noise_vector (6,)
    Output: Fourier coefficient corrections (20,)
    """

    def __init__(self, n_noise: int = 6, n_params: int = 20, hidden_dims: List[int] = [128, 64]):
        super().__init__()

        layers = []
        prev_dim = n_noise
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, n_params))

        self.network = nn.Sequential(*layers)

    def forward(self, noise_vec: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        noise_vec : torch.Tensor
            Shape (batch, 6)

        Returns
        -------
        correction : torch.Tensor
            Shape (batch, 20)
        """
        return self.network(noise_vec)


class NoiseEstimatorDiff(nn.Module):
    """Estimate noise from density matrix trajectory (differentiable version).

    Input: ρ(t) trajectory, shape (batch, k_calib+1, 4, 4)
    Output: noise estimate (batch, 6)
    """

    def __init__(self, k_calib: int = 10, hidden_dims: List[int] = [256, 128]):
        super().__init__()

        self.k_calib = k_calib
        # Input: (k_calib+1) density matrices, each 4x4 complex → flatten to real/imag
        # 4x4 complex = 32 real values
        input_dim = (k_calib + 1) * 32

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 6))

        self.network = nn.Sequential(*layers)

    def forward(self, rho_traj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        rho_traj : torch.Tensor
            Shape (batch, k_calib+1, 4, 4) complex

        Returns
        -------
        noise_est : torch.Tensor
            Shape (batch, 6)
        """
        batch_size = rho_traj.shape[0]

        # Flatten real and imag parts
        real = rho_traj.real.reshape(batch_size, -1)
        imag = rho_traj.imag.reshape(batch_size, -1)
        x = torch.cat([real, imag], dim=-1) if self.k_calib > 0 else real

        # Actually, just flatten all: (k_calib+1) * 4 * 4 * 2 floats
        x = torch.cat([
            rho_traj.real.reshape(batch_size, -1),
            rho_traj.imag.reshape(batch_size, -1)
        ], dim=-1).float()  # Cast to float32 for network compatibility

        return self.network(x)


# ===================================================================
# Phase A: Oracle Test
# ===================================================================

def phase_a_oracle(
    cmaes_params: np.ndarray,
    noise_scales: List[float],
    batch_size: int = 32,
    n_opt_steps: int = 100,
    device: torch.device = None,
    verbose: bool = True,
) -> Dict:
    """Test: can per-noise gradient optimization beat CMA-ES average?

    Uses batched optimization: optimize a batch of noise realizations in parallel,
    each with its own pulse parameters (params differ across batch dimension).

    Parameters
    ----------
    cmaes_params : np.ndarray
        CMA-ES optimized Fourier parameters (20,)
    noise_scales : list of float
        Noise levels to test
    batch_size : int
        Batch of noise realizations optimized in parallel
    n_opt_steps : int
        Optimization steps per batch
    device : torch.device
        Compute device
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Oracle fidelities per noise level
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print("\n" + "=" * 60)
        print("Phase A: Oracle Test (batched per-noise gradient optimization)")
        print("=" * 60)
        print(f"  Batch size: {batch_size}, opt steps: {n_opt_steps}")

    diff_sim = DifferentiableLindblad(scenario="C", device=device)
    decoder = FourierPulseDecoder(n_steps=60, n_fourier=5, device=device)

    # Convert CMA-ES params to torch
    base_params = torch.tensor(cmaes_params, dtype=torch.float32, device=device)

    results = {}

    for alpha in noise_scales:
        if verbose:
            print(f"\n  alpha = {alpha}:")

        t0 = time.perf_counter()

        # Sample a batch of noise realizations
        rng = np.random.default_rng(10000 + int(alpha * 1000))
        noise = sample_noise_batch(batch_size, noise_scale=alpha, device=device, rng=rng)

        # Evaluate CMA-ES baseline for this batch
        with torch.no_grad():
            cmaes_actions = decoder(base_params.unsqueeze(0).expand(batch_size, -1))
            cmaes_fids, _ = diff_sim.simulate(cmaes_actions, noise, n_steps=60)
            cmaes_mean = cmaes_fids.mean().item()
            cmaes_std = cmaes_fids.std().item()

        if verbose:
            print(f"    CMA-ES baseline: {cmaes_mean:.4f} ± {cmaes_std:.4f}")

        # Each noise realization gets its OWN set of pulse parameters
        # Shape: (batch_size, 20)
        params = base_params.unsqueeze(0).expand(batch_size, -1).clone().detach().requires_grad_(True)
        optimizer = optim.Adam([params], lr=0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_opt_steps, eta_min=0.01)

        best_fids = cmaes_fids.detach().clone()

        for step in range(n_opt_steps):
            optimizer.zero_grad()

            # Decode and simulate - each sample uses its own params
            actions = decoder(params)  # (batch, 60, 2)
            fids, _ = diff_sim.simulate(actions, noise, n_steps=60)

            # Loss: maximize mean fidelity
            loss = -fids.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Track best per sample
            improved = fids > best_fids
            best_fids[improved] = fids[improved].detach()

            if verbose and (step + 1) % 20 == 0:
                print(f"    Step {step+1}/{n_opt_steps}: mean_F = {fids.mean().item():.4f}, "
                      f"best_mean = {best_fids.mean().item():.4f}")

        elapsed = time.perf_counter() - t0
        oracle_mean = best_fids.mean().item()
        oracle_std = best_fids.std().item()
        gap = oracle_mean - cmaes_mean

        results[alpha] = {
            "oracle_mean_F": float(oracle_mean),
            "oracle_std_F": float(oracle_std),
            "cmaes_mean_F": float(cmaes_mean),
            "cmaes_std_F": float(cmaes_std),
            "gap": float(gap),
            "min_F": float(best_fids.min().item()),
            "oracle_fidelities": best_fids.tolist(),
            "cmaes_fidelities": cmaes_fids.tolist(),
            "wall_time": elapsed,
        }

        symbol = "✓" if gap > 0.005 else ("~" if gap > -0.005 else "✗")
        if verbose:
            print(f"    Oracle: {oracle_mean:.4f} ± {oracle_std:.4f}")
            print(f"    Gap:    {gap:+.4f} {symbol}  ({elapsed:.1f}s)")

    return results


# ===================================================================
# Phase B: Noise-Conditioned Generator
# ===================================================================

def phase_b_train(
    cmaes_params: np.ndarray,
    noise_scale_range: Tuple[float, float] = (0.5, 5.0),
    n_batches: int = 5000,
    batch_size: int = 64,
    lr: float = 1e-3,
    correction_scale: float = 0.2,
    device: torch.device = None,
    verbose: bool = True,
) -> Tuple[NoiseConditionedCorrector, Dict]:
    """Train noise-conditioned pulse generator with exact gradients.

    Parameters
    ----------
    cmaes_params : np.ndarray
        CMA-ES baseline params (20,)
    noise_scale_range : tuple
        (min_alpha, max_alpha) for training
    n_batches : int
        Number of training batches
    batch_size : int
        Batch size
    lr : float
        Learning rate
    correction_scale : float
        Scale of corrections: params = base + scale * correction
    device : torch.device
        Compute device
    verbose : bool
        Print progress

    Returns
    -------
    corrector : NoiseConditionedCorrector
        Trained network
    history : dict
        Training history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print("\n" + "=" * 60)
        print("Phase B: Training Noise-Conditioned Generator")
        print("=" * 60)

    diff_sim = DifferentiableLindblad(scenario="C", device=device)
    decoder = FourierPulseDecoder(n_steps=60, n_fourier=5, device=device)

    # Base params (CMA-ES)
    base_params = torch.tensor(cmaes_params, dtype=torch.float32, device=device)

    # Network
    corrector = NoiseConditionedCorrector().to(device)
    optimizer = optim.Adam(corrector.parameters(), lr=lr)

    history = {"loss": [], "mean_F": []}
    t0 = time.perf_counter()

    # Normalization: use typical scales for noise at alpha=3 (mid-range)
    # These should match the typical magnitude of noise_to_vector output
    noise_normalizer = torch.tensor(DNAAC_NOISE_NORMALIZER, device=device)

    for batch in range(n_batches):
        # Sample noise uniformly from range
        alpha_min, alpha_max = noise_scale_range
        rng = np.random.default_rng(batch)
        alpha = rng.uniform(alpha_min, alpha_max)
        noise = sample_noise_batch(batch_size, noise_scale=alpha, device=device, rng=rng)

        # Get noise vector for network input
        noise_vec = noise_to_vector(noise)

        # Normalize noise for network input
        noise_vec_norm = (noise_vec / noise_normalizer).float()

        # Get correction
        correction = corrector(noise_vec_norm)

        # Final params
        params = base_params.unsqueeze(0) + correction_scale * correction

        # Simulate
        actions = decoder(params)
        fids, _ = diff_sim.simulate(actions, noise, n_steps=60)

        # Loss: maximize fidelity
        loss = -fids.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["loss"].append(-loss.item())
        history["mean_F"].append(fids.mean().item())

        if verbose and (batch + 1) % 500 == 0:
            mean_f = np.mean(history["mean_F"][-500:])
            print(f"  Batch {batch+1}/{n_batches}: mean_F = {mean_f:.4f}")

    history["wall_time"] = time.perf_counter() - t0

    if verbose:
        print(f"\n  Training complete in {history['wall_time']:.1f}s")
        print(f"  Final mean_F = {np.mean(history['mean_F'][-100:]):.4f}")

    return corrector, history


# ===================================================================
# Phase C: End-to-End Estimator
# ===================================================================

def phase_c_train(
    corrector: NoiseConditionedCorrector,
    cmaes_params: np.ndarray,
    k_calib: int = 10,
    noise_scale_range: Tuple[float, float] = (0.5, 5.0),
    n_batches: int = 5000,
    batch_size: int = 32,
    lr: float = 1e-3,
    correction_scale: float = 0.2,
    device: torch.device = None,
    verbose: bool = True,
) -> Tuple[NoiseEstimatorDiff, Dict]:
    """Train estimator end-to-end for fidelity (not MSE on noise).

    Parameters
    ----------
    corrector : NoiseConditionedCorrector
        Pre-trained corrector from Phase B
    cmaes_params : np.ndarray
        CMA-ES baseline params
    k_calib : int
        Calibration steps
    noise_scale_range : tuple
        Noise range for training
    n_batches : int
        Number of training batches
    batch_size : int
        Batch size
    lr : float
        Learning rate
    correction_scale : float
        Scale of corrections
    device : torch.device
        Compute device
    verbose : bool
        Print progress

    Returns
    -------
    estimator : NoiseEstimatorDiff
        Trained estimator
    history : dict
        Training history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print("\n" + "=" * 60)
        print("Phase C: Training End-to-End Estimator")
        print("=" * 60)

    diff_sim = DifferentiableLindblad(scenario="C", device=device)
    decoder = FourierPulseDecoder(n_steps=60, n_fourier=5, device=device)

    # Base params
    base_params = torch.tensor(cmaes_params, dtype=torch.float32, device=device)

    # Networks
    estimator = NoiseEstimatorDiff(k_calib=k_calib).to(device)
    corrector = corrector.to(device)

    # Train estimator, fine-tune corrector
    optimizer = optim.Adam(
        list(estimator.parameters()) + list(corrector.parameters()),
        lr=lr
    )

    # Calibration pulse (constant, learnable later if needed)
    calib_omega = torch.ones(k_calib, device=device) * 0.5
    calib_delta = torch.zeros(k_calib, device=device)
    calib_actions = torch.stack([calib_omega, calib_delta], dim=-1).unsqueeze(0)  # (1, k_calib, 2)

    # Normalization: use typical scales for noise at alpha=3
    noise_normalizer = torch.tensor(DNAAC_NOISE_NORMALIZER, device=device)

    history = {"loss": [], "mean_F": [], "estimation_error": []}
    t0 = time.perf_counter()

    for batch in range(n_batches):
        rng = np.random.default_rng(50000 + batch)
        alpha_min, alpha_max = noise_scale_range
        alpha = rng.uniform(alpha_min, alpha_max)
        noise = sample_noise_batch(batch_size, noise_scale=alpha, device=device, rng=rng)

        # 1. Run calibration phase (differentiable)
        calib_actions_batch = calib_actions.expand(batch_size, -1, -1)
        _, calib_traj = diff_sim.simulate_partial(
            calib_actions_batch, noise, n_steps=k_calib, start_step=0
        )  # calib_traj: (batch, k_calib+1, 4, 4)

        # 2. Estimate noise from trajectory
        noise_est = estimator(calib_traj)

        # 3. Get correction from estimated noise
        noise_est_norm = (noise_est / noise_normalizer).float()
        correction = corrector(noise_est_norm)

        # 4. Get adapted pulse
        params = base_params.unsqueeze(0) + correction_scale * correction

        # 5. Continue from calibration state to end
        # Adaptive actions for remaining steps
        adaptive_actions = decoder(params)[:, k_calib:, :]  # (batch, 60-k_calib, 2)

        # Get final state from calibration
        rho_post_calib = calib_traj[:, -1, :, :]  # (batch, 4, 4)

        # Simulate remaining steps
        _, adapt_traj = diff_sim.simulate_partial(
            adaptive_actions, noise, n_steps=60 - k_calib,
            start_step=k_calib, rho_init=rho_post_calib
        )

        # Final fidelity
        rho_final = adapt_traj[:, -1, :, :]
        fids = diff_sim.compute_fidelity(rho_final)

        # Loss
        loss = -fids.mean()

        # Track estimation error for analysis
        true_noise_vec = noise_to_vector(noise)
        est_error = (noise_est - true_noise_vec).pow(2).mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["loss"].append(-loss.item())
        history["mean_F"].append(fids.mean().item())
        history["estimation_error"].append(est_error)

        if verbose and (batch + 1) % 500 == 0:
            mean_f = np.mean(history["mean_F"][-500:])
            mean_err = np.mean(history["estimation_error"][-500:])
            print(f"  Batch {batch+1}/{n_batches}: mean_F = {mean_f:.4f}, est_err = {mean_err:.2e}")

    history["wall_time"] = time.perf_counter() - t0

    if verbose:
        print(f"\n  Training complete in {history['wall_time']:.1f}s")
        print(f"  Final mean_F = {np.mean(history['mean_F'][-100:]):.4f}")

    return estimator, history


# ===================================================================
# Phase D: Evaluation
# ===================================================================

def phase_d_evaluate(
    corrector: NoiseConditionedCorrector,
    corrector_finetuned: Optional[NoiseConditionedCorrector],
    estimator: Optional[NoiseEstimatorDiff],
    cmaes_params: np.ndarray,
    noise_scales: List[float],
    k_calib: int = 10,
    n_test: int = 200,
    correction_scale: float = 0.2,
    device: torch.device = None,
    verbose: bool = True,
) -> Dict:
    """Evaluate DNAAC against baselines using original numpy environment.

    Parameters
    ----------
    corrector : NoiseConditionedCorrector
        Original corrector from Phase B (for Phase B eval with true noise)
    corrector_finetuned : NoiseConditionedCorrector or None
        Fine-tuned corrector from Phase C (for Phase C eval with estimator)
    estimator : NoiseEstimatorDiff or None
        Trained estimator from Phase C (None to skip)
    cmaes_params : np.ndarray
        CMA-ES baseline params
    noise_scales : list
        Noise levels to test
    k_calib : int
        Calibration steps
    n_test : int
        Test episodes per noise level
    correction_scale : float
        Scale of corrections
    device : torch.device
        Compute device
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Evaluation results
    """
    from src.environments.rydberg_env import RydbergBellEnv
    from src.physics.differentiable_lindblad import FourierPulseDecoder as FPD

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print("\n" + "=" * 60)
        print("Phase D: Evaluation on Numpy Environment")
        print("=" * 60)

    diff_sim = DifferentiableLindblad(scenario="C", device=device)
    decoder = FourierPulseDecoder(n_steps=60, n_fourier=5, device=device)
    base_params = torch.tensor(cmaes_params, dtype=torch.float32, device=device)

    # Normalization: use typical scales for noise at alpha=3
    noise_normalizer = torch.tensor(DNAAC_NOISE_NORMALIZER, device=device)

    results = {}

    for alpha in noise_scales:
        if verbose:
            print(f"\n  alpha = {alpha}:")

        cmaes_fids = []
        phase_b_fids = []
        phase_c_fids = [] if estimator is not None else None

        env = RydbergBellEnv(
            scenario="C", n_steps=60, use_noise=True, noise_scale=alpha
        )

        for i in range(n_test):
            # Reset env and extract noise ONCE
            env.reset(seed=50000 + i)
            noise_params_np = env._noise_params
            ou_series_snapshot = env._ou_series[:60].copy() if env._ou_series is not None else np.zeros(60)

            # Get CMA-ES actions
            cmaes_actions = decoder(base_params.unsqueeze(0)).detach().cpu().numpy()[0]

            # --- CMA-ES baseline ---
            env.reset(seed=50000 + i)
            for step in range(60):
                _, _, _, _, info = env.step(cmaes_actions[step])
            cmaes_fids.append(info["fidelity"])

            # --- Phase B: known noise ---
            # Extract the ACTUAL noise realization (not distribution params)
            # to match what noise_to_vector() produces during training
            delta_doppler = noise_params_np.get("delta_doppler", [0.0, 0.0])
            delta_R = noise_params_np.get("delta_R", [0.0, 0.0])
            phase_noise = noise_params_np.get("phase_noise", 0.0)
            # Use actual OU series mean from the snapshot
            ou_mean = float(ou_series_snapshot.mean())

            noise_vec = torch.tensor([
                delta_doppler[0],
                delta_doppler[1],
                delta_R[0],
                delta_R[1],
                phase_noise / env.T_gate,  # Convert to detuning rate (rad/s)
                ou_mean,
            ], dtype=torch.float32, device=device).unsqueeze(0)

            noise_vec_norm = (noise_vec / noise_normalizer).float()

            with torch.no_grad():
                correction = corrector(noise_vec_norm)
                adapted_params = base_params + correction_scale * correction.squeeze()
                adapted_actions = decoder(adapted_params.unsqueeze(0)).cpu().numpy()[0]

            env.reset(seed=50000 + i)
            for step in range(60):
                _, _, _, _, info = env.step(adapted_actions[step])
            phase_b_fids.append(info["fidelity"])

            # --- Phase C: estimated noise ---
            if estimator is not None:
                # Run calibration in numpy env, get trajectory
                env.reset(seed=50000 + i)
                calib_traj = []
                calib_traj.append(env._rho_np.copy())
                for step in range(k_calib):
                    action = np.array([0.5, 0.0], dtype=np.float32)  # Calibration pulse
                    env.step(action)
                    calib_traj.append(env._rho_np.copy())

                # Convert to torch
                calib_traj_np = np.array(calib_traj)
                calib_traj_torch = torch.tensor(calib_traj_np, dtype=torch.complex128, device=device)
                calib_traj_torch = calib_traj_torch.unsqueeze(0)  # (1, k_calib+1, 4, 4)

                # Estimate noise - use finetuned corrector for Phase C
                c_for_phase_c = corrector_finetuned if corrector_finetuned is not None else corrector
                with torch.no_grad():
                    noise_est = estimator(calib_traj_torch)
                    noise_est_norm = (noise_est / noise_normalizer).float()
                    correction = c_for_phase_c(noise_est_norm)
                    adapted_params = base_params + correction_scale * correction.squeeze()
                    adapted_actions = decoder(adapted_params.unsqueeze(0)).cpu().numpy()[0]

                # Continue with adapted pulse (already did k_calib steps)
                for step in range(k_calib, 60):
                    _, _, _, _, info = env.step(adapted_actions[step])
                phase_c_fids.append(info["fidelity"])

            if verbose and (i + 1) % 50 == 0:
                print(f"    {i+1}/{n_test} done")

        # Aggregate results
        results[alpha] = {
            "cmaes": {
                "mean_F": float(np.mean(cmaes_fids)),
                "std_F": float(np.std(cmaes_fids)),
            },
            "phase_b": {
                "mean_F": float(np.mean(phase_b_fids)),
                "std_F": float(np.std(phase_b_fids)),
            },
        }

        if estimator is not None:
            results[alpha]["phase_c"] = {
                "mean_F": float(np.mean(phase_c_fids)),
                "std_F": float(np.std(phase_c_fids)),
            }

        if verbose:
            print(f"    CMA-ES:   {results[alpha]['cmaes']['mean_F']:.4f} ± {results[alpha]['cmaes']['std_F']:.4f}")
            print(f"    Phase B:  {results[alpha]['phase_b']['mean_F']:.4f} ± {results[alpha]['phase_b']['std_F']:.4f}")
            if estimator is not None:
                print(f"    Phase C:  {results[alpha]['phase_c']['mean_F']:.4f} ± {results[alpha]['phase_c']['std_F']:.4f}")

    return results


# ===================================================================
# Main
# ===================================================================

def load_cmaes_baseline(alpha: float = 1.0) -> np.ndarray:
    """Load CMA-ES optimized parameters for given noise level."""
    cmaes_path = ROOT / "results" / "noise_scaling" / "cmaes_sweep.json"
    with open(cmaes_path) as f:
        data = json.load(f)

    for entry in data["noise_levels"]:
        if abs(entry["noise_scale"] - alpha) < 0.01:
            return np.array(entry["best_params"])

    # Default to alpha=1.0 if exact match not found
    for entry in data["noise_levels"]:
        if abs(entry["noise_scale"] - 1.0) < 0.01:
            return np.array(entry["best_params"])

    raise ValueError(f"No CMA-ES params found for alpha={alpha}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DNAAC Training")
    parser.add_argument("--phase", choices=["a", "b", "c", "d", "all"], default="all",
                        help="Phase to run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-batches", type=int, default=5000, help="Training batches")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--k-calib", type=int, default=10, help="Calibration steps")
    parser.add_argument("--correction-scale", type=float, default=0.2, help="Correction scale")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load baseline
    cmaes_params = load_cmaes_baseline(alpha=1.0)
    print(f"Loaded CMA-ES params: shape={cmaes_params.shape}")

    noise_scales = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    results = {}
    save_dir = ROOT / "results" / "dnaac"
    save_dir.mkdir(exist_ok=True, parents=True)
    model_dir = ROOT / "models" / "dnaac"
    model_dir.mkdir(exist_ok=True, parents=True)

    # Phase A: Oracle test
    if args.phase in ["a", "all"]:
        print("\n" + "=" * 70)
        print("PHASE A: Oracle Test")
        print("=" * 70)

        oracle_results = phase_a_oracle(
            cmaes_params=cmaes_params,
            noise_scales=noise_scales,
            batch_size=32,
            n_opt_steps=100,
            device=device,
        )
        results["phase_a"] = oracle_results

        # Save
        with open(save_dir / "phase_a_oracle.json", "w") as f:
            json.dump(oracle_results, f, indent=2)

        # Check if oracle beats CMA-ES
        print("\n  Summary: Oracle vs CMA-ES")
        print("  " + "-" * 40)
        with open(ROOT / "results" / "noise_scaling" / "cmaes_sweep.json") as f:
            cmaes_data = json.load(f)

        for entry in cmaes_data["noise_levels"]:
            alpha = entry["noise_scale"]
            cmaes_f = entry["mean_F"]
            if alpha in oracle_results:
                oracle_f = oracle_results[alpha]["oracle_mean_F"]
                gap = oracle_f - cmaes_f
                symbol = "✓" if gap > 0.005 else ("~" if gap > -0.005 else "✗")
                print(f"  α={alpha}: Oracle={oracle_f:.4f}, CMA-ES={cmaes_f:.4f}, gap={gap:+.4f} {symbol}")

    # Phase B: Noise-conditioned generator
    corrector = None
    if args.phase in ["b", "all"]:
        print("\n" + "=" * 70)
        print("PHASE B: Noise-Conditioned Generator")
        print("=" * 70)

        corrector, history_b = phase_b_train(
            cmaes_params=cmaes_params,
            noise_scale_range=(0.5, 5.0),
            n_batches=args.n_batches,
            batch_size=args.batch_size,
            correction_scale=args.correction_scale,
            device=device,
        )
        results["phase_b"] = history_b

        # Save model and history
        torch.save(corrector.state_dict(), model_dir / "corrector.pt")
        with open(save_dir / "phase_b_history.json", "w") as f:
            json.dump({k: v if not isinstance(v, list) else v[-100:] for k, v in history_b.items()}, f, indent=2)

    # Phase C: End-to-end estimator
    estimator = None
    if args.phase in ["c", "all"]:
        print("\n" + "=" * 70)
        print("PHASE C: End-to-End Estimator")
        print("=" * 70)

        # Load corrector if not trained in this run
        if corrector is None:
            corrector = NoiseConditionedCorrector().to(device)
            corrector.load_state_dict(torch.load(model_dir / "corrector.pt"))

        estimator, history_c = phase_c_train(
            corrector=corrector,
            cmaes_params=cmaes_params,
            k_calib=args.k_calib,
            noise_scale_range=(0.5, 5.0),
            n_batches=args.n_batches,
            batch_size=args.batch_size,
            correction_scale=args.correction_scale,
            device=device,
        )
        results["phase_c"] = history_c

        # Save model and history
        torch.save(estimator.state_dict(), model_dir / "estimator.pt")
        torch.save(corrector.state_dict(), model_dir / "corrector_finetuned.pt")
        with open(save_dir / "phase_c_history.json", "w") as f:
            json.dump({k: v if not isinstance(v, list) else v[-100:] for k, v in history_c.items()}, f, indent=2)

    # Phase D: Evaluation
    if args.phase in ["d", "all"]:
        print("\n" + "=" * 70)
        print("PHASE D: Evaluation")
        print("=" * 70)

        # Load original corrector (Phase B) — always needed
        if corrector is None:
            corrector = NoiseConditionedCorrector().to(device)
            corrector.load_state_dict(torch.load(model_dir / "corrector.pt"))

        # Load finetuned corrector (Phase C) — for Phase C eval
        corrector_finetuned = None
        if (model_dir / "corrector_finetuned.pt").exists():
            corrector_finetuned = NoiseConditionedCorrector().to(device)
            corrector_finetuned.load_state_dict(torch.load(model_dir / "corrector_finetuned.pt"))

        if estimator is None and (model_dir / "estimator.pt").exists():
            estimator = NoiseEstimatorDiff(k_calib=args.k_calib).to(device)
            estimator.load_state_dict(torch.load(model_dir / "estimator.pt"))

        eval_results = phase_d_evaluate(
            corrector=corrector,
            corrector_finetuned=corrector_finetuned,
            estimator=estimator,
            cmaes_params=cmaes_params,
            noise_scales=noise_scales,
            k_calib=args.k_calib,
            n_test=200,
            correction_scale=args.correction_scale,
            device=device,
        )
        results["phase_d"] = eval_results

        # Save
        with open(save_dir / "phase_d_eval.json", "w") as f:
            json.dump(eval_results, f, indent=2)

        # Summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"{'alpha':<8} {'CMA-ES':<12} {'Phase B':<12} {'Phase C':<12} {'Gap B':<10} {'Gap C':<10}")
        print("-" * 70)
        for alpha in noise_scales:
            if alpha in eval_results:
                cmaes_f = eval_results[alpha]["cmaes"]["mean_F"]
                phase_b_f = eval_results[alpha]["phase_b"]["mean_F"]
                gap_b = phase_b_f - cmaes_f

                if "phase_c" in eval_results[alpha]:
                    phase_c_f = eval_results[alpha]["phase_c"]["mean_F"]
                    gap_c = phase_c_f - cmaes_f
                    print(f"{alpha:<8} {cmaes_f:<12.4f} {phase_b_f:<12.4f} {phase_c_f:<12.4f} {gap_b:<+10.4f} {gap_c:<+10.4f}")
                else:
                    print(f"{alpha:<8} {cmaes_f:<12.4f} {phase_b_f:<12.4f} {'N/A':<12} {gap_b:<+10.4f} {'N/A':<10}")

    print("\nDone!")


if __name__ == "__main__":
    main()
