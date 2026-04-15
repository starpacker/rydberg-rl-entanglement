"""Train NAAC v3: Warm-start with CMA-ES, only train correction + estimator.

Key insight: Learning the base pulse from scratch via REINFORCE is extremely hard.
Solution: Initialize base pulse with CMA-ES solution, then train:
1. Noise estimator (supervised)
2. Correction network (to adapt to specific noise realization)
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.algorithms.naac import NAAC, numpy_to_torch_rho
from src.environments.rydberg_env_naac import BatchRydbergEnvNAAC


def load_cmaes_params() -> Dict[str, np.ndarray]:
    """Load best CMA-ES parameters for each noise level."""
    results_path = ROOT / "results" / "noise_scaling" / "cmaes_sweep.json"

    with open(results_path) as f:
        data = json.load(f)

    # Average parameters across noise levels for a robust initialization
    all_params = []
    for level in data["noise_levels"]:
        params = np.array(level["best_params"])
        all_params.append(params)

    avg_params = np.mean(all_params, axis=0)

    # Split into omega (first 10) and delta (last 10)
    # CMA-ES uses 20 Fourier params: [a0_Ω, b0_Ω, a1_Ω, b1_Ω, ..., a0_Δ, b0_Δ, ...]
    n_fourier = 5
    omega_params = avg_params[:2*n_fourier]  # First 10
    delta_params = avg_params[2*n_fourier:]  # Last 10

    return {
        "omega": omega_params,
        "delta": delta_params,
    }


def initialize_naac_with_cmaes(naac: NAAC, cmaes_params: Dict) -> None:
    """Initialize NAAC Fourier parameters with CMA-ES solution."""
    # Initialize Fourier parameters in the generator
    with torch.no_grad():
        naac.generator.fourier_omega.copy_(
            torch.tensor(cmaes_params["omega"], dtype=torch.float32)
        )
        naac.generator.fourier_delta.copy_(
            torch.tensor(cmaes_params["delta"], dtype=torch.float32)
        )

    print("Initialized NAAC with CMA-ES Fourier parameters")
    print(f"  Omega: mean={cmaes_params['omega'].mean():.3f}, std={cmaes_params['omega'].std():.3f}")
    print(f"  Delta: mean={cmaes_params['delta'].mean():.3f}, std={cmaes_params['delta'].std():.3f}")


def rollout_naac_with_gradients(
    naac: NAAC,
    envs: BatchRydbergEnvNAAC,
    k_calib: int,
    n_steps: int,
    device: torch.device,
    exploration_std: float = 0.05,
    freeze_base: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, List[torch.Tensor], torch.Tensor]:
    """Rollout NAAC with gradient tracking.

    Parameters
    ----------
    freeze_base : bool
        If True, only correction network is trained (base pulse frozen)
    """
    n_envs = envs.n_envs
    n_adaptive = n_steps - k_calib

    obs, _ = envs.reset()

    # Phase 1: Calibration (deterministic)
    calib_actions = naac.get_calibration_pulse(batch_size=n_envs)
    calib_actions_np = calib_actions.detach().cpu().numpy()

    for step in range(k_calib):
        actions = calib_actions_np[:, step, :]
        obs, _, _, _, _ = envs.step(actions)

    rho_calib_np = envs.get_trajectories()[:, :k_calib, :, :]
    rho_calib = numpy_to_torch_rho(rho_calib_np).to(device)

    noise_true_np = envs.get_noise_vectors()
    noise_true = torch.from_numpy(noise_true_np).float().to(device)

    noise_est = naac.estimate_noise(rho_calib)

    # Phase 3: Adaptive execution
    rho_adaptive_list = []
    log_probs = []

    for step in range(n_adaptive):
        rho_current_np = np.array([env._rho_np for env in envs.envs])
        rho_current = numpy_to_torch_rho(rho_current_np).to(device)
        rho_adaptive_list.append(rho_current)

        t = torch.tensor([step / n_adaptive] * n_envs, device=device)

        # Generate base action (potentially frozen)
        if freeze_base:
            with torch.no_grad():
                # Get Fourier base (frozen)
                basis = naac.generator.fourier_basis(t)
                omega_base = torch.matmul(basis, naac.generator.fourier_omega)
                delta_base = torch.matmul(basis, naac.generator.fourier_delta)
        else:
            basis = naac.generator.fourier_basis(t)
            omega_base = torch.matmul(basis, naac.generator.fourier_omega)
            delta_base = torch.matmul(basis, naac.generator.fourier_delta)

        # Get correction (always trainable)
        rho_flat = rho_current.reshape(n_envs, -1)
        if t.dim() == 1:
            t_input = t.unsqueeze(1)
        else:
            t_input = t
        correction_input = torch.cat([t_input, noise_est.detach(), rho_flat], dim=1)
        correction = naac.generator.correction_network(correction_input)
        omega_corr, delta_corr = correction[:, 0], correction[:, 1]

        # Combine
        actions_mean = torch.stack([
            torch.tanh(omega_base + omega_corr),
            torch.tanh(delta_base + delta_corr)
        ], dim=1)

        # Add exploration noise
        dist = Normal(actions_mean, exploration_std)
        actions_torch = dist.sample()
        log_prob = dist.log_prob(actions_torch).sum(dim=1)
        log_probs.append(log_prob)

        actions_torch = torch.clamp(actions_torch, -1.0, 1.0)
        actions_np = actions_torch.detach().cpu().numpy()

        obs, _, terminated, truncated, infos = envs.step(actions_np)

    fidelities = np.array([info.get("fidelity", 0.0) for info in infos])
    rho_adaptive = torch.stack(rho_adaptive_list, dim=1)

    return rho_calib, rho_adaptive, noise_true, fidelities, noise_est, log_probs


def train_naac_v3(
    scenario: str = "C",
    n_steps: int = 60,
    k_calib: int = 10,
    n_envs: int = 16,
    n_episodes: int = 50000,
    lr: float = 1e-4,
    lambda_estimator: float = 1.0,
    lambda_policy: float = 0.1,
    exploration_std: float = 0.05,
    freeze_base: bool = True,
    noise_scale_range: Tuple[float, float] = (0.5, 5.0),
    save_dir: str = "models/naac_v3",
    log_dir: str = "logs/naac_v3",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> Dict:
    """Train NAAC v3 with CMA-ES warm-start."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device)
    print(f"Training on device: {device}")

    save_path = Path(ROOT) / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    log_path = Path(ROOT) / log_dir
    log_path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_path))

    # Initialize NAAC
    naac = NAAC(
        k_calib=k_calib,
        n_fourier=5,
        estimator_hidden=[256, 128],
        generator_hidden=[128, 64],
        n_noise_params=6,
    ).to(device)

    # Warm-start with CMA-ES
    cmaes_params = load_cmaes_params()
    initialize_naac_with_cmaes(naac, cmaes_params)

    # Optimizer: if freezing base, only train estimator + correction
    if freeze_base:
        trainable_params = list(naac.estimator.parameters()) + \
                          list(naac.generator.correction_network.parameters()) + \
                          [naac.calib_omega, naac.calib_delta]
        print(f"Freezing base pulse. Training {sum(p.numel() for p in trainable_params)} params")
    else:
        trainable_params = naac.parameters()
        print(f"Training all {sum(p.numel() for p in naac.parameters())} params")

    optimizer = optim.Adam(trainable_params, lr=lr)

    stats = {
        "episode": [],
        "noise_scale": [],
        "mean_fidelity": [],
        "loss_total": [],
        "loss_policy": [],
        "loss_estimator": [],
        "noise_est_error": [],
    }

    print(f"\n{'='*70}")
    print(f"{'NAAC v3 Training (CMA-ES Warm-start)':^70}")
    print(f"{'='*70}")
    print(f"Scenario: {scenario}, n_steps: {n_steps}, k_calib: {k_calib}")
    print(f"Noise scale range: {noise_scale_range}")
    print(f"Total episodes: {n_episodes}, batch size: {n_envs}")
    print(f"Freeze base: {freeze_base}, Exploration std: {exploration_std}")
    print(f"{'='*70}\n")

    episode = 0
    t_start = time.time()

    while episode < n_episodes:
        noise_scale = np.random.uniform(*noise_scale_range)

        envs = BatchRydbergEnvNAAC(
            n_envs=n_envs,
            scenario=scenario,
            n_steps=n_steps,
            use_noise=True,
            noise_scale=noise_scale,
            record_trajectory=True,
        )

        rho_calib, rho_adaptive, noise_true, fidelities, noise_est, log_probs = \
            rollout_naac_with_gradients(
                naac, envs, k_calib, n_steps, device, exploration_std, freeze_base
            )

        # Losses
        fidelity_tensor = torch.from_numpy(fidelities).float().to(device)
        returns = fidelity_tensor
        returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs_stacked = torch.stack(log_probs, dim=1)
        policy_loss = -(log_probs_stacked.mean(dim=1) * returns_norm).mean()

        noise_scales = torch.tensor([
            2 * np.pi * 50e3, 2 * np.pi * 50e3, 0.1, 0.1, 2 * np.pi * 1e3, 0.02
        ], device=device)
        noise_est_norm = noise_est / noise_scales
        noise_true_norm = noise_true / noise_scales
        loss_estimator = nn.functional.mse_loss(noise_est_norm, noise_true_norm)

        loss_total = lambda_policy * policy_loss + lambda_estimator * loss_estimator

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        episode += n_envs
        mean_fid = fidelities.mean()
        noise_error = torch.abs(noise_est_norm - noise_true_norm).mean().item()

        stats["episode"].append(episode)
        stats["noise_scale"].append(noise_scale)
        stats["mean_fidelity"].append(float(mean_fid))
        stats["loss_total"].append(loss_total.item())
        stats["loss_policy"].append(policy_loss.item())
        stats["loss_estimator"].append(loss_estimator.item())
        stats["noise_est_error"].append(noise_error)

        writer.add_scalar("train/fidelity", mean_fid, episode)
        writer.add_scalar("train/loss_total", loss_total.item(), episode)
        writer.add_scalar("train/loss_policy", policy_loss.item(), episode)
        writer.add_scalar("train/loss_estimator", loss_estimator.item(), episode)
        writer.add_scalar("train/noise_est_error", noise_error, episode)
        writer.add_scalar("train/noise_scale", noise_scale, episode)

        if episode % 1000 == 0 or episode >= n_episodes:
            elapsed = time.time() - t_start
            eps_per_sec = episode / elapsed
            print(f"Episode {episode:6d} | "
                  f"α={noise_scale:.2f} | "
                  f"F={mean_fid:.4f} | "
                  f"L_tot={loss_total.item():.4f} | "
                  f"L_pol={policy_loss.item():.4f} | "
                  f"L_est={loss_estimator.item():.4f} | "
                  f"err={noise_error:.4f} | "
                  f"{eps_per_sec:.1f} eps/s")

        if episode % 10000 == 0:
            ckpt_path = save_path / f"naac_ep{episode}.pt"
            torch.save({
                "episode": episode,
                "model_state_dict": naac.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stats": stats,
                "cmaes_params": cmaes_params,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    final_path = save_path / "naac_final.pt"
    torch.save({
        "episode": episode,
        "model_state_dict": naac.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stats": stats,
        "cmaes_params": cmaes_params,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")

    writer.close()

    stats_path = save_path / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time/3600:.2f} hours")
    print(f"Final mean fidelity: {stats['mean_fidelity'][-1]:.4f}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NAAC v3 with CMA-ES warm-start")
    parser.add_argument("--scenario", default="C")
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--k-calib", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-episodes", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-est", type=float, default=1.0)
    parser.add_argument("--lambda-pol", type=float, default=0.1)
    parser.add_argument("--exploration-std", type=float, default=0.05)
    parser.add_argument("--no-freeze-base", action="store_true")
    parser.add_argument("--noise-min", type=float, default=0.5)
    parser.add_argument("--noise-max", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_naac_v3(
        scenario=args.scenario,
        n_steps=args.n_steps,
        k_calib=args.k_calib,
        n_envs=args.n_envs,
        n_episodes=args.n_episodes,
        lr=args.lr,
        lambda_estimator=args.lambda_est,
        lambda_policy=args.lambda_pol,
        exploration_std=args.exploration_std,
        freeze_base=not args.no_freeze_base,
        noise_scale_range=(args.noise_min, args.noise_max),
        device=args.device,
        seed=args.seed,
    )
