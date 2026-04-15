"""Train NAAC v2 with proper policy gradients.

Key fix: Use REINFORCE to backprop through action generation.
- Actions are sampled from Gaussian policy
- Log probabilities tracked for gradient computation
- Fidelity used as reward signal
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


def rollout_naac_with_gradients(
    naac: NAAC,
    envs: BatchRydbergEnvNAAC,
    k_calib: int,
    n_steps: int,
    device: torch.device,
    exploration_std: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, List[torch.Tensor]]:
    """Rollout NAAC with action log probabilities for REINFORCE.

    Returns
    -------
    rho_calib : torch.Tensor
    rho_adaptive : torch.Tensor
    noise_true : torch.Tensor
    fidelities : np.ndarray
    log_probs : List[torch.Tensor]
        Log probabilities of actions for policy gradient
    """
    n_envs = envs.n_envs
    n_adaptive = n_steps - k_calib

    # Reset environments
    obs, _ = envs.reset()

    # Phase 1: Calibration (deterministic)
    calib_actions = naac.get_calibration_pulse(batch_size=n_envs)
    calib_actions_np = calib_actions.detach().cpu().numpy()

    for step in range(k_calib):
        actions = calib_actions_np[:, step, :]
        obs, _, _, _, _ = envs.step(actions)

    # Get calibration trajectories
    rho_calib_np = envs.get_trajectories()[:, :k_calib, :, :]
    rho_calib = numpy_to_torch_rho(rho_calib_np).to(device)

    # Get ground-truth noise
    noise_true_np = envs.get_noise_vectors()
    noise_true = torch.from_numpy(noise_true_np).float().to(device)

    # Phase 2: Estimate noise
    noise_est = naac.estimate_noise(rho_calib)

    # Phase 3: Adaptive execution with exploration
    rho_adaptive_list = []
    log_probs = []

    for step in range(n_adaptive):
        # Get current ρ
        rho_current_np = np.array([env._rho_np for env in envs.envs])
        rho_current = numpy_to_torch_rho(rho_current_np).to(device)
        rho_adaptive_list.append(rho_current)

        # Generate action (mean from network)
        t = torch.tensor([step / n_adaptive] * n_envs, device=device)
        actions_mean = naac.generate_action(t, noise_est.detach(), rho_current)

        # Add exploration noise and compute log prob
        dist = Normal(actions_mean, exploration_std)
        actions_torch = dist.sample()
        log_prob = dist.log_prob(actions_torch).sum(dim=1)  # Sum over action dims
        log_probs.append(log_prob)

        # Clip actions to [-1, 1]
        actions_torch = torch.clamp(actions_torch, -1.0, 1.0)
        actions_np = actions_torch.detach().cpu().numpy()

        # Step environments
        obs, _, terminated, truncated, infos = envs.step(actions_np)

    # Get final fidelities
    fidelities = np.array([info.get("fidelity", 0.0) for info in infos])

    rho_adaptive = torch.stack(rho_adaptive_list, dim=1)

    return rho_calib, rho_adaptive, noise_true, fidelities, noise_est, log_probs


def train_naac_v2(
    scenario: str = "C",
    n_steps: int = 60,
    k_calib: int = 10,
    n_envs: int = 16,
    n_episodes: int = 50000,
    lr: float = 3e-4,
    lambda_estimator: float = 1.0,
    lambda_policy: float = 1.0,
    exploration_std: float = 0.1,
    noise_scale_range: Tuple[float, float] = (0.5, 5.0),
    save_dir: str = "models/naac_v2",
    log_dir: str = "logs/naac_v2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> Dict:
    """Train NAAC v2 with policy gradients."""

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device)
    print(f"Training on device: {device}")

    # Create directories
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

    optimizer = optim.Adam(naac.parameters(), lr=lr)

    # Training statistics
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
    print(f"{'NAAC v2 Training (with Policy Gradients)':^70}")
    print(f"{'='*70}")
    print(f"Scenario: {scenario}, n_steps: {n_steps}, k_calib: {k_calib}")
    print(f"Noise scale range: {noise_scale_range}")
    print(f"Total episodes: {n_episodes}, batch size: {n_envs}")
    print(f"Exploration std: {exploration_std}")
    print(f"{'='*70}\n")

    episode = 0
    t_start = time.time()

    while episode < n_episodes:
        # Sample noise scale for this batch
        noise_scale = np.random.uniform(*noise_scale_range)

        # Create batch environments
        envs = BatchRydbergEnvNAAC(
            n_envs=n_envs,
            scenario=scenario,
            n_steps=n_steps,
            use_noise=True,
            noise_scale=noise_scale,
            record_trajectory=True,
        )

        # Rollout with gradient tracking
        rho_calib, rho_adaptive, noise_true, fidelities, noise_est, log_probs = \
            rollout_naac_with_gradients(naac, envs, k_calib, n_steps, device, exploration_std)

        # Compute losses
        # 1. Policy gradient loss (REINFORCE)
        fidelity_tensor = torch.from_numpy(fidelities).float().to(device)
        returns = fidelity_tensor  # Use fidelity as reward

        # Normalize returns for stability
        returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy loss: -log_prob * return (maximize fidelity)
        log_probs_stacked = torch.stack(log_probs, dim=1)  # (n_envs, n_adaptive)
        policy_loss = -(log_probs_stacked.mean(dim=1) * returns_norm).mean()

        # 2. Estimator loss (supervised)
        noise_scales = torch.tensor([
            2 * np.pi * 50e3,
            2 * np.pi * 50e3,
            0.1,
            0.1,
            2 * np.pi * 1e3,
            0.02,
        ], device=device)

        noise_est_norm = noise_est / noise_scales
        noise_true_norm = noise_true / noise_scales
        loss_estimator = nn.functional.mse_loss(noise_est_norm, noise_true_norm)

        # Total loss
        loss_total = lambda_policy * policy_loss + lambda_estimator * loss_estimator

        # Backward
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(naac.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        episode += n_envs
        mean_fid = fidelities.mean()
        noise_error = torch.abs(noise_est_norm - noise_true_norm).mean().item()

        stats["episode"].append(episode)
        stats["noise_scale"].append(noise_scale)
        stats["mean_fidelity"].append(mean_fid)
        stats["loss_total"].append(loss_total.item())
        stats["loss_policy"].append(policy_loss.item())
        stats["loss_estimator"].append(loss_estimator.item())
        stats["noise_est_error"].append(noise_error)

        # Logging
        writer.add_scalar("train/fidelity", mean_fid, episode)
        writer.add_scalar("train/loss_total", loss_total.item(), episode)
        writer.add_scalar("train/loss_policy", policy_loss.item(), episode)
        writer.add_scalar("train/loss_estimator", loss_estimator.item(), episode)
        writer.add_scalar("train/noise_est_error", noise_error, episode)
        writer.add_scalar("train/noise_scale", noise_scale, episode)

        # Print progress
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

        # Save checkpoint
        if episode % 10000 == 0:
            ckpt_path = save_path / f"naac_ep{episode}.pt"
            torch.save({
                "episode": episode,
                "model_state_dict": naac.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stats": stats,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = save_path / "naac_final.pt"
    torch.save({
        "episode": episode,
        "model_state_dict": naac.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stats": stats,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")

    writer.close()

    # Save training stats
    stats_path = save_path / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time/3600:.2f} hours")
    print(f"Final mean fidelity: {stats['mean_fidelity'][-1]:.4f}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NAAC v2 with policy gradients")
    parser.add_argument("--scenario", default="C", help="Scenario key")
    parser.add_argument("--n-steps", type=int, default=60, help="Total control steps")
    parser.add_argument("--k-calib", type=int, default=10, help="Calibration steps")
    parser.add_argument("--n-envs", type=int, default=16, help="Batch size")
    parser.add_argument("--n-episodes", type=int, default=50000, help="Total episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lambda-est", type=float, default=1.0, help="Estimator loss weight")
    parser.add_argument("--lambda-pol", type=float, default=1.0, help="Policy loss weight")
    parser.add_argument("--exploration-std", type=float, default=0.1, help="Exploration noise std")
    parser.add_argument("--noise-min", type=float, default=0.5, help="Min noise scale")
    parser.add_argument("--noise-max", type=float, default=5.0, help="Max noise scale")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    stats = train_naac_v2(
        scenario=args.scenario,
        n_steps=args.n_steps,
        k_calib=args.k_calib,
        n_envs=args.n_envs,
        n_episodes=args.n_episodes,
        lr=args.lr,
        lambda_estimator=args.lambda_est,
        lambda_policy=args.lambda_pol,
        exploration_std=args.exploration_std,
        noise_scale_range=(args.noise_min, args.noise_max),
        device=args.device,
        seed=args.seed,
    )
