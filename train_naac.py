"""Train NAAC (Noise-Aware Adaptive Control) via meta-learning.

Training strategy:
1. Sample noise_scale ~ Uniform(0.5, 5.0) for each batch
2. Collect trajectories with NAAC policy
3. Compute losses:
   - L_fidelity: negative final fidelity
   - L_estimator: MSE between estimated and true noise
4. Meta-update to ensure generalization across noise scales

Key innovation: Single model works across full noise spectrum.
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
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.algorithms.naac import NAAC, numpy_to_torch_rho
from src.environments.rydberg_env_naac import BatchRydbergEnvNAAC


# ===================================================================
# Training Loop
# ===================================================================

def rollout_naac(
    naac: NAAC,
    envs: BatchRydbergEnvNAAC,
    k_calib: int,
    n_steps: int,
    device: torch.device,
    use_oracle_noise: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Rollout NAAC policy in batch environments.

    Parameters
    ----------
    use_oracle_noise : bool
        If True, use ground-truth noise instead of estimated (for curriculum)

    Returns
    -------
    rho_calib : torch.Tensor
        Calibration trajectory, shape (n_envs, k_calib, 4, 4, 2)
    rho_adaptive : torch.Tensor
        Adaptive trajectory, shape (n_envs, n_adaptive, 4, 4, 2)
    noise_true : torch.Tensor
        Ground-truth noise, shape (n_envs, 6)
    fidelities : np.ndarray
        Final fidelities, shape (n_envs,)
    """
    n_envs = envs.n_envs
    n_adaptive = n_steps - k_calib

    # Reset environments
    obs, _ = envs.reset()

    # Phase 1: Calibration
    calib_actions = naac.get_calibration_pulse(batch_size=n_envs)  # (n_envs, k_calib, 2)
    calib_actions_np = calib_actions.detach().cpu().numpy()

    for step in range(k_calib):
        actions = calib_actions_np[:, step, :]  # (n_envs, 2)
        obs, _, _, _, _ = envs.step(actions)

    # Get calibration trajectories
    rho_calib_np = envs.get_trajectories()[:, :k_calib+1, :, :]  # (n_envs, k_calib+1, 4, 4)
    rho_calib = numpy_to_torch_rho(rho_calib_np[:, :k_calib, :, :]).to(device)  # Use first k_calib

    # Get ground-truth noise
    noise_true_np = envs.get_noise_vectors()  # (n_envs, 6)
    noise_true = torch.from_numpy(noise_true_np).float().to(device)

    # Phase 2: Estimate noise (or use oracle)
    noise_est = naac.estimate_noise(rho_calib)

    if use_oracle_noise:
        noise_for_control = noise_true
    else:
        noise_for_control = noise_est.detach()  # Don't backprop through control

    # Phase 3: Adaptive execution
    rho_adaptive_list = []
    for step in range(n_adaptive):
        # Get current ρ
        rho_current_np = np.array([env._rho_np for env in envs.envs])  # (n_envs, 4, 4)
        rho_current = numpy_to_torch_rho(rho_current_np).to(device)
        rho_adaptive_list.append(rho_current)

        # Generate action
        t = torch.tensor([step / n_adaptive] * n_envs, device=device)
        with torch.no_grad():
            actions_torch = naac.generate_action(t, noise_for_control, rho_current)
        actions_np = actions_torch.cpu().numpy()

        # Step environments
        obs, _, terminated, truncated, infos = envs.step(actions_np)

    # Get final fidelities
    fidelities = np.array([info.get("fidelity", 0.0) for info in infos])

    rho_adaptive = torch.stack(rho_adaptive_list, dim=1)  # (n_envs, n_adaptive, 4, 4, 2)

    return rho_calib, rho_adaptive, noise_true, fidelities, noise_est


def train_naac(
    scenario: str = "C",
    n_steps: int = 60,
    k_calib: int = 10,
    n_envs: int = 16,
    n_episodes: int = 50000,
    batch_size: int = 16,
    lr: float = 3e-4,
    lambda_estimator: float = 1.0,
    noise_scale_range: Tuple[float, float] = (0.5, 5.0),
    save_dir: str = "models/naac",
    log_dir: str = "logs/naac",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> Dict:
    """Train NAAC via meta-learning.

    Parameters
    ----------
    scenario : str
        Base scenario
    n_steps : int
        Total control steps
    k_calib : int
        Calibration steps
    n_envs : int
        Batch size for parallel environments
    n_episodes : int
        Total training episodes
    batch_size : int
        Gradient accumulation batch size
    lr : float
        Learning rate
    lambda_estimator : float
        Weight for estimator loss
    noise_scale_range : tuple
        (min, max) noise scale for meta-learning
    save_dir : str
        Model checkpoint directory
    log_dir : str
        TensorBoard log directory
    device : str
        'cuda' or 'cpu'
    seed : int
        Random seed

    Returns
    -------
    result : dict
        Training statistics
    """
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
        "loss_fidelity": [],
        "loss_estimator": [],
        "noise_est_error": [],
    }

    print(f"\n{'='*70}")
    print(f"{'NAAC Training':^70}")
    print(f"{'='*70}")
    print(f"Scenario: {scenario}, n_steps: {n_steps}, k_calib: {k_calib}")
    print(f"Noise scale range: {noise_scale_range}")
    print(f"Total episodes: {n_episodes}, batch size: {batch_size}")
    print(f"{'='*70}\n")

    episode = 0
    t_start = time.time()

    while episode < n_episodes:
        # Sample noise scale for this batch (meta-learning)
        noise_scale = np.random.uniform(*noise_scale_range)

        # Create batch environments with this noise scale
        envs = BatchRydbergEnvNAAC(
            n_envs=n_envs,
            scenario=scenario,
            n_steps=n_steps,
            use_noise=True,
            noise_scale=noise_scale,
            record_trajectory=True,
        )

        # Rollout
        rho_calib, rho_adaptive, noise_true, fidelities, noise_est = rollout_naac(
            naac, envs, k_calib, n_steps, device
        )

        # Compute losses
        # 1. Fidelity loss (negative mean fidelity)
        fidelity_tensor = torch.from_numpy(fidelities).float().to(device)
        loss_fidelity = -fidelity_tensor.mean()

        # 2. Estimator loss (MSE with normalized noise parameters)
        # Normalize noise parameters to similar scales for stable training
        noise_scales = torch.tensor([
            2 * np.pi * 50e3,  # δ_doppler scale
            2 * np.pi * 50e3,
            0.1,                # δ_R scale
            0.1,
            2 * np.pi * 1e3,   # δ_phase scale
            0.02,               # η_OU scale
        ], device=device)

        noise_est_norm = noise_est / noise_scales
        noise_true_norm = noise_true / noise_scales
        loss_estimator = nn.functional.mse_loss(noise_est_norm, noise_true_norm)

        # Total loss
        loss_total = loss_fidelity + lambda_estimator * loss_estimator

        # Backward
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(naac.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        episode += n_envs
        mean_fid = fidelities.mean()
        # Compute error in normalized space for interpretability
        noise_error = torch.abs(noise_est_norm - noise_true_norm).mean().item()

        stats["episode"].append(episode)
        stats["noise_scale"].append(noise_scale)
        stats["mean_fidelity"].append(mean_fid)
        stats["loss_total"].append(loss_total.item())
        stats["loss_fidelity"].append(loss_fidelity.item())
        stats["loss_estimator"].append(loss_estimator.item())
        stats["noise_est_error"].append(noise_error)

        # Logging
        writer.add_scalar("train/fidelity", mean_fid, episode)
        writer.add_scalar("train/loss_total", loss_total.item(), episode)
        writer.add_scalar("train/loss_fidelity", loss_fidelity.item(), episode)
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


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NAAC")
    parser.add_argument("--scenario", default="C", help="Scenario key")
    parser.add_argument("--n-steps", type=int, default=60, help="Total control steps")
    parser.add_argument("--k-calib", type=int, default=10, help="Calibration steps")
    parser.add_argument("--n-envs", type=int, default=16, help="Batch size")
    parser.add_argument("--n-episodes", type=int, default=50000, help="Total episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lambda-est", type=float, default=1.0, help="Estimator loss weight")
    parser.add_argument("--noise-min", type=float, default=0.5, help="Min noise scale")
    parser.add_argument("--noise-max", type=float, default=5.0, help="Max noise scale")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    stats = train_naac(
        scenario=args.scenario,
        n_steps=args.n_steps,
        k_calib=args.k_calib,
        n_envs=args.n_envs,
        n_episodes=args.n_episodes,
        lr=args.lr,
        lambda_estimator=args.lambda_est,
        noise_scale_range=(args.noise_min, args.noise_max),
        device=args.device,
        seed=args.seed,
    )
