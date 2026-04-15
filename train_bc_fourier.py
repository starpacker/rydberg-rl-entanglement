"""Behavior Cloning from DNAAC with Fourier parameterization.

Key insight: DNAAC corrector outputs 20-dim Fourier params, not 60×2 actions.
BC should learn: noise → Fourier params, not noise × t → action.

This matches DNAAC architecture exactly:
    noise → corrector → base_params + 0.2*correction → decoder → actions

We distill the corrector via supervised learning (MSE on Fourier params),
then use the same FourierPulseDecoder at inference.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.environments.rydberg_env import RydbergBellEnv
from src.physics.differentiable_lindblad import DNAAC_NOISE_NORMALIZER, FourierPulseDecoder

CORRECTION_SCALE = 0.2  # From train_dnaac.py


def load_dnaac_corrector(device):
    from train_dnaac import NoiseConditionedCorrector
    corrector = NoiseConditionedCorrector().to(device)
    corrector.load_state_dict(torch.load(ROOT / "models" / "dnaac" / "corrector.pt", map_location=device))
    corrector.eval()
    return corrector


def load_cmaes_params(device):
    with open(ROOT / "results" / "noise_scaling" / "cmaes_sweep.json") as f:
        data = json.load(f)
    for entry in data["noise_levels"]:
        if entry["noise_scale"] == 1.0:
            return torch.tensor(entry["best_params"], dtype=torch.float32, device=device)
    raise ValueError("Not found")


def env_noise_to_dnaac_noise(env) -> np.ndarray:
    """Extract raw noise from env and return in DNAAC's normalized format."""
    delta_doppler = env._noise_params.get("delta_doppler", [0.0, 0.0])
    delta_R = env._noise_params.get("delta_R", [0.0, 0.0])
    phase_noise = env._noise_params.get("phase_noise", 0.0)
    ou_mean = float(env._ou_series.mean()) if env._ou_series is not None else 0.0

    # Convert phase_noise to detuning rate (rad/s)
    phase_rate = phase_noise / env.T_gate if env.T_gate > 0 else 0.0

    raw = np.array([
        delta_doppler[0],
        delta_doppler[1],
        delta_R[0],
        delta_R[1],
        phase_rate,
        ou_mean,
    ])
    return raw / DNAAC_NOISE_NORMALIZER


def env_noise_to_policy_noise(env) -> np.ndarray:
    """Get noise vector in env's normalized format (for policy input)."""
    return env._get_noise_vector_normalized()


def generate_expert_data(n_samples=10000, device=None, use_dnaac_normalized=False):
    """Generate (noise_normalized, expert_fourier_params) pairs.

    Args:
        n_samples: Number of samples to generate
        device: Compute device
        use_dnaac_normalized: If True, output noise in DNAAC format (for estimator compatibility)
                              If False, output noise in env format (for oracle mode)

    For each noise realization:
    1. Sample noise via numpy env
    2. Get expert Fourier params from DNAAC corrector
    3. Store (noise_normalized, final_fourier_params) pair
    """
    corrector = load_dnaac_corrector(device)
    cmaes_params = load_cmaes_params(device)

    all_noise = []          # (n_samples, 6)
    all_fourier_params = [] # (n_samples, 20)

    rng = np.random.default_rng(42)
    mode_str = "DNAAC-normalized" if use_dnaac_normalized else "env-normalized"
    print(f"Generating {n_samples} expert demos ({mode_str})...")

    for i in range(n_samples):
        alpha = rng.uniform(0.5, 5.0)
        env = RydbergBellEnv(scenario="C", n_steps=60, use_noise=True,
                             obs_mode="noise_conditioned", noise_scale=alpha)
        env.reset(seed=20000 + i)

        # Get DNAAC-normalized noise for corrector
        dnaac_noise = env_noise_to_dnaac_noise(env)  # (6,)
        dnaac_noise_t = torch.tensor(dnaac_noise, dtype=torch.float32, device=device).unsqueeze(0)

        # Get expert Fourier params
        with torch.no_grad():
            correction = corrector(dnaac_noise_t)
            fourier_params = cmaes_params + CORRECTION_SCALE * correction.squeeze()

        # Store noise in requested format
        if use_dnaac_normalized:
            all_noise.append(dnaac_noise)  # DNAAC format (compatible with estimator)
        else:
            env_noise = env_noise_to_policy_noise(env)  # Env format (for oracle)
            all_noise.append(env_noise)

        all_fourier_params.append(fourier_params.cpu().numpy())

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n_samples}")

    noise_out = np.array(all_noise)       # (n_samples, 6)
    params_out = np.array(all_fourier_params) # (n_samples, 20)
    print(f"Done: noise {noise_out.shape}, params {params_out.shape}")
    return noise_out, params_out


class BCFourierPolicy(nn.Module):
    """Maps noise params to Fourier pulse params.

    Architecture matches DNAAC corrector: 6 → [128, 64] → 20
    but input is env-normalized noise instead of DNAAC-normalized.
    """

    def __init__(self, hidden=[128, 64]):
        super().__init__()
        layers = []
        prev = 6
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 20))
        self.net = nn.Sequential(*layers)

    def forward(self, noise):
        return self.net(noise)


def train_bc(noise_params, fourier_params, n_epochs=500, batch_size=256, device=None):
    """Behavior cloning: learn noise → Fourier params from DNAAC expert."""
    n_samples = len(noise_params)

    noise_t = torch.tensor(noise_params, dtype=torch.float32, device=device)
    params_t = torch.tensor(fourier_params, dtype=torch.float32, device=device)

    print(f"BC Fourier training: {n_samples} samples, {n_epochs} epochs")

    policy = BCFourierPolicy().to(device)
    opt = optim.Adam(policy.parameters(), lr=3e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_b = (n_samples + batch_size - 1) // batch_size

        for b in range(n_b):
            s, e = b * batch_size, min((b + 1) * batch_size, n_samples)
            idx = perm[s:e]
            pred = policy(noise_t[idx])
            loss = nn.functional.mse_loss(pred, params_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        sched.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/n_b:.6f}")

    return policy


def evaluate(policy, noise_levels=None, n_traj=200, device=None):
    """Evaluate BC Fourier policy on numpy env."""
    if noise_levels is None:
        noise_levels = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    policy.eval()
    decoder = FourierPulseDecoder(n_steps=60, n_fourier=5, device=device)
    results = {}

    for alpha in noise_levels:
        print(f"\n  alpha = {alpha}:")
        env = RydbergBellEnv(scenario="C", n_steps=60, use_noise=True,
                             reward_shaping_alpha=0.0, obs_mode="noise_conditioned",
                             noise_scale=alpha)

        fids = []
        for i in range(n_traj):
            obs, _ = env.reset(seed=50000 + i)
            noise_vec = env_noise_to_policy_noise(env)

            # Get Fourier params from BC policy
            with torch.no_grad():
                noise_t = torch.tensor(noise_vec, dtype=torch.float32, device=device).unsqueeze(0)
                fourier_params = policy(noise_t)
                actions = decoder(fourier_params).cpu().numpy()[0]  # (60, 2)

            # Replay actions through env
            done = False
            t = 0
            while not done:
                _, _, term, trunc, info = env.step(actions[t])
                t += 1
                done = term or trunc
            fids.append(info["fidelity"])

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{n_traj}: mean_F={np.mean(fids):.4f}")

        fids = np.array(fids)
        results[alpha] = {
            "mean_F": float(fids.mean()),
            "std_F": float(fids.std()),
        }
        print(f"    => {fids.mean():.4f} +/- {fids.std():.4f}")

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Train TWO models:
    # 1. env-normalized (for oracle mode evaluation)
    # 2. DNAAC-normalized (for estimator mode deployment)

    print("\n" + "="*70)
    print("TRAINING: BC-Fourier (env-normalized, for oracle mode)")
    print("="*70)
    noise_env, fourier_params = generate_expert_data(n_samples=20000, device=device, use_dnaac_normalized=False)
    policy_env = train_bc(noise_env, fourier_params, n_epochs=500, device=device)
    torch.save(policy_env.state_dict(), ROOT / "models" / "bc_fourier_policy.pt")

    print("\n" + "="*70)
    print("TRAINING: BC-Fourier (DNAAC-normalized, for estimator mode)")
    print("="*70)
    noise_dnaac, _ = generate_expert_data(n_samples=20000, device=device, use_dnaac_normalized=True)
    policy_dnaac = train_bc(noise_dnaac, fourier_params, n_epochs=500, device=device)
    torch.save(policy_dnaac.state_dict(), ROOT / "models" / "bc_fourier_policy_dnaac.pt")

    # Evaluate env-normalized policy with oracle noise
    print("\n" + "="*60)
    print("EVALUATION: BC-Fourier (Oracle Noise)")
    print("="*60)
    results_oracle = evaluate(policy_env, device=device)

    with open(ROOT / "results" / "bc_fourier.json", "w") as f:
        json.dump(results_oracle, f, indent=2)

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON: BC-Fourier vs DNAAC Phase B vs CMA-ES")
    print("="*70)
    print(f"{'alpha':<8} {'CMA-ES':<12} {'DNAAC B':<12} {'BC-Fourier':<12} {'Gap(BC-CMA)':<12}")
    print("-"*56)

    # Load reference results
    try:
        with open(ROOT / "results" / "noise_scaling" / "noise_scaling_all.json") as f:
            cmaes_all = json.load(f)
        cmaes_results = {}
        for entry in cmaes_all.get("cmaes", {}).get("noise_levels", []):
            cmaes_results[entry["noise_scale"]] = entry["mean_F"]
    except FileNotFoundError:
        cmaes_results = {}

    # DNAAC Phase B results
    dnaac_b_results = {
        0.5: 0.9979, 1.0: 0.9978, 1.5: 0.9976,
        2.0: 0.9971, 3.0: 0.9964, 5.0: 0.9908
    }

    for alpha in sorted(results_oracle.keys()):
        bc_f = results_oracle[alpha]["mean_F"]
        cmaes_f = cmaes_results.get(alpha, 0)
        dnaac_f = dnaac_b_results.get(alpha, "N/A")
        gap = bc_f - cmaes_f if cmaes_f else "N/A"
        print(f"{alpha:<8} {cmaes_f:<12.4f} {dnaac_f:<12.4f} {bc_f:<12.4f} {gap:<+12.4f}" if isinstance(gap, float) else f"{alpha:<8} {'N/A':<12} {dnaac_f:<12} {bc_f:<12.4f} {gap}")


if __name__ == "__main__":
    main()
