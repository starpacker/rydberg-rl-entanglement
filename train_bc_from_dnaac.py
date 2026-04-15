"""Behavior Cloning from DNAAC to Open-loop Policy.

Uses DNAAC Phase B corrector to generate expert open-loop pulses,
then trains a simple MLP via BC to map (t/T, noise_normalized) -> action.

The env's noise_conditioned mode provides obs = [t/T, noise_normalized(6)].
The DNAAC corrector expects noise_raw / noise_normalizer.
We bridge the gap by converting between the two normalizations.
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

# Env normalizer (from _get_noise_vector_normalized)
ENV_NORMALIZER = np.array([
    1e6,    # delta_doppler / 1e6
    1e6,    # delta_doppler / 1e6
    0.1,    # delta_R / 0.1
    0.1,    # delta_R / 0.1
    0.1,    # ou_mean / 0.1
    0.1,    # phase_noise / 0.1
])

# NOTE: DNAAC order: [doppler1, doppler2, R1, R2, phase, ou]
# ENV order:         [doppler1, doppler2, R1, R2, ou, phase]
# Need to swap last two when converting!


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
    """Extract raw noise from env and return in DNAAC's normalized format.

    Returns (6,) array in DNAAC's order: [doppler1, doppler2, R1, R2, phase_rate, ou_mean]

    CRITICAL: phase_noise must be converted to detuning rate (rad/s) by dividing
    by T_gate before normalizing. This matches Phase D evaluation in train_dnaac.py.
    """
    delta_doppler = env._noise_params.get("delta_doppler", [0.0, 0.0])
    delta_R = env._noise_params.get("delta_R", [0.0, 0.0])
    phase_noise = env._noise_params.get("phase_noise", 0.0)
    ou_mean = float(env._ou_series.mean()) if env._ou_series is not None else 0.0

    # Convert phase_noise to detuning rate (rad/s) - critical for DNAAC compatibility
    phase_rate = phase_noise / env.T_gate if env.T_gate > 0 else 0.0

    raw = np.array([
        delta_doppler[0],
        delta_doppler[1],
        delta_R[0],
        delta_R[1],
        phase_rate,  # NOT phase_noise!
        ou_mean,
    ])
    return raw / DNAAC_NOISE_NORMALIZER


def generate_expert_data(n_samples=10000, device=None):
    """Generate (env_obs, expert_actions) pairs.

    For each noise realization:
    1. Sample noise via numpy env (get env-normalized noise vector)
    2. Convert to DNAAC normalization, get expert pulse
    3. Store (env_noise_normalized, expert_actions) pair
    """
    corrector = load_dnaac_corrector(device)
    cmaes_params = load_cmaes_params(device)
    decoder = FourierPulseDecoder(n_steps=60, n_fourier=5, device=device)

    dnaac_normalizer_t = torch.tensor(DNAAC_NOISE_NORMALIZER, dtype=torch.float32, device=device)

    all_env_noise = []  # (n_samples, 6) - env normalized
    all_actions = []    # (n_samples, 60, 2)

    rng = np.random.default_rng(42)
    print(f"Generating {n_samples} expert demos...")

    for i in range(n_samples):
        alpha = rng.uniform(0.5, 5.0)
        env = RydbergBellEnv(scenario="C", n_steps=60, use_noise=True,
                             obs_mode="noise_conditioned", noise_scale=alpha)
        env.reset(seed=20000 + i)

        # Get env-normalized noise (what policy will see)
        env_noise = env._get_noise_vector_normalized()  # (6,)

        # Get DNAAC-normalized noise for corrector
        dnaac_noise = env_noise_to_dnaac_noise(env)  # (6,)
        dnaac_noise_t = torch.tensor(dnaac_noise, dtype=torch.float32, device=device).unsqueeze(0)

        # Get expert actions
        with torch.no_grad():
            correction = corrector(dnaac_noise_t)
            params = cmaes_params.unsqueeze(0) + 0.2 * correction  # correction_scale = 0.2 (matches train_dnaac.py)
            actions = decoder(params).cpu().numpy()[0]  # (60, 2)

        all_env_noise.append(env_noise)
        all_actions.append(actions)

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n_samples}")

    noise_out = np.array(all_env_noise)   # (n_samples, 6)
    actions_out = np.array(all_actions)    # (n_samples, 60, 2)
    print(f"Done: noise {noise_out.shape}, actions {actions_out.shape}")
    return noise_out, actions_out


class BCPolicy(nn.Module):
    """Simple MLP: (t/T, noise_6) -> action_2."""

    def __init__(self, hidden=[256, 256]):
        super().__init__()
        layers = []
        prev = 7
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()])
            prev = h
        layers.extend([nn.Linear(prev, 2), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_bc(noise_params, actions, n_epochs=200, batch_size=2048, device=None):
    """Behavior cloning: learn (t/T, noise) -> action from DNAAC expert data."""
    n_samples, n_steps, _ = actions.shape

    # Build dataset: (n_samples * n_steps) samples
    t_fracs = np.tile(np.arange(n_steps) / n_steps, n_samples)
    noise_rep = np.repeat(noise_params, n_steps, axis=0)
    actions_flat = actions.reshape(-1, 2)

    obs = torch.tensor(np.column_stack([t_fracs, noise_rep]), dtype=torch.float32, device=device)
    act = torch.tensor(actions_flat, dtype=torch.float32, device=device)

    N = len(obs)
    print(f"BC training: {N} samples, {n_epochs} epochs")

    policy = BCPolicy().to(device)
    opt = optim.Adam(policy.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        n_b = (N + batch_size - 1) // batch_size

        for b in range(n_b):
            s, e = b * batch_size, min((b + 1) * batch_size, N)
            idx = perm[s:e]
            pred = policy(obs[idx])
            loss = nn.functional.mse_loss(pred, act[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        sched.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/n_b:.6f}")

    return policy


def evaluate(policy, noise_levels=None, n_traj=200, device=None):
    """Evaluate on original numpy env with oracle noise."""
    if noise_levels is None:
        noise_levels = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    policy.eval()
    results = {}

    for alpha in noise_levels:
        print(f"\n  alpha = {alpha}:")
        env = RydbergBellEnv(scenario="C", n_steps=60, use_noise=True,
                             reward_shaping_alpha=0.0, obs_mode="noise_conditioned",
                             noise_scale=alpha)

        fids = []
        for i in range(n_traj):
            obs, _ = env.reset(seed=50000 + i)
            done = False
            while not done:
                with torch.no_grad():
                    a = policy(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                obs, _, term, trunc, info = env.step(a.cpu().numpy()[0])
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

    # Generate expert data
    noise, actions = generate_expert_data(n_samples=10000, device=device)

    # Train
    policy = train_bc(noise, actions, n_epochs=200, device=device)
    torch.save(policy.state_dict(), ROOT / "models" / "bc_from_dnaac.pt")

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION: BC Policy (oracle noise)")
    print("="*60)
    results = evaluate(policy, device=device)

    with open(ROOT / "results" / "bc_from_dnaac.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON: BC vs DNAAC Phase B vs CMA-ES")
    print("="*70)
    print(f"{'alpha':<8} {'CMA-ES':<12} {'DNAAC B':<12} {'BC Policy':<12} {'Gap(BC-CMA)':<12}")
    print("-"*56)

    # Load DNAAC results
    try:
        with open(ROOT / "results" / "dnaac" / "phase_d_eval.json") as f:
            dnaac = json.load(f)
    except FileNotFoundError:
        dnaac = {}

    # Load CMA-ES results
    try:
        with open(ROOT / "results" / "noise_scaling" / "noise_scaling_all.json") as f:
            cmaes_all = json.load(f)
        cmaes_results = {}
        for entry in cmaes_all.get("cmaes", {}).get("noise_levels", []):
            cmaes_results[entry["noise_scale"]] = entry["mean_F"]
    except FileNotFoundError:
        cmaes_results = {}

    for alpha in sorted(results.keys()):
        bc_f = results[alpha]["mean_F"]
        cmaes_f = cmaes_results.get(alpha, 0)
        gap = bc_f - cmaes_f if cmaes_f else "N/A"
        dnaac_f = "N/A"  # Would need to parse phase_d_eval.json
        if isinstance(gap, float):
            print(f"{alpha:<8} {cmaes_f:<12.4f} {dnaac_f:<12} {bc_f:<12.4f} {gap:<+12.4f}")
        else:
            print(f"{alpha:<8} {cmaes_f:<12} {dnaac_f:<12} {bc_f:<12.4f} {gap}")


if __name__ == "__main__":
    main()
