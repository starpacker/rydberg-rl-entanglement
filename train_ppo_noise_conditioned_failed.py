"""Noise-conditioned Open-loop PPO training.

The policy observes (t/T, noise_params) where noise_params are oracle (training)
or estimated (inference). This enables open-loop pulse adaptation to noise.

Training:
    obs = [t/T, noise_params...]  <- oracle noise (7-dim)
    Policy learns: given noise parameters, what's the optimal pulse at time t?

Inference:
    1. Run calibration trajectory (5-10 steps with fixed probe pulse)
    2. Use noise estimator to estimate noise_params from rho trajectory
    3. Use estimated noise_params to condition policy
    4. Execute open-loop (actions depend only on t and estimated noise)

This bridges the gap between:
    - Pure open-loop (time_only): no noise adaptation
    - DNAAC: noise adaptation via Fourier correction network
    - This: noise adaptation via PPO policy

Advantage over DNAAC Phase B: learns full action trajectory, not just Fourier correction.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.environments.rydberg_env import RydbergBellEnv


class FidelityLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.fidelities: List[float] = []
        self.timesteps_log: List[int] = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "fidelity" in info:
                self.fidelities.append(info["fidelity"])
                self.timesteps_log.append(self.num_timesteps)
                if self.verbose >= 1 and len(self.fidelities) % 200 == 0:
                    recent = self.fidelities[-200:]
                    print(f"  [Step {self.num_timesteps}] Ep {len(self.fidelities)}: "
                          f"mean F = {np.mean(recent):.4f}, max F = {np.max(recent):.4f}")
        return True


def train(
    total_timesteps: int = 3_000_000,
    n_seeds: int = 3,
    env_n_steps: int = 60,
    scenario: str = "C",
    noise_scale: float = None,  # None = uniform sampling over [0.5, 5.0]
):
    """Train noise-conditioned open-loop PPO.

    The policy sees obs = [t/T, noise_params_normalized] at each step.
    Since noise_params are constant within an episode, the policy learns
    a time-dependent pulse that varies based on noise realization.
    """
    models_dir = ROOT / "models"
    results_dir = ROOT / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    best_model = None
    best_fid = -1.0
    all_logs = []

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 111
        print(f"\n{'='*60}")
        print(f"Noise-conditioned PPO on Scenario {scenario}, seed={seed}")
        print(f"  {total_timesteps/1e6:.0f}M steps, {env_n_steps} ctrl steps")
        print(f"  obs_mode='noise_conditioned' -> policy sees [t/T, noise_params]")
        print(f"{'='*60}")

        env = RydbergBellEnv(
            scenario=scenario,
            n_steps=env_n_steps,
            use_noise=True,
            reward_shaping_alpha=0.15,
            obs_mode="noise_conditioned",
            noise_scale=noise_scale,
        )

        # Linear LR schedule: 3e-4 -> 5e-5
        def lr_schedule(progress_remaining: float) -> float:
            return 5e-5 + (3e-4 - 5e-5) * progress_remaining

        model = PPO(
            "MlpPolicy", env,
            learning_rate=lr_schedule,
            n_steps=4096,
            batch_size=512,
            n_epochs=10,
            gamma=1.0,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=seed,
            verbose=0,
            policy_kwargs={"net_arch": [512, 256]},
        )

        cb = FidelityLogCallback(verbose=1)
        t0 = time.perf_counter()
        model.learn(total_timesteps=total_timesteps, callback=cb)
        wall_time = time.perf_counter() - t0

        final_fids = cb.fidelities[-200:] if len(cb.fidelities) >= 200 else cb.fidelities
        final_mean = float(np.mean(final_fids)) if final_fids else 0.0

        print(f"  Seed {seed}: {wall_time:.1f}s, final mean F = {final_mean:.4f}")

        model_path = models_dir / f"ppo_noise_conditioned_seed{seed}"
        model.save(str(model_path))

        seed_log = {
            "seed": seed,
            "wall_time": wall_time,
            "n_episodes": len(cb.fidelities),
            "final_mean_fidelity": final_mean,
            "fidelities": cb.fidelities,
            "timesteps": cb.timesteps_log,
        }
        all_logs.append(seed_log)

        if final_mean > best_fid:
            best_fid = final_mean
            best_model = model

    # Save best model
    if best_model is not None:
        best_path = models_dir / "ppo_noise_conditioned_best"
        best_model.save(str(best_path))
        print(f"\nBest noise-conditioned model: F = {best_fid:.4f}")

    # Save training logs
    log_path = results_dir / "training_logs_ppo_noise_conditioned.json"
    with open(log_path, "w") as f:
        json.dump({
            "scenario": scenario,
            "total_timesteps": total_timesteps,
            "env_n_steps": env_n_steps,
            "n_seeds": n_seeds,
            "obs_mode": "noise_conditioned",
            "seeds": [{
                "seed": s["seed"],
                "wall_time": s["wall_time"],
                "n_episodes": s["n_episodes"],
                "final_mean_fidelity": s["final_mean_fidelity"],
            } for s in all_logs],
        }, f, indent=2)
    print(f"Training logs saved to {log_path}")

    return best_model


def evaluate_oracle(model=None, noise_levels: List[float] = None, n_traj: int = 200):
    """Evaluate with oracle noise params (upper bound on performance)."""
    if noise_levels is None:
        noise_levels = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    scenario = "C"
    env_n_steps = 60

    if model is None:
        best_path = ROOT / "models" / "ppo_noise_conditioned_best"
        model = PPO.load(str(best_path))
        print(f"Loaded model from {best_path}")

    print(f"\n{'='*60}")
    print(f"Evaluate Noise-conditioned PPO with Oracle Noise")
    print(f"{'='*60}")

    results = {}

    for alpha in noise_levels:
        print(f"\n  alpha = {alpha}:")
        env = RydbergBellEnv(
            scenario=scenario, n_steps=env_n_steps,
            use_noise=True, reward_shaping_alpha=0.0,
            obs_mode="noise_conditioned",
            noise_scale=alpha,
        )

        fids = []
        for i in range(n_traj):
            obs, _ = env.reset(seed=5000 + i)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            fids.append(info.get("fidelity", 0.0))

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{n_traj}: mean_F = {np.mean(fids):.4f}")

        fids = np.array(fids)
        results[alpha] = {
            "mean_F": float(fids.mean()),
            "std_F": float(fids.std()),
            "F_05": float(np.percentile(fids, 5)),
            "min_F": float(fids.min()),
        }
        print(f"    Result: {fids.mean():.4f} +/- {fids.std():.4f}")

    return results


def evaluate_with_estimator(
    model=None,
    estimator_path: Path = None,
    noise_levels: List[float] = None,
    n_traj: int = 200,
    k_calib: int = 10,
):
    """Evaluate using DNAAC's noise estimator instead of oracle noise.

    Steps:
    1. Run k_calib calibration steps
    2. Estimate noise from trajectory
    3. Use estimated noise to condition the policy
    4. Execute remaining steps open-loop
    """
    if noise_levels is None:
        noise_levels = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    scenario = "C"
    env_n_steps = 60

    # Load models
    if model is None:
        best_path = ROOT / "models" / "ppo_noise_conditioned_best"
        model = PPO.load(str(best_path))
        print(f"Loaded PPO from {best_path}")

    if estimator_path is None:
        estimator_path = ROOT / "models" / "dnaac" / "estimator.pt"

    # Load estimator (from DNAAC)
    sys.path.insert(0, str(ROOT))
    from train_dnaac import NoiseEstimatorDiff

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estimator = NoiseEstimatorDiff(k_calib=k_calib).to(device)
    estimator.load_state_dict(torch.load(estimator_path, map_location=device))
    estimator.eval()
    print(f"Loaded estimator from {estimator_path}")

    print(f"\n{'='*60}")
    print(f"Evaluate Noise-conditioned PPO with Estimated Noise")
    print(f"  k_calib = {k_calib}")
    print(f"{'='*60}")

    # Calibration pulse: simple fixed pulse
    def get_calib_actions(n_steps):
        """Return calibration pulse actions (simple ramp)."""
        actions = []
        for t in range(n_steps):
            frac = t / n_steps
            omega_norm = 0.5 * (1 - np.cos(np.pi * frac))  # Smooth ramp
            delta_norm = 0.0
            actions.append(np.array([omega_norm * 2 - 1, delta_norm], dtype=np.float32))
        return actions

    calib_actions = get_calib_actions(k_calib)

    results = {}

    for alpha in noise_levels:
        print(f"\n  alpha = {alpha}:")
        env = RydbergBellEnv(
            scenario=scenario, n_steps=env_n_steps,
            use_noise=True, reward_shaping_alpha=0.0,
            obs_mode="full",  # Need full obs for calibration
            obs_include_time=True,
            noise_scale=alpha,
        )

        fids = []
        for i in range(n_traj):
            obs, _ = env.reset(seed=5000 + i)

            # Phase 1: Calibration (k_calib steps)
            rho_traj = [env._rho_np.copy()]
            for t in range(k_calib):
                obs, _, _, _, _ = env.step(calib_actions[t])
                rho_traj.append(env._rho_np.copy())

            # Estimate noise from calibration trajectory
            rho_traj_np = np.array(rho_traj)  # (k_calib+1, 4, 4) complex
            rho_traj_torch = torch.tensor(rho_traj_np, dtype=torch.complex64, device=device)
            rho_traj_torch = rho_traj_torch.unsqueeze(0)  # (1, k_calib+1, 4, 4)

            with torch.no_grad():
                noise_est = estimator(rho_traj_torch).cpu().numpy()[0]  # (6,)

            # Phase 2: Execute remaining steps with noise-conditioned policy
            for t in range(k_calib, env_n_steps):
                time_frac = t / env_n_steps
                # Construct noise-conditioned observation
                obs_nc = np.concatenate([[time_frac], noise_est]).astype(np.float32)
                action, _ = model.predict(obs_nc, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

            fids.append(info.get("fidelity", 0.0))

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{n_traj}: mean_F = {np.mean(fids):.4f}")

        fids = np.array(fids)
        results[alpha] = {
            "mean_F": float(fids.mean()),
            "std_F": float(fids.std()),
            "F_05": float(np.percentile(fids, 5)),
            "min_F": float(fids.min()),
        }
        print(f"    Result: {fids.mean():.4f} +/- {fids.std():.4f}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval-oracle", action="store_true", help="Evaluate with oracle noise")
    parser.add_argument("--eval-estimated", action="store_true", help="Evaluate with estimated noise")
    parser.add_argument("--timesteps", type=int, default=3_000_000, help="Training timesteps")
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--n-traj", type=int, default=200, help="Evaluation trajectories per noise level")
    parser.add_argument("--k-calib", type=int, default=10, help="Calibration steps for estimator mode")
    args = parser.parse_args()

    model = None

    if args.train:
        model = train(total_timesteps=args.timesteps, n_seeds=args.n_seeds)

    if args.eval_oracle:
        results = evaluate_oracle(model=model, n_traj=args.n_traj)
        out_path = ROOT / "results" / "ppo_noise_conditioned_oracle.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nOracle results saved to {out_path}")

    if args.eval_estimated:
        results = evaluate_with_estimator(model=model, n_traj=args.n_traj, k_calib=args.k_calib)
        out_path = ROOT / "results" / "ppo_noise_conditioned_estimated.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nEstimated results saved to {out_path}")

    if not args.train and not args.eval_oracle and not args.eval_estimated:
        # Default: train and evaluate both
        model = train(total_timesteps=args.timesteps, n_seeds=args.n_seeds)

        print("\n" + "="*60)
        print("EVALUATION PHASE")
        print("="*60)

        results_oracle = evaluate_oracle(model=model, n_traj=args.n_traj)
        results_estimated = evaluate_with_estimator(model=model, n_traj=args.n_traj, k_calib=args.k_calib)

        # Save combined results
        combined = {
            "oracle": results_oracle,
            "estimated": results_estimated,
        }
        out_path = ROOT / "results" / "ppo_noise_conditioned_all.json"
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2)

        # Print summary comparison
        print("\n" + "="*60)
        print("SUMMARY: Noise-conditioned PPO")
        print("="*60)
        print(f"{'alpha':<8} {'Oracle F':<12} {'Estimated F':<12} {'Gap':<10}")
        print("-"*42)
        for alpha in sorted(results_oracle.keys()):
            oracle_f = results_oracle[alpha]["mean_F"]
            est_f = results_estimated[alpha]["mean_F"]
            gap = est_f - oracle_f
            print(f"{alpha:<8} {oracle_f:<12.4f} {est_f:<12.4f} {gap:<+10.4f}")


if __name__ == "__main__":
    main()
