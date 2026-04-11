"""Open-loop PPO v2: Fourier features + improved hyperparameters.

Fixes for naive open-loop PPO (F=0.495):
1. Fourier feature encoding: obs = [sin(2πkt), cos(2πkt)] for k=0..7 (16-dim)
2. Parallel environments: n_envs=16 to reduce gradient variance
3. Larger batch: n_steps=8192 for more stable updates
4. More exploration: ent_coef=0.01
5. Smaller network: [256, 128] (Fourier features do the heavy lifting)

Target: beat GRAPE's F=0.803 on Scenario C.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.environments.rydberg_env import RydbergBellEnv
from src.environments.rydberg_env_fourier import FourierFeatureWrapper


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
                if self.verbose >= 1 and len(self.fidelities) % 100 == 0:
                    recent = self.fidelities[-100:]
                    print(f"  [Step {self.num_timesteps}] Ep {len(self.fidelities)}: "
                          f"mean F = {np.mean(recent):.4f}, max F = {np.max(recent):.4f}")
        return True


def make_env(scenario: str, n_steps: int, n_fourier: int, seed: int):
    """Factory for creating Fourier-wrapped environments."""
    def _init():
        env = RydbergBellEnv(
            scenario=scenario,
            n_steps=n_steps,
            use_noise=True,
            reward_shaping_alpha=0.15,
            obs_mode="time_only",
        )
        env = FourierFeatureWrapper(env, n_fourier=n_fourier)
        env.reset(seed=seed)
        return env
    return _init


def train():
    scenario = "C"
    total_timesteps = 3_000_000
    n_seeds = 3
    env_n_steps = 60
    n_fourier = 8  # Fourier frequencies k=0..7 -> 16-dim obs
    n_envs = 16    # Parallel environments

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
        print(f"Open-loop PPO v2 (Fourier) on Scenario {scenario}, seed={seed}")
        print(f"  {total_timesteps/1e6:.0f}M steps, {env_n_steps} ctrl steps")
        print(f"  Fourier features: k=0..{n_fourier-1} -> {2*n_fourier}-dim obs")
        print(f"  Parallel envs: {n_envs}")
        print(f"{'='*60}")

        # Create vectorized environment
        env_fns = [make_env(scenario, env_n_steps, n_fourier, seed + i)
                   for i in range(n_envs)]
        env = SubprocVecEnv(env_fns)

        # Linear LR schedule: 3e-4 -> 5e-5
        def lr_schedule(progress_remaining: float) -> float:
            return 5e-5 + (3e-4 - 5e-5) * progress_remaining

        model = PPO(
            "MlpPolicy", env,
            learning_rate=lr_schedule,
            n_steps=8192,        # Larger batch (was 4096)
            batch_size=512,
            n_epochs=10,
            gamma=1.0,
            clip_range=0.2,
            ent_coef=0.01,       # More exploration (was 0.005)
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=seed,
            verbose=0,
            policy_kwargs={"net_arch": [256, 128]},  # Smaller (Fourier does work)
        )

        cb = FidelityLogCallback(verbose=1)
        t0 = time.perf_counter()
        model.learn(total_timesteps=total_timesteps, callback=cb)
        wall_time = time.perf_counter() - t0

        env.close()

        final_fids = cb.fidelities[-200:] if len(cb.fidelities) >= 200 else cb.fidelities
        final_mean = float(np.mean(final_fids)) if final_fids else 0.0

        print(f"  Seed {seed}: {wall_time:.1f}s, final mean F = {final_mean:.4f}")

        model_path = models_dir / f"ppo_C_openloop_v2_seed{seed}"
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
        best_path = models_dir / "ppo_C_openloop_v2_best"
        best_model.save(str(best_path))
        print(f"\nBest open-loop v2 model: F = {best_fid:.4f}")

    # Save training logs
    log_path = results_dir / "training_logs_C_openloop_v2.json"
    with open(log_path, "w") as f:
        json.dump({
            "scenario": scenario,
            "total_timesteps": total_timesteps,
            "env_n_steps": env_n_steps,
            "n_seeds": n_seeds,
            "n_fourier": n_fourier,
            "n_envs": n_envs,
            "obs_mode": "time_only",
            "architecture": "Fourier features + MLP[256,128]",
            "seeds": [{
                "seed": s["seed"],
                "wall_time": s["wall_time"],
                "n_episodes": s["n_episodes"],
                "final_mean_fidelity": s["final_mean_fidelity"],
                "fidelities": s["fidelities"],
                "timesteps": s["timesteps"],
            } for s in all_logs],
        }, f)
    print(f"Training logs saved to {log_path}")

    return best_model


def evaluate(model=None):
    """Evaluate open-loop PPO v2 on 200 noisy trajectories."""
    scenario = "C"
    env_n_steps = 60
    n_fourier = 8
    n_traj = 200
    results_dir = ROOT / "results"

    if model is None:
        best_path = ROOT / "models" / "ppo_C_openloop_v2_best"
        model = PPO.load(str(best_path))
        print(f"Loaded open-loop v2 model from {best_path}")

    print(f"\n--- Open-loop PPO v2 evaluation ({n_traj} trajectories) ---")

    # Single env for evaluation
    env = RydbergBellEnv(
        scenario=scenario, n_steps=env_n_steps,
        use_noise=True, reward_shaping_alpha=0.0,
        obs_mode="time_only",
    )
    env = FourierFeatureWrapper(env, n_fourier=n_fourier)

    fids = []
    # Sanity check: verify actions are identical across seeds
    first_actions = None

    for i in range(n_traj):
        obs, _ = env.reset(seed=1000 + i)
        done = False
        actions_this = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions_this.append(action.copy())
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        fids.append(info.get("fidelity", 0.0))

        if first_actions is None:
            first_actions = actions_this
        else:
            if i == 1:
                match = all(
                    np.allclose(a1, a2) for a1, a2 in zip(first_actions, actions_this)
                )
                if match:
                    print("  [SANITY CHECK PASSED] Actions identical across noise seeds")
                else:
                    print("  [WARNING] Actions differ across noise seeds!")

        if (i + 1) % 50 == 0:
            print(f"  trajectory {i+1}/{n_traj}: mean_F = {np.mean(fids):.4f}")

    fids = np.array(fids)
    result = {
        "method": "ppo_openloop_v2_fourier",
        "scenario": scenario,
        "obs_mode": "time_only",
        "n_fourier": n_fourier,
        "architecture": "Fourier features + MLP[256,128]",
        "mean_F": float(fids.mean()),
        "std_F": float(fids.std()),
        "F_05": float(np.percentile(fids, 5)),
        "min_F": float(fids.min()),
        "max_F": float(fids.max()),
        "n_trajectories": n_traj,
    }

    out_path = results_dir / "ppo_C_openloop_v2.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Open-loop PPO v2: mean_F = {result['mean_F']:.4f} +/- {result['std_F']:.4f}")
    print(f"  F_05 = {result['F_05']:.4f}, min = {result['min_F']:.4f}, max = {result['max_F']:.4f}")
    print(f"  Results saved to {out_path}")

    # Compare to GRAPE baseline
    print(f"\n  GRAPE baseline: F = 0.803 +/- 0.163")
    gap = result['mean_F'] - 0.803
    if gap > 0.01:
        print(f"  >> SUCCESS: Open-loop PPO v2 beats GRAPE by {gap:+.4f}!")
    elif gap > -0.01:
        print(f"  >> CLOSE: Open-loop PPO v2 ~ GRAPE (gap = {gap:+.4f})")
    else:
        print(f"  >> GRAPE still wins by {-gap:.4f}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate existing model, don't train")
    args = parser.parse_args()

    if args.eval_only:
        evaluate()
    else:
        model = train()
        evaluate(model)
