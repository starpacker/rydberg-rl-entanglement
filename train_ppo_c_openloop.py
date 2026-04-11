"""Open-loop PPO training for Scenario C.

The policy observes ONLY t/T (1-dim), not rho(t).
This means the policy can only learn a time-dependent pulse shape:
    pi(a | t)   not   pi(a | rho(t), t)

Any performance gain over GRAPE must come from domain randomization
(DR) during training, NOT from mid-gate state feedback.

Hyperparameters match train_ppo_c_improved.py (v2 closed-loop):
- 3M timesteps, LR 3e-4 -> 5e-5, [512, 256] MLP
- 60 control steps, reward_shaping_alpha=0.15
- 3 seeds

After training, runs 200-trajectory MC evaluation under noise.
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
                if self.verbose >= 1 and len(self.fidelities) % 100 == 0:
                    recent = self.fidelities[-100:]
                    print(f"  [Step {self.num_timesteps}] Ep {len(self.fidelities)}: "
                          f"mean F = {np.mean(recent):.4f}, max F = {np.max(recent):.4f}")
        return True


def train():
    scenario = "C"
    total_timesteps = 3_000_000
    n_seeds = 3
    env_n_steps = 60

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
        print(f"Open-loop PPO on Scenario {scenario}, seed={seed}, "
              f"{total_timesteps/1e6:.0f}M steps, {env_n_steps} ctrl steps")
        print(f"obs_mode='time_only' -> policy sees only t/T")
        print(f"{'='*60}")

        env = RydbergBellEnv(
            scenario=scenario,
            n_steps=env_n_steps,
            use_noise=True,
            reward_shaping_alpha=0.15,
            obs_mode="time_only",
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

        model_path = models_dir / f"ppo_C_openloop_seed{seed}"
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
        best_path = models_dir / "ppo_C_openloop_best"
        best_model.save(str(best_path))
        print(f"\nBest open-loop model: F = {best_fid:.4f}")

    # Save training logs
    log_path = results_dir / "training_logs_C_openloop.json"
    with open(log_path, "w") as f:
        json.dump({
            "scenario": scenario,
            "total_timesteps": total_timesteps,
            "env_n_steps": env_n_steps,
            "n_seeds": n_seeds,
            "obs_mode": "time_only",
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
    """Evaluate open-loop PPO on 200 noisy trajectories."""
    scenario = "C"
    env_n_steps = 60
    n_traj = 200
    results_dir = ROOT / "results"

    if model is None:
        best_path = ROOT / "models" / "ppo_C_openloop_best"
        model = PPO.load(str(best_path))
        print(f"Loaded open-loop model from {best_path}")

    print(f"\n--- Open-loop PPO evaluation ({n_traj} trajectories) ---")
    env = RydbergBellEnv(
        scenario=scenario, n_steps=env_n_steps,
        use_noise=True, reward_shaping_alpha=0.0,
        obs_mode="time_only",
    )

    fids = []
    # Sanity check: verify actions are identical across seeds
    # (since obs = t/T is deterministic, policy output should be too)
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
            # Check first trajectory's actions match (open-loop sanity check)
            if i == 1:
                match = all(
                    np.allclose(a1, a2) for a1, a2 in zip(first_actions, actions_this)
                )
                if match:
                    print("  [SANITY CHECK PASSED] Actions identical across noise seeds (open-loop confirmed)")
                else:
                    print("  [WARNING] Actions differ across noise seeds! Policy may still use state info")

        if (i + 1) % 50 == 0:
            print(f"  trajectory {i+1}/{n_traj}: mean_F = {np.mean(fids):.4f}")

    fids = np.array(fids)
    result = {
        "method": "ppo_openloop",
        "scenario": scenario,
        "obs_mode": "time_only",
        "mean_F": float(fids.mean()),
        "std_F": float(fids.std()),
        "F_05": float(np.percentile(fids, 5)),
        "min_F": float(fids.min()),
        "max_F": float(fids.max()),
        "n_trajectories": n_traj,
    }

    out_path = results_dir / "ppo_C_openloop.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Open-loop PPO: mean_F = {result['mean_F']:.4f} +/- {result['std_F']:.4f}")
    print(f"  F_05 = {result['F_05']:.4f}, min = {result['min_F']:.4f}, max = {result['max_F']:.4f}")
    print(f"  Results saved to {out_path}")

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
