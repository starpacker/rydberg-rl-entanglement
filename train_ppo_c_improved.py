"""Improved PPO training for Scenario C with enhanced hyperparameters.

Key improvements over previous training:
1. 3M timesteps (was 1M) with linear LR annealing
2. Larger network [512, 256] (was [256, 256])
3. 60 control steps (was 30) for finer-grained control
4. n_steps=4096 for better advantage estimation
5. Reward shaping alpha=0.15 (was 0.1)
6. 3 seeds for robustness

After training, evaluates all methods (PPO, STIRAP, GRAPE) on Scenario C
with the same n_steps=60 for fair comparison.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.environments.rydberg_env import RydbergBellEnv
from src.physics.constants import SCENARIOS
from src.physics.noise_model import NoiseModel
from src.baselines.stirap import run_stirap
from src.baselines.grape import run_grape, run_grape_eval
from src.baselines.evaluate import evaluate_policy


# ===================================================================
# Callback
# ===================================================================

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


# ===================================================================
# Training
# ===================================================================

def train():
    scenario = "C"
    total_timesteps = 3_000_000
    n_seeds = 3
    env_n_steps = 60  # finer control (was 30)

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
        print(f"Training PPO on Scenario {scenario}, seed={seed}, "
              f"{total_timesteps/1e6:.0f}M steps, {env_n_steps} ctrl steps")
        print(f"{'='*60}")

        env = RydbergBellEnv(
            scenario=scenario,
            n_steps=env_n_steps,
            use_noise=True,
            reward_shaping_alpha=0.15,
            obs_include_time=True,
        )

        # Linear LR schedule: from 3e-4 → 5e-5
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

        model_path = models_dir / f"ppo_C_v2_seed{seed}"
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
        best_path = models_dir / "ppo_C_v2_best"
        best_model.save(str(best_path))
        print(f"\nBest model: F = {best_fid:.4f}")

    # Save training logs
    log_path = results_dir / "training_logs_C_v2.json"
    with open(log_path, "w") as f:
        json.dump({
            "scenario": scenario,
            "total_timesteps": total_timesteps,
            "env_n_steps": env_n_steps,
            "n_seeds": n_seeds,
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


# ===================================================================
# Evaluation
# ===================================================================

def evaluate_all(ppo_model=None):
    """Evaluate all 3 methods on Scenario C with 60 control steps."""
    scenario = "C"
    n_traj = 200
    env_n_steps = 60
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Evaluating all methods on Scenario {scenario} (n_steps={env_n_steps})")
    print(f"{'='*60}")

    # --- PPO ---
    if ppo_model is None:
        best_path = ROOT / "models" / "ppo_C_v2_best"
        if best_path.exists() or (best_path.parent / f"{best_path.name}.zip").exists():
            ppo_model = PPO.load(str(best_path))
            print(f"Loaded PPO model from {best_path}")
        else:
            print("WARNING: No PPO model found")

    if ppo_model is not None:
        print(f"\n--- PPO evaluation ({n_traj} trajectories) ---")
        env = RydbergBellEnv(
            scenario=scenario, n_steps=env_n_steps,
            use_noise=True, reward_shaping_alpha=0.0,
            obs_include_time=True,
        )
        ppo_fids = []
        for i in range(n_traj):
            obs, _ = env.reset(seed=1000 + i)
            done = False
            while not done:
                action, _ = ppo_model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            ppo_fids.append(info.get("fidelity", 0.0))
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  trajectory {i+1}/{n_traj}: mean_F = {np.mean(ppo_fids):.4f}")

        ppo_arr = np.array(ppo_fids)
        ppo_res = {
            "method": "ppo",
            "scenario": scenario,
            "mean_F": float(ppo_arr.mean()),
            "std_F": float(ppo_arr.std()),
            "F_05": float(np.percentile(ppo_arr, 5)),
            "n_trajectories": n_traj,
        }
        with open(results_dir / "ppo_C_v2.json", "w") as f:
            json.dump(ppo_res, f, indent=2)
        print(f"  PPO: mean_F = {ppo_res['mean_F']:.4f} +/- {ppo_res['std_F']:.4f}, "
              f"F_05 = {ppo_res['F_05']:.4f}")

    # --- STIRAP ---
    print(f"\n--- STIRAP evaluation ({n_traj} trajectories) ---")
    # STIRAP uses its own n_steps (200 by default for mesolve), not our env n_steps
    stirap_res = evaluate_policy(run_stirap, scenario, n_trajectories=n_traj)
    stirap_res["method"] = "stirap"
    stirap_res["scenario"] = scenario
    with open(results_dir / "stirap_C_v2.json", "w") as f:
        json.dump({k: v for k, v in stirap_res.items() if k != "fidelities"}, f, indent=2)
    print(f"  STIRAP: mean_F = {stirap_res['mean_F']:.4f} +/- {stirap_res['std_F']:.4f}, "
          f"F_05 = {stirap_res['F_05']:.4f}")

    # --- GRAPE (with 60 steps for fair comparison) ---
    print(f"\n--- GRAPE optimization + evaluation ({n_traj} trajectories) ---")
    print("  Optimizing GRAPE (noiseless, 60 steps, 1000 iterations)...")
    fid_noiseless, omega_grape, delta_grape = run_grape(
        scenario, n_steps=60, n_iter=1000, verbose=True
    )
    print(f"  GRAPE noiseless F = {fid_noiseless:.6f}")

    def grape_eval(sc, noise_params=None, **kw):
        return run_grape_eval(sc, omega_grape, delta_grape, noise_params)

    print(f"  Evaluating GRAPE under noise...")
    grape_res = evaluate_policy(grape_eval, scenario, n_trajectories=n_traj)
    grape_res["method"] = "grape"
    grape_res["scenario"] = scenario
    grape_res["noiseless_F"] = fid_noiseless
    grape_save = {k: v for k, v in grape_res.items() if k != "fidelities"}
    grape_save["omega_pulse"] = omega_grape.tolist()
    grape_save["delta_pulse"] = delta_grape.tolist()
    with open(results_dir / "grape_C_v2.json", "w") as f:
        json.dump(grape_save, f, indent=2)
    print(f"  GRAPE: mean_F = {grape_res['mean_F']:.4f} +/- {grape_res['std_F']:.4f}, "
          f"F_05 = {grape_res['F_05']:.4f}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"{'SCENARIO C RESULTS':^60}")
    print(f"{'='*60}")
    print(f"{'Method':<12} {'mean_F':>10} {'std_F':>10} {'F_05':>10}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    if ppo_model is not None:
        print(f"{'PPO':<12} {ppo_res['mean_F']:>10.4f} {ppo_res['std_F']:>10.4f} {ppo_res['F_05']:>10.4f}")
    print(f"{'STIRAP':<12} {stirap_res['mean_F']:>10.4f} {stirap_res['std_F']:>10.4f} {stirap_res['F_05']:>10.4f}")
    print(f"{'GRAPE':<12} {grape_res['mean_F']:>10.4f} {grape_res['std_F']:>10.4f} {grape_res['F_05']:>10.4f}")

    # --- Robustness sweep on C ---
    print(f"\n--- Robustness sweep on C ---")
    delta_pcts = [0, 1, 2, 3, 4, 5]
    rob_results = {"delta_pct": delta_pcts, "stirap": [], "grape": [], "ppo": []}
    nm = NoiseModel(scenario)
    n_rob = 100

    for dp in delta_pcts:
        bias = dp / 100.0
        print(f"  delta_Omega = {dp}%")

        # STIRAP
        rng = np.random.default_rng(42)
        stir_fids = []
        for i in range(n_rob):
            noise = nm.sample(rng)
            noise["amplitude_bias"] = noise.get("amplitude_bias", 0.0) + bias
            fid, _ = run_stirap(scenario, noise_params=noise)
            stir_fids.append(fid)
        rob_results["stirap"].append(float(np.mean(stir_fids)))

        # GRAPE
        rng = np.random.default_rng(42)
        gr_fids = []
        for i in range(n_rob):
            noise = nm.sample(rng)
            noise["amplitude_bias"] = noise.get("amplitude_bias", 0.0) + bias
            fid = run_grape_eval(scenario, omega_grape, delta_grape, noise)
            gr_fids.append(fid)
        rob_results["grape"].append(float(np.mean(gr_fids)))

        # PPO
        if ppo_model is not None:
            env = RydbergBellEnv(
                scenario=scenario, n_steps=env_n_steps,
                use_noise=True, reward_shaping_alpha=0.0,
                obs_include_time=True,
            )
            pp_fids = []
            for i in range(n_rob):
                obs, _ = env.reset(seed=2000 + i)
                env._amplitude_bias += bias
                done = False
                while not done:
                    action, _ = ppo_model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                pp_fids.append(info.get("fidelity", 0.0))
            rob_results["ppo"].append(float(np.mean(pp_fids)))
        else:
            rob_results["ppo"].append(None)

        print(f"    STIRAP={rob_results['stirap'][-1]:.4f}, "
              f"GRAPE={rob_results['grape'][-1]:.4f}, "
              f"PPO={rob_results['ppo'][-1]}")

    with open(ROOT / "results" / "robustness_sweep_C_v2.json", "w") as f:
        json.dump(rob_results, f, indent=2)

    print("\nAll evaluations complete!")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_all()
    else:
        model = train()
        evaluate_all(model)
