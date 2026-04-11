"""Quick PPO training for Scenario C — 1 seed, 1.5M steps, with checkpointing.

Same hyperparameters as train_ppo_c_improved.py but faster:
- 1 seed only (seed=42)
- 1.5M steps (enough for convergence based on learning curve)
- Saves checkpoint every 250k steps
- Runs evaluation immediately after training
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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


class CheckpointCallback(BaseCallback):
    def __init__(self, save_dir, save_freq=250_000, verbose=1):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_freq = save_freq
        self.fidelities = []
        self.timesteps_log = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "fidelity" in info:
                self.fidelities.append(info["fidelity"])
                self.timesteps_log.append(self.num_timesteps)
                if self.verbose >= 1 and len(self.fidelities) % 100 == 0:
                    recent = self.fidelities[-100:]
                    print(f"  [Step {self.num_timesteps}] Ep {len(self.fidelities)}: "
                          f"mean F = {np.mean(recent):.4f}, max F = {np.max(recent):.4f}")

        if self.num_timesteps % self.save_freq < self.locals.get("n_steps", 4096):
            path = self.save_dir / f"ppo_C_v3_ckpt_{self.num_timesteps}"
            self.model.save(str(path))
            if self.verbose >= 1:
                print(f"  >>> Checkpoint saved at step {self.num_timesteps}")

        return True


def train():
    scenario = "C"
    total_timesteps = 1_500_000
    env_n_steps = 60
    seed = 42

    models_dir = ROOT / "models"
    results_dir = ROOT / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    print(f"Training PPO on Scenario {scenario}, seed={seed}, "
          f"{total_timesteps/1e6:.1f}M steps, {env_n_steps} ctrl steps")

    env = RydbergBellEnv(
        scenario=scenario,
        n_steps=env_n_steps,
        use_noise=True,
        reward_shaping_alpha=0.15,
        obs_include_time=True,
    )

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

    cb = CheckpointCallback(models_dir, save_freq=250_000, verbose=1)
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=cb)
    wall_time = time.perf_counter() - t0

    final_fids = cb.fidelities[-200:] if len(cb.fidelities) >= 200 else cb.fidelities
    final_mean = float(np.mean(final_fids)) if final_fids else 0.0
    print(f"\nTraining done: {wall_time:.1f}s, final mean F = {final_mean:.4f}")

    model.save(str(models_dir / "ppo_C_v3_best"))

    # Save training log
    with open(results_dir / "training_logs_C_v3.json", "w") as f:
        json.dump({
            "seed": seed,
            "wall_time": wall_time,
            "n_episodes": len(cb.fidelities),
            "final_mean_fidelity": final_mean,
            "fidelities": cb.fidelities,
            "timesteps": cb.timesteps_log,
        }, f)

    return model


def evaluate_all(ppo_model=None):
    """Evaluate all 3 methods on Scenario C."""
    scenario = "C"
    n_traj = 200
    env_n_steps = 60
    results_dir = ROOT / "results"

    print(f"\n{'='*60}")
    print(f"Evaluating all methods on Scenario {scenario}")
    print(f"{'='*60}")

    # --- PPO ---
    if ppo_model is None:
        for name in ["ppo_C_v3_best", "ppo_C_v2_best"]:
            path = ROOT / "models" / name
            if path.exists() or (path.parent / f"{name}.zip").exists():
                ppo_model = PPO.load(str(path))
                print(f"Loaded PPO model from {path}")
                break
        if ppo_model is None:
            print("WARNING: No PPO model found")

    results = {}

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
            if (i + 1) % 50 == 0:
                print(f"  trajectory {i+1}/{n_traj}: mean_F = {np.mean(ppo_fids):.4f}")

        ppo_arr = np.array(ppo_fids)
        results["ppo"] = {
            "method": "ppo",
            "scenario": scenario,
            "mean_F": float(ppo_arr.mean()),
            "std_F": float(ppo_arr.std()),
            "F_05": float(np.percentile(ppo_arr, 5)),
            "n_trajectories": n_traj,
        }
        with open(results_dir / "ppo_C_v3.json", "w") as f:
            json.dump(results["ppo"], f, indent=2)
        print(f"  PPO: mean_F = {results['ppo']['mean_F']:.4f} +/- {results['ppo']['std_F']:.4f}, "
              f"F_05 = {results['ppo']['F_05']:.4f}")

        # Also extract pulse and population data for figures
        print("\n--- Extracting PPO pulse + populations ---")
        env2 = RydbergBellEnv(
            scenario=scenario, n_steps=env_n_steps,
            use_noise=False, reward_shaping_alpha=0.0,
            obs_include_time=True,
        )
        obs, _ = env2.reset(seed=0)
        pulse_omega, pulse_delta = [], []
        populations = {"gg": [], "gr": [], "rg": [], "rr": []}
        rho = env2._rho_np
        diag = np.diag(rho).real
        populations["gg"].append(float(diag[0]))
        populations["gr"].append(float(diag[1]))
        populations["rg"].append(float(diag[2]))
        populations["rr"].append(float(diag[3]))

        done = False
        while not done:
            action, _ = ppo_model.predict(obs, deterministic=True)
            a = np.clip(action, -1.0, 1.0)
            Omega_val = float((a[0] + 1.0) / 2.0 * 2.0 * env2.Omega_max)
            Delta_val = float(a[1] * env2.Omega_max)
            pulse_omega.append(Omega_val / (2 * np.pi * 1e6))  # MHz
            pulse_delta.append(Delta_val / (2 * np.pi * 1e6))  # MHz

            obs, _, terminated, truncated, info = env2.step(action)
            done = terminated or truncated

            rho = env2._rho_np
            diag = np.diag(rho).real
            populations["gg"].append(float(diag[0]))
            populations["gr"].append(float(diag[1]))
            populations["rg"].append(float(diag[2]))
            populations["rr"].append(float(diag[3]))

        T_gate = env2.T_gate
        t_pulse = [i * T_gate / env_n_steps * 1e6 for i in range(env_n_steps)]
        t_pop = [i * T_gate / env_n_steps * 1e6 for i in range(env_n_steps + 1)]

        with open(results_dir / "ppo_pulse_C.json", "w") as f:
            json.dump({"time_us": t_pulse, "omega_MHz": pulse_omega, "delta_MHz": pulse_delta}, f, indent=2)
        with open(results_dir / "ppo_populations_C.json", "w") as f:
            json.dump({"time_us": t_pop, **populations}, f, indent=2)
        print(f"  Noiseless PPO fidelity: {info.get('fidelity', 0.0):.6f}")

    # --- STIRAP ---
    print(f"\n--- STIRAP evaluation ({n_traj} trajectories) ---")
    stirap_res = evaluate_policy(run_stirap, scenario, n_trajectories=n_traj)
    stirap_res["method"] = "stirap"
    stirap_res["scenario"] = scenario
    results["stirap"] = stirap_res
    with open(results_dir / "stirap_C_v3.json", "w") as f:
        json.dump({k: v for k, v in stirap_res.items() if k != "fidelities"}, f, indent=2)
    print(f"  STIRAP: mean_F = {stirap_res['mean_F']:.4f} +/- {stirap_res['std_F']:.4f}, "
          f"F_05 = {stirap_res['F_05']:.4f}")

    # --- GRAPE ---
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
    results["grape"] = grape_res
    grape_save = {k: v for k, v in grape_res.items() if k != "fidelities"}
    grape_save["omega_pulse"] = omega_grape.tolist()
    grape_save["delta_pulse"] = delta_grape.tolist()
    with open(results_dir / "grape_C_v3.json", "w") as f:
        json.dump(grape_save, f, indent=2)
    print(f"  GRAPE: mean_F = {grape_res['mean_F']:.4f} +/- {grape_res['std_F']:.4f}, "
          f"F_05 = {grape_res['F_05']:.4f}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"{'SCENARIO C RESULTS':^60}")
    print(f"{'='*60}")
    print(f"{'Method':<12} {'mean_F':>10} {'std_F':>10} {'F_05':>10}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for m in ["ppo", "stirap", "grape"]:
        if m in results:
            r = results[m]
            print(f"{m.upper():<12} {r['mean_F']:>10.4f} {r['std_F']:>10.4f} {r['F_05']:>10.4f}")

    # --- Robustness sweep ---
    print(f"\n--- Robustness sweep ---")
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

    with open(results_dir / "robustness_sweep_C_v3.json", "w") as f:
        json.dump(rob_results, f, indent=2)

    print("\nAll evaluations complete!")


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
