#!/usr/bin/env python
"""Side-by-side evaluation: open-loop PPO vs GRAPE vs closed-loop PPO.

All three evaluated on the same 200 noise seeds for fair comparison.
Open-loop PPO and GRAPE are both fixed-pulse methods (no state feedback).
Closed-loop PPO (state feedback) is included as an upper bound reference.
"""
import sys, os, json, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from src.environments.rydberg_env import RydbergBellEnv
from src.baselines.grape import run_grape, run_grape_eval
from src.baselines.evaluate import evaluate_policy
from src.physics.noise_model import NoiseModel


def eval_ppo_closed_loop(model, scenario, env_n_steps, seeds):
    """Closed-loop PPO: policy sees rho(t) at each step."""
    env = RydbergBellEnv(
        scenario=scenario, n_steps=env_n_steps,
        use_noise=True, reward_shaping_alpha=0.0,
        obs_include_time=True, obs_mode="full",
    )
    fids = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        fids.append(info.get("fidelity", 0.0))
    return np.array(fids)


def eval_ppo_open_loop(model, scenario, env_n_steps, seeds):
    """Open-loop PPO: policy sees only t/T."""
    env = RydbergBellEnv(
        scenario=scenario, n_steps=env_n_steps,
        use_noise=True, reward_shaping_alpha=0.0,
        obs_mode="time_only",
    )
    fids = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        fids.append(info.get("fidelity", 0.0))
    return np.array(fids)


def eval_grape(scenario, env_n_steps, seeds, n_iter=1000):
    """GRAPE: noiseless-optimized fixed pulse evaluated under noise."""
    print("  Optimizing GRAPE (noiseless)...")
    fid_noiseless, omega, delta = run_grape(scenario, n_steps=env_n_steps, n_iter=n_iter)
    print(f"  GRAPE noiseless F = {fid_noiseless:.6f}")

    nm = NoiseModel(scenario)
    rng = np.random.default_rng(42)
    fids = []
    for seed in seeds:
        rng_seed = np.random.default_rng(seed)
        noise = nm.sample(rng_seed)
        fid = run_grape_eval(scenario, omega, delta, noise)
        fids.append(fid)
    return np.array(fids), fid_noiseless


def print_stats(name, fids, width=30):
    print(f"  {name:<{width}} mean={np.mean(fids):.4f} +/- {np.std(fids):.4f}  "
          f"F_05={np.percentile(fids, 5):.4f}  min={np.min(fids):.4f}")


def main():
    scenario = "C"
    env_n_steps = 60
    n_traj = 200
    seeds = [1000 + i for i in range(n_traj)]
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    results = {"scenario": scenario, "env_n_steps": env_n_steps, "n_traj": n_traj}

    # --- Open-loop PPO ---
    openloop_path = ROOT / "models" / "ppo_C_openloop_best"
    if openloop_path.exists() or (openloop_path.parent / f"{openloop_path.name}.zip").exists():
        print("\n--- Open-loop PPO ---")
        model_ol = PPO.load(str(openloop_path))
        t0 = time.perf_counter()
        fids_ol = eval_ppo_open_loop(model_ol, scenario, env_n_steps, seeds)
        t_ol = time.perf_counter() - t0
        print_stats("Open-loop PPO (obs=t/T)", fids_ol)
        print(f"    time: {t_ol:.1f}s")
        results["openloop_ppo"] = {
            "mean_F": float(fids_ol.mean()),
            "std_F": float(fids_ol.std()),
            "F_05": float(np.percentile(fids_ol, 5)),
            "fidelities": fids_ol.tolist(),
        }
    else:
        print(f"\n  Open-loop PPO model not found at {openloop_path}")
        print("  Run train_ppo_c_openloop.py first")
        fids_ol = None

    # --- GRAPE ---
    print("\n--- GRAPE ---")
    t0 = time.perf_counter()
    fids_grape, grape_noiseless = eval_grape(scenario, env_n_steps, seeds)
    t_grape = time.perf_counter() - t0
    print_stats("GRAPE (noiseless-optimized)", fids_grape)
    print(f"    time: {t_grape:.1f}s")
    results["grape"] = {
        "mean_F": float(fids_grape.mean()),
        "std_F": float(fids_grape.std()),
        "F_05": float(np.percentile(fids_grape, 5)),
        "noiseless_F": float(grape_noiseless),
        "fidelities": fids_grape.tolist(),
    }

    # --- Closed-loop PPO (reference) ---
    closed_path = ROOT / "models" / "ppo_C_v2_best"
    if closed_path.exists() or (closed_path.parent / f"{closed_path.name}.zip").exists():
        print("\n--- Closed-loop PPO (reference) ---")
        model_cl = PPO.load(str(closed_path))
        t0 = time.perf_counter()
        fids_cl = eval_ppo_closed_loop(model_cl, scenario, env_n_steps, seeds)
        t_cl = time.perf_counter() - t0
        print_stats("Closed-loop PPO (obs=rho+t)", fids_cl)
        print(f"    time: {t_cl:.1f}s")
        results["closed_loop_ppo"] = {
            "mean_F": float(fids_cl.mean()),
            "std_F": float(fids_cl.std()),
            "F_05": float(np.percentile(fids_cl, 5)),
            "fidelities": fids_cl.tolist(),
        }
    else:
        print(f"\n  Closed-loop PPO model not found at {closed_path}")
        fids_cl = None

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY: Scenario C (open-loop methods + reference)")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'mean_F':>8} {'std_F':>8} {'F_05':>8}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8}")

    if fids_ol is not None:
        print(f"{'Open-loop PPO (obs=t/T)':<30} {fids_ol.mean():>8.4f} {fids_ol.std():>8.4f} "
              f"{np.percentile(fids_ol, 5):>8.4f}")
    print(f"{'GRAPE (noiseless-opt)':<30} {fids_grape.mean():>8.4f} {fids_grape.std():>8.4f} "
          f"{np.percentile(fids_grape, 5):>8.4f}")
    if fids_cl is not None:
        print(f"{'Closed-loop PPO (reference)':<30} {fids_cl.mean():>8.4f} {fids_cl.std():>8.4f} "
              f"{np.percentile(fids_cl, 5):>8.4f}")

    if fids_ol is not None:
        gap = float(fids_ol.mean() - fids_grape.mean())
        print(f"\n  Open-loop PPO - GRAPE gap: {gap:+.4f}")
        if gap > 0.01:
            print(f"  >> DR training gives open-loop PPO a genuine advantage over GRAPE!")
        elif gap > -0.01:
            print(f"  >> Open-loop PPO ~ GRAPE (no significant difference)")
        else:
            print(f"  >> GRAPE outperforms open-loop PPO")

    # Save
    out_path = results_dir / "openloop_comparison_C.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
