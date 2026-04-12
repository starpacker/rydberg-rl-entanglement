"""Noise-scaling phase diagram: sweep noise_scale for GRAPE, CMA-ES+DR, closed-loop PPO.

This is the core contribution experiment. It answers:
  "At what noise level does GRAPE fail, and how do DR methods compare?"

Uses Scenario B as base physics (T_gate=0.3μs, all 5 noise sources).
noise_scale overrides the noise amplification factor.

Three methods evaluated at each noise_scale:
  1. GRAPE: noiseless-optimized pulse, evaluated under noise
  2. CMA-ES+DR: Fourier pulse optimized against noise at each level
  3. Closed-loop PPO: RL policy with obs=ρ(t)+t, trained at each noise level
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environments.rydberg_env import RydbergBellEnv
from src.physics.noise_model import NoiseModel
from src.baselines.grape import run_grape, run_grape_eval
from src.baselines.evaluate import evaluate_policy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENARIO = "C"  # Base physics (T_gate=1.0μs, all 5 noise sources)
N_STEPS = 60    # Control steps (matches Scenario C training)
N_TEST = 200    # Test trajectories per noise level
NOISE_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
# Note: Scenario C has noise_amplification=3.0 by default.
# noise_scale overrides this — so noise_scale=3.0 reproduces Scenario C,
# noise_scale=1.0 gives Scenario-B-level noise on C's longer gate time.

# CMA-ES config (lighter than Scenario C experiment for sweep efficiency)
CMAES_N_FOURIER = 5
CMAES_POPSIZE = 15
CMAES_MAX_GEN = 200
CMAES_N_EVAL = 20

# PPO config (lighter for sweep)
PPO_TOTAL_STEPS = 1_000_000
PPO_NET_ARCH = [256, 256]

RESULTS_DIR = ROOT / "results" / "noise_scaling"


# ---------------------------------------------------------------------------
# GRAPE: optimize once noiseless, evaluate at each noise level
# ---------------------------------------------------------------------------
def run_grape_sweep(noise_levels: List[float], n_test: int = N_TEST) -> Dict:
    """Evaluate noiseless-optimized GRAPE pulse across noise levels."""
    print(f"\n{'='*60}")
    print("GRAPE: noiseless optimization + noise sweep evaluation")
    print(f"{'='*60}")

    # Optimize GRAPE pulse (noiseless, once)
    print("Optimizing GRAPE pulse (noiseless)...")
    fid_noiseless, omega_grape, delta_grape = run_grape(
        SCENARIO, n_steps=N_STEPS, n_iter=500, verbose=True
    )
    print(f"  Noiseless fidelity: {fid_noiseless:.6f}")

    results = {
        "method": "grape",
        "noiseless_F": float(fid_noiseless),
        "omega_pulse": omega_grape.tolist(),
        "delta_pulse": delta_grape.tolist(),
        "noise_levels": [],
    }

    for ns in noise_levels:
        print(f"\n--- GRAPE eval at noise_scale={ns:.1f} ({n_test} trajectories) ---")
        t0 = time.perf_counter()

        # Create noise model with overridden scale
        nm = NoiseModel(SCENARIO, noise_scale=ns)
        rng = np.random.default_rng(42)

        fids = []
        for i in range(n_test):
            noise = nm.sample(rng)
            try:
                fid = run_grape_eval(SCENARIO, omega_grape, delta_grape, noise)
            except Exception:
                fid = 0.0
            fids.append(fid)

        fids = np.array(fids)
        wall_time = time.perf_counter() - t0

        level_result = {
            "noise_scale": ns,
            "mean_F": float(fids.mean()),
            "std_F": float(fids.std()),
            "F_05": float(np.percentile(fids, 5)),
            "min_F": float(fids.min()),
            "wall_time": wall_time,
        }
        results["noise_levels"].append(level_result)
        print(f"  noise_scale={ns:.1f}: F = {level_result['mean_F']:.4f} "
              f"± {level_result['std_F']:.4f}  ({wall_time:.0f}s)")

    return results


# ---------------------------------------------------------------------------
# CMA-ES: optimize at each noise level
# ---------------------------------------------------------------------------
def run_cmaes_sweep(noise_levels: List[float], n_test: int = N_TEST) -> Dict:
    """Run CMA-ES optimization at each noise level."""
    import cma

    from optimize_cmaes_openloop import FourierPulseParameterization

    print(f"\n{'='*60}")
    print("CMA-ES+DR: optimize at each noise level")
    print(f"{'='*60}")

    results = {
        "method": "cmaes_dr",
        "n_fourier": CMAES_N_FOURIER,
        "popsize": CMAES_POPSIZE,
        "max_gen": CMAES_MAX_GEN,
        "noise_levels": [],
    }

    param = FourierPulseParameterization(N_STEPS, CMAES_N_FOURIER)

    for ns in noise_levels:
        print(f"\n--- CMA-ES at noise_scale={ns:.1f} ---")
        t0 = time.perf_counter()

        # Fitness function for this noise level
        def evaluate_pulse_ns(params, seed_offset=0):
            omega_norm, delta_norm = param.decode(params)
            env = RydbergBellEnv(
                scenario=SCENARIO, n_steps=N_STEPS,
                use_noise=True, reward_shaping_alpha=0.0,
                obs_mode="time_only", noise_scale=ns,
            )
            fids = []
            for i in range(CMAES_N_EVAL):
                env.reset(seed=seed_offset + i)
                for step_idx in range(N_STEPS):
                    action = np.array([omega_norm[step_idx], delta_norm[step_idx]],
                                      dtype=np.float32)
                    _, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                fids.append(info.get("fidelity", 0.0))
            return float(np.mean(fids))

        # CMA-ES optimization
        x0 = np.zeros(param.n_params)
        opts = {
            'seed': 42,
            'popsize': CMAES_POPSIZE,
            'maxiter': CMAES_MAX_GEN,
            'verb_disp': 50,
            'verb_log': 0,
        }
        es = cma.CMAEvolutionStrategy(x0, 0.5, opts)

        gen = 0
        best_fitness = -np.inf
        best_params = None

        while not es.stop():
            solutions = es.ask()
            fitnesses = []
            for sol in solutions:
                seed_offset = 10000 + gen * 1000
                fit = evaluate_pulse_ns(sol, seed_offset)
                fitnesses.append(fit)
            es.tell(solutions, [-f for f in fitnesses])

            gen_best = max(fitnesses)
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_params = solutions[np.argmax(fitnesses)].copy()

            if gen % 50 == 0:
                print(f"  Gen {gen:3d}: best_F={gen_best:.4f}, "
                      f"mean_F={np.mean(fitnesses):.4f}")
            gen += 1

        opt_time = time.perf_counter() - t0
        print(f"  CMA-ES done: best_fitness={best_fitness:.4f} ({opt_time:.0f}s)")

        # Final evaluation
        omega_best, delta_best = param.decode(best_params)
        env_test = RydbergBellEnv(
            scenario=SCENARIO, n_steps=N_STEPS,
            use_noise=True, reward_shaping_alpha=0.0,
            obs_mode="time_only", noise_scale=ns,
        )
        test_fids = []
        for i in range(n_test):
            env_test.reset(seed=50000 + i)
            for step_idx in range(N_STEPS):
                action = np.array([omega_best[step_idx], delta_best[step_idx]],
                                  dtype=np.float32)
                _, _, terminated, truncated, info = env_test.step(action)
                if terminated or truncated:
                    break
            test_fids.append(info.get("fidelity", 0.0))

        test_fids = np.array(test_fids)
        total_time = time.perf_counter() - t0

        level_result = {
            "noise_scale": ns,
            "mean_F": float(test_fids.mean()),
            "std_F": float(test_fids.std()),
            "F_05": float(np.percentile(test_fids, 5)),
            "min_F": float(test_fids.min()),
            "best_train_fitness": float(best_fitness),
            "generations": gen,
            "wall_time": total_time,
            "best_params": best_params.tolist(),
        }
        results["noise_levels"].append(level_result)
        print(f"  noise_scale={ns:.1f}: test F = {level_result['mean_F']:.4f} "
              f"± {level_result['std_F']:.4f}  ({total_time:.0f}s)")

        # Save incremental results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "cmaes_sweep_partial.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Closed-loop PPO: train at each noise level
# ---------------------------------------------------------------------------
def run_ppo_sweep(noise_levels: List[float], n_test: int = N_TEST) -> Dict:
    """Train and evaluate closed-loop PPO at each noise level."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    print(f"\n{'='*60}")
    print("Closed-loop PPO: train at each noise level")
    print(f"{'='*60}")

    class QuietFidelityCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.fidelities = []
        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "fidelity" in info:
                    self.fidelities.append(info["fidelity"])
            return True

    results = {
        "method": "ppo_closed_loop",
        "total_timesteps": PPO_TOTAL_STEPS,
        "net_arch": PPO_NET_ARCH,
        "noise_levels": [],
    }

    for ns in noise_levels:
        print(f"\n--- PPO training at noise_scale={ns:.1f} ---")
        t0 = time.perf_counter()

        env = RydbergBellEnv(
            scenario=SCENARIO, n_steps=N_STEPS,
            use_noise=True, reward_shaping_alpha=0.15,
            obs_include_time=True, obs_mode="full",
            noise_scale=ns,
        )

        def lr_schedule(progress_remaining: float) -> float:
            return 5e-5 + (3e-4 - 5e-5) * progress_remaining

        model = PPO(
            "MlpPolicy", env,
            learning_rate=lr_schedule,
            n_steps=4096, batch_size=256, n_epochs=10,
            gamma=1.0, clip_range=0.2, ent_coef=0.005,
            vf_coef=0.5, max_grad_norm=0.5,
            seed=42, verbose=0,
            policy_kwargs={"net_arch": PPO_NET_ARCH},
        )

        cb = QuietFidelityCallback()
        model.learn(total_timesteps=PPO_TOTAL_STEPS, callback=cb)
        train_time = time.perf_counter() - t0

        # Training curve summary
        if cb.fidelities:
            final_fids = cb.fidelities[-200:] if len(cb.fidelities) >= 200 else cb.fidelities
            train_final_F = float(np.mean(final_fids))
        else:
            train_final_F = 0.0
        print(f"  Training done: {train_time:.0f}s, final mean F = {train_final_F:.4f}")

        # Evaluation
        env_test = RydbergBellEnv(
            scenario=SCENARIO, n_steps=N_STEPS,
            use_noise=True, reward_shaping_alpha=0.0,
            obs_include_time=True, obs_mode="full",
            noise_scale=ns,
        )
        test_fids = []
        for i in range(n_test):
            obs, _ = env_test.reset(seed=50000 + i)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env_test.step(action)
                done = terminated or truncated
            test_fids.append(info.get("fidelity", 0.0))

        test_fids = np.array(test_fids)
        total_time = time.perf_counter() - t0

        level_result = {
            "noise_scale": ns,
            "mean_F": float(test_fids.mean()),
            "std_F": float(test_fids.std()),
            "F_05": float(np.percentile(test_fids, 5)),
            "min_F": float(test_fids.min()),
            "train_final_F": train_final_F,
            "train_n_episodes": len(cb.fidelities),
            "wall_time": total_time,
        }
        results["noise_levels"].append(level_result)
        print(f"  noise_scale={ns:.1f}: test F = {level_result['mean_F']:.4f} "
              f"± {level_result['std_F']:.4f}  ({total_time:.0f}s)")

        # Save model
        model_dir = ROOT / "models" / "noise_scaling"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir / f"ppo_ns{ns:.1f}"))

        # Save incremental results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "ppo_sweep_partial.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Noise-scaling phase diagram experiment")
    parser.add_argument("--method", choices=["grape", "cmaes", "ppo", "all"],
                        default="all", help="Which method(s) to run")
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=NOISE_LEVELS, help="Noise scale values to sweep")
    parser.add_argument("--n-test", type=int, default=N_TEST,
                        help="Test trajectories per noise level")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    if args.method in ("grape", "all"):
        grape_results = run_grape_sweep(args.noise_levels, args.n_test)
        all_results["grape"] = grape_results
        with open(RESULTS_DIR / "grape_sweep.json", "w") as f:
            json.dump(grape_results, f, indent=2)

    if args.method in ("cmaes", "all"):
        cmaes_results = run_cmaes_sweep(args.noise_levels, args.n_test)
        all_results["cmaes"] = cmaes_results
        with open(RESULTS_DIR / "cmaes_sweep.json", "w") as f:
            json.dump(cmaes_results, f, indent=2)

    if args.method in ("ppo", "all"):
        ppo_results = run_ppo_sweep(args.noise_levels, args.n_test)
        all_results["ppo"] = ppo_results
        with open(RESULTS_DIR / "ppo_sweep.json", "w") as f:
            json.dump(ppo_results, f, indent=2)

    # Combined results
    with open(RESULTS_DIR / "noise_scaling_all.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'NOISE-SCALING PHASE DIAGRAM RESULTS':^70}")
    print(f"{'='*70}")
    print(f"{'noise_scale':>11} | {'GRAPE':>10} | {'CMA-ES+DR':>10} | {'PPO(CL)':>10}")
    print(f"{'-'*11}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for ns in args.noise_levels:
        grape_f = cmaes_f = ppo_f = "---"
        if "grape" in all_results:
            for l in all_results["grape"]["noise_levels"]:
                if abs(l["noise_scale"] - ns) < 0.01:
                    grape_f = f"{l['mean_F']:.4f}"
        if "cmaes" in all_results:
            for l in all_results["cmaes"]["noise_levels"]:
                if abs(l["noise_scale"] - ns) < 0.01:
                    cmaes_f = f"{l['mean_F']:.4f}"
        if "ppo" in all_results:
            for l in all_results["ppo"]["noise_levels"]:
                if abs(l["noise_scale"] - ns) < 0.01:
                    ppo_f = f"{l['mean_F']:.4f}"
        print(f"{ns:>11.1f} | {grape_f:>10} | {cmaes_f:>10} | {ppo_f:>10}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
