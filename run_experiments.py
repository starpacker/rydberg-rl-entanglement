"""Full experiment pipeline: Train PPO + Evaluate all methods on Scenarios A/B/C.

Usage:
    python3 run_experiments.py                 # Run all experiments
    python3 run_experiments.py --scenario B    # Single scenario
    python3 run_experiments.py --eval-only     # Skip training, evaluate existing models
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.environments.rydberg_env import RydbergBellEnv
from src.physics.constants import C6_53S, SCENARIOS
from src.physics.noise_model import NoiseModel
from src.baselines.stirap import run_stirap
from src.baselines.grape import run_grape, run_grape_eval
from src.baselines.evaluate import evaluate_policy


# ===================================================================
# Training callback
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
                          f"mean F = {np.mean(recent):.4f}, max = {np.max(recent):.4f}")
        return True


# ===================================================================
# PPO Training
# ===================================================================

def train_ppo_for_scenario(
    scenario: str,
    total_timesteps: int = 1_000_000,
    n_seeds: int = 2,
    env_n_steps: int = 30,
    models_dir: Path = ROOT / "models",
) -> Tuple[PPO, Dict[str, Any]]:
    """Train PPO on a given scenario with multiple seeds. Returns best model."""
    models_dir.mkdir(exist_ok=True)

    config = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 5,
        "gamma": 1.0,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "policy_kwargs": {"net_arch": [256, 256]},
    }

    best_model = None
    best_fid = -1.0
    all_logs = []

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 111
        print(f"\n{'='*60}")
        print(f"Training PPO on Scenario {scenario}, seed={seed}")
        print(f"{'='*60}")

        env = RydbergBellEnv(
            scenario=scenario,
            n_steps=env_n_steps,
            use_noise=True,
            reward_shaping_alpha=0.1,
        )

        model = PPO(
            "MlpPolicy", env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            seed=seed,
            verbose=0,
            policy_kwargs=config["policy_kwargs"],
        )

        cb = FidelityLogCallback(verbose=1)
        t0 = time.perf_counter()
        model.learn(total_timesteps=total_timesteps, callback=cb)
        wall_time = time.perf_counter() - t0

        # Evaluate this seed
        final_fids = cb.fidelities[-100:] if len(cb.fidelities) >= 100 else cb.fidelities
        final_mean = float(np.mean(final_fids)) if final_fids else 0.0

        print(f"  Seed {seed}: {wall_time:.1f}s, final mean F = {final_mean:.4f}")

        model_path = models_dir / f"ppo_{scenario}_seed{seed}"
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
        best_path = models_dir / f"ppo_{scenario}_best"
        best_model.save(str(best_path))
        print(f"\nBest model for scenario {scenario}: F = {best_fid:.4f}")

    training_log = {
        "scenario": scenario,
        "total_timesteps": total_timesteps,
        "n_seeds": n_seeds,
        "config": config,
        "seeds": all_logs,
    }
    return best_model, training_log


# ===================================================================
# Evaluation
# ===================================================================

def evaluate_ppo_model(
    model: PPO,
    scenario: str,
    n_traj: int = 200,
    env_n_steps: int = 30,
) -> Dict[str, Any]:
    """Evaluate a trained PPO model."""
    env = RydbergBellEnv(
        scenario=scenario, n_steps=env_n_steps,
        use_noise=True, reward_shaping_alpha=0.0,
    )

    fidelities = []
    for i in range(n_traj):
        obs, _ = env.reset(seed=1000 + i)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if "fidelity" in info:
            fidelities.append(info["fidelity"])

    arr = np.array(fidelities)
    return {
        "method": "ppo",
        "scenario": scenario,
        "mean_F": float(arr.mean()),
        "std_F": float(arr.std()),
        "F_05": float(np.percentile(arr, 5)),
        "n_trajectories": len(fidelities),
        "fidelities": fidelities,
    }


def evaluate_stirap(scenario: str, n_traj: int = 200) -> Dict[str, Any]:
    """Monte Carlo eval of STIRAP."""
    print(f"  Evaluating STIRAP on scenario {scenario}...")
    result = evaluate_policy(run_stirap, scenario, n_trajectories=n_traj)
    result["method"] = "stirap"
    result["scenario"] = scenario
    return result


def evaluate_grape_method(scenario: str, n_traj: int = 200) -> Dict[str, Any]:
    """Optimize GRAPE noiseless, then evaluate under noise."""
    print(f"  Optimizing GRAPE on scenario {scenario} (noiseless)...")
    fid_noiseless, omega, delta = run_grape(
        scenario, n_steps=30, n_iter=500, verbose=True
    )
    print(f"  GRAPE noiseless F = {fid_noiseless:.6f}")

    def grape_eval(sc, noise_params=None, **kw):
        return run_grape_eval(sc, omega, delta, noise_params)

    print(f"  Evaluating GRAPE under noise ({n_traj} trajectories)...")
    result = evaluate_policy(grape_eval, scenario, n_trajectories=n_traj)
    result["method"] = "grape"
    result["scenario"] = scenario
    result["noiseless_F"] = fid_noiseless
    result["omega_pulse"] = omega.tolist()
    result["delta_pulse"] = delta.tolist()
    return result


# ===================================================================
# Supplementary experiments (for figures)
# ===================================================================

def run_noise_channel_sweep(scenario: str = "B", n_traj: int = 200) -> Dict[str, Any]:
    """Run STIRAP with single noise channels enabled (for Fig 07)."""
    channels = ["doppler", "position", "amplitude", "phase", "decay"]
    results = {}

    rng_base = np.random.default_rng(42)
    nm = NoiseModel(scenario)

    for ch in channels + ["all"]:
        print(f"  Noise channel sweep: {ch}...")
        fidelities = []
        for i in range(n_traj):
            noise = nm.sample(rng_base)
            if ch != "all":
                # Zero out all channels except the target
                if ch != "doppler":
                    noise["delta_doppler"] = [0.0] * nm.n_atoms
                if ch != "position":
                    noise["delta_R"] = [0.0] * nm.n_atoms
                if ch != "amplitude":
                    noise["ou_sigma"] = 0.0
                if ch != "phase":
                    noise["phase_noise"] = 0.0
                # Control decay via include_decay flag
                noise["include_decay"] = (ch == "decay")
            else:
                noise["include_decay"] = True

            fid, _ = run_stirap(scenario, noise_params=noise)
            fidelities.append(fid)

        arr = np.array(fidelities)
        results[ch] = {
            "mean_F": float(arr.mean()),
            "std_F": float(arr.std()),
            "infidelity": float(1 - arr.mean()),
        }
        print(f"    {ch}: 1-F = {1-arr.mean():.6f}")

    # Also compute noiseless baseline
    fid_clean, _ = run_stirap(scenario, noise_params=None)
    results["noiseless"] = {"mean_F": float(fid_clean), "infidelity": float(1 - fid_clean)}

    return results


def run_gate_time_sweep(n_traj: int = 100) -> Dict[str, Any]:
    """Sweep T_gate for STIRAP and GRAPE (for Fig 08)."""
    T_gates = [0.1e-6, 0.2e-6, 0.3e-6, 0.5e-6, 1.0e-6, 2.0e-6, 5.0e-6]
    results = {"T_gates": [t * 1e6 for t in T_gates], "stirap": [], "grape": []}

    for T in T_gates:
        print(f"\n  T_gate = {T*1e6:.1f} us")
        # Create temporary scenario with modified T_gate
        import copy
        cfg_tmp = copy.deepcopy(SCENARIOS["B"])
        cfg_tmp["T_gate"] = T

        # Temporarily patch SCENARIOS
        SCENARIOS["_sweep"] = cfg_tmp

        try:
            # STIRAP
            stirap_fids = []
            nm = NoiseModel("_sweep")
            rng = np.random.default_rng(42)
            for i in range(n_traj):
                noise = nm.sample(rng)
                fid, _ = run_stirap("_sweep", noise_params=noise)
                stirap_fids.append(fid)
            stirap_mean = float(np.mean(stirap_fids))
            results["stirap"].append(stirap_mean)
            print(f"    STIRAP: F = {stirap_mean:.4f}")

            # GRAPE
            fid_noiseless, omega, delta = run_grape("_sweep", n_steps=30, n_iter=300, verbose=False)
            grape_fids = []
            for i in range(n_traj):
                noise = nm.sample(rng)
                fid = run_grape_eval("_sweep", omega, delta, noise)
                grape_fids.append(fid)
            grape_mean = float(np.mean(grape_fids))
            results["grape"].append(grape_mean)
            print(f"    GRAPE:  F = {grape_mean:.4f}")
        finally:
            del SCENARIOS["_sweep"]

    return results


def run_robustness_sweep(
    scenario: str = "B",
    ppo_model: Optional[PPO] = None,
    n_traj: int = 100,
) -> Dict[str, Any]:
    """Sweep systematic amplitude perturbation (for Fig 11)."""
    delta_pcts = [0, 1, 2, 3, 4, 5]
    results = {
        "delta_pct": delta_pcts,
        "stirap": [], "grape": [], "ppo": [],
    }

    # First, get GRAPE pulse for this scenario
    print("  Optimizing GRAPE pulse for robustness sweep...")
    _, omega_grape, delta_grape = run_grape(scenario, n_steps=30, n_iter=500, verbose=False)

    nm = NoiseModel(scenario)

    for dp in delta_pcts:
        print(f"\n  delta_Omega/Omega = {dp}%")
        bias = dp / 100.0

        # STIRAP
        stirap_fids = []
        rng = np.random.default_rng(42)
        for i in range(n_traj):
            noise = nm.sample(rng)
            noise["amplitude_bias"] = noise.get("amplitude_bias", 0.0) + bias
            # For STIRAP, amplitude bias modifies Omega_max
            # We need to patch this into the run_stirap call
            # Simplification: apply bias as additional OU mean
            fid, _ = run_stirap(scenario, noise_params=noise)
            stirap_fids.append(fid)
        results["stirap"].append(float(np.mean(stirap_fids)))
        print(f"    STIRAP: F = {np.mean(stirap_fids):.4f}")

        # GRAPE
        grape_fids = []
        rng = np.random.default_rng(42)
        for i in range(n_traj):
            noise = nm.sample(rng)
            noise["amplitude_bias"] = noise.get("amplitude_bias", 0.0) + bias
            fid = run_grape_eval(scenario, omega_grape, delta_grape, noise)
            grape_fids.append(fid)
        results["grape"].append(float(np.mean(grape_fids)))
        print(f"    GRAPE:  F = {np.mean(grape_fids):.4f}")

        # PPO
        if ppo_model is not None:
            env = RydbergBellEnv(
                scenario=scenario, n_steps=30,
                use_noise=True, reward_shaping_alpha=0.0,
            )
            ppo_fids = []
            for i in range(n_traj):
                obs, _ = env.reset(seed=2000 + i)
                # Inject additional bias
                env._amplitude_bias += bias
                done = False
                while not done:
                    action, _ = ppo_model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                ppo_fids.append(info.get("fidelity", 0.0))
            results["ppo"].append(float(np.mean(ppo_fids)))
            print(f"    PPO:    F = {np.mean(ppo_fids):.4f}")
        else:
            results["ppo"].append(None)

    return results


def extract_ppo_pulse(model: PPO, scenario: str = "B") -> Dict[str, Any]:
    """Extract real pulse from PPO model rollout (for Fig 13)."""
    env = RydbergBellEnv(scenario=scenario, n_steps=30, use_noise=False, reward_shaping_alpha=0.0)
    obs, _ = env.reset(seed=0)

    omegas, deltas = [], []
    for step in range(30):
        action, _ = model.predict(obs, deterministic=True)
        a = np.clip(action, -1, 1)
        Omega = float((a[0] + 1) / 2 * 2 * env.Omega_max)
        Delta = float(a[1] * env.Omega_max)
        omegas.append(Omega)
        deltas.append(Delta)
        obs, _, done, _, info = env.step(action)

    return {
        "omega": omegas,
        "delta": deltas,
        "final_fidelity": info.get("fidelity", 0.0),
        "scenario": scenario,
        "T_gate": env.T_gate,
        "Omega_max": env.Omega_max,
    }


def extract_ppo_populations(model: PPO, scenario: str = "B") -> Dict[str, Any]:
    """Extract population evolution from PPO rollout (for Fig 14)."""
    env = RydbergBellEnv(scenario=scenario, n_steps=30, use_noise=False, reward_shaping_alpha=0.0)
    obs, _ = env.reset(seed=0)

    # Target: |W> = (|gr> + |rg>)/sqrt(2)
    # Populations: P_gg, P_W, P_rr
    # P_gg = rho[0,0], P_rr = rho[3,3]
    # P_W = <W|rho|W> where |W> = (|01> + |10>)/sqrt(2)
    populations = {"P_gg": [], "P_W": [], "P_rr": [], "t": []}
    dt = env.dt

    # Record initial state
    rho = env._rho_np.copy()
    P_gg = rho[0, 0].real
    P_rr = rho[3, 3].real
    P_W = 0.5 * (rho[1, 1].real + rho[2, 2].real + rho[1, 2].real + rho[2, 1].real)
    populations["P_gg"].append(P_gg)
    populations["P_W"].append(P_W)
    populations["P_rr"].append(P_rr)
    populations["t"].append(0.0)

    for step in range(30):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)

        rho = env._rho_np.copy()
        P_gg = rho[0, 0].real
        P_rr = rho[3, 3].real
        P_W = 0.5 * (rho[1, 1].real + rho[2, 2].real + rho[1, 2].real + rho[2, 1].real)
        populations["P_gg"].append(P_gg)
        populations["P_W"].append(P_W)
        populations["P_rr"].append(P_rr)
        populations["t"].append((step + 1) * dt)

    populations["final_fidelity"] = info.get("fidelity", 0.0)
    populations["scenario"] = scenario
    return populations


# ===================================================================
# Main orchestrator
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default=None, help="Run single scenario")
    parser.add_argument("--eval-only", action="store_true", help="Skip training")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-traj", type=int, default=200)
    parser.add_argument("--skip-figures", action="store_true", help="Skip supplementary figure data")
    args = parser.parse_args()

    results_dir = ROOT / "results"
    models_dir = ROOT / "models"
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    scenarios = [args.scenario] if args.scenario else ["A", "B", "C"]

    all_results = {}
    ppo_models = {}

    for sc in scenarios:
        print(f"\n{'#'*60}")
        print(f"# SCENARIO {sc}: {SCENARIOS[sc]['description']}")
        print(f"{'#'*60}")

        # --- Train PPO ---
        if not args.eval_only:
            model, train_log = train_ppo_for_scenario(
                sc,
                total_timesteps=args.timesteps,
                n_seeds=2,
                models_dir=models_dir,
            )
            ppo_models[sc] = model

            # Save training logs
            log_path = results_dir / f"training_logs_{sc}.json"
            # Make JSON-serializable
            serializable_log = {
                "scenario": train_log["scenario"],
                "total_timesteps": train_log["total_timesteps"],
                "n_seeds": train_log["n_seeds"],
                "seeds": [{
                    "seed": s["seed"],
                    "wall_time": s["wall_time"],
                    "n_episodes": s["n_episodes"],
                    "final_mean_fidelity": s["final_mean_fidelity"],
                    "fidelities": s["fidelities"],
                    "timesteps": s["timesteps"],
                } for s in train_log["seeds"]],
            }
            with open(log_path, "w") as f:
                json.dump(serializable_log, f)
            print(f"  Training logs saved to {log_path}")
        else:
            # Load existing model
            best_path = models_dir / f"ppo_{sc}_best"
            if best_path.exists() or (best_path.parent / f"{best_path.name}.zip").exists():
                ppo_models[sc] = PPO.load(str(best_path))
                print(f"  Loaded model from {best_path}")
            else:
                print(f"  WARNING: No model found for scenario {sc}")
                ppo_models[sc] = None

        # --- Evaluate all methods ---
        print(f"\n--- Evaluating methods on Scenario {sc} ---")

        # STIRAP
        stirap_res = evaluate_stirap(sc, n_traj=args.n_traj)
        with open(results_dir / f"stirap_{sc}.json", "w") as f:
            json.dump({k: v for k, v in stirap_res.items() if k != "fidelities"}, f, indent=2)
        print(f"  STIRAP {sc}: F = {stirap_res['mean_F']:.4f} +/- {stirap_res['std_F']:.4f}")

        # GRAPE
        grape_res = evaluate_grape_method(sc, n_traj=args.n_traj)
        save_grape = {k: v for k, v in grape_res.items() if k != "fidelities"}
        with open(results_dir / f"grape_{sc}.json", "w") as f:
            json.dump(save_grape, f, indent=2)
        print(f"  GRAPE  {sc}: F = {grape_res['mean_F']:.4f} +/- {grape_res['std_F']:.4f}")

        # PPO
        if ppo_models.get(sc) is not None:
            ppo_res = evaluate_ppo_model(ppo_models[sc], sc, n_traj=args.n_traj)
            with open(results_dir / f"ppo_{sc}.json", "w") as f:
                json.dump({k: v for k, v in ppo_res.items() if k != "fidelities"}, f, indent=2)
            print(f"  PPO    {sc}: F = {ppo_res['mean_F']:.4f} +/- {ppo_res['std_F']:.4f}")
        else:
            ppo_res = None

        all_results[sc] = {
            "stirap": stirap_res,
            "grape": grape_res,
            "ppo": ppo_res,
        }

    # --- Print summary table ---
    print(f"\n{'='*70}")
    print(f"{'SUMMARY TABLE':^70}")
    print(f"{'='*70}")
    print(f"{'Scenario':<12} {'STIRAP F':>12} {'GRAPE F':>12} {'PPO F':>12}")
    print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for sc in scenarios:
        r = all_results[sc]
        stirap_f = f"{r['stirap']['mean_F']:.4f}" if r['stirap'] else "N/A"
        grape_f = f"{r['grape']['mean_F']:.4f}" if r['grape'] else "N/A"
        ppo_f = f"{r['ppo']['mean_F']:.4f}" if r.get('ppo') else "N/A"
        print(f"{sc:<12} {stirap_f:>12} {grape_f:>12} {ppo_f:>12}")

    # --- Supplementary data for figures ---
    if not args.skip_figures:
        print(f"\n{'='*60}")
        print("Generating supplementary figure data...")
        print(f"{'='*60}")

        # Fig 07: Noise channel sweep
        print("\n--- Fig 07: Noise channel sweep ---")
        noise_sweep = run_noise_channel_sweep("B", n_traj=200)
        with open(results_dir / "noise_channel_sweep_B.json", "w") as f:
            json.dump(noise_sweep, f, indent=2)

        # Fig 08: Gate time sweep
        print("\n--- Fig 08: Gate time sweep ---")
        gate_sweep = run_gate_time_sweep(n_traj=100)
        with open(results_dir / "gate_time_sweep.json", "w") as f:
            json.dump(gate_sweep, f, indent=2)

        # Fig 11: Robustness sweep
        print("\n--- Fig 11: Robustness sweep ---")
        ppo_b = ppo_models.get("B")
        robust_sweep = run_robustness_sweep("B", ppo_model=ppo_b, n_traj=100)
        with open(results_dir / "robustness_sweep_B.json", "w") as f:
            json.dump(robust_sweep, f, indent=2)

        # Fig 13: PPO pulse extraction
        if ppo_b is not None:
            print("\n--- Fig 13: PPO pulse extraction ---")
            pulse_data = extract_ppo_pulse(ppo_b, "B")
            with open(results_dir / "ppo_pulse_B.json", "w") as f:
                json.dump(pulse_data, f, indent=2)

            # Fig 14: Population evolution
            print("\n--- Fig 14: Population evolution ---")
            pop_data = extract_ppo_populations(ppo_b, "B")
            with open(results_dir / "ppo_populations_B.json", "w") as f:
                json.dump(pop_data, f, indent=2)

    print("\nAll experiments complete!")


if __name__ == "__main__":
    main()
