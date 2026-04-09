"""Unified evaluation interface for control baselines.

Provides Monte Carlo noise evaluation and JSON result saving
for STIRAP, GRAPE, and (later) PPO policies.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

# Allow running as a script
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.physics.constants import SCENARIOS
from src.physics.noise_model import NoiseModel


# ===================================================================
# Core evaluation
# ===================================================================

def evaluate_policy(
    run_func: Callable,
    scenario: str,
    n_trajectories: int = 200,
    seed: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run Monte Carlo evaluation under sampled noise.

    Parameters
    ----------
    run_func : callable
        run_func(scenario, noise_params=..., **kwargs) -> (fidelity, ...)
        Must return fidelity as the first element of its return value.
    scenario : str
        Scenario key ("A", "B", "D").
    n_trajectories : int
        Number of noise realisations.
    seed : int
        RNG seed for reproducibility.
    **kwargs
        Extra arguments forwarded to run_func.

    Returns
    -------
    dict with keys:
        mean_F   : float -- mean fidelity
        F_05     : float -- 5th percentile fidelity
        std_F    : float -- standard deviation
        fidelities : list[float] -- all fidelity values
    """
    rng = np.random.default_rng(seed)
    nm = NoiseModel(scenario)

    fidelities = []
    t0 = time.time()

    for i in range(n_trajectories):
        noise = nm.sample(rng)
        result = run_func(scenario, noise_params=noise, **kwargs)
        # run_func may return (fidelity, ...) or just fidelity
        if isinstance(result, tuple):
            fid = float(result[0])
        else:
            fid = float(result)
        fidelities.append(fid)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  trajectory {i+1}/{n_trajectories}  "
                  f"mean_F = {np.mean(fidelities):.4f}  "
                  f"({elapsed:.1f}s)")

    fidelities_arr = np.array(fidelities)
    return {
        "mean_F": float(np.mean(fidelities_arr)),
        "F_05": float(np.percentile(fidelities_arr, 5)),
        "std_F": float(np.std(fidelities_arr)),
        "fidelities": [float(f) for f in fidelities_arr],
    }


# ===================================================================
# Save results
# ===================================================================

def save_results(
    results: Dict[str, Any],
    method: str,
    scenario: str,
) -> Path:
    """Save evaluation results to results/{method}_{scenario}.json.

    Returns the path to the saved file.
    """
    results_dir = Path(_ROOT) / "results"
    results_dir.mkdir(exist_ok=True)

    # Don't save the full fidelity list in the summary (keep it lean)
    summary = {
        "method": method,
        "scenario": scenario,
        "mean_F": results["mean_F"],
        "F_05": results["F_05"],
        "std_F": results["std_F"],
        "n_trajectories": len(results["fidelities"]),
    }

    path = results_dir / f"{method}_{scenario}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {path}")
    return path


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    from src.baselines.stirap import run_stirap
    from src.baselines.grape import run_grape, run_grape_eval

    N_TRAJ = 100  # use 100 for speed during development

    # --- STIRAP on scenario A ---
    print("=" * 60)
    print("STIRAP  scenario A  (with noise)")
    print("=" * 60)
    res_stirap_A = evaluate_policy(run_stirap, "A", n_trajectories=N_TRAJ)
    save_results(res_stirap_A, "stirap", "A")
    print(f"  => mean_F = {res_stirap_A['mean_F']:.4f}, "
          f"F_05 = {res_stirap_A['F_05']:.4f}, "
          f"std = {res_stirap_A['std_F']:.4f}\n")

    # --- STIRAP on scenario B ---
    print("=" * 60)
    print("STIRAP  scenario B  (with noise)")
    print("=" * 60)
    res_stirap_B = evaluate_policy(run_stirap, "B", n_trajectories=N_TRAJ)
    save_results(res_stirap_B, "stirap", "B")
    print(f"  => mean_F = {res_stirap_B['mean_F']:.4f}, "
          f"F_05 = {res_stirap_B['F_05']:.4f}, "
          f"std = {res_stirap_B['std_F']:.4f}\n")

    # --- GRAPE on scenario B (optimise once, then evaluate) ---
    print("=" * 60)
    print("GRAPE  scenario B  (optimise noiseless, eval with noise)")
    print("=" * 60)
    print("Optimising GRAPE pulse (noiseless)...")
    fid_grape, omega_grape, delta_grape = run_grape(
        "B", n_steps=30, n_iter=500, noise_params=None, verbose=True,
    )
    print(f"  GRAPE noiseless fidelity: {fid_grape:.6f}\n")

    # Evaluate the optimised pulse under noise
    def grape_eval_B(scenario, noise_params=None, **kw):
        return run_grape_eval(scenario, omega_grape, delta_grape, noise_params)

    print("Evaluating GRAPE under noise...")
    res_grape_B = evaluate_policy(grape_eval_B, "B", n_trajectories=N_TRAJ)
    # Also record the noiseless fidelity
    res_grape_B["noiseless_F"] = fid_grape
    save_results(res_grape_B, "grape", "B")
    print(f"  => mean_F = {res_grape_B['mean_F']:.4f}, "
          f"F_05 = {res_grape_B['F_05']:.4f}, "
          f"std = {res_grape_B['std_F']:.4f}")
