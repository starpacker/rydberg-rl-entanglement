"""Evaluate NAAC on noise-scaling sweep and compare to baselines.

Evaluation protocol:
1. Load trained NAAC model
2. For each noise_scale in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
   - Run 200 test episodes
   - Compute mean F, std F, F_05
3. Compare to GRAPE, CMA-ES, PPO baselines
4. Generate comparison plots
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithms.naac import NAAC, numpy_to_torch_rho
from src.environments.rydberg_env_naac import RydbergBellEnvNAAC


# ===================================================================
# Evaluation
# ===================================================================

def evaluate_naac_at_noise_level(
    naac: NAAC,
    scenario: str,
    n_steps: int,
    k_calib: int,
    noise_scale: float,
    n_test: int = 200,
    device: torch.device = torch.device("cpu"),
    seed_offset: int = 50000,
) -> Dict:
    """Evaluate NAAC at a specific noise level.

    Parameters
    ----------
    naac : NAAC
        Trained model
    scenario : str
        Scenario key
    n_steps : int
        Total control steps
    k_calib : int
        Calibration steps
    noise_scale : float
        Noise amplification factor
    n_test : int
        Number of test episodes
    device : torch.device
        Device for inference
    seed_offset : int
        Seed offset for test episodes

    Returns
    -------
    result : dict
        Evaluation statistics
    """
    naac.eval()

    env = RydbergBellEnvNAAC(
        scenario=scenario,
        n_steps=n_steps,
        use_noise=True,
        noise_scale=noise_scale,
        record_trajectory=True,
    )

    fidelities = []
    noise_errors = []

    n_adaptive = n_steps - k_calib

    for i in range(n_test):
        # Reset
        obs, _ = env.reset(seed=seed_offset + i)

        # Phase 1: Calibration
        calib_actions = naac.get_calibration_pulse(batch_size=1)  # (1, k_calib, 2)
        calib_actions_np = calib_actions.squeeze(0).detach().cpu().numpy()  # (k_calib, 2)

        for step in range(k_calib):
            action = calib_actions_np[step]
            obs, _, _, _, _ = env.step(action)

        # Get calibration trajectory
        rho_calib_np = env.get_trajectory()[:k_calib, :, :]  # (k_calib, 4, 4)
        rho_calib = numpy_to_torch_rho(rho_calib_np).unsqueeze(0).to(device)  # (1, k_calib, 4, 4, 2)

        # Estimate noise
        with torch.no_grad():
            noise_est = naac.estimate_noise(rho_calib)  # (1, 6)

        # Ground-truth noise
        noise_true = env.get_noise_vector()  # (6,)
        noise_error = np.abs(noise_est.cpu().numpy()[0] - noise_true).mean()
        noise_errors.append(noise_error)

        # Phase 2: Adaptive execution
        for step in range(n_adaptive):
            # Get current ρ
            rho_current_np = env._rho_np
            rho_current = numpy_to_torch_rho(rho_current_np).unsqueeze(0).to(device)  # (1, 4, 4, 2)

            # Generate action
            t = torch.tensor([step / n_adaptive], device=device)
            with torch.no_grad():
                action_torch = naac.generate_action(t, noise_est, rho_current)  # (1, 2)
            action = action_torch.cpu().numpy()[0]

            # Step
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        fidelities.append(info.get("fidelity", 0.0))

    fidelities = np.array(fidelities)
    noise_errors = np.array(noise_errors)

    result = {
        "noise_scale": noise_scale,
        "mean_F": float(fidelities.mean()),
        "std_F": float(fidelities.std()),
        "F_05": float(np.percentile(fidelities, 5)),
        "min_F": float(fidelities.min()),
        "max_F": float(fidelities.max()),
        "mean_noise_error": float(noise_errors.mean()),
        "std_noise_error": float(noise_errors.std()),
    }

    return result


def run_naac_sweep(
    model_path: str,
    scenario: str = "C",
    n_steps: int = 60,
    k_calib: int = 10,
    noise_levels: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    n_test: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """Run NAAC evaluation across noise levels.

    Parameters
    ----------
    model_path : str
        Path to trained NAAC checkpoint
    scenario : str
        Scenario key
    n_steps : int
        Total control steps
    k_calib : int
        Calibration steps
    noise_levels : list
        Noise scales to evaluate
    n_test : int
        Test episodes per level
    device : str
        Device for inference

    Returns
    -------
    results : dict
        Full evaluation results
    """
    device = torch.device(device)
    print(f"Evaluating on device: {device}")

    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    naac = NAAC(
        k_calib=k_calib,
        n_fourier=5,
        estimator_hidden=[256, 128],
        generator_hidden=[128, 64],
        n_noise_params=6,
    ).to(device)

    naac.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded (trained for {checkpoint['episode']} episodes)")

    # Evaluate at each noise level
    results = {
        "method": "naac",
        "model_path": model_path,
        "k_calib": k_calib,
        "n_steps": n_steps,
        "noise_levels": [],
    }

    print(f"\n{'='*70}")
    print(f"{'NAAC Noise-Scaling Evaluation':^70}")
    print(f"{'='*70}")

    for noise_scale in noise_levels:
        print(f"\n--- Evaluating at noise_scale={noise_scale:.1f} ({n_test} episodes) ---")
        t0 = time.time()

        result = evaluate_naac_at_noise_level(
            naac, scenario, n_steps, k_calib, noise_scale, n_test, device
        )

        elapsed = time.time() - t0
        result["wall_time"] = elapsed

        results["noise_levels"].append(result)

        print(f"  F = {result['mean_F']:.4f} ± {result['std_F']:.4f}")
        print(f"  F_05 = {result['F_05']:.4f}, min = {result['min_F']:.4f}")
        print(f"  Noise est error = {result['mean_noise_error']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

    return results


# ===================================================================
# Comparison & Plotting
# ===================================================================

def load_baseline_results() -> Dict:
    """Load baseline results from noise_scaling sweep."""
    results_dir = ROOT / "results" / "noise_scaling"

    baselines = {}

    # GRAPE
    grape_path = results_dir / "grape_sweep.json"
    if grape_path.exists():
        with open(grape_path) as f:
            baselines["grape"] = json.load(f)

    # CMA-ES
    cmaes_path = results_dir / "cmaes_sweep.json"
    if cmaes_path.exists():
        with open(cmaes_path) as f:
            baselines["cmaes"] = json.load(f)
    else:
        # Try partial
        cmaes_partial = results_dir / "cmaes_sweep_partial.json"
        if cmaes_partial.exists():
            with open(cmaes_partial) as f:
                baselines["cmaes"] = json.load(f)

    # PPO
    ppo_path = results_dir / "ppo_sweep.json"
    if ppo_path.exists():
        with open(ppo_path) as f:
            baselines["ppo"] = json.load(f)
    else:
        # Try partial
        ppo_partial = results_dir / "ppo_sweep_partial.json"
        if ppo_partial.exists():
            with open(ppo_partial) as f:
                baselines["ppo"] = json.load(f)

    return baselines


def plot_comparison(naac_results: Dict, baselines: Dict, save_path: Path):
    """Plot NAAC vs baselines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Extract data
    noise_scales = np.array([l["noise_scale"] for l in naac_results["noise_levels"]])
    naac_F = np.array([l["mean_F"] for l in naac_results["noise_levels"]])
    naac_std = np.array([l["std_F"] for l in naac_results["noise_levels"]])

    colors = {
        "grape": "#1f77b4",
        "cmaes": "#ff7f0e",
        "ppo": "#2ca02c",
        "naac": "#d62728",
    }

    labels = {
        "grape": "GRAPE (noiseless-optimized)",
        "cmaes": "CMA-ES + Domain Randomization",
        "ppo": "PPO Closed-loop",
        "naac": "NAAC (Ours)",
    }

    # Left: Main comparison
    for method, data in baselines.items():
        if "noise_levels" in data:
            ns = np.array([l["noise_scale"] for l in data["noise_levels"]])
            mean_F = np.array([l["mean_F"] for l in data["noise_levels"]])
            std_F = np.array([l["std_F"] for l in data["noise_levels"]])

            ax1.plot(ns, mean_F, 'o-', color=colors[method], label=labels[method],
                    linewidth=2, markersize=6, alpha=0.8)
            ax1.fill_between(ns, mean_F - std_F, mean_F + std_F,
                            color=colors[method], alpha=0.15)

    # NAAC (highlight)
    ax1.plot(noise_scales, naac_F, 'o-', color=colors["naac"], label=labels["naac"],
            linewidth=3, markersize=8)
    ax1.fill_between(noise_scales, naac_F - naac_std, naac_F + naac_std,
                    color=colors["naac"], alpha=0.2)

    ax1.set_xlabel('Noise amplification factor α', fontsize=13)
    ax1.set_ylabel('Bell state fidelity F', fontsize=13)
    ax1.set_title('NAAC vs Baselines: Noise-Scaling Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.55, 1.0)

    # Right: Advantage over CMA-ES
    if "cmaes" in baselines and "noise_levels" in baselines["cmaes"]:
        cmaes_ns = np.array([l["noise_scale"] for l in baselines["cmaes"]["noise_levels"]])
        cmaes_F = np.array([l["mean_F"] for l in baselines["cmaes"]["noise_levels"]])

        # Interpolate to match noise scales
        naac_advantage = []
        for ns in noise_scales:
            idx = np.where(cmaes_ns == ns)[0]
            if len(idx) > 0:
                gap = naac_F[noise_scales == ns][0] - cmaes_F[idx[0]]
                naac_advantage.append(gap)
            else:
                naac_advantage.append(np.nan)

        naac_advantage = np.array(naac_advantage)

        ax2.plot(noise_scales, naac_advantage * 100, 'o-', color=colors["naac"],
                linewidth=3, markersize=8)
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.fill_between(noise_scales, 0, naac_advantage * 100,
                        where=(naac_advantage > 0), color=colors["naac"], alpha=0.3,
                        label='NAAC advantage')
        ax2.fill_between(noise_scales, naac_advantage * 100, 0,
                        where=(naac_advantage < 0), color='gray', alpha=0.3,
                        label='CMA-ES advantage')

        ax2.set_xlabel('Noise amplification factor α', fontsize=13)
        ax2.set_ylabel('ΔF = NAAC - CMA-ES (%)', fontsize=13)
        ax2.set_title('NAAC Advantage over CMA-ES+DR', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved: {save_path}")
    plt.close()


def print_comparison_table(naac_results: Dict, baselines: Dict):
    """Print comparison table."""
    print(f"\n{'='*90}")
    print(f"{'NAAC vs BASELINES COMPARISON':^90}")
    print(f"{'='*90}")
    print(f"{'α':>6} | {'GRAPE':>10} | {'CMA-ES':>10} | {'PPO':>10} | {'NAAC':>10} | {'Δ(CMA)':>10} | {'Δ(PPO)':>10}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for level in naac_results["noise_levels"]:
        ns = level["noise_scale"]
        naac_f = level["mean_F"]

        # Find baseline values
        grape_f = cmaes_f = ppo_f = None

        if "grape" in baselines:
            for l in baselines["grape"]["noise_levels"]:
                if abs(l["noise_scale"] - ns) < 0.01:
                    grape_f = l["mean_F"]

        if "cmaes" in baselines:
            for l in baselines["cmaes"]["noise_levels"]:
                if abs(l["noise_scale"] - ns) < 0.01:
                    cmaes_f = l["mean_F"]

        if "ppo" in baselines:
            for l in baselines["ppo"]["noise_levels"]:
                if abs(l["noise_scale"] - ns) < 0.01:
                    ppo_f = l["mean_F"]

        grape_str = f"{grape_f:.4f}" if grape_f is not None else "---"
        cmaes_str = f"{cmaes_f:.4f}" if cmaes_f is not None else "---"
        ppo_str = f"{ppo_f:.4f}" if ppo_f is not None else "---"
        naac_str = f"{naac_f:.4f}"

        delta_cmaes = f"{(naac_f - cmaes_f)*100:+.2f}%" if cmaes_f is not None else "---"
        delta_ppo = f"{(naac_f - ppo_f)*100:+.2f}%" if ppo_f is not None else "---"

        print(f"{ns:>6.1f} | {grape_str:>10} | {cmaes_str:>10} | {ppo_str:>10} | "
              f"{naac_str:>10} | {delta_cmaes:>10} | {delta_ppo:>10}")

    print(f"{'='*90}\n")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NAAC on noise sweep")
    parser.add_argument("--model", required=True, help="Path to NAAC checkpoint")
    parser.add_argument("--scenario", default="C", help="Scenario key")
    parser.add_argument("--n-steps", type=int, default=60, help="Total control steps")
    parser.add_argument("--k-calib", type=int, default=10, help="Calibration steps")
    parser.add_argument("--n-test", type=int, default=200, help="Test episodes per level")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Run evaluation
    naac_results = run_naac_sweep(
        model_path=args.model,
        scenario=args.scenario,
        n_steps=args.n_steps,
        k_calib=args.k_calib,
        n_test=args.n_test,
        device=args.device,
    )

    # Save results
    results_dir = ROOT / "results" / "naac"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "naac_sweep.json"
    with open(results_path, "w") as f:
        json.dump(naac_results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Load baselines
    baselines = load_baseline_results()
    print(f"\nLoaded baselines: {list(baselines.keys())}")

    # Comparison table
    print_comparison_table(naac_results, baselines)

    # Plot
    figures_dir = ROOT / "figures" / "naac"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_path = figures_dir / "naac_vs_baselines.png"
    plot_comparison(naac_results, baselines, plot_path)

    print("\nEvaluation complete!")
