"""Plot noise-scaling phase diagram from sweep results.

Produces publication-ready figures showing:
  1. Main figure: F vs noise_scale for GRAPE, CMA-ES+DR, PPO closed-loop
  2. Gap analysis: (CMA-ES - GRAPE) and (PPO - CMA-ES) vs noise_scale
  3. Regime classification with crossover annotation
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results" / "noise_scaling"
FIGURES_DIR = ROOT / "figures" / "noise_scaling"


def load_results() -> Dict:
    """Load all sweep results."""
    results = {}

    for method in ["grape", "cmaes", "ppo"]:
        path = RESULTS_DIR / f"{method}_sweep.json"
        if path.exists():
            with open(path) as f:
                results[method] = json.load(f)

    return results


def extract_curves(results: Dict) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Extract noise_scale and fidelity curves for each method.

    Returns
    -------
    noise_scales : array
        Common noise_scale values
    curves : dict
        {method: {'mean': array, 'std': array, 'F_05': array}}
    """
    # Get noise scales from GRAPE (should be same for all)
    if "grape" in results:
        noise_scales = np.array([l["noise_scale"] for l in results["grape"]["noise_levels"]])
    else:
        raise ValueError("GRAPE results required for noise_scale axis")

    curves = {}
    for method in ["grape", "cmaes", "ppo"]:
        if method not in results:
            continue

        levels = results[method]["noise_levels"]
        mean_F = np.array([l["mean_F"] for l in levels])
        std_F = np.array([l["std_F"] for l in levels])
        F_05 = np.array([l["F_05"] for l in levels])

        curves[method] = {
            "mean": mean_F,
            "std": std_F,
            "F_05": F_05,
        }

    return noise_scales, curves


def find_crossover(noise_scales: np.ndarray,
                   grape_F: np.ndarray,
                   cmaes_F: np.ndarray) -> float:
    """Find noise_scale where CMA-ES overtakes GRAPE."""
    # Linear interpolation to find crossover
    diff = cmaes_F - grape_F

    # Find first point where CMA-ES > GRAPE
    for i in range(len(diff) - 1):
        if diff[i] <= 0 and diff[i+1] > 0:
            # Linear interpolation
            x0, x1 = noise_scales[i], noise_scales[i+1]
            y0, y1 = diff[i], diff[i+1]
            crossover = x0 - y0 * (x1 - x0) / (y1 - y0)
            return crossover

    # If CMA-ES always better or always worse
    if diff[0] > 0:
        return noise_scales[0]  # CMA-ES better from start
    else:
        return noise_scales[-1]  # GRAPE better throughout


def plot_main_figure(noise_scales: np.ndarray, curves: Dict, save_path: Path):
    """Main figure: F vs noise_scale with all three methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "grape": "#1f77b4",      # blue
        "cmaes": "#ff7f0e",      # orange
        "ppo": "#2ca02c",        # green
    }
    labels = {
        "grape": "GRAPE (noiseless-optimized)",
        "cmaes": "CMA-ES + Domain Randomization",
        "ppo": "Closed-loop PPO (state feedback)",
    }

    # Plot curves
    for method in ["grape", "cmaes", "ppo"]:
        if method not in curves:
            continue

        mean = curves[method]["mean"]
        std = curves[method]["std"]

        ax.plot(noise_scales, mean, 'o-', color=colors[method],
                label=labels[method], linewidth=2, markersize=6)
        ax.fill_between(noise_scales, mean - std, mean + std,
                        color=colors[method], alpha=0.2)

    # Find and annotate crossover
    if "grape" in curves and "cmaes" in curves:
        crossover = find_crossover(noise_scales, curves["grape"]["mean"],
                                   curves["cmaes"]["mean"])
        ax.axvline(crossover, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(crossover + 0.1, 0.65, f'Crossover\n$\\alpha$ = {crossover:.2f}',
                fontsize=11, color='gray', ha='left')

    # Regime labels
    if "grape" in curves and "cmaes" in curves:
        # Low noise regime
        ax.text(0.7, 0.99, 'Analytical\nsufficient', fontsize=10,
                ha='center', va='top', color='gray', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='gray', alpha=0.7))

        # High noise regime
        ax.text(4.5, 0.99, 'DR essential', fontsize=10,
                ha='center', va='top', color='gray', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='gray', alpha=0.7))

    ax.set_xlabel('Noise amplification factor $\\alpha$', fontsize=13)
    ax.set_ylabel('Bell state fidelity $F$', fontsize=13)
    ax.set_title('Noise-Scaling Phase Diagram: Rydberg Bell State Preparation',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(noise_scales[0] - 0.2, noise_scales[-1] + 0.2)
    ax.set_ylim(0.55, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved main figure: {save_path}")
    plt.close()


def plot_gap_analysis(noise_scales: np.ndarray, curves: Dict, save_path: Path):
    """Gap analysis: advantage of DR and closed-loop feedback."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: CMA-ES - GRAPE (DR advantage)
    if "grape" in curves and "cmaes" in curves:
        gap_dr = curves["cmaes"]["mean"] - curves["grape"]["mean"]
        ax1.plot(noise_scales, gap_dr, 'o-', color='#d62728', linewidth=2, markersize=6)
        ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax1.fill_between(noise_scales, 0, gap_dr, where=(gap_dr > 0),
                        color='#d62728', alpha=0.2, label='CMA-ES advantage')
        ax1.fill_between(noise_scales, gap_dr, 0, where=(gap_dr < 0),
                        color='#1f77b4', alpha=0.2, label='GRAPE advantage')

        ax1.set_xlabel('Noise amplification factor $\\alpha$', fontsize=12)
        ax1.set_ylabel('$\\Delta F$ = CMA-ES $-$ GRAPE', fontsize=12)
        ax1.set_title('Domain Randomization Advantage', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

    # Right: PPO - CMA-ES (closed-loop advantage)
    if "cmaes" in curves and "ppo" in curves:
        gap_cl = curves["ppo"]["mean"] - curves["cmaes"]["mean"]
        ax2.plot(noise_scales, gap_cl, 'o-', color='#9467bd', linewidth=2, markersize=6)
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax2.fill_between(noise_scales, 0, gap_cl, where=(gap_cl > 0),
                        color='#9467bd', alpha=0.2, label='Closed-loop advantage')

        ax2.set_xlabel('Noise amplification factor $\\alpha$', fontsize=12)
        ax2.set_ylabel('$\\Delta F$ = PPO $-$ CMA-ES', fontsize=12)
        ax2.set_title('Closed-Loop Feedback Advantage', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved gap analysis: {save_path}")
    plt.close()


def print_summary_table(noise_scales: np.ndarray, curves: Dict):
    """Print summary table."""
    print(f"\n{'='*80}")
    print(f"{'NOISE-SCALING PHASE DIAGRAM SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'α':>6} | {'GRAPE':>12} | {'CMA-ES+DR':>12} | {'PPO(CL)':>12} | "
          f"{'ΔF(DR)':>10} | {'ΔF(CL)':>10}")
    print(f"{'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")

    for i, ns in enumerate(noise_scales):
        grape_f = f"{curves['grape']['mean'][i]:.4f}" if "grape" in curves else "---"
        cmaes_f = f"{curves['cmaes']['mean'][i]:.4f}" if "cmaes" in curves else "---"
        ppo_f = f"{curves['ppo']['mean'][i]:.4f}" if "ppo" in curves else "---"

        gap_dr = "---"
        if "grape" in curves and "cmaes" in curves:
            gap_dr = f"{curves['cmaes']['mean'][i] - curves['grape']['mean'][i]:+.4f}"

        gap_cl = "---"
        if "cmaes" in curves and "ppo" in curves:
            gap_cl = f"{curves['ppo']['mean'][i] - curves['cmaes']['mean'][i]:+.4f}"

        print(f"{ns:>6.1f} | {grape_f:>12} | {cmaes_f:>12} | {ppo_f:>12} | "
              f"{gap_dr:>10} | {gap_cl:>10}")

    # Crossover point
    if "grape" in curves and "cmaes" in curves:
        crossover = find_crossover(noise_scales, curves["grape"]["mean"],
                                   curves["cmaes"]["mean"])
        print(f"\nCrossover point (CMA-ES overtakes GRAPE): α = {crossover:.2f}")

    # Average gaps
    if "grape" in curves and "cmaes" in curves:
        avg_gap_dr = np.mean(curves["cmaes"]["mean"] - curves["grape"]["mean"])
        print(f"Average DR advantage: ΔF = {avg_gap_dr:+.4f}")

    if "cmaes" in curves and "ppo" in curves:
        avg_gap_cl = np.mean(curves["ppo"]["mean"] - curves["cmaes"]["mean"])
        print(f"Average closed-loop advantage: ΔF = {avg_gap_cl:+.4f}")


def main():
    print("Loading noise-scaling sweep results...")
    results = load_results()

    if not results:
        print("ERROR: No results found. Run experiments/noise_scaling_sweep.py first.")
        return

    print(f"Found results for: {list(results.keys())}")

    noise_scales, curves = extract_curves(results)

    # Create figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_main_figure(noise_scales, curves, FIGURES_DIR / "phase_diagram.png")
    plot_gap_analysis(noise_scales, curves, FIGURES_DIR / "gap_analysis.png")

    # Print summary
    print_summary_table(noise_scales, curves)

    print(f"\nFigures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
