"""Open-loop pulse optimization via CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

Why CMA-ES instead of PPO for open-loop control:
1. Direct pulse parameterization: optimize θ = [Ω(t₀), Δ(t₀), ..., Ω(t₅₉), Δ(t₅₉)]
2. Fitness = E_noise[F(θ, noise)] — directly optimizes what we care about
3. Population-based search handles noise variance better than policy gradients
4. No need for value function or advantage estimation

CMA-ES is the gold standard for derivative-free optimization and has been
successfully used for quantum control (e.g., Doria et al. PRL 2011).

Pulse parameterization options:
- Raw: 120 params (Ω, Δ for each of 60 steps) — high-dim, flexible
- Fourier: ~20 params (Fourier coefficients) — low-dim, smooth
- Piecewise: ~10 params (control points + interpolation) — very low-dim

We'll use Fourier parameterization for smoothness and dimensionality reduction.
"""

import sys
import json
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
import cma  # pip install cma

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.environments.rydberg_env import RydbergBellEnv


class FourierPulseParameterization:
    """Parameterize pulse as Fourier series: Ω(t) = Σ aₖ sin(2πkt/T) + bₖ cos(2πkt/T)."""

    def __init__(self, n_steps: int, n_fourier: int = 5):
        """
        Parameters
        ----------
        n_steps : int
            Number of control steps
        n_fourier : int
            Number of Fourier components (k=0,1,...,n_fourier-1)
            Total params = 4 * n_fourier (sin/cos for Ω and Δ each)
        """
        self.n_steps = n_steps
        self.n_fourier = n_fourier
        self.n_params = 4 * n_fourier  # [a_Ω, b_Ω, a_Δ, b_Δ] for each k

        # Time points
        self.t = np.linspace(0, 1, n_steps, endpoint=False)  # t/T ∈ [0, 1)

        # Fourier basis: [sin(2πkt), cos(2πkt)] for k=0,1,...,K-1
        self.basis = np.zeros((n_steps, 2 * n_fourier))
        for k in range(n_fourier):
            self.basis[:, 2*k] = np.sin(2 * np.pi * k * self.t)
            self.basis[:, 2*k+1] = np.cos(2 * np.pi * k * self.t)

    def decode(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Fourier coefficients to pulse arrays.

        Parameters
        ----------
        params : array of shape (4*n_fourier,)
            [a_Ω₀, b_Ω₀, ..., a_Ωₖ, b_Ωₖ, a_Δ₀, b_Δ₀, ..., a_Δₖ, b_Δₖ]

        Returns
        -------
        omega_norm : array of shape (n_steps,)
            Normalized Ω ∈ [-1, 1]
        delta_norm : array of shape (n_steps,)
            Normalized Δ ∈ [-1, 1]
        """
        half = 2 * self.n_fourier
        omega_coeffs = params[:half]
        delta_coeffs = params[half:]

        omega_raw = self.basis @ omega_coeffs
        delta_raw = self.basis @ delta_coeffs

        # Clip to [-1, 1] (env expects normalized actions)
        omega_norm = np.clip(omega_raw, -1, 1)
        delta_norm = np.clip(delta_raw, -1, 1)

        return omega_norm, delta_norm


def evaluate_pulse(params: np.ndarray,
                   parameterization: FourierPulseParameterization,
                   scenario: str,
                   n_steps: int,
                   n_eval: int = 20,
                   seed_offset: int = 0) -> float:
    """Evaluate pulse fitness = mean fidelity over noise realizations.

    Parameters
    ----------
    params : array
        Pulse parameters (Fourier coefficients)
    parameterization : FourierPulseParameterization
        Decoder
    scenario : str
        Scenario key
    n_steps : int
        Number of control steps
    n_eval : int
        Number of noise samples for fitness evaluation
    seed_offset : int
        Seed offset for reproducibility

    Returns
    -------
    fitness : float
        Mean fidelity (to be maximized)
    """
    omega_norm, delta_norm = parameterization.decode(params)

    env = RydbergBellEnv(
        scenario=scenario,
        n_steps=n_steps,
        use_noise=True,
        reward_shaping_alpha=0.0,
        obs_mode="time_only",  # obs not used, but need to reset
    )

    fids = []
    for i in range(n_eval):
        env.reset(seed=seed_offset + i)
        for step_idx in range(n_steps):
            action = np.array([omega_norm[step_idx], delta_norm[step_idx]], dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        fids.append(info.get("fidelity", 0.0))

    return float(np.mean(fids))


def optimize_cmaes(scenario: str = "C",
                   n_steps: int = 60,
                   n_fourier: int = 5,
                   n_eval_per_candidate: int = 20,
                   max_generations: int = 300,
                   popsize: int = 20,
                   seed: int = 42) -> dict:
    """Run CMA-ES to optimize open-loop pulse.

    Parameters
    ----------
    scenario : str
        Scenario key
    n_steps : int
        Number of control steps
    n_fourier : int
        Number of Fourier components (params = 4*n_fourier)
    n_eval_per_candidate : int
        Noise samples per fitness evaluation
    max_generations : int
        Maximum CMA-ES generations
    popsize : int
        Population size per generation
    seed : int
        Random seed

    Returns
    -------
    result : dict
        Optimization results
    """
    param = FourierPulseParameterization(n_steps, n_fourier)
    n_params = param.n_params

    print(f"\n{'='*60}")
    print(f"CMA-ES open-loop pulse optimization")
    print(f"  Scenario: {scenario}, n_steps: {n_steps}")
    print(f"  Fourier components: {n_fourier} -> {n_params} params")
    print(f"  Population size: {popsize}, max generations: {max_generations}")
    print(f"  Fitness evals per candidate: {n_eval_per_candidate}")
    print(f"{'='*60}\n")

    # Initialize CMA-ES
    x0 = np.zeros(n_params)  # Start from zero pulse
    sigma0 = 0.5  # Initial step size

    opts = {
        'seed': seed,
        'popsize': popsize,
        'maxiter': max_generations,
        'verb_disp': 10,  # Print every 10 generations
        'verb_log': 0,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    generation = 0
    best_fitness = -np.inf
    best_params = None
    fitness_history = []

    t0 = time.perf_counter()

    while not es.stop():
        solutions = es.ask()

        # Evaluate population (can be parallelized)
        fitnesses = []
        for sol in solutions:
            # Use different seed offset per generation for diversity
            seed_offset = 10000 + generation * 1000
            fit = evaluate_pulse(sol, param, scenario, n_steps,
                                n_eval_per_candidate, seed_offset)
            fitnesses.append(fit)

        # CMA-ES minimizes, so negate fitness
        es.tell(solutions, [-f for f in fitnesses])

        # Track best
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]
        gen_mean_fit = np.mean(fitnesses)

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_params = solutions[gen_best_idx].copy()

        fitness_history.append({
            'generation': generation,
            'best_fitness': float(gen_best_fit),
            'mean_fitness': float(gen_mean_fit),
            'std_fitness': float(np.std(fitnesses)),
        })

        if generation % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  Gen {generation:3d}: best_F = {gen_best_fit:.4f}, "
                  f"mean_F = {gen_mean_fit:.4f} ± {np.std(fitnesses):.4f}  "
                  f"({elapsed:.0f}s)")

        generation += 1

    wall_time = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"CMA-ES optimization complete")
    print(f"  Best fitness: {best_fitness:.4f}")
    print(f"  Generations: {generation}")
    print(f"  Wall time: {wall_time:.1f}s")
    print(f"{'='*60}\n")

    # Decode best pulse
    omega_best, delta_best = param.decode(best_params)

    result = {
        'method': 'cmaes_fourier',
        'scenario': scenario,
        'n_steps': n_steps,
        'n_fourier': n_fourier,
        'n_params': n_params,
        'popsize': popsize,
        'max_generations': max_generations,
        'n_eval_per_candidate': n_eval_per_candidate,
        'best_fitness': float(best_fitness),
        'best_params': best_params.tolist(),
        'best_omega_norm': omega_best.tolist(),
        'best_delta_norm': delta_best.tolist(),
        'fitness_history': fitness_history,
        'wall_time': wall_time,
        'seed': seed,
    }

    return result


def evaluate_final(result: dict, n_test: int = 200) -> dict:
    """Evaluate optimized pulse on test set.

    Parameters
    ----------
    result : dict
        CMA-ES optimization result
    n_test : int
        Number of test trajectories

    Returns
    -------
    eval_result : dict
        Test evaluation results
    """
    scenario = result['scenario']
    n_steps = result['n_steps']
    omega_norm = np.array(result['best_omega_norm'])
    delta_norm = np.array(result['best_delta_norm'])

    print(f"\n--- Final evaluation on {n_test} test trajectories ---")

    env = RydbergBellEnv(
        scenario=scenario, n_steps=n_steps,
        use_noise=True, reward_shaping_alpha=0.0,
        obs_mode="time_only",
    )

    fids = []
    for i in range(n_test):
        env.reset(seed=50000 + i)  # Different seed range from training
        for step_idx in range(n_steps):
            action = np.array([omega_norm[step_idx], delta_norm[step_idx]], dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        fids.append(info.get("fidelity", 0.0))

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_test}: mean_F = {np.mean(fids):.4f}")

    fids = np.array(fids)

    eval_result = {
        'mean_F': float(fids.mean()),
        'std_F': float(fids.std()),
        'F_05': float(np.percentile(fids, 5)),
        'min_F': float(fids.min()),
        'max_F': float(fids.max()),
        'n_test': n_test,
        'fidelities': fids.tolist(),
    }

    print(f"\n  CMA-ES pulse: mean_F = {eval_result['mean_F']:.4f} +/- {eval_result['std_F']:.4f}")
    print(f"  F_05 = {eval_result['F_05']:.4f}, min = {eval_result['min_F']:.4f}")

    print(f"\n  GRAPE baseline: F = 0.803 +/- 0.163")
    gap = eval_result['mean_F'] - 0.803
    if gap > 0.01:
        print(f"  >> SUCCESS: CMA-ES beats GRAPE by {gap:+.4f}!")
    elif gap > -0.01:
        print(f"  >> CLOSE: CMA-ES ~ GRAPE (gap = {gap:+.4f})")
    else:
        print(f"  >> GRAPE still wins by {-gap:.4f}")

    return eval_result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="C", help="Scenario key")
    parser.add_argument("--n-steps", type=int, default=60, help="Control steps")
    parser.add_argument("--n-fourier", type=int, default=5, help="Fourier components")
    parser.add_argument("--n-eval", type=int, default=20, help="Noise samples per fitness eval")
    parser.add_argument("--max-gen", type=int, default=300, help="Max CMA-ES generations")
    parser.add_argument("--popsize", type=int, default=20, help="Population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Optimize
    result = optimize_cmaes(
        scenario=args.scenario,
        n_steps=args.n_steps,
        n_fourier=args.n_fourier,
        n_eval_per_candidate=args.n_eval,
        max_generations=args.max_gen,
        popsize=args.popsize,
        seed=args.seed,
    )

    # Save optimization result
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    opt_path = results_dir / f"cmaes_openloop_{args.scenario}.json"
    with open(opt_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Optimization result saved to {opt_path}")

    # Final evaluation
    eval_result = evaluate_final(result, n_test=200)
    result['test_evaluation'] = eval_result

    # Save combined result
    with open(opt_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Final result saved to {opt_path}")
