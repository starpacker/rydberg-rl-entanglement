"""PPO training pipeline for Rydberg Bell state preparation.

Usage
-----
    python -m src.training.train_ppo          # from project root
    python src/training/train_ppo.py          # also works (adds project root to path)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.environments.rydberg_env import RydbergBellEnv
from src.training.config import PPO_CONFIG


# ===================================================================
# Callback: log terminal fidelities
# ===================================================================

class FidelityLogCallback(BaseCallback):
    """Log episode terminal fidelities during PPO training."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.fidelities: List[float] = []
        self.timesteps_log: List[int] = []

    def _on_step(self) -> bool:
        # Check for episode end in infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if "fidelity" in info:
                self.fidelities.append(info["fidelity"])
                self.timesteps_log.append(self.num_timesteps)
                if self.verbose >= 1 and len(self.fidelities) % 50 == 0:
                    recent = self.fidelities[-50:]
                    print(
                        f"  [Step {self.num_timesteps}] "
                        f"Ep {len(self.fidelities)}: "
                        f"mean F = {np.mean(recent):.4f}, "
                        f"max F = {np.max(recent):.4f}"
                    )
        return True


# ===================================================================
# Training
# ===================================================================

def train_single_seed(
    seed: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Train PPO for one seed. Returns dict with model, callback, timing."""
    print(f"\n{'='*60}")
    print(f"Training seed={seed}")
    print(f"{'='*60}")

    # Create environment
    env = RydbergBellEnv(
        scenario=config["scenario"],
        n_steps=config["env_n_steps"],
        use_noise=True,
        reward_shaping_alpha=config.get("reward_shaping_alpha", 0.1),
    )

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        seed=seed,
        verbose=0,
        policy_kwargs=config.get("policy_kwargs", None),
    )

    callback = FidelityLogCallback(verbose=1)

    t0 = time.perf_counter()
    model.learn(total_timesteps=config["total_timesteps"], callback=callback)
    wall_time = time.perf_counter() - t0

    print(f"  Seed {seed} done in {wall_time:.1f}s")
    if callback.fidelities:
        last_n = min(50, len(callback.fidelities))
        print(
            f"  Last {last_n} episodes: "
            f"mean F = {np.mean(callback.fidelities[-last_n:]):.4f}"
        )

    return {
        "model": model,
        "callback": callback,
        "wall_time": wall_time,
        "seed": seed,
    }


# ===================================================================
# Evaluation
# ===================================================================

def evaluate_ppo(
    model: PPO,
    scenario: str = "B",
    n_traj: int = 200,
    use_noise: bool = True,
    env_n_steps: int = 30,
) -> Dict[str, Any]:
    """Evaluate a trained PPO model over n_traj episodes."""
    env = RydbergBellEnv(
        scenario=scenario,
        n_steps=env_n_steps,
        use_noise=use_noise,
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

    fidelities_arr = np.array(fidelities)
    results = {
        "n_traj": n_traj,
        "use_noise": use_noise,
        "scenario": scenario,
        "mean_fidelity": float(np.mean(fidelities_arr)),
        "std_fidelity": float(np.std(fidelities_arr)),
        "median_fidelity": float(np.median(fidelities_arr)),
        "max_fidelity": float(np.max(fidelities_arr)),
        "min_fidelity": float(np.min(fidelities_arr)),
        "fidelities": fidelities_arr.tolist(),
    }
    return results


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    config = PPO_CONFIG.copy()

    # Directories
    models_dir = Path(_PROJECT_ROOT) / "models"
    results_dir = Path(_PROJECT_ROOT) / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    n_seeds = config["n_seeds"]
    all_results: List[Dict[str, Any]] = []
    best_model = None
    best_final_fidelity = -1.0

    total_start = time.perf_counter()

    for i in range(n_seeds):
        seed = 42 + i * 111
        result = train_single_seed(seed, config)

        # Save model
        model_path = models_dir / f"ppo_{config['scenario']}_seed{seed}"
        result["model"].save(str(model_path))
        print(f"  Model saved to {model_path}")

        # Collect training log
        cb = result["callback"]
        fids = cb.fidelities
        final_fidelity = float(np.mean(fids[-50:])) if len(fids) >= 50 else (
            float(np.mean(fids)) if fids else 0.0
        )

        seed_log = {
            "seed": seed,
            "wall_time": result["wall_time"],
            "n_episodes": len(fids),
            "final_mean_fidelity": final_fidelity,
            "fidelities": fids,
            "timesteps": cb.timesteps_log,
        }
        all_results.append(seed_log)

        if final_fidelity > best_final_fidelity:
            best_final_fidelity = final_fidelity
            best_model = result["model"]

    total_wall = time.perf_counter() - total_start
    print(f"\nTotal training time: {total_wall:.1f}s")

    # Save training logs
    training_log = {
        "config": {k: v for k, v in config.items()},
        "total_wall_time": total_wall,
        "seeds": [
            {
                "seed": r["seed"],
                "wall_time": r["wall_time"],
                "n_episodes": r["n_episodes"],
                "final_mean_fidelity": r["final_mean_fidelity"],
                "fidelities": r["fidelities"],
                "timesteps": r["timesteps"],
            }
            for r in all_results
        ],
    }
    log_path = results_dir / "training_logs.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Training logs saved to {log_path}")

    # Evaluate best model
    if best_model is not None:
        print(f"\nEvaluating best model (final mean F = {best_final_fidelity:.4f})...")
        eval_results = evaluate_ppo(
            best_model,
            scenario=config["scenario"],
            n_traj=200,
            use_noise=True,
            env_n_steps=config["env_n_steps"],
        )
        eval_results["best_seed_final_fidelity"] = best_final_fidelity

        eval_path = results_dir / f"ppo_{config['scenario']}.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"Evaluation results saved to {eval_path}")
        print(
            f"  Eval: mean F = {eval_results['mean_fidelity']:.4f} "
            f"+/- {eval_results['std_fidelity']:.4f}, "
            f"median = {eval_results['median_fidelity']:.4f}, "
            f"max = {eval_results['max_fidelity']:.4f}"
        )

        # Also save best model with a clear name
        best_path = models_dir / f"ppo_{config['scenario']}_best"
        best_model.save(str(best_path))
        print(f"Best model saved to {best_path}")


if __name__ == "__main__":
    main()
