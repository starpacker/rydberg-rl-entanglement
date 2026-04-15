"""Fair Noise-Level-Conditioned Open-Loop PPO Training.

NO ORACLE ACCESS to noise realization. The policy only sees the noise
LEVEL (alpha), not the specific noise parameters. This is a fair
comparison to CMA-ES per-alpha, which also only knows the noise level.

The policy architecture:
    obs = [alpha_normalized] (1-dim)
    policy outputs: Fourier_correction (20-dim)
    actions = FourierDecode(base_params + scale * correction)

Key difference from train_ppo_openloop_real.py:
    - obs is 1-dim (alpha only), NOT 6-dim (noise vector)
    - Policy cannot adapt to individual noise realizations
    - Fair comparison: same info as CMA-ES per-alpha
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from src.environments.rydberg_env import RydbergBellEnv
from src.physics.differentiable_lindblad import FourierPulseDecoder


# =============================================================================
# Fourier-parameterized environment wrapper
# =============================================================================

class FourierOpenLoopEnv(gym.Env):
    """Single-step environment for fair noise-level-conditioned open-loop control.

    This is a bandit/one-step MDP:
    - Observation: [alpha_normalized] (1-dim) — noise LEVEL only, no oracle access
    - Action: 20-dim Fourier correction
    - Reward: fidelity after executing decoded pulse

    The full 60-step simulation happens inside a single env.step().
    Fair comparison: same info as CMA-ES per-alpha.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "C",
        n_steps: int = 60,
        noise_scale: float = None,
        noise_scale_range: Tuple[float, float] = (0.5, 5.0),
        base_params: np.ndarray = None,
        correction_scale: float = 0.2,
    ):
        super().__init__()

        self.scenario = scenario
        self.n_steps = n_steps
        self.noise_scale = noise_scale
        self.noise_scale_range = noise_scale_range
        self.correction_scale = correction_scale

        # Load base params (CMA-ES)
        if base_params is None:
            with open(ROOT / "results" / "noise_scaling" / "cmaes_sweep.json") as f:
                data = json.load(f)
            for entry in data["noise_levels"]:
                if entry["noise_scale"] == 1.0:
                    base_params = np.array(entry["best_params"], dtype=np.float32)
                    break
        self.base_params = base_params

        # Fourier decoder (numpy version for speed)
        # endpoint=False to match CMA-ES convention
        self.n_fourier = 5
        t = np.linspace(0, 1, n_steps, endpoint=False)
        self.basis = np.zeros((n_steps, 2 * self.n_fourier), dtype=np.float32)
        for k in range(self.n_fourier):
            self.basis[:, 2*k] = np.sin(2 * np.pi * k * t)
            self.basis[:, 2*k+1] = np.cos(2 * np.pi * k * t)

        # Observation: alpha_normalized only (1-dim) — NO oracle noise access
        # Normalized to [0, 1] from noise_scale_range
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Action: 20-dim Fourier correction
        self.action_space = spaces.Box(
            low=-2.0, high=2.0, shape=(20,), dtype=np.float32
        )

        self._env: Optional[RydbergBellEnv] = None
        self._alpha_obs: Optional[np.ndarray] = None
        self._current_noise_scale: float = 1.0
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample noise scale
        if self.noise_scale is not None:
            self._current_noise_scale = self.noise_scale
        else:
            low, high = self.noise_scale_range
            self._current_noise_scale = self._rng.uniform(low, high)

        # Create internal env
        self._env = RydbergBellEnv(
            scenario=self.scenario,
            n_steps=self.n_steps,
            use_noise=True,
            reward_shaping_alpha=0.0,
            obs_mode="noise_conditioned",
            noise_scale=self._current_noise_scale,
        )
        self._env.reset(seed=seed)

        # Observation: only the noise LEVEL (alpha), normalized to [0, 1]
        # Same info as CMA-ES per-alpha: knows which noise level, not the realization
        alpha_norm = (self._current_noise_scale - 0.5) / 4.5  # maps [0.5, 5.0] → [0, 1]
        self._alpha_obs = np.array([np.clip(alpha_norm, 0.0, 1.0)], dtype=np.float32)

        return self._alpha_obs.copy(), {}

    def step(self, action: np.ndarray):
        """Single step = full simulation with decoded Fourier pulse.

        The env was already reset() with noise sampled. We restore the initial
        state and run the full simulation with the given Fourier correction.
        """
        correction = np.clip(action, -2.0, 2.0)
        params = self.base_params + self.correction_scale * correction
        actions = self._decode_fourier(params)

        # Restore initial state (|gg>) without re-sampling noise
        from src.physics.hamiltonian import get_ground_state
        gg = get_ground_state(2)
        self._env._rho_np = (gg * gg.dag()).full()
        self._env._step_count = 0
        self._env._prev_fidelity = self._env._compute_fidelity_np(self._env._rho_np)

        # Run full simulation with same noise realization
        for t in range(self.n_steps):
            _, _, terminated, truncated, info = self._env.step(actions[t])
            if terminated or truncated:
                break

        fidelity = info.get("fidelity", 0.0)

        # Guard against NaN from numerical overflow in Lindblad propagation
        if not np.isfinite(fidelity):
            fidelity = 0.0

        # Single-step MDP: always done after one action
        return self._alpha_obs.copy(), fidelity, True, False, {"fidelity": fidelity}

    def _decode_fourier(self, params: np.ndarray) -> np.ndarray:
        """Decode 20-dim Fourier params to (n_steps, 2) actions."""
        half = 2 * self.n_fourier
        omega_coeffs = params[:half]
        delta_coeffs = params[half:]

        omega_raw = self.basis @ omega_coeffs
        delta_raw = self.basis @ delta_coeffs

        omega_norm = np.clip(omega_raw, -1, 1)
        delta_norm = np.clip(delta_raw, -1, 1)

        return np.stack([omega_norm, delta_norm], axis=-1).astype(np.float32)


# =============================================================================
# Curriculum callback
# =============================================================================

class CurriculumCallback(BaseCallback):
    """Gradually increase noise scale during training."""

    def __init__(
        self,
        initial_range: Tuple[float, float] = (0.3, 1.0),
        final_range: Tuple[float, float] = (0.5, 5.0),
        curriculum_steps: int = 1_000_000,
        update_every: int = 1024,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.initial_range = initial_range
        self.final_range = final_range
        self.curriculum_steps = curriculum_steps
        self.update_every = update_every
        self.fidelities: List[float] = []
        self.timesteps_log: List[int] = []
        self._last_update_step = -1

    def _on_step(self) -> bool:
        # Update noise range only periodically (IPC calls are expensive with SubprocVecEnv)
        if self.num_timesteps - self._last_update_step >= self.update_every:
            self._last_update_step = self.num_timesteps
            progress = min(1.0, self.num_timesteps / self.curriculum_steps)

            low = self.initial_range[0] + progress * (self.final_range[0] - self.initial_range[0])
            high = self.initial_range[1] + progress * (self.final_range[1] - self.initial_range[1])

            self.training_env.env_method("set_noise_range", (low, high))

        # Log fidelities - every step is a complete episode in single-step env
        for info in self.locals.get("infos", []):
            if "fidelity" in info:
                self.fidelities.append(info["fidelity"])
                self.timesteps_log.append(self.num_timesteps)

                if self.verbose >= 1 and len(self.fidelities) % 500 == 0:
                    recent = self.fidelities[-500:]
                    progress = min(1.0, self.num_timesteps / self.curriculum_steps)
                    low = self.initial_range[0] + progress * (self.final_range[0] - self.initial_range[0])
                    high = self.initial_range[1] + progress * (self.final_range[1] - self.initial_range[1])
                    print(f"  [Step {self.num_timesteps}] Ep {len(self.fidelities)}: "
                          f"mean F = {np.mean(recent):.4f}, noise_range = ({low:.2f}, {high:.2f})",
                          flush=True)

        return True


# Add method to env for curriculum
def set_noise_range(self, noise_range):
    self.noise_scale_range = noise_range

FourierOpenLoopEnv.set_noise_range = set_noise_range


# =============================================================================
# BC Warm-start
# =============================================================================

def load_bc_weights_into_ppo(ppo_model, bc_policy_path: Path, device):
    """Load BC policy weights as initialization for PPO policy.

    BC policy: 6 -> [128, 64] -> 20
    PPO policy: 7 -> [256, 128] -> 20 (via MlpPolicy)

    We need to adapt the architecture or retrain BC with matching arch.
    For now, we'll train a matching BC first.
    """
    # This is complex due to architecture mismatch
    # Skip for now, use simpler approach: train from scratch with curriculum
    pass


# =============================================================================
# Training
# =============================================================================

def make_env(scenario, n_steps, noise_scale_range, correction_scale, seed=None):
    """Factory for SubprocVecEnv."""
    def _init():
        env = FourierOpenLoopEnv(
            scenario=scenario,
            n_steps=n_steps,
            noise_scale_range=noise_scale_range,
            correction_scale=correction_scale,
        )
        return env
    return _init


def train_ppo_openloop(
    total_timesteps: int = 5_000_000,
    n_seeds: int = 3,
    n_envs: int = 8,
    use_curriculum: bool = True,
    verbose: int = 1,
):
    """Train real noise-conditioned open-loop PPO with parallel envs."""

    models_dir = ROOT / "models"
    results_dir = ROOT / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    best_model = None
    best_fid = -1.0
    all_logs = []

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 111
        print(f"\n{'='*70}")
        print(f"FAIR Noise-Level-Conditioned Open-Loop PPO, seed={seed}")
        print(f"  {total_timesteps/1e6:.1f}M steps, Fourier action space (20-dim), obs=alpha(1-dim)")
        print(f"  Curriculum: {use_curriculum}, n_envs: {n_envs}")
        print(f"{'='*70}", flush=True)

        # Create env with curriculum starting range
        if use_curriculum:
            initial_range = (0.3, 1.0)
        else:
            initial_range = (0.5, 5.0)

        # Parallel envs for speedup
        env = SubprocVecEnv([
            make_env("C", 60, initial_range, 0.2, seed + i)
            for i in range(n_envs)
        ])

        # PPO for contextual bandit (single-step MDP)
        # gamma=0 since there's no future reward (single step)
        def lr_schedule(progress_remaining: float) -> float:
            return 1e-4 + (3e-4 - 1e-4) * progress_remaining

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,
            n_steps=128,  # Episodes per env per rollout (single-step each)
            batch_size=256,
            n_epochs=10,
            gamma=0.0,  # Contextual bandit: no discounting needed
            gae_lambda=1.0,
            clip_range=0.2,
            ent_coef=0.02,  # Higher entropy for exploration in 20-dim space
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=seed,
            verbose=0,
            device="cpu",
            policy_kwargs={
                "net_arch": dict(pi=[128, 64], vf=[128, 64]),
            },
        )

        # Callbacks
        callbacks = []
        if use_curriculum:
            curriculum_cb = CurriculumCallback(
                initial_range=(0.3, 1.0),
                final_range=(0.5, 5.0),
                curriculum_steps=int(total_timesteps * 0.6),
                verbose=verbose,
            )
            callbacks.append(curriculum_cb)
        else:
            curriculum_cb = CurriculumCallback(
                initial_range=(0.5, 5.0),
                final_range=(0.5, 5.0),
                curriculum_steps=1,
                verbose=verbose,
            )
            callbacks.append(curriculum_cb)

        t0 = time.perf_counter()
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        wall_time = time.perf_counter() - t0

        env.close()

        # Extract fidelities from callback
        fids = curriculum_cb.fidelities
        final_fids = fids[-1000:] if len(fids) >= 1000 else fids
        final_mean = float(np.mean(final_fids)) if final_fids else 0.0

        print(f"\n  Seed {seed}: {wall_time:.1f}s, final mean F = {final_mean:.4f}")

        # Save model
        model_path = models_dir / f"ppo_openloop_fair_seed{seed}"
        model.save(str(model_path))

        seed_log = {
            "seed": seed,
            "wall_time": wall_time,
            "n_episodes": len(fids),
            "final_mean_fidelity": final_mean,
        }
        all_logs.append(seed_log)

        if final_mean > best_fid:
            best_fid = final_mean
            best_model = model

    # Save best
    if best_model is not None:
        best_path = models_dir / "ppo_openloop_fair_best"
        best_model.save(str(best_path))
        print(f"\nBest model: F = {best_fid:.4f}, saved to {best_path}")

    # Save logs
    log_path = results_dir / "ppo_openloop_fair_training.json"
    with open(log_path, "w") as f:
        json.dump({
            "total_timesteps": total_timesteps,
            "n_seeds": n_seeds,
            "use_curriculum": use_curriculum,
            "seeds": all_logs,
        }, f, indent=2)

    return best_model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_ppo_openloop(
    model=None,
    noise_levels: List[float] = None,
    n_traj: int = 200,
):
    """Evaluate PPO open-loop policy."""
    if noise_levels is None:
        noise_levels = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    if model is None:
        model_path = ROOT / "models" / "ppo_openloop_fair_best"
        model = PPO.load(str(model_path))
        print(f"Loaded model from {model_path}")

    print(f"\n{'='*60}")
    print("Evaluate FAIR Open-Loop PPO (obs=alpha only)")
    print("="*60)

    results = {}

    for alpha in noise_levels:
        print(f"\n  alpha = {alpha}:")

        env = FourierOpenLoopEnv(
            scenario="C",
            n_steps=60,
            noise_scale=alpha,
            correction_scale=0.2,
        )

        fids = []
        for i in range(n_traj):
            obs, _ = env.reset(seed=50000 + i)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            fids.append(info.get("fidelity", 0.0))

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{n_traj}: mean_F = {np.mean(fids):.4f}")

        fids = np.array(fids)
        results[alpha] = {
            "mean_F": float(fids.mean()),
            "std_F": float(fids.std()),
        }
        print(f"    => {fids.mean():.4f} +/- {fids.std():.4f}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--no-curriculum", action="store_true")
    args = parser.parse_args()

    model = None

    if args.train:
        model = train_ppo_openloop(
            total_timesteps=args.timesteps,
            n_seeds=args.n_seeds,
            n_envs=args.n_envs,
            use_curriculum=not args.no_curriculum,
        )

    if args.eval or not args.train:
        results = evaluate_ppo_openloop(model=model)

        out_path = ROOT / "results" / "ppo_openloop_fair_eval.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

        # Print comparison
        print("\n" + "="*70)
        print("COMPARISON: Fair PPO Open-Loop vs Baselines")
        print("="*70)

        # Load baselines
        try:
            with open(ROOT / "results" / "open_loop_comparison.json") as f:
                baselines = json.load(f)
        except FileNotFoundError:
            baselines = {}

        print(f"{'alpha':<8} {'CMA-ES':<12} {'BC-Oracle':<12} {'PPO-OpenLoop':<12} {'Gap(PPO-CMA)':<12}")
        print("-"*56)

        for alpha in sorted(results.keys()):
            ppo_f = results[alpha]["mean_F"]
            cmaes_f = baselines.get(str(alpha), {}).get("cmaes", 0)
            if cmaes_f == 0:
                cmaes_f = baselines.get(alpha, {}).get("cmaes", 0)
            bc_f = baselines.get(str(alpha), {}).get("bc_fourier_oracle", 0)
            if bc_f == 0:
                bc_f = baselines.get(alpha, {}).get("bc_fourier_oracle", 0)

            gap = ppo_f - cmaes_f if cmaes_f else 0
            print(f"{alpha:<8} {cmaes_f:<12.4f} {bc_f:<12.4f} {ppo_f:<12.4f} {gap:<+12.4f}")


if __name__ == "__main__":
    main()
