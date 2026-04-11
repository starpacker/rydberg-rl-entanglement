#!/usr/bin/env python
"""Experiment: Open-loop vs closed-loop PPO evaluation.

Quantifies how much of PPO's advantage comes from mid-gate state feedback
(observing rho(t)) vs. domain randomization training itself.

Three evaluation modes:
  1. Closed-loop (current): policy sees rho(t) at each step -> different actions per noise realization
  2. Open-loop (noiseless ref): record actions from noiseless rollout, replay fixed on all noisy trajectories
  3. Open-loop (avg ref): record actions from one noisy rollout, replay fixed on all noisy trajectories
"""
import sys, os, json, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from src.environments.rydberg_env import RydbergBellEnv


def extract_actions_closed_loop(model, env):
    """Run one episode with the model observing rho(t), return (actions, fidelity)."""
    obs, _ = env.reset()
    actions = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return actions, info.get("fidelity", 0.0)


def replay_fixed_actions(actions, env, seed):
    """Replay a fixed action sequence on a noisy environment."""
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(actions[step])
        step += 1
        done = terminated or truncated
    return info.get("fidelity", 0.0)


def evaluate_closed_loop(model, scenario, n_traj, env_n_steps):
    """Standard closed-loop evaluation (policy sees rho(t) each step)."""
    env = RydbergBellEnv(
        scenario=scenario,
        n_steps=env_n_steps,
        use_noise=True,
        reward_shaping_alpha=0.0,
        obs_include_time=True,
    )
    fids = []
    for i in range(n_traj):
        obs, _ = env.reset(seed=2000 + i)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        fids.append(info.get("fidelity", 0.0))
    return np.array(fids)


def evaluate_open_loop(actions_fixed, scenario, n_traj, env_n_steps):
    """Open-loop: replay fixed actions across noisy trajectories."""
    env = RydbergBellEnv(
        scenario=scenario,
        n_steps=env_n_steps,
        use_noise=True,
        reward_shaping_alpha=0.0,
        obs_include_time=True,  # doesn't matter, obs not used
    )
    fids = []
    for i in range(n_traj):
        fid = replay_fixed_actions(actions_fixed, env, seed=2000 + i)
        fids.append(fid)
    return np.array(fids)


def print_stats(name, fids):
    print(f"  {name}:")
    print(f"    mean F  = {np.mean(fids):.4f}")
    print(f"    std F   = {np.std(fids):.4f}")
    print(f"    F_05    = {np.percentile(fids, 5):.4f}")
    print(f"    min F   = {np.min(fids):.4f}")
    print(f"    max F   = {np.max(fids):.4f}")


def main():
    scenario = "C"
    env_n_steps = 60
    n_traj = 200

    model_path = ROOT / "models" / "ppo_C_v2_best"
    print(f"Loading model: {model_path}")
    model = PPO.load(str(model_path))

    # =================================================================
    # 1. Extract reference pulse from NOISELESS rollout
    # =================================================================
    print("\n--- Extracting reference pulse (noiseless) ---")
    env_noiseless = RydbergBellEnv(
        scenario=scenario,
        n_steps=env_n_steps,
        use_noise=False,
        reward_shaping_alpha=0.0,
        obs_include_time=True,
    )
    actions_noiseless, fid_noiseless = extract_actions_closed_loop(model, env_noiseless)
    print(f"  Noiseless fidelity: {fid_noiseless:.6f}")
    print(f"  Num actions: {len(actions_noiseless)}")

    # Print the pulse for reference
    Omega_max = env_noiseless.Omega_max
    print(f"\n  Noiseless pulse (Omega/2pi MHz, Delta/2pi MHz):")
    for k, a in enumerate(actions_noiseless):
        Omega = (a[0] + 1.0) / 2.0 * 2.0 * Omega_max
        Delta = a[1] * Omega_max
        print(f"    step {k:2d}: Omega/2pi = {Omega/(2*np.pi*1e6):6.3f} MHz, "
              f"Delta/2pi = {Delta/(2*np.pi*1e6):+7.3f} MHz")

    # =================================================================
    # 2. Closed-loop evaluation (standard)
    # =================================================================
    print(f"\n--- Closed-loop evaluation ({n_traj} trajectories) ---")
    t0 = time.perf_counter()
    fids_closed = evaluate_closed_loop(model, scenario, n_traj, env_n_steps)
    t_closed = time.perf_counter() - t0
    print_stats("Closed-loop (policy sees rho(t))", fids_closed)
    print(f"    time: {t_closed:.1f}s")

    # =================================================================
    # 3. Open-loop evaluation with noiseless reference pulse
    # =================================================================
    print(f"\n--- Open-loop evaluation: noiseless ref ({n_traj} trajectories) ---")
    t0 = time.perf_counter()
    fids_open_noiseless = evaluate_open_loop(actions_noiseless, scenario, n_traj, env_n_steps)
    t_open = time.perf_counter() - t0
    print_stats("Open-loop (noiseless reference pulse)", fids_open_noiseless)
    print(f"    time: {t_open:.1f}s")

    # =================================================================
    # 4. Open-loop with multiple reference pulses (average over a few)
    # =================================================================
    print(f"\n--- Open-loop evaluation: avg of 5 noisy ref pulses ---")
    # Extract actions from 5 different noisy rollouts, evaluate each
    env_noisy_ref = RydbergBellEnv(
        scenario=scenario,
        n_steps=env_n_steps,
        use_noise=True,
        reward_shaping_alpha=0.0,
        obs_include_time=True,
    )
    fids_multi_open = []
    for ref_idx in range(5):
        env_noisy_ref.reset(seed=9000 + ref_idx)
        # Re-create to get fresh noise
        actions_ref, fid_ref = extract_actions_closed_loop(model, env_noisy_ref)
        fids_this_ref = evaluate_open_loop(actions_ref, scenario, n_traj, env_n_steps)
        fids_multi_open.append(fids_this_ref)
        print(f"  Ref pulse {ref_idx} (F_ref={fid_ref:.4f}): "
              f"mean open-loop F = {np.mean(fids_this_ref):.4f} ± {np.std(fids_this_ref):.4f}")

    fids_multi_open_all = np.concatenate(fids_multi_open)
    print_stats("Open-loop (avg over 5 ref pulses)", fids_multi_open_all)

    # =================================================================
    # 5. Summary and gap analysis
    # =================================================================
    print(f"\n{'='*60}")
    print(f"SUMMARY: Open-loop vs Closed-loop on Scenario C")
    print(f"{'='*60}")
    print(f"  Noiseless F (single trajectory):     {fid_noiseless:.4f}")
    print(f"  Closed-loop <F> (state feedback):    {np.mean(fids_closed):.4f} ± {np.std(fids_closed):.4f}")
    print(f"  Open-loop <F> (noiseless ref):       {np.mean(fids_open_noiseless):.4f} ± {np.std(fids_open_noiseless):.4f}")
    print(f"  Open-loop <F> (5-ref average):       {np.mean(fids_multi_open_all):.4f} ± {np.std(fids_multi_open_all):.4f}")

    gap = np.mean(fids_closed) - np.mean(fids_open_noiseless)
    print(f"\n  Closed - Open gap (feedback advantage): {gap:+.4f}")
    print(f"  This gap = how much performance comes from observing rho(t)")

    if abs(gap) < 0.005:
        print(f"\n  >> Gap is SMALL (<0.5%): PPO advantage is mostly from DR training, not state feedback.")
        print(f"  >> Fair to compare against ensemble GRAPE.")
    elif gap > 0.05:
        print(f"\n  >> Gap is LARGE (>{gap*100:.0f}%): PPO advantage is mostly from state feedback (closed-loop).")
        print(f"  >> Comparison with ensemble GRAPE would be UNFAIR.")
    else:
        print(f"\n  >> Gap is MODERATE: both DR training and state feedback contribute.")

    # Save results
    results = {
        "scenario": scenario,
        "n_traj": n_traj,
        "env_n_steps": env_n_steps,
        "noiseless_F": float(fid_noiseless),
        "closed_loop": {
            "mean_F": float(np.mean(fids_closed)),
            "std_F": float(np.std(fids_closed)),
            "F_05": float(np.percentile(fids_closed, 5)),
            "fidelities": fids_closed.tolist(),
        },
        "open_loop_noiseless_ref": {
            "mean_F": float(np.mean(fids_open_noiseless)),
            "std_F": float(np.std(fids_open_noiseless)),
            "F_05": float(np.percentile(fids_open_noiseless, 5)),
            "fidelities": fids_open_noiseless.tolist(),
        },
        "open_loop_multi_ref": {
            "mean_F": float(np.mean(fids_multi_open_all)),
            "std_F": float(np.std(fids_multi_open_all)),
            "F_05": float(np.percentile(fids_multi_open_all, 5)),
        },
        "feedback_gap": float(gap),
        "noiseless_pulse": [
            {"Omega_norm": float(a[0]), "Delta_norm": float(a[1])}
            for a in actions_noiseless
        ],
    }
    out_path = ROOT / "results" / "open_vs_closed_loop_C.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
