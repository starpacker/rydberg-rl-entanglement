"""Evaluate PPO v2 model on Scenario C with STIRAP and GRAPE baselines.

Loads ppo_C_v2_best.zip and runs:
1. PPO evaluation (200 trajectories)
2. PPO pulse + population extraction (noiseless)
3. STIRAP evaluation (200 trajectories)
4. GRAPE optimization + evaluation (200 trajectories)
5. Robustness sweep (0-5% amplitude error)

Saves all results to results/*_C_v3.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from src.environments.rydberg_env import RydbergBellEnv
from src.physics.constants import SCENARIOS
from src.physics.noise_model import NoiseModel
from src.baselines.stirap import run_stirap
from src.baselines.grape import run_grape, run_grape_eval
from src.baselines.evaluate import evaluate_policy

results_dir = ROOT / "results"
results_dir.mkdir(exist_ok=True)

scenario = "C"
n_traj = 200
env_n_steps = 60

# Load model
model_path = ROOT / "models" / "ppo_C_v2_best"
print(f"Loading PPO model from {model_path}")
ppo_model = PPO.load(str(model_path))

# ===================================================================
# 1. PPO Evaluation
# ===================================================================
print(f"\n--- PPO evaluation ({n_traj} trajectories) ---")
env = RydbergBellEnv(
    scenario=scenario, n_steps=env_n_steps,
    use_noise=True, reward_shaping_alpha=0.0,
    obs_include_time=True,
)
ppo_fids = []
for i in range(n_traj):
    obs, _ = env.reset(seed=1000 + i)
    done = False
    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    ppo_fids.append(info.get("fidelity", 0.0))
    if (i + 1) % 50 == 0:
        print(f"  trajectory {i+1}/{n_traj}: mean_F = {np.mean(ppo_fids):.4f}")

ppo_arr = np.array(ppo_fids)
ppo_res = {
    "method": "ppo",
    "scenario": scenario,
    "mean_F": float(ppo_arr.mean()),
    "std_F": float(ppo_arr.std()),
    "F_05": float(np.percentile(ppo_arr, 5)),
    "n_trajectories": n_traj,
}
with open(results_dir / "ppo_C_v3.json", "w") as f:
    json.dump(ppo_res, f, indent=2)
print(f"  PPO: mean_F = {ppo_res['mean_F']:.4f} +/- {ppo_res['std_F']:.4f}, "
      f"F_05 = {ppo_res['F_05']:.4f}")

# ===================================================================
# 2. PPO Pulse + Population Extraction (noiseless)
# ===================================================================
print("\n--- PPO pulse + populations (noiseless) ---")
env2 = RydbergBellEnv(
    scenario=scenario, n_steps=env_n_steps,
    use_noise=False, reward_shaping_alpha=0.0,
    obs_include_time=True,
)
obs, _ = env2.reset(seed=0)
pulse_omega, pulse_delta = [], []
populations = {"gg": [], "gr": [], "rg": [], "rr": []}
rho = env2._rho_np
diag = np.diag(rho).real
populations["gg"].append(float(diag[0]))
populations["gr"].append(float(diag[1]))
populations["rg"].append(float(diag[2]))
populations["rr"].append(float(diag[3]))

done = False
while not done:
    action, _ = ppo_model.predict(obs, deterministic=True)
    a = np.clip(action, -1.0, 1.0)
    Omega_val = float((a[0] + 1.0) / 2.0 * 2.0 * env2.Omega_max)
    Delta_val = float(a[1] * env2.Omega_max)
    pulse_omega.append(Omega_val / (2 * np.pi * 1e6))
    pulse_delta.append(Delta_val / (2 * np.pi * 1e6))

    obs, _, terminated, truncated, info = env2.step(action)
    done = terminated or truncated

    rho = env2._rho_np
    diag = np.diag(rho).real
    populations["gg"].append(float(diag[0]))
    populations["gr"].append(float(diag[1]))
    populations["rg"].append(float(diag[2]))
    populations["rr"].append(float(diag[3]))

T_gate = env2.T_gate
t_pulse = [i * T_gate / env_n_steps * 1e6 for i in range(env_n_steps)]
t_pop = [i * T_gate / env_n_steps * 1e6 for i in range(env_n_steps + 1)]

with open(results_dir / "ppo_pulse_C.json", "w") as f:
    json.dump({"time_us": t_pulse, "omega_MHz": pulse_omega, "delta_MHz": pulse_delta}, f, indent=2)
with open(results_dir / "ppo_populations_C.json", "w") as f:
    json.dump({"time_us": t_pop, **populations}, f, indent=2)
print(f"  Noiseless PPO fidelity: {info.get('fidelity', 0.0):.6f}")

# ===================================================================
# 3. STIRAP Evaluation
# ===================================================================
print(f"\n--- STIRAP evaluation ({n_traj} trajectories) ---")
stirap_res = evaluate_policy(run_stirap, scenario, n_trajectories=n_traj)
stirap_res["method"] = "stirap"
stirap_res["scenario"] = scenario
with open(results_dir / "stirap_C_v3.json", "w") as f:
    json.dump({k: v for k, v in stirap_res.items() if k != "fidelities"}, f, indent=2)
print(f"  STIRAP: mean_F = {stirap_res['mean_F']:.4f} +/- {stirap_res['std_F']:.4f}, "
      f"F_05 = {stirap_res['F_05']:.4f}")

# ===================================================================
# 4. GRAPE Optimization + Evaluation
# ===================================================================
print(f"\n--- GRAPE optimization + evaluation ({n_traj} trajectories) ---")
print("  Optimizing GRAPE (noiseless, 60 steps, 1000 iterations)...")
fid_noiseless, omega_grape, delta_grape = run_grape(
    scenario, n_steps=60, n_iter=1000, verbose=True
)
print(f"  GRAPE noiseless F = {fid_noiseless:.6f}")

def grape_eval(sc, noise_params=None, **kw):
    return run_grape_eval(sc, omega_grape, delta_grape, noise_params)

print(f"  Evaluating GRAPE under noise...")
grape_res = evaluate_policy(grape_eval, scenario, n_trajectories=n_traj)
grape_res["method"] = "grape"
grape_res["scenario"] = scenario
grape_res["noiseless_F"] = fid_noiseless
grape_save = {k: v for k, v in grape_res.items() if k != "fidelities"}
grape_save["omega_pulse"] = omega_grape.tolist()
grape_save["delta_pulse"] = delta_grape.tolist()
with open(results_dir / "grape_C_v3.json", "w") as f:
    json.dump(grape_save, f, indent=2)
print(f"  GRAPE: mean_F = {grape_res['mean_F']:.4f} +/- {grape_res['std_F']:.4f}, "
      f"F_05 = {grape_res['F_05']:.4f}")

# ===================================================================
# 5. Robustness Sweep
# ===================================================================
print(f"\n--- Robustness sweep ---")
delta_pcts = [0, 1, 2, 3, 4, 5]
rob_results = {"delta_pct": delta_pcts, "stirap": [], "grape": [], "ppo": []}
nm = NoiseModel(scenario)
n_rob = 100

for dp in delta_pcts:
    bias = dp / 100.0
    print(f"  delta_Omega = {dp}%")

    rng = np.random.default_rng(42)
    stir_fids = []
    for i in range(n_rob):
        noise = nm.sample(rng)
        noise["amplitude_bias"] = noise.get("amplitude_bias", 0.0) + bias
        fid, _ = run_stirap(scenario, noise_params=noise)
        stir_fids.append(fid)
    rob_results["stirap"].append(float(np.mean(stir_fids)))

    rng = np.random.default_rng(42)
    gr_fids = []
    for i in range(n_rob):
        noise = nm.sample(rng)
        noise["amplitude_bias"] = noise.get("amplitude_bias", 0.0) + bias
        fid = run_grape_eval(scenario, omega_grape, delta_grape, noise)
        gr_fids.append(fid)
    rob_results["grape"].append(float(np.mean(gr_fids)))

    env = RydbergBellEnv(
        scenario=scenario, n_steps=env_n_steps,
        use_noise=True, reward_shaping_alpha=0.0,
        obs_include_time=True,
    )
    pp_fids = []
    for i in range(n_rob):
        obs, _ = env.reset(seed=2000 + i)
        env._amplitude_bias += bias
        done = False
        while not done:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        pp_fids.append(info.get("fidelity", 0.0))
    rob_results["ppo"].append(float(np.mean(pp_fids)))

    print(f"    STIRAP={rob_results['stirap'][-1]:.4f}, "
          f"GRAPE={rob_results['grape'][-1]:.4f}, "
          f"PPO={rob_results['ppo'][-1]:.4f}")

with open(results_dir / "robustness_sweep_C_v3.json", "w") as f:
    json.dump(rob_results, f, indent=2)

# ===================================================================
# Summary
# ===================================================================
print(f"\n{'='*60}")
print(f"{'SCENARIO C RESULTS':^60}")
print(f"{'='*60}")
print(f"{'Method':<12} {'mean_F':>10} {'std_F':>10} {'F_05':>10}")
print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10}")
print(f"{'PPO':<12} {ppo_res['mean_F']:>10.4f} {ppo_res['std_F']:>10.4f} {ppo_res['F_05']:>10.4f}")
print(f"{'STIRAP':<12} {stirap_res['mean_F']:>10.4f} {stirap_res['std_F']:>10.4f} {stirap_res['F_05']:>10.4f}")
print(f"{'GRAPE':<12} {grape_res['mean_F']:>10.4f} {grape_res['std_F']:>10.4f} {grape_res['F_05']:>10.4f}")
print(f"\nPPO advantage over STIRAP: {ppo_res['mean_F'] - stirap_res['mean_F']:+.4f}")
print(f"PPO advantage over GRAPE:  {ppo_res['mean_F'] - grape_res['mean_F']:+.4f}")
print("\nAll evaluations complete!")
