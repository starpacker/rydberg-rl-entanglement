"""Re-run all evaluations after audit fixes.

Fixes applied:
1. Fourier basis endpoint=False (matches CMA-ES)
2. Amplitude bias NOT scaled by noise_scale (matches numpy env)
3. BC-Fourier estimated uses correct calibration pulse
4. GRAPE eval uses unified seeds (50000+i)
5. Shared DNAAC_NOISE_NORMALIZER constant

Steps:
1. Retrain DNAAC (B+C) with fixed diff sim
2. Evaluate DNAAC (Phase D) in numpy env
3. Retrain PPO-OL fair with fixed Fourier basis
4. Re-evaluate GRAPE with unified seeds
5. Compile final results
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
t0_total = time.perf_counter()

# Load CMA-ES params
with open(ROOT / "results" / "noise_scaling" / "cmaes_sweep.json") as f:
    cmaes_data = json.load(f)
cmaes_params = None
for entry in cmaes_data["noise_levels"]:
    if entry["noise_scale"] == 1.0:
        cmaes_params = np.array(entry["best_params"], dtype=np.float32)
        break
print(f"Loaded CMA-ES base params (20-dim)")

noise_scales = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

# ══════════════════════════════════════════════════════════════
# STEP 1: Retrain DNAAC
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 1: Retrain DNAAC with fixed simulator")
print("="*70)

from train_dnaac import phase_a_oracle, phase_b_train, phase_c_train, phase_d_evaluate

# Phase A: Oracle test
print("\n--- Phase A ---")
oracle_results = phase_a_oracle(
    cmaes_params, noise_scales=noise_scales,
    batch_size=32, n_opt_steps=200, device=device
)
print("Oracle results:")
for alpha in noise_scales:
    print(f"  alpha={alpha}: F={oracle_results.get(alpha, {}).get('oracle_mean_F', 'N/A')}")

# Phase B
print("\n--- Phase B ---")
corrector, b_history = phase_b_train(
    cmaes_params, n_batches=10000, batch_size=64, device=device
)

# Phase C
print("\n--- Phase C ---")
estimator, c_history = phase_c_train(
    corrector, cmaes_params, k_calib=10,
    n_batches=5000, batch_size=32, device=device
)

# Save models
models_dir = ROOT / "models" / "dnaac"
models_dir.mkdir(parents=True, exist_ok=True)
torch.save(corrector.state_dict(), models_dir / "corrector.pt")
torch.save(estimator.state_dict(), models_dir / "estimator.pt")
print(f"Models saved to {models_dir}")

# Phase D: Evaluate in numpy env
print("\n--- Phase D ---")
results_d = phase_d_evaluate(
    corrector=corrector,
    corrector_finetuned=corrector,
    estimator=estimator,
    cmaes_params=cmaes_params,
    noise_scales=noise_scales,
    k_calib=10, n_test=200, device=device,
)

results_dir = ROOT / "results" / "dnaac"
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "phase_d_eval.json", "w") as f:
    json.dump(results_d, f, indent=2)
print(f"DNAAC results saved")

# ══════════════════════════════════════════════════════════════
# STEP 2: Retrain PPO-OL fair (basis changed, must retrain)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 2: Retrain PPO-OL fair with fixed Fourier basis")
print("="*70)

import subprocess
result = subprocess.run(
    [sys.executable, "train_ppo_openloop_fair.py",
     "--train", "--eval", "--timesteps", "1000000",
     "--n-seeds", "3", "--n-envs", "8"],
    cwd=str(ROOT),
    capture_output=False,
    timeout=7200,  # 2 hour timeout
)
print(f"PPO-OL fair training exit code: {result.returncode}")

# ══════════════════════════════════════════════════════════════
# STEP 3: Re-evaluate GRAPE with unified seeds
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 3: Re-evaluate GRAPE with unified seeds")
print("="*70)

from src.baselines.grape import run_grape, run_grape_eval
from src.physics.noise_model import NoiseModel

_, omega_grape, delta_grape = run_grape("C", n_steps=60)
grape_results = {"method": "grape", "noise_levels": []}

for alpha in noise_scales:
    nm = NoiseModel("C", noise_scale=alpha)
    fids = []
    for i in range(200):
        rng = np.random.default_rng(50000 + i)
        noise = nm.sample(rng)
        try:
            fid = run_grape_eval("C", omega_grape, delta_grape, noise)
        except Exception:
            fid = 0.0
        fids.append(fid)

    fids_arr = np.array(fids)
    grape_results["noise_levels"].append({
        "noise_scale": alpha,
        "mean_F": float(fids_arr.mean()),
        "std_F": float(fids_arr.std()),
    })
    print(f"  alpha={alpha}: F = {fids_arr.mean():.4f} +/- {fids_arr.std():.4f}")

with open(ROOT / "results" / "noise_scaling" / "grape_sweep.json", "w") as f:
    json.dump(grape_results, f, indent=2)
print("GRAPE results saved")

# ══════════════════════════════════════════════════════════════
# STEP 4: Compile final results
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 4: Compile final results")
print("="*70)

# Run compile_results.py as a subprocess to avoid stale imports
subprocess.run([sys.executable, "compile_results.py"], cwd=str(ROOT))

total_time = time.perf_counter() - t0_total
print(f"\n{'='*70}")
print(f"TOTAL RE-EVALUATION TIME: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"{'='*70}")
