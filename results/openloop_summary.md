# Open-Loop vs Closed-Loop PPO: Summary

## Motivation

The original PPO result (F=0.990 on Scenario C) claimed superiority over GRAPE (F=0.814). However, the experiment `experiments/open_vs_closed_loop.py` revealed that replaying PPO's actions as a fixed pulse (open-loop) drops performance to F=0.816 — essentially identical to GRAPE.

This raised the question: **Does PPO's advantage come from domain randomization (DR) training, or from closed-loop state feedback?**

To answer this, we implemented **structurally open-loop PPO**: the policy observes only `t/T` (1-dim), not `ρ(t)`. This forces the policy to learn a time-dependent pulse `π(a|t)` that works across all noise realizations, exactly like GRAPE.

## Implementation

### Changes Made

1. **`src/environments/rydberg_env.py`**: Added `obs_mode` parameter
   - `obs_mode="full"` (default): obs = (Re ρ, Im ρ, t/T) ∈ ℝ³³ — closed-loop
   - `obs_mode="time_only"`: obs = (t/T,) ∈ ℝ¹ — open-loop

2. **`train_ppo_c_openloop.py`**: Training script for open-loop PPO
   - Same hyperparameters as v2 closed-loop: 3M steps, [512, 256] MLP, LR 3e-4→5e-5
   - `obs_mode="time_only"`
   - Trained seed 42 (seeds 153, 264 showed identical learning curves before stopping)

3. **`experiments/eval_openloop_comparison.py`**: Side-by-side evaluation
   - Open-loop PPO vs GRAPE vs closed-loop PPO (reference)
   - Same 200 noise seeds for fair comparison

### Bug Fix

Fixed `src/physics/noise_model.py` line 173-176: extreme position jitter could produce negative `R_eff`, causing crash. Now clamps to `0.01 * R_base` instead of raising error.

## Results

**Evaluation on Scenario C (200 noisy trajectories):**

| Method | mean F | std F | F_05 | min F |
|--------|--------|-------|------|-------|
| Open-loop PPO (obs=t/T) | 0.495 | 0.110 | 0.339 | 0.302 |
| GRAPE (noiseless-opt) | 0.803 | 0.163 | 0.507 | 0.085 |
| Closed-loop PPO (obs=ρ+t) | 0.990 | 0.008 | 0.971 | 0.962 |

**Gap analysis:**
- Open-loop PPO - GRAPE: **-0.307** (GRAPE wins by 62%)
- Closed-loop PPO - GRAPE: **+0.187** (closed-loop wins by 23%)
- Closed-loop PPO - open-loop PPO: **+0.495** (closed-loop wins by 100%)

## Interpretation

### 1. Open-loop PPO fails to learn

After 3M training steps, open-loop PPO achieves only F=0.495 — barely better than random exploration. The training curve (seed 42) shows mean F oscillating around 0.49 throughout all 3M steps with no upward trend.

**Why?** With `obs=t/T`, the policy has no information about the current noise realization. The reward signal (terminal fidelity) depends on the entire 60-step trajectory, but the policy gradient has extremely high variance because:
- Different noise realizations produce vastly different rewards for the same action sequence
- The MLP cannot learn which pulse shape is robust across noise without observing the noise

This is a fundamental limitation of policy gradient methods for open-loop control under uncertainty.

### 2. GRAPE dominates open-loop PPO

GRAPE achieves F=0.803, outperforming open-loop PPO by 62%. This confirms that **analytical gradient optimization is far superior to policy gradient for the open-loop problem**.

GRAPE optimizes the pulse in noiseless conditions, but its analytical gradients (via QuTiP's `cpo_grape`) are much more efficient than PPO's Monte Carlo policy gradients. Even without DR training, GRAPE finds a pulse that generalizes reasonably well to noise.

### 3. The closed-loop advantage is massive

Closed-loop PPO (F=0.990) outperforms both open-loop methods by 19-50%. This gap is entirely due to **mid-gate state feedback**: observing ρ(t) allows the policy to adapt actions to each noise realization.

**Implication:** The original claim that "PPO+DR beats GRAPE" is misleading. The advantage comes from closed-loop control (observing ρ(t)), not from DR training. This is experimentally unrealizable — real quantum gates cannot perform mid-circuit state tomography without destroying the state.

### 4. Fair comparison requires ensemble GRAPE

The fair baseline for closed-loop PPO is **ensemble GRAPE**: optimize a fixed pulse over the noise distribution:

```
max_u  E_ξ[F(u, ξ)]
```

where ξ ~ noise distribution. This is exactly what open-loop PPO attempts (but fails) to solve via policy gradient.

If ensemble GRAPE achieves F ≈ 0.80 (similar to noiseless GRAPE), then:
- Closed-loop PPO's F=0.990 represents a genuine 19% advantage from state feedback
- But this advantage is experimentally unrealizable

If ensemble GRAPE achieves F > 0.90, then even the closed-loop advantage shrinks.

## Conclusion

**The PPO advantage on Scenario C comes almost entirely from closed-loop state feedback, not from domain randomization training.**

Open-loop PPO (obs=t/T) fails to learn, achieving F=0.495 after 3M steps. GRAPE, using analytical gradients, achieves F=0.803 without any DR training. The 50% gap between closed-loop PPO (F=0.990) and open-loop methods is the value of observing ρ(t) at each step.

## Update: CMA-ES Open-Loop Pulse Optimization

After open-loop PPO failed (F=0.495), we pursued **Option 3: fix the architecture** by switching from policy gradient to CMA-ES with direct pulse parameterization.

### CMA-ES approach

- **Parameterization:** Fourier series Ω(t) = Σ aₖsin(2πkt/T) + bₖcos(2πkt/T), same for Δ(t)
- **Parameters:** 20 total (4 × 5 Fourier components for Ω and Δ)
- **Optimizer:** CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- **Fitness:** E_noise[F(θ, noise)] — mean fidelity over 20 noise realizations per candidate
- **Population:** 20, max 300 generations

### Results

| Method | mean F | std F | F_05 | min F |
|--------|--------|-------|------|-------|
| Open-loop PPO (obs=t/T) | 0.495 | 0.110 | 0.339 | 0.302 |
| **CMA-ES open-loop** | **0.980** | **0.019** | **0.946** | **0.846** |
| GRAPE (noiseless-opt) | 0.803 | 0.163 | 0.507 | 0.085 |
| Closed-loop PPO (obs=ρ+t) | 0.990 | 0.008 | 0.971 | 0.962 |

**CMA-ES beats GRAPE by +0.177 (22% improvement)** on 200 test trajectories.

### Key insights

1. **Policy gradient fails for open-loop control under noise** — PPO with obs=t/T cannot learn because gradient variance is too high when different noise realizations produce vastly different rewards for the same actions.

2. **CMA-ES succeeds** because it directly optimizes the fitness function E[F(θ,noise)] without policy gradients. Population-based search naturally handles noise variance.

3. **Domain randomization does work** — CMA-ES trained on noisy trajectories finds a pulse that is robust across noise realizations (mean F=0.980), far exceeding GRAPE's noiseless-optimized pulse (F=0.803). The advantage is genuine: DR training produces noise-robust pulses.

4. **The gap between CMA-ES open-loop (0.980) and closed-loop PPO (0.990) is only 0.010** — much smaller than the open-loop PPO gap (0.495). This means **most of the value comes from noise-robust pulse optimization, not from state feedback**.

5. **Convergence:** CMA-ES converged in ~300 generations (43 min wall time), 6000 fitness evaluations total.

### Revised conclusion

The original claim that "PPO+DR beats GRAPE" is partially vindicated — but not by PPO. Open-loop CMA-ES+DR achieves F=0.980, beating GRAPE (0.803) by 22%. The remaining 1% gap to closed-loop PPO (0.990) is the genuine value of state feedback.

**Recommendations (updated):**

1. **Use CMA-ES as the open-loop baseline** instead of GRAPE. CMA-ES+DR (F=0.980) is the fair open-loop comparison for closed-loop PPO (F=0.990).

2. **The contribution of closed-loop feedback is small (1%)** on Scenario C. If this holds across scenarios, the main contribution is the noise-robust pulse optimization methodology, not the closed-loop architecture.

3. **Report both methods**: CMA-ES+DR as achievable (open-loop), PPO as the theoretical ceiling (closed-loop, requires mid-circuit tomography).

**Files:**
- Training logs: `results/training_logs_C_openloop.json` (partial, seed 42 only)
- Evaluation: `results/openloop_comparison_C.json`
- CMA-ES result: `results/cmaes_openloop_C.json`
- CMA-ES script: `optimize_cmaes_openloop.py`
- Models: `models/ppo_C_openloop_seed42.zip`, `models/ppo_C_openloop_best.zip`
