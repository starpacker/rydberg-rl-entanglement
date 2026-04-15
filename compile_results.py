"""Compile all method results into a unified comparison table.

All evaluations use seed=50000+i for fair comparison.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    noise_levels = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    # ── CMA-ES per-alpha (canonical baseline) ──
    cmaes_sweep = load_json(ROOT / "results" / "noise_scaling" / "cmaes_sweep.json")
    cmaes = {}
    if cmaes_sweep:
        for entry in cmaes_sweep["noise_levels"]:
            cmaes[entry["noise_scale"]] = entry["mean_F"]

    # ── GRAPE ──
    grape_sweep = load_json(ROOT / "results" / "noise_scaling" / "grape_sweep.json")
    grape = {}
    if grape_sweep:
        for entry in grape_sweep.get("noise_levels", []):
            grape[entry["noise_scale"]] = entry["mean_F"]

    # ── PPO closed-loop ──
    ppo_partial = load_json(ROOT / "results" / "noise_scaling" / "ppo_sweep_partial.json")
    ppo_cl = {}
    if ppo_partial:
        for entry in ppo_partial.get("noise_levels", []):
            ppo_cl[entry["noise_scale"]] = entry["mean_F"]

    # ── BC-Fourier (oracle noise) ──
    bc_oracle = load_json(ROOT / "results" / "bc_fourier_eval.json")
    if bc_oracle is None:
        bc_oracle = load_json(ROOT / "results" / "bc_fourier.json")
    bc_ora = {}
    if bc_oracle:
        for k, v in bc_oracle.items():
            bc_ora[float(k)] = v["mean_F"]

    # ── BC-Fourier (estimated noise) ──
    bc_est = load_json(ROOT / "results" / "bc_fourier_estimated_eval.json")
    if bc_est is None:
        bc_est = load_json(ROOT / "results" / "bc_fourier_estimated.json")
    bc_estimated = {}
    if bc_est:
        for k, v in bc_est.items():
            bc_estimated[float(k)] = v["mean_F"]

    # ── DNAAC Phase B (oracle noise) ──
    dnaac_eval = load_json(ROOT / "results" / "dnaac" / "phase_d_eval.json")
    dnaac_b = {}
    dnaac_c = {}
    if dnaac_eval:
        for k, v in dnaac_eval.items():
            alpha = float(k)
            dnaac_b[alpha] = v["phase_b"]["mean_F"]
            if "phase_c" in v:
                dnaac_c[alpha] = v["phase_c"]["mean_F"]

    # ── PPO-OL oracle (6-dim noise obs) ──
    ppo_ol_oracle = load_json(ROOT / "results" / "ppo_openloop_real_eval.json")
    ppo_ora = {}
    if ppo_ol_oracle:
        for k, v in ppo_ol_oracle.items():
            ppo_ora[float(k)] = v["mean_F"]

    # ── PPO-OL fair (1-dim alpha obs) ──
    ppo_ol_fair = load_json(ROOT / "results" / "ppo_openloop_fair_eval.json")
    ppo_fair = {}
    if ppo_ol_fair:
        for k, v in ppo_ol_fair.items():
            ppo_fair[float(k)] = v["mean_F"]

    # ── NAAC v3 ──
    naac = load_json(ROOT / "results" / "naac" / "naac_sweep.json")
    naac_v3 = {}
    if naac:
        for entry in naac.get("noise_levels", []):
            naac_v3[entry["noise_scale"]] = entry["mean_F"]

    # ════════════════════════════════════════════════════════════
    # Print comparison table
    # ════════════════════════════════════════════════════════════
    print()
    print("=" * 110)
    print("COMPREHENSIVE METHOD COMPARISON — Bell State Fidelity F vs Noise Scale α")
    print("All evaluations: 200 episodes, seed=50000+i, scenario C")
    print("=" * 110)

    # ── Table 1: Fair methods (no oracle noise access) ──
    print()
    print("TABLE 1: FAIR METHODS (no oracle noise access)")
    print("-" * 90)
    header = f"{'α':<6} {'GRAPE':<10} {'CMA-ES/α':<10} {'PPO-CL':<10} {'DNAAC-C':<10} {'PPO-OL':<10} {'NAAC-v3':<10}"
    print(header)
    print("-" * 90)

    for alpha in noise_levels:
        g = f"{grape.get(alpha, 0):.4f}" if grape.get(alpha) else "  ---  "
        c = f"{cmaes.get(alpha, 0):.4f}" if cmaes.get(alpha) else "  ---  "
        p = f"{ppo_cl.get(alpha, 0):.4f}" if ppo_cl.get(alpha) else "  ---  "
        d = f"{dnaac_c.get(alpha, 0):.4f}" if dnaac_c.get(alpha) else "  ---  "
        o = f"{ppo_fair.get(alpha, 0):.4f}" if ppo_fair.get(alpha) else " (run) "
        n = f"{naac_v3.get(alpha, 0):.4f}" if naac_v3.get(alpha) else "  ---  "
        print(f"{alpha:<6} {g:<10} {c:<10} {p:<10} {d:<10} {o:<10} {n:<10}")

    # ── Table 2: Oracle methods (uses true noise params) ──
    print()
    print("TABLE 2: ORACLE METHODS (uses true noise params — upper bounds)")
    print("-" * 70)
    header2 = f"{'α':<6} {'BC-Four.':<10} {'DNAAC-B':<10} {'PPO-OL*':<10} {'CMA-ES/α':<10}"
    print(header2)
    print("-" * 70)

    for alpha in noise_levels:
        b = f"{bc_ora.get(alpha, 0):.4f}" if bc_ora.get(alpha) else "  ---  "
        d = f"{dnaac_b.get(alpha, 0):.4f}" if dnaac_b.get(alpha) else "  ---  "
        o = f"{ppo_ora.get(alpha, 0):.4f}" if ppo_ora.get(alpha) else "  ---  "
        c = f"{cmaes.get(alpha, 0):.4f}" if cmaes.get(alpha) else "  ---  "
        print(f"{alpha:<6} {b:<10} {d:<10} {o:<10} {c:<10}")

    print()
    print("* PPO-OL oracle uses 6-dim noise vector as observation (cheating)")
    print("  BC-Fourier and DNAAC-B also use oracle noise access")

    # ── Gap analysis ──
    print()
    print("TABLE 3: GAP vs CMA-ES per-α (positive = beats CMA-ES)")
    print("-" * 80)
    header3 = f"{'α':<6} {'DNAAC-C':<12} {'PPO-OL fair':<12} {'DNAAC-B*':<12} {'BC-Four.*':<12}"
    print(header3)
    print("-" * 80)

    for alpha in noise_levels:
        c_base = cmaes.get(alpha, 0)
        dc = f"{dnaac_c.get(alpha, 0) - c_base:+.4f}" if dnaac_c.get(alpha) and c_base else "   ---   "
        pf = f"{ppo_fair.get(alpha, 0) - c_base:+.4f}" if ppo_fair.get(alpha) and c_base else "  (run)  "
        db = f"{dnaac_b.get(alpha, 0) - c_base:+.4f}" if dnaac_b.get(alpha) and c_base else "   ---   "
        bc = f"{bc_ora.get(alpha, 0) - c_base:+.4f}" if bc_ora.get(alpha) and c_base else "   ---   "
        print(f"{alpha:<6} {dc:<12} {pf:<12} {db:<12} {bc:<12}")

    print()
    print("* Oracle methods (upper bounds, not fair comparison)")

    # ── Save unified JSON ──
    unified = {
        "description": "Unified comparison, all eval with seed=50000+i",
        "noise_levels": noise_levels,
        "methods": {}
    }

    def add_method(name, data, fair):
        unified["methods"][name] = {
            "fair": fair,
            "results": {alpha: data.get(alpha) for alpha in noise_levels if data.get(alpha)},
        }

    add_method("grape", grape, True)
    add_method("cmaes_per_alpha", cmaes, True)
    add_method("ppo_closed_loop", ppo_cl, True)
    add_method("dnaac_phase_c", dnaac_c, True)
    add_method("ppo_openloop_fair", ppo_fair, True)
    add_method("naac_v3", naac_v3, True)
    add_method("bc_fourier_oracle", bc_ora, False)
    add_method("dnaac_phase_b", dnaac_b, False)
    add_method("ppo_openloop_oracle", ppo_ora, False)

    out_path = ROOT / "results" / "unified_comparison.json"
    with open(out_path, "w") as f:
        json.dump(unified, f, indent=2)
    print(f"\nSaved unified results to {out_path}")


if __name__ == "__main__":
    main()
