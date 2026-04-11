#!/usr/bin/env python
"""Generate all Scenario C figures from evaluation results.

Creates figures 15-18 showing PPO advantage on Scenario C:
- fig15: Scenario C comparison bar chart (PPO vs STIRAP vs GRAPE)
- fig16: Scenario C training curve
- fig17: Scenario C robustness sweep
- fig18: Scenario C pulse comparison (PPO vs STIRAP)
- fig19: Scenario C population evolution (PPO vs STIRAP)

Reads from results/*_C_v3.json (or *_C_v2.json as fallback).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *
from src.physics.constants import SCENARIOS

BASE = os.path.join(os.path.dirname(__file__), '../..')
RESULTS = os.path.join(BASE, 'results')
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def load_json(name):
    """Load JSON, trying v3 then v2 then base."""
    for suffix in ['_v3.json', '_v2.json', '.json']:
        path = os.path.join(RESULTS, name.replace('.json', '') + suffix)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f"  Loaded {path}")
            return data
    raise FileNotFoundError(f"No data found for {name}")


# ===================================================================
# Fig 15: Scenario C Algorithm Comparison Bar Chart
# ===================================================================
def fig15_comparison():
    print("\n--- Fig 15: Scenario C comparison ---")
    ppo = load_json('ppo_C')
    stirap = load_json('stirap_C')
    grape = load_json('grape_C')

    methods = ['STIRAP', 'GRAPE', 'PPO+DR']
    mean_F = [stirap['mean_F'], grape['mean_F'], ppo['mean_F']]
    std_F = [stirap['std_F'], grape['std_F'], ppo['std_F']]
    F_05 = [stirap['F_05'], grape['F_05'], ppo['F_05']]

    yerr_lo = [m - f5 for m, f5 in zip(mean_F, F_05)]
    yerr_hi = std_F
    yerr = [yerr_lo, yerr_hi]

    bar_colors = [COLORS['blue'], COLORS['orange'], COLORS['green']]

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    x = np.arange(len(methods))
    bars = ax.bar(x, mean_F, width=0.5, color=bar_colors, edgecolor='black',
                  linewidth=0.5, yerr=yerr, capsize=5, error_kw={'linewidth': 1.2},
                  zorder=3)

    for i, (bar, val) in enumerate(zip(bars, mean_F)):
        y_top = bar.get_height() + std_F[i] + 0.008
        ax.text(bar.get_x() + bar.get_width() / 2, y_top,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel(r'Fidelity $F$')
    ax.set_title('Scenario C: High-Noise Regime')
    y_min = min(mean_F) - max(yerr_lo) - 0.05
    ax.set_ylim(max(0, y_min), 1.02)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    outpath = os.path.join(FIGDIR, 'fig15_scenario_C_comparison.png')
    fig.savefig(outpath)
    fig.savefig(outpath.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  Saved {outpath}")
    return mean_F, std_F, F_05


# ===================================================================
# Fig 16: Scenario C Training Curve
# ===================================================================
def fig16_training_curve():
    print("\n--- Fig 16: Training curve ---")
    for name in ['training_logs_C_v3.json', 'training_logs_C_v2.json', 'training_logs_C.json']:
        path = os.path.join(RESULTS, name)
        if os.path.exists(path):
            with open(path) as f:
                logs = json.load(f)
            print(f"  Loaded {path}")
            break
    else:
        print("  No training logs found!")
        return

    def rolling_mean(x, window=50):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        result = np.zeros(len(x))
        for i in range(len(x)):
            lo = max(0, i - window + 1)
            result[i] = (cumsum[i + 1] - cumsum[lo]) / (i - lo + 1)
        return result

    fig, ax = plt.subplots(figsize=DOUBLE_COL)

    # Handle both single-seed and multi-seed formats
    if 'seeds' in logs:
        seeds_data = logs['seeds']
    else:
        seeds_data = [logs]  # single seed

    seed_colors = [COLORS['blue'], COLORS['orange'], COLORS['red']]

    for idx, sd in enumerate(seeds_data):
        fids = np.array(sd['fidelities'])
        timesteps = np.array(sd.get('timesteps', np.arange(len(fids))))

        # Use timesteps on x-axis for better interpretability
        smoothed = rolling_mean(fids, window=100)
        seed_val = sd.get('seed', idx)
        label = f"Seed {seed_val}"

        ax.plot(timesteps, fids, color=seed_colors[idx % 3], alpha=0.06, linewidth=0.3)
        ax.plot(timesteps, smoothed, color=seed_colors[idx % 3], linewidth=1.5, label=label)

    # Reference lines for STIRAP and GRAPE
    try:
        stirap = load_json('stirap_C')
        grape = load_json('grape_C')
        ax.axhline(stirap['mean_F'], color=COLORS['blue'], linestyle='--',
                    linewidth=1.0, alpha=0.7, zorder=1)
        ax.text(timesteps[-1] * 0.02, stirap['mean_F'] + 0.005,
                f"STIRAP = {stirap['mean_F']:.3f}", fontsize=9, color=COLORS['blue'])

        ax.axhline(grape['mean_F'], color=COLORS['orange'], linestyle='--',
                    linewidth=1.0, alpha=0.7, zorder=1)
        ax.text(timesteps[-1] * 0.02, grape['mean_F'] - 0.025,
                f"GRAPE = {grape['mean_F']:.3f}", fontsize=9, color=COLORS['orange'])
    except FileNotFoundError:
        pass

    ax.set_xlabel('Training timestep')
    ax.set_ylabel('Episode fidelity')
    ax.set_title('PPO training on Scenario C (high-noise regime)')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    outpath = os.path.join(FIGDIR, 'fig16_training_curve_C.png')
    fig.savefig(outpath)
    fig.savefig(outpath.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  Saved {outpath}")


# ===================================================================
# Fig 17: Robustness Sweep
# ===================================================================
def fig17_robustness():
    print("\n--- Fig 17: Robustness sweep ---")
    data = load_json('robustness_sweep_C')

    delta_pct = np.array(data['delta_pct'])
    F_stirap = np.array(data['stirap'])
    F_grape = np.array(data['grape'])
    F_ppo_raw = data['ppo']
    has_ppo = all(v is not None for v in F_ppo_raw)
    F_ppo = np.array(F_ppo_raw) if has_ppo else None

    fig, ax = plt.subplots(figsize=DOUBLE_COL)

    ax.plot(delta_pct, F_stirap, 'o-', color=COLORS['blue'], linewidth=1.8,
            markersize=6, label='STIRAP')
    ax.plot(delta_pct, F_grape, 's-', color=COLORS['orange'], linewidth=1.8,
            markersize=6, label='GRAPE')
    if F_ppo is not None:
        ax.plot(delta_pct, F_ppo, '^-', color=COLORS['green'], linewidth=1.8,
                markersize=6, label='PPO+DR')

    ax.set_xlabel(r'Additional amplitude miscalibration $\delta\Omega / \Omega$ (%)')
    ax.set_ylabel(r'Mean fidelity $\langle F \rangle$')
    ax.set_title('Robustness to amplitude error (Scenario C, high-noise)')
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim(-0.2, max(delta_pct) + 0.2)
    y_min = min(F_stirap.min(), F_grape.min())
    if F_ppo is not None:
        y_min = min(y_min, F_ppo.min())
    ax.set_ylim(max(0, y_min - 0.05), 1.005)
    ax.grid(True, alpha=0.3)

    outpath = os.path.join(FIGDIR, 'fig17_robustness_C.png')
    fig.savefig(outpath)
    fig.savefig(outpath.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  Saved {outpath}")


# ===================================================================
# Fig 18: Pulse Comparison (STIRAP vs PPO on Scenario C)
# ===================================================================
def fig18_pulse_comparison():
    print("\n--- Fig 18: Pulse comparison ---")
    pulse_path = os.path.join(RESULTS, 'ppo_pulse_C.json')
    if not os.path.exists(pulse_path):
        print(f"  {pulse_path} not found, skipping")
        return

    with open(pulse_path) as f:
        pulse_data = json.load(f)

    # STIRAP pulse for Scenario C
    cfg = SCENARIOS['C']
    T_gate = cfg['T_gate']
    n_atoms = cfg['n_atoms']
    Omega_max_stirap = 2 * np.pi / (np.sqrt(n_atoms) * T_gate)

    n_pts = 300
    t_s_stirap = np.linspace(0, T_gate, n_pts)
    t_us_stirap = t_s_stirap * 1e6
    Omega_stirap_MHz = Omega_max_stirap * np.sin(np.pi * t_s_stirap / T_gate) ** 2 / (2 * np.pi * 1e6)
    Delta_stirap_MHz = np.zeros_like(t_s_stirap)

    # PPO pulse
    t_us_ppo = np.array(pulse_data['time_us'])
    Omega_ppo_MHz = np.array(pulse_data['omega_MHz'])
    Delta_ppo_MHz = np.array(pulse_data['delta_MHz'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COL[0], DOUBLE_COL[1] * 1.1),
                                    sharex=True)

    ax1.plot(t_us_stirap, Omega_stirap_MHz, '-', color=COLORS['blue'], linewidth=1.8,
             label='STIRAP')
    ax1.step(t_us_ppo, Omega_ppo_MHz, '-', color=COLORS['red'], linewidth=1.8,
             where='mid', label='PPO')
    ax1.set_ylabel(r'$\Omega(t)/2\pi$ (MHz)')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_title('Pulse shape comparison (Scenario C)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.plot(t_us_stirap, Delta_stirap_MHz, '-', color=COLORS['blue'], linewidth=1.8,
             label='STIRAP')
    ax2.step(t_us_ppo, Delta_ppo_MHz, '-', color=COLORS['red'], linewidth=1.8,
             where='mid', label='PPO')
    ax2.set_xlabel(r'Time ($\mu$s)')
    ax2.set_ylabel(r'$\Delta(t)/2\pi$ (MHz)')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linewidth=0.5, linestyle='-')

    fig.tight_layout()

    outpath = os.path.join(FIGDIR, 'fig18_pulse_comparison_C.png')
    fig.savefig(outpath)
    fig.savefig(outpath.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  Saved {outpath}")


# ===================================================================
# Fig 19: Population Evolution (STIRAP vs PPO on Scenario C)
# ===================================================================
def fig19_population_evolution():
    print("\n--- Fig 19: Population evolution ---")
    pop_path = os.path.join(RESULTS, 'ppo_populations_C.json')
    if not os.path.exists(pop_path):
        print(f"  {pop_path} not found, skipping")
        return

    with open(pop_path) as f:
        pop_data = json.load(f)

    # Simulate STIRAP on Scenario C
    try:
        from src.baselines.stirap import run_stirap
        from src.physics.hamiltonian import get_target_state, get_ground_state
        import qutip

        fid, result = run_stirap('C', noise_params=None, n_steps=200)
        cfg = SCENARIOS['C']
        T_gate = cfg['T_gate']
        tlist = np.linspace(0, T_gate, len(result.states))
        t_us_stirap = tlist * 1e6

        gg = get_ground_state(2)
        W = get_target_state(2)
        rr = qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 1))

        P_gg_s = np.zeros(len(result.states))
        P_W_s = np.zeros(len(result.states))
        P_rr_s = np.zeros(len(result.states))

        for i, state in enumerate(result.states):
            rho = state if state.isoper else qutip.ket2dm(state)
            P_gg_s[i] = float(np.real((gg.dag() * rho * gg).full()[0, 0]))
            P_W_s[i] = float(np.real((W.dag() * rho * W).full()[0, 0]))
            P_rr_s[i] = float(np.real((rr.dag() * rho * rr).full()[0, 0]))

        stirap_ok = True
        print(f"  STIRAP noiseless F = {fid:.4f}")
    except Exception as e:
        print(f"  STIRAP simulation failed: {e}")
        stirap_ok = False

    # PPO data
    t_us_ppo = np.array(pop_data['time_us'])
    P_gg_p = np.array(pop_data['gg'])
    P_W_p = (np.array(pop_data['gr']) + np.array(pop_data['rg']))  # |W> = |gr>+|rg>
    P_rr_p = np.array(pop_data['rr'])

    c_gg = COLORS['green']
    c_W = COLORS['blue']
    c_rr = COLORS['red']

    if stirap_ok:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL[0] * 1.2, DOUBLE_COL[1] * 0.85),
                                        sharey=True)

        ax1.plot(t_us_stirap, P_gg_s, '-', color=c_gg, linewidth=1.5, label=r'$P_{|gg\rangle}$')
        ax1.plot(t_us_stirap, P_W_s, '-', color=c_W, linewidth=1.5, label=r'$P_{|W\rangle}$')
        ax1.plot(t_us_stirap, P_rr_s, '-', color=c_rr, linewidth=1.5, label=r'$P_{|rr\rangle}$')
        ax1.set_xlabel(r'Time ($\mu$s)')
        ax1.set_ylabel('Population')
        ax1.set_title(f'STIRAP (F = {fid:.4f})')
        ax1.legend(loc='center right', fontsize=9)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)

        ppo_fid = P_W_p[-1]
        ax2.plot(t_us_ppo, P_gg_p, '-', color=c_gg, linewidth=1.5, label=r'$P_{|gg\rangle}$')
        ax2.plot(t_us_ppo, P_W_p, '-', color=c_W, linewidth=1.5, label=r'$P_{|W\rangle}$')
        ax2.plot(t_us_ppo, P_rr_p, '-', color=c_rr, linewidth=1.5, label=r'$P_{|rr\rangle}$')
        ax2.set_xlabel(r'Time ($\mu$s)')
        ax2.set_title(f'PPO (F = {ppo_fid:.4f})')
        ax2.legend(loc='center right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.suptitle('Population dynamics (Scenario C, noiseless)', fontsize=13, y=1.02)
    else:
        fig, ax = plt.subplots(figsize=DOUBLE_COL)
        ppo_fid = P_W_p[-1]
        ax.plot(t_us_ppo, P_gg_p, '-', color=c_gg, linewidth=1.5, label=r'$P_{|gg\rangle}$')
        ax.plot(t_us_ppo, P_W_p, '-', color=c_W, linewidth=1.5, label=r'$P_{|W\rangle}$')
        ax.plot(t_us_ppo, P_rr_p, '-', color=c_rr, linewidth=1.5, label=r'$P_{|rr\rangle}$')
        ax.set_xlabel(r'Time ($\mu$s)')
        ax.set_ylabel('Population')
        ax.set_title(f'PPO population dynamics (F = {ppo_fid:.4f})')
        ax.legend(loc='center right', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    outpath = os.path.join(FIGDIR, 'fig19_population_evolution_C.png')
    fig.savefig(outpath)
    fig.savefig(outpath.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  Saved {outpath}")


# ===================================================================
# Main
# ===================================================================
if __name__ == '__main__':
    print("Generating Scenario C figures...")
    try:
        mean_F, std_F, F_05 = fig15_comparison()
    except FileNotFoundError as e:
        print(f"  Skipping fig15: {e}")
        mean_F = None

    try:
        fig16_training_curve()
    except Exception as e:
        print(f"  Skipping fig16: {e}")

    try:
        fig17_robustness()
    except FileNotFoundError as e:
        print(f"  Skipping fig17: {e}")

    fig18_pulse_comparison()
    fig19_population_evolution()

    if mean_F:
        print(f"\n{'='*50}")
        print(f"SCENARIO C SUMMARY")
        print(f"{'='*50}")
        print(f"  STIRAP: F = {mean_F[0]:.4f} +/- {std_F[0]:.4f}")
        print(f"  GRAPE:  F = {mean_F[1]:.4f} +/- {std_F[1]:.4f}")
        print(f"  PPO:    F = {mean_F[2]:.4f} +/- {std_F[2]:.4f}")
        if mean_F[2] > mean_F[0] and mean_F[2] > mean_F[1]:
            print(f"  >>> PPO outperforms by {mean_F[2] - max(mean_F[0], mean_F[1]):.4f} <<<")

    print("\nDone!")
