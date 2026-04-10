#!/usr/bin/env python
"""Fig.14 -- Population dynamics comparison: STIRAP vs PPO.

Two side-by-side panels showing P_gg(t), P_W(t), P_rr(t) evolution.
Left panel: STIRAP via run_stirap (real simulation).
Right panel: PPO from real rollout data in results/ppo_populations_B.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.plotting.plot_config import *
from src.physics.constants import SCENARIOS

cfg = SCENARIOS['B']
T_gate = cfg['T_gate']

# ---------------------------------------------------------------
# STIRAP: always simulate (fast, deterministic)
# ---------------------------------------------------------------
stirap_simulated = False
try:
    from src.baselines.stirap import run_stirap
    from src.physics.hamiltonian import get_target_state, get_ground_state
    import qutip

    fid, result = run_stirap('B', noise_params=None, n_steps=200)
    tlist = np.linspace(0, T_gate, len(result.states))
    t_us_stirap = tlist * 1e6

    gg = get_ground_state(2)
    W = get_target_state(2)
    rr = qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 1))

    P_gg_stirap = np.zeros(len(result.states))
    P_W_stirap = np.zeros(len(result.states))
    P_rr_stirap = np.zeros(len(result.states))

    def _expect(bra_ket, rho):
        val = bra_ket.dag() * rho * bra_ket
        if isinstance(val, (complex, float, int, np.floating)):
            return float(np.real(val))
        elif isinstance(val, qutip.Qobj):
            return float(np.real(val.full()[0, 0]))
        return float(np.real(val))

    for i, state in enumerate(result.states):
        rho = state if state.isoper else qutip.ket2dm(state)
        P_gg_stirap[i] = _expect(gg, rho)
        P_W_stirap[i] = _expect(W, rho)
        P_rr_stirap[i] = _expect(rr, rho)

    stirap_simulated = True
    print(f"STIRAP simulation succeeded: F = {fid:.4f}")
except Exception as e:
    print(f"STIRAP simulation failed ({e}), cannot generate left panel")
    sys.exit(1)

# ---------------------------------------------------------------
# PPO: load real population data
# ---------------------------------------------------------------
data_path = os.path.join(os.path.dirname(__file__), '../../results/ppo_populations_B.json')

if os.path.exists(data_path):
    with open(data_path) as f:
        pop_data = json.load(f)

    t_us_ppo = np.array(pop_data['t']) * 1e6
    P_gg_ppo = np.array(pop_data['P_gg'])
    P_W_ppo = np.array(pop_data['P_W'])
    P_rr_ppo = np.array(pop_data['P_rr'])
    ppo_fid = pop_data.get('final_fidelity', P_W_ppo[-1])
    print(f"Loaded PPO population data from {data_path} (final F={ppo_fid:.4f})")
else:
    print(f"WARNING: {data_path} not found. Run: python run_experiments.py --scenario B")
    sys.exit(1)

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL[0] * 1.2, DOUBLE_COL[1] * 0.85),
                                sharey=True)

c_gg = COLORS['green']
c_W = COLORS['blue']
c_rr = COLORS['red']

# Left: STIRAP
ax1.plot(t_us_stirap, P_gg_stirap, '-', color=c_gg, linewidth=1.5, label=r'$P_{|gg\rangle}$')
ax1.plot(t_us_stirap, P_W_stirap, '-', color=c_W, linewidth=1.5, label=r'$P_{|W\rangle}$')
ax1.plot(t_us_stirap, P_rr_stirap, '-', color=c_rr, linewidth=1.5, label=r'$P_{|rr\rangle}$')
ax1.set_xlabel(r'Time ($\mu$s)')
ax1.set_ylabel('Population')
ax1.set_title(f'STIRAP (F = {fid:.4f})')
ax1.legend(loc='center right', fontsize=9)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.3)

# Right: PPO
ax2.plot(t_us_ppo, P_gg_ppo, '-', color=c_gg, linewidth=1.5, label=r'$P_{|gg\rangle}$')
ax2.plot(t_us_ppo, P_W_ppo, '-', color=c_W, linewidth=1.5, label=r'$P_{|W\rangle}$')
ax2.plot(t_us_ppo, P_rr_ppo, '-', color=c_rr, linewidth=1.5, label=r'$P_{|rr\rangle}$')
ax2.set_xlabel(r'Time ($\mu$s)')
ax2.set_title(f'PPO (F = {ppo_fid:.4f})')
ax2.legend(loc='center right', fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle('Population dynamics comparison (Scenario B)', fontsize=13, y=1.02)
fig.tight_layout()

outpath = os.path.join(os.path.dirname(__file__), '../../', FIGURE_DIR, 'fig14_population_evolution.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
plt.close(fig)
print(f"Saved {os.path.abspath(outpath)}")
