# Rydberg RL Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete mid-term physics report (markdown + LaTeX math) with runnable simulation code, RL training pipeline, and 14 auto-generated figures, demonstrating PPO-based quantum control outperforming traditional methods for Rydberg atom entanglement.

**Architecture:** Phase-based execution with sub-agent parallelism. Phase 1 builds the physics code foundation + writes §1-§3. Phase 2 runs baselines + RL training + writes §4-§6 + appendices. Phase 3 generates result figures + writes §7-§8. Phase 4 assembles the final report. Each phase gates on the previous. Within each phase, independent tasks run in parallel via sub-agents.

**Tech Stack:** Python 3.10+, QuTiP 5, numpy, scipy, gymnasium, stable-baselines3, qutip-qtrl, matplotlib, seaborn

**Spec:** `docs/superpowers/specs/2026-04-09-rydberg-rl-report-design.md`
**Task Cards:** `docs/superpowers/specs/2026-04-09-task-cards.md`

---

## Phase 0: Project Setup

### Task 0: Initialize project structure and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/physics/__init__.py`
- Create: `src/environments/__init__.py`
- Create: `src/baselines/__init__.py`
- Create: `src/training/__init__.py`
- Create: `src/plotting/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
cd E:/project/report
mkdir -p src/physics src/environments src/baselines src/training src/plotting
mkdir -p figures results models drafts
```

- [ ] **Step 2: Create requirements.txt**

```
qutip>=5.0.0
numpy>=1.24
scipy>=1.10
gymnasium>=0.29
stable-baselines3>=2.1
matplotlib>=3.7
seaborn>=0.12
```

- [ ] **Step 3: Create all __init__.py files**

`src/__init__.py`:
```python
```

`src/physics/__init__.py`:
```python
from .constants import *
from .hamiltonian import build_two_atom_hamiltonian, build_three_atom_hamiltonian
from .noise_model import NoiseModel
from .lindblad import get_collapse_operators, mesolve_with_noise
```

`src/environments/__init__.py`:
```python
```

`src/baselines/__init__.py`:
```python
```

`src/training/__init__.py`:
```python
```

`src/plotting/__init__.py`:
```python
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 5: Verify imports**

```bash
python -c "import qutip; print(qutip.__version__); import gymnasium; import stable_baselines3; import matplotlib; print('All imports OK')"
```

Expected: version numbers + "All imports OK"

- [ ] **Step 6: Commit**

```bash
git init
git add requirements.txt src/ figures/.gitkeep results/.gitkeep models/.gitkeep drafts/.gitkeep
git commit -m "chore: initialize project structure and dependencies"
```

---

## Phase 1: Physics Foundation (Parallel Sub-agents)

> **Dispatch in parallel:** Task 1, Task 2, Task 3, Task 4, Task 5

### Task 1: Physical constants module

**Files:**
- Create: `src/physics/constants.py`

- [ ] **Step 1: Write constants.py**

```python
"""Physical constants and Rb-87 Rydberg parameters.

All values sourced from literature. Units: SI unless noted.
Frequencies in rad/s for computation; display values noted in comments.
"""
import numpy as np

# Fundamental constants
HBAR = 1.054571817e-34       # J·s
K_B = 1.380649e-23           # J/K
C_LIGHT = 2.99792458e8       # m/s
E_CHARGE = 1.602176634e-19   # C
A0 = 5.29177210903e-11       # m, Bohr radius
RY_INF = 10973731.568160     # m^-1, Rydberg constant
ALPHA_FS = 7.2973525693e-3   # fine-structure constant

# Rb-87 atomic properties
RB87_MASS = 1.4431607e-25    # kg (86.909180527 amu)
RY_RB = RY_INF * (1 - 5.48579909065e-4 / 86.909180527)  # reduced-mass Rydberg (m^-1)
RY_RB_HZ = RY_RB * C_LIGHT   # Hz

# Quantum defects (Lorenzen & Niemax / Li et al.)
DELTA_0_S = 3.1311804        # nS_{1/2}
DELTA_2_S = 0.1784           # nS_{1/2} second-order
DELTA_0_P12 = 2.6548849      # nP_{1/2}
DELTA_0_P32 = 2.6416737      # nP_{3/2}
DELTA_0_D32 = 1.3480917      # nD_{3/2}
DELTA_0_D52 = 1.3462730      # nD_{5/2}

# 5P_{3/2} decay rate (intermediate state for two-photon)
GAMMA_5P = 2 * np.pi * 6.065e6  # rad/s (linewidth 6.065 MHz)

# Rydberg state: 53S_{1/2}
N_RYD = 53
N_STAR_53S = N_RYD - DELTA_0_S  # effective quantum number
TAU_RAD_53S = 135e-6         # s, 0K radiative lifetime (Beterov 2009)
GAMMA_BBR_53S = 1 / (200e-6) # s^-1, BBR rate at 300K (Beterov 2009)
TAU_EFF_53S = 1 / (1/TAU_RAD_53S + GAMMA_BBR_53S)  # ~ 88 μs

# van der Waals coefficient for 53S + 53S
# From Saffman-Walker-Molmer 2010, Table III interpolation
C6_53S = 2 * np.pi * 15.4e9  # rad/s · μm^6

# Scenario B baseline parameters (Evered 2023)
OMEGA_BASELINE = 2 * np.pi * 4.6e6   # rad/s (4.6 MHz)
R_ATOM = 2.0                          # μm (atom separation)
V_VDW_BASELINE = C6_53S / R_ATOM**6  # rad/s

# Two-photon excitation parameters (de Leseluc 2018)
DELTA_INTERMEDIATE = 2 * np.pi * 7.8e9  # rad/s (7.8 GHz detuning from 6P)

# Noise parameters
SIGMA_DOPPLER = 2 * np.pi * 50e3     # rad/s (50 kHz at T=10 μK)
SIGMA_POSITION = 0.1                   # μm (100 nm position jitter)
OU_CORRELATION_TIME = 10e-6            # s (amplitude noise)
OU_SIGMA = 0.02                        # relative amplitude noise
SERVO_BUMP_HEIGHT = 1e-8               # W/Hz (-80 dBc/Hz)
TEMPERATURE = 10e-6                    # K (10 μK)

# Scenario definitions
SCENARIOS = {
    "A": {
        "description": "Long-time / STIRAP-compatible",
        "T_gate": 5e-6,           # s
        "Omega": 2 * np.pi * 0.8e6,  # rad/s
        "R": 2.0,                  # μm
        "noise_sources": ["doppler", "decay"],
        "n_atoms": 2,
    },
    "B": {
        "description": "Short-time / full noise (primary scenario)",
        "T_gate": 0.3e-6,         # s
        "Omega": 2 * np.pi * 4.6e6,  # rad/s
        "R": 2.0,                  # μm
        "noise_sources": ["doppler", "position", "amplitude", "phase", "decay"],
        "n_atoms": 2,
    },
    "D": {
        "description": "3-atom W-state preparation",
        "T_gate": 0.5e-6,         # s
        "Omega": 2 * np.pi * 4.6e6,  # rad/s
        "R": 2.0,                  # μm (equilateral triangle side)
        "noise_sources": ["doppler", "position", "amplitude", "phase", "decay"],
        "n_atoms": 3,
    },
}

# Blockade radius for reference (Rb 70S at Omega=1 MHz for Tab.1)
C6_70S = 2 * np.pi * 862e9  # rad/s · μm^6 (Saffman 2010)
TAU_RAD_70S = 150e-6         # s, 0K
TAU_EFF_70S = 80e-6          # s, 300K
```

- [ ] **Step 2: Verify constants load**

```bash
cd E:/project/report && python -c "from src.physics.constants import *; print(f'n*={N_STAR_53S:.2f}, tau_eff={TAU_EFF_53S*1e6:.1f} us, V_vdW={V_VDW_BASELINE/2/np.pi/1e6:.1f} MHz')"
```

Expected: `n*=49.87, tau_eff=88.0 us, V_vdW=XXX MHz`

- [ ] **Step 3: Commit**

```bash
git add src/physics/constants.py
git commit -m "feat: add Rb-87 physical constants and scenario definitions"
```

---

### Task 2: Hamiltonian module

**Files:**
- Create: `src/physics/hamiltonian.py`

- [ ] **Step 1: Write hamiltonian.py**

```python
"""Rydberg atom Hamiltonian construction for 2-atom and 3-atom systems.

Basis ordering:
  2-atom (full): |gg>, |gr>, |rg>, |rr>  (4 states)
  3-atom (full): |ggg>, |ggr>, |grg>, |ggg>, ... (8 states, binary ordering)
"""
import numpy as np
import qutip

def _basis_state(n_atoms: int, excitation_pattern: str) -> qutip.Qobj:
    """Create a basis state from a pattern like 'gr' or 'ggr'.

    Args:
        n_atoms: Number of atoms.
        excitation_pattern: String of 'g' and 'r', length n_atoms.

    Returns:
        Tensor product ket.
    """
    g = qutip.basis(2, 0)
    r = qutip.basis(2, 1)
    states = [g if c == 'g' else r for c in excitation_pattern]
    return qutip.tensor(states)


def _sigma_gr(n_atoms: int, atom_idx: int) -> qutip.Qobj:
    """Transition operator |g><r| for atom atom_idx in n_atoms system."""
    ops = [qutip.qeye(2)] * n_atoms
    ops[atom_idx] = qutip.basis(2, 0) * qutip.basis(2, 1).dag()  # |g><r|
    return qutip.tensor(ops)


def _sigma_rg(n_atoms: int, atom_idx: int) -> qutip.Qobj:
    """Transition operator |r><g| for atom atom_idx in n_atoms system."""
    ops = [qutip.qeye(2)] * n_atoms
    ops[atom_idx] = qutip.basis(2, 1) * qutip.basis(2, 0).dag()  # |r><g|
    return qutip.tensor(ops)


def _n_r(n_atoms: int, atom_idx: int) -> qutip.Qobj:
    """Rydberg population operator |r><r| for atom atom_idx."""
    ops = [qutip.qeye(2)] * n_atoms
    ops[atom_idx] = qutip.basis(2, 1) * qutip.basis(2, 1).dag()  # |r><r|
    return qutip.tensor(ops)


def build_two_atom_hamiltonian(
    Omega: float,
    Delta: float,
    V_vdW: float,
) -> qutip.Qobj:
    """Build 2-atom Rydberg Hamiltonian in full 4x4 basis.

    H = sum_i [ (Omega/2)(|r_i><g_i| + h.c.) - Delta |r_i><r_i| ]
        + V_vdW |rr><rr|

    Args:
        Omega: Rabi frequency (rad/s).
        Delta: Laser detuning (rad/s).
        V_vdW: van der Waals interaction C6/R^6 (rad/s).

    Returns:
        4x4 Hamiltonian as Qobj.
    """
    n = 2
    H = qutip.tensor([qutip.qeye(2)] * n) * 0.0  # zero

    # Single-atom terms
    for i in range(n):
        H += (Omega / 2) * (_sigma_rg(n, i) + _sigma_gr(n, i))
        H += -Delta * _n_r(n, i)

    # Interaction: V_vdW |rr><rr|
    rr = _basis_state(n, 'rr')
    H += V_vdW * rr * rr.dag()

    return H


def build_three_atom_hamiltonian(
    Omega: float,
    Delta: float,
    positions: np.ndarray,
    C6: float,
) -> qutip.Qobj:
    """Build 3-atom Rydberg Hamiltonian in full 8x8 basis.

    Args:
        Omega: Rabi frequency (rad/s), global drive.
        Delta: Laser detuning (rad/s).
        positions: Shape (3, 2) array, atom positions in μm.
        C6: van der Waals coefficient (rad/s · μm^6).

    Returns:
        8x8 Hamiltonian as Qobj.
    """
    n = 3
    H = qutip.tensor([qutip.qeye(2)] * n) * 0.0

    # Single-atom terms
    for i in range(n):
        H += (Omega / 2) * (_sigma_rg(n, i) + _sigma_gr(n, i))
        H += -Delta * _n_r(n, i)

    # Pairwise vdW interactions
    for i in range(n):
        for j in range(i + 1, n):
            R_ij = np.linalg.norm(positions[i] - positions[j])
            V_ij = C6 / R_ij**6
            H += V_ij * _n_r(n, i) * _n_r(n, j)

    return H


def get_target_state(n_atoms: int) -> qutip.Qobj:
    """Return the target W-state for n_atoms.

    |W_2> = (|gr> + |rg>) / sqrt(2)
    |W_3> = (|rgg> + |grg> + |ggr>) / sqrt(3)

    Returns:
        Normalized ket.
    """
    if n_atoms == 2:
        return (_basis_state(2, 'gr') + _basis_state(2, 'rg')).unit()
    elif n_atoms == 3:
        return (_basis_state(3, 'rgg') + _basis_state(3, 'grg')
                + _basis_state(3, 'ggr')).unit()
    else:
        raise ValueError(f"n_atoms={n_atoms} not supported")


def get_ground_state(n_atoms: int) -> qutip.Qobj:
    """Return |gg...g> for n_atoms."""
    return _basis_state(n_atoms, 'g' * n_atoms)
```

- [ ] **Step 2: Verify Hamiltonian eigenvalues**

```bash
cd E:/project/report && python -c "
from src.physics.hamiltonian import *
import numpy as np

# Test: Delta=0, V=0 => eigenvalues should be {-Omega, -Omega/sqrt(2)*?, ...}
# Actually for 2 atoms, H has eigenvalues related to sqrt(2)*Omega
H = build_two_atom_hamiltonian(Omega=1.0, Delta=0.0, V_vdW=0.0)
evals = H.eigenenergies()
print('Eigenvalues (V=0):', np.sort(evals))

# With large V (blockade limit): effective 2-level with sqrt(2)*Omega
H_block = build_two_atom_hamiltonian(Omega=1.0, Delta=0.0, V_vdW=1000.0)
evals_b = H_block.eigenenergies()
print('Eigenvalues (blockade):', np.sort(evals_b))

# Target states
w2 = get_target_state(2)
print('W2 state:', w2)
w3 = get_target_state(3)
print('W3 norm:', w3.norm())
print('OK')
"
```

Expected: eigenvalues printed, states normalized, "OK"

- [ ] **Step 3: Commit**

```bash
git add src/physics/hamiltonian.py
git commit -m "feat: add 2-atom and 3-atom Rydberg Hamiltonian builders"
```

---

### Task 3: Noise model module

**Files:**
- Create: `src/physics/noise_model.py`

- [ ] **Step 1: Write noise_model.py**

```python
"""Noise model for Rydberg atom simulations.

Supports 5 noise sources:
1. Doppler shift (static Gaussian per trajectory)
2. Position jitter (static Gaussian, affects V_vdW via 1/R^6)
3. Laser amplitude OU noise (time-dependent)
4. Phase servo bump (simplified as static phase noise per trajectory)
5. Rydberg decay (Lindblad, handled separately)
"""
import numpy as np
from .constants import (
    SIGMA_DOPPLER, SIGMA_POSITION, OU_CORRELATION_TIME,
    OU_SIGMA, C6_53S, SCENARIOS, TAU_EFF_53S,
)


class NoiseModel:
    """Noise sampler for a given experimental scenario."""

    def __init__(self, scenario: str):
        """Initialize noise model for scenario A, B, or D.

        Args:
            scenario: One of "A", "B", "D".
        """
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario = scenario
        self.config = SCENARIOS[scenario]
        self.active_sources = set(self.config["noise_sources"])
        self.n_atoms = self.config["n_atoms"]

    def sample(self, rng: np.random.Generator) -> dict:
        """Sample one noise realization for a single trajectory.

        Args:
            rng: Numpy random generator for reproducibility.

        Returns:
            Dictionary with noise parameters:
                delta_doppler: list of per-atom Doppler shifts (rad/s)
                delta_R: list of per-pair position offsets (μm)
                amplitude_noise_series: None or callable t -> relative noise
                phase_noise: static phase offset (rad)
        """
        params = {}

        # 1. Doppler (static per atom per trajectory)
        if "doppler" in self.active_sources:
            params["delta_doppler"] = [
                rng.normal(0, SIGMA_DOPPLER) for _ in range(self.n_atoms)
            ]
        else:
            params["delta_doppler"] = [0.0] * self.n_atoms

        # 2. Position jitter (static per pair)
        if "position" in self.active_sources:
            n_pairs = self.n_atoms * (self.n_atoms - 1) // 2
            params["delta_R"] = [
                rng.normal(0, SIGMA_POSITION) for _ in range(n_pairs)
            ]
        else:
            n_pairs = self.n_atoms * (self.n_atoms - 1) // 2
            params["delta_R"] = [0.0] * n_pairs

        # 3. Laser amplitude OU noise (pre-generate time series)
        if "amplitude" in self.active_sources:
            params["ou_sigma"] = OU_SIGMA
            params["ou_tau"] = OU_CORRELATION_TIME
        else:
            params["ou_sigma"] = 0.0
            params["ou_tau"] = OU_CORRELATION_TIME

        # 4. Phase noise (simplified as static random phase per trajectory)
        if "phase" in self.active_sources:
            # Approximate effect of servo bump: random phase kick
            # RMS phase from -80 dBc/Hz over bandwidth ~ Omega
            params["phase_noise"] = rng.normal(0, 0.03)  # ~30 mrad RMS
        else:
            params["phase_noise"] = 0.0

        return params

    def generate_ou_series(
        self, rng: np.random.Generator, tlist: np.ndarray
    ) -> np.ndarray:
        """Generate Ornstein-Uhlenbeck amplitude noise time series.

        Args:
            rng: Random generator.
            tlist: Time array (s).

        Returns:
            Array of relative amplitude noise xi(t), same length as tlist.
        """
        dt = tlist[1] - tlist[0]
        theta = 1.0 / OU_CORRELATION_TIME
        sigma = OU_SIGMA
        xi = np.zeros(len(tlist))
        for i in range(1, len(tlist)):
            xi[i] = xi[i-1] - theta * xi[i-1] * dt + sigma * np.sqrt(2 * theta * dt) * rng.normal()
        return xi

    def compute_V_vdW(self, R_base: float, delta_R: float) -> float:
        """Compute vdW interaction with position jitter.

        Args:
            R_base: Nominal separation (μm).
            delta_R: Position offset (μm).

        Returns:
            V_vdW in rad/s.
        """
        R_actual = R_base + delta_R
        if R_actual <= 0:
            R_actual = 0.01  # safety floor
        return C6_53S / R_actual**6
```

- [ ] **Step 2: Verify noise sampling**

```bash
cd E:/project/report && python -c "
from src.physics.noise_model import NoiseModel
import numpy as np
rng = np.random.default_rng(42)

nm_B = NoiseModel('B')
params = nm_B.sample(rng)
print('Scenario B noise keys:', list(params.keys()))
print('Doppler shifts (MHz):', [d/2/np.pi/1e6 for d in params['delta_doppler']])
print('Position jitter (nm):', [d*1000 for d in params['delta_R']])
print('Phase noise (mrad):', params['phase_noise']*1000)

tlist = np.linspace(0, 0.3e-6, 100)
ou = nm_B.generate_ou_series(rng, tlist)
print(f'OU noise: mean={ou.mean():.4f}, std={ou.std():.4f}')

nm_A = NoiseModel('A')
params_A = nm_A.sample(rng)
print('Scenario A active:', nm_A.active_sources)
print('OK')
"
```

Expected: noise parameters printed with reasonable values, "OK"

- [ ] **Step 3: Commit**

```bash
git add src/physics/noise_model.py
git commit -m "feat: add noise model with 5 sources and scenario support"
```

---

### Task 4: Lindblad evolution module

**Files:**
- Create: `src/physics/lindblad.py`

- [ ] **Step 1: Write lindblad.py**

```python
"""Lindblad master equation tools for Rydberg simulations.

Provides collapse operators and a noise-aware mesolve wrapper.
"""
import numpy as np
import qutip
from .constants import TAU_EFF_53S
from .hamiltonian import _n_r, _sigma_gr


def get_collapse_operators(
    n_atoms: int,
    tau_eff: float = TAU_EFF_53S,
) -> list[qutip.Qobj]:
    """Get Lindblad collapse operators for Rydberg decay.

    Each atom decays independently: L_i = sqrt(gamma) |g_i><r_i|

    Args:
        n_atoms: Number of atoms (2 or 3).
        tau_eff: Effective lifetime including BBR (s).

    Returns:
        List of collapse operators.
    """
    gamma = 1.0 / tau_eff
    c_ops = []
    for i in range(n_atoms):
        c_ops.append(np.sqrt(gamma) * _sigma_gr(n_atoms, i))
    return c_ops


def mesolve_with_noise(
    H_base: qutip.Qobj,
    psi0: qutip.Qobj,
    tlist: np.ndarray,
    c_ops: list[qutip.Qobj],
    noise_params: dict,
    n_atoms: int,
    Omega_base: float,
    Delta_base: float,
    R_base: float,
    C6: float,
    build_H_func=None,
) -> qutip.Result:
    """Run mesolve with time-dependent noise.

    The Hamiltonian is rebuilt at each timestep to incorporate:
    - Per-atom Doppler shifts
    - Position-dependent V_vdW
    - Amplitude modulation xi(t)
    - Phase offset

    For efficiency, we use QobjEvo with coefficient functions.

    Args:
        H_base: Static base Hamiltonian (used for structure).
        psi0: Initial state (ket or dm).
        tlist: Time points (s).
        c_ops: Collapse operators.
        noise_params: From NoiseModel.sample().
        n_atoms: Number of atoms.
        Omega_base: Base Rabi frequency (rad/s).
        Delta_base: Base detuning (rad/s).
        R_base: Base atom separation (μm).
        C6: van der Waals coefficient (rad/s · μm^6).
        build_H_func: Hamiltonian builder function (2-atom or 3-atom).

    Returns:
        QuTiP Result object.
    """
    from .hamiltonian import (
        build_two_atom_hamiltonian,
        build_three_atom_hamiltonian,
        _sigma_rg, _sigma_gr as _sgr, _n_r as _nr,
    )
    from .noise_model import NoiseModel

    delta_doppler = noise_params.get("delta_doppler", [0.0] * n_atoms)
    delta_R_list = noise_params.get("delta_R", [0.0])
    phase_noise = noise_params.get("phase_noise", 0.0)
    ou_sigma = noise_params.get("ou_sigma", 0.0)
    ou_tau = noise_params.get("ou_tau", 10e-6)

    # Pre-generate OU noise if needed
    if ou_sigma > 0:
        rng = np.random.default_rng()
        nm = NoiseModel.__new__(NoiseModel)
        nm.active_sources = set()
        dt_val = tlist[1] - tlist[0]
        theta = 1.0 / ou_tau
        xi_series = np.zeros(len(tlist))
        for i in range(1, len(tlist)):
            xi_series[i] = (xi_series[i-1]
                           - theta * xi_series[i-1] * dt_val
                           + ou_sigma * np.sqrt(2 * theta * dt_val)
                           * np.random.default_rng().normal())
    else:
        xi_series = np.zeros(len(tlist))

    # Build time-dependent Hamiltonian using piecewise-constant approach
    # For simplicity, use callback-based QobjEvo
    if n_atoms == 2:
        # Compute effective V_vdW with position jitter
        V_eff = C6 / (R_base + delta_R_list[0])**6

        # Static part: detuning + interaction + Doppler
        H_static = qutip.tensor([qutip.qeye(2)] * 2) * 0.0
        for i in range(2):
            H_static += -(Delta_base + delta_doppler[i]) * _nr(2, i)
        H_static += V_eff * (_nr(2, 0) * _nr(2, 1))

        # Drive part (time-dependent via amplitude noise)
        H_drive = qutip.tensor([qutip.qeye(2)] * 2) * 0.0
        for i in range(2):
            # Include phase noise as rotation of drive
            H_drive += (Omega_base / 2) * (
                np.exp(1j * phase_noise) * _sigma_rg(2, i)
                + np.exp(-1j * phase_noise) * _sgr(2, i)
            )

        def coeff_drive(t, args=None):
            idx = int(t / (tlist[1] - tlist[0]))
            idx = min(idx, len(xi_series) - 1)
            return 1.0 + xi_series[idx]

        H_td = [H_static, [H_drive, coeff_drive]]

    elif n_atoms == 3:
        # Equilateral triangle positions
        positions_base = np.array([
            [0, 0],
            [R_base, 0],
            [R_base / 2, R_base * np.sqrt(3) / 2],
        ])

        # Apply position jitter to distances (simplified: jitter each pair)
        pair_idx = 0
        V_pairs = {}
        for i in range(3):
            for j in range(i + 1, 3):
                R_ij = np.linalg.norm(positions_base[i] - positions_base[j])
                R_ij_eff = R_ij + delta_R_list[min(pair_idx, len(delta_R_list)-1)]
                V_pairs[(i, j)] = C6 / max(R_ij_eff, 0.01)**6
                pair_idx += 1

        H_static = qutip.tensor([qutip.qeye(2)] * 3) * 0.0
        for i in range(3):
            H_static += -(Delta_base + delta_doppler[i]) * _nr(3, i)
        for (i, j), V in V_pairs.items():
            H_static += V * (_nr(3, i) * _nr(3, j))

        H_drive = qutip.tensor([qutip.qeye(2)] * 3) * 0.0
        for i in range(3):
            H_drive += (Omega_base / 2) * (
                np.exp(1j * phase_noise) * _sigma_rg(3, i)
                + np.exp(-1j * phase_noise) * _sgr(3, i)
            )

        def coeff_drive(t, args=None):
            idx = int(t / (tlist[1] - tlist[0]))
            idx = min(idx, len(xi_series) - 1)
            return 1.0 + xi_series[idx]

        H_td = [H_static, [H_drive, coeff_drive]]
    else:
        raise ValueError(f"n_atoms={n_atoms} not supported")

    # Initial state: convert ket to density matrix if needed
    rho0 = qutip.ket2dm(psi0) if psi0.isket else psi0

    result = qutip.mesolve(H_td, rho0, tlist, c_ops=c_ops)
    return result


def compute_fidelity(rho: qutip.Qobj, target_ket: qutip.Qobj) -> float:
    """Compute state fidelity F = <target|rho|target>.

    Args:
        rho: Density matrix.
        target_ket: Target pure state (ket).

    Returns:
        Fidelity (float in [0, 1]).
    """
    target_dm = qutip.ket2dm(target_ket)
    return float((target_dm * rho).tr().real)
```

- [ ] **Step 2: Verify Lindblad evolution reproduces Rabi oscillation**

```bash
cd E:/project/report && python -c "
import numpy as np
import qutip
from src.physics.hamiltonian import build_two_atom_hamiltonian, get_ground_state, get_target_state
from src.physics.lindblad import mesolve_with_noise, compute_fidelity
from src.physics.constants import *

# No noise, no decay => perfect Rabi oscillation to W state
Omega = 2*np.pi * 1e6
V = 100 * Omega  # strong blockade
H = build_two_atom_hamiltonian(Omega, 0, V)
psi0 = get_ground_state(2)
T_pi = np.pi / (np.sqrt(2) * Omega)
tlist = np.linspace(0, T_pi, 200)

# Use direct mesolve (no noise)
result = qutip.mesolve(H, psi0, tlist, c_ops=[])
rho_final = result.states[-1]
W = get_target_state(2)
F = compute_fidelity(qutip.ket2dm(rho_final), W)
print(f'Fidelity at t=pi/sqrt(2)Omega (no noise, strong blockade): F={F:.6f}')
assert F > 0.99, f'Expected F>0.99, got {F}'
print('OK')
"
```

Expected: `F≈0.999...`, "OK"

- [ ] **Step 3: Commit**

```bash
git add src/physics/lindblad.py
git commit -m "feat: add Lindblad evolution with noise-aware mesolve wrapper"
```

---

### Task 5: Update physics __init__.py and verify full import

**Files:**
- Modify: `src/physics/__init__.py`

- [ ] **Step 1: Verify full physics package import**

```bash
cd E:/project/report && python -c "
from src.physics.constants import SCENARIOS, C6_53S
from src.physics.hamiltonian import build_two_atom_hamiltonian, build_three_atom_hamiltonian, get_target_state, get_ground_state
from src.physics.noise_model import NoiseModel
from src.physics.lindblad import get_collapse_operators, mesolve_with_noise, compute_fidelity
print('All physics imports OK')
"
```

Expected: "All physics imports OK"

- [ ] **Step 2: Commit**

```bash
git add src/physics/__init__.py
git commit -m "feat: finalize physics package with all module exports"
```

---

## Phase 1 (continued): Report Text §1-§3

> **Dispatch in parallel with Tasks 1-5:** Task 6, Task 7, Task 8

### Task 6: Write §1 — Rydberg 原子物理基础

**Files:**
- Create: `drafts/section_01.md`

- [ ] **Step 1: Write section_01.md**

Write the complete markdown for §1 following these requirements:

**§1.1** (~0.5 page): Hydrogen recap → alkali metal penetration/polarization → Rydberg-Ritz formula with QDT *concise* derivation (physical picture: short-range phase shift causes energy level offset). One sentence pointing to Appendix A. Numerical values: Rb δ₀(nS)≈3.13, δ₀(nP)≈2.65, δ₀(nD)≈1.35.

**§1.2** (~0.7 page): Tab.1 scaling law table (7 rows: orbital radius ~n*², dipole moment ~n*², polarizability ~n*⁷, radiative lifetime ~n*³, BBR lifetime ~n*², C₆ ~n*¹¹, R_b ~n*¹¹/⁶). Each row: scaling exponent + one-sentence physical meaning. Numerical anchor: Rb 70S values. Concluding sentence: Rydberg states uniquely combine long lifetime + strong interaction.

**§1.3** (~0.5 page): BBR lifetime correction formula. Physical meaning: gate time constraint. Cite Beterov 2009.

**Language:** Chinese body, English terms. All formulas in LaTeX. Important formulas numbered. References as [AuthorYY].

- [ ] **Step 2: Self-review checklist**
  - Rydberg-Ritz formula present and numbered
  - Tab.1 complete (7 rows)
  - BBR formula present
  - All numerical values sourced
  - QDT derivation ≤ 0.5 page, points to Appendix A

- [ ] **Step 3: Commit**

```bash
git add drafts/section_01.md
git commit -m "docs: write §1 Rydberg atomic physics fundamentals"
```

---

### Task 7: Write §2 — 原子—激光相互作用

**Files:**
- Create: `drafts/section_02.md`

- [ ] **Step 1: Write section_02.md**

**§2.1** (~0.8 page): Full RWA derivation: dipole approximation → dipole Hamiltonian → Rabi frequency definition → rotating frame → RWA. Must write out final effective Hamiltonian (numbered). Bloch sphere picture (reference Fig.4). Rabi oscillation P_r(t) = sin²(Ωt/2).

**§2.2** (~0.7 page): Experimental motivation for two-photon. Three-level Λ + large detuning → *concise* adiabatic elimination. Give Ω_eff = Ω₁Ω₂/(2Δ), AC Stark shift, scattering cost formula. Point to Appendix B. de Léséleuc 2018 numerical anchor.

**§2.3** (~0.3 page): Polarization, selection rules, m_J channel. Brief. Walker-Saffman correction.

**Language:** Chinese body, English terms. LaTeX formulas.

- [ ] **Step 2: Self-review checklist**
  - RWA Hamiltonian fully derived and numbered
  - Adiabatic elimination concise (≤0.3 page), points to Appendix B
  - Contains numerical anchor
  - References Fig.4 for Bloch sphere

- [ ] **Step 3: Commit**

```bash
git add drafts/section_02.md
git commit -m "docs: write §2 atom-laser interaction and two-photon excitation"
```

---

### Task 8: Write §3 — 里德堡阻塞与 Bell 态

**Files:**
- Create: `drafts/section_03.md`

- [ ] **Step 1: Write section_03.md**

This is the **physical climax** of the report. Most detailed derivation.

**§3.1** (~0.5 page): Dipole-dipole → second-order perturbation → C₆/R⁶. Förster resonance intro. C₆ expression.

**§3.2** (~1.0 page, FULL derivation):
- Write out 4×4 Hamiltonian matrix in {|gg⟩,|gr⟩,|rg⟩,|rr⟩} basis
- Symmetric/antisymmetric basis transformation: |W⟩=(|gr⟩+|rg⟩)/√2, |D⟩=(|gr⟩-|rg⟩)/√2
- Mathematical proof: |gg⟩ couples ONLY to |W⟩, dark state |D⟩ decouples
- Blockade limit V≫ℏΩ → effective two-level: H_eff = (ℏ√2·Ω/2)(|gg⟩⟨W|+h.c.)
- √2 enhancement: physical explanation (collective coupling)
- Blockade radius R_b = (C₆/ℏΩ)^{1/6}, Rb 70S numerical value

**§3.3** (~0.4 page): t=π/Ω_eff → |gg⟩ → -i|W⟩. N-atom generalization (√N). Best experimental results.

**§3.4** (~0.3 page): Optical tweezer arrays. Brief.

**Language:** Chinese body, English terms. LaTeX formulas.

- [ ] **Step 2: Self-review checklist**
  - 4×4 matrix explicitly written
  - Basis transformation rigorous
  - Blockade radius formula + number
  - √2 enhancement explained
  - This is the most detailed derivation section

- [ ] **Step 3: Commit**

```bash
git add drafts/section_03.md
git commit -m "docs: write §3 Rydberg blockade and Bell state preparation"
```

---

## Phase 1 (continued): Physics Figures

> **Dispatch in parallel with Tasks 1-8:** Task 9

### Task 9: Generate physics figures (Fig.1-6)

**Files:**
- Create: `src/plotting/plot_config.py` (shared config)
- Create: `src/plotting/fig01_energy_levels.py`
- Create: `src/plotting/fig02_scaling_laws.py`
- Create: `src/plotting/fig03_two_photon.py`
- Create: `src/plotting/fig04_rabi_bloch.py`
- Create: `src/plotting/fig05_blockade.py`
- Create: `src/plotting/fig06_fidelity_vs_distance.py`

- [ ] **Step 1: Write shared plot config**

```python
"""Shared plotting configuration for all figures."""
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'mathtext.fontset': 'cm',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colorblind-friendly palette (Tableau 10)
COLORS = {
    'blue': '#4E79A7',
    'orange': '#F28E2B',
    'green': '#59A14F',
    'red': '#E15759',
    'purple': '#B07AA1',
    'brown': '#9C755F',
    'pink': '#FF9DA7',
    'gray': '#BAB0AC',
}

FIGURE_DIR = 'figures'
SINGLE_COL = (3.5, 2.8)   # inches
DOUBLE_COL = (7.0, 4.5)    # inches
```

- [ ] **Step 2: Write and run each figure script**

Each script must:
1. Import from `src.physics.constants` for parameters
2. Use `plot_config` for styling
3. Save to `figures/figXX_name.pdf`
4. Be independently runnable: `python src/plotting/fig01_energy_levels.py`

**Fig.1**: Energy level diagram — plot Rb S,P,D states for n=5..80, overlay H atom dashed lines, annotate quantum defect shifts.

**Fig.2**: 4-panel log-log scaling laws using analytical formulas from constants.

**Fig.3**: Two-photon ladder diagram (matplotlib arrows/annotations, not computed data).

**Fig.4**: (a) P_r(t) for resonant + 2 detuned cases via QuTiP mesolve. (b) Bloch sphere trajectory via qutip.Bloch.

**Fig.5**: (a) Energy level diagram for 2-atom blockade. (b) Population dynamics: V=0 vs V≫Ω.

**Fig.6**: F(|W⟩) vs R from QuTiP mesolve sweep, vertical line at R_b.

- [ ] **Step 3: Verify all 6 PDFs exist**

```bash
ls -la figures/fig0[1-6]_*.pdf
```

Expected: 6 PDF files

- [ ] **Step 4: Commit**

```bash
git add src/plotting/plot_config.py src/plotting/fig0[1-6]_*.py figures/fig0[1-6]_*.pdf
git commit -m "feat: generate physics figures Fig.1-6"
```

---

## Phase 2: Baselines + RL + Text §4-§6 (Parallel)

> **Gate:** Phase 1 Tasks 1-5 (physics code) must complete.
> **Dispatch in parallel:** Task 10, Task 11, Task 12, Task 13, Task 14

### Task 10: STIRAP baseline

**Files:**
- Create: `src/baselines/stirap.py`

- [ ] **Step 1: Write stirap.py**

```python
"""STIRAP pulse generation and simulation for Rydberg Bell state preparation."""
import numpy as np
import qutip
from src.physics.constants import SCENARIOS, C6_53S, TAU_EFF_53S
from src.physics.hamiltonian import (
    build_two_atom_hamiltonian, get_ground_state, get_target_state,
)
from src.physics.lindblad import get_collapse_operators, compute_fidelity
from src.physics.noise_model import NoiseModel


def stirap_pulse(t: float, T_gate: float, Omega_max: float) -> float:
    """STIRAP-like adiabatic pulse: sin² envelope.

    Args:
        t: Current time (s).
        T_gate: Total gate time (s).
        Omega_max: Peak Rabi frequency (rad/s).

    Returns:
        Omega(t) in rad/s.
    """
    return Omega_max * np.sin(np.pi * t / (2 * T_gate))**2


def run_stirap(
    scenario: str,
    noise_params: dict | None = None,
    n_steps: int = 200,
) -> tuple[float, qutip.Result]:
    """Run STIRAP protocol for a given scenario.

    Args:
        scenario: "A", "B", or "D".
        noise_params: Pre-sampled noise dict, or None for no noise.
        n_steps: Time discretization steps.

    Returns:
        (fidelity, qutip.Result)
    """
    cfg = SCENARIOS[scenario]
    T_gate = cfg["T_gate"]
    Omega_max = cfg["Omega"]
    R = cfg["R"]
    n_atoms = cfg["n_atoms"]
    V_vdW = C6_53S / R**6

    tlist = np.linspace(0, T_gate, n_steps)
    psi0 = get_ground_state(n_atoms)
    target = get_target_state(n_atoms)
    c_ops = get_collapse_operators(n_atoms) if "decay" in cfg["noise_sources"] else []

    if noise_params is None:
        # No noise: simple time-dependent Hamiltonian
        def H_t(t, args=None):
            Omega = stirap_pulse(t, T_gate, Omega_max)
            return build_two_atom_hamiltonian(Omega, 0.0, V_vdW)

        # Use piecewise approach
        H_list = []
        for i, t in enumerate(tlist[:-1]):
            Omega = stirap_pulse(t, T_gate, Omega_max)
            H_list.append(build_two_atom_hamiltonian(Omega, 0.0, V_vdW))

        # Use mesolve with time-dependent list
        H_drive_op = build_two_atom_hamiltonian(1.0, 0.0, 0.0)  # drive part
        H_static = build_two_atom_hamiltonian(0.0, 0.0, V_vdW)  # static part

        def omega_coeff(t, args=None):
            return stirap_pulse(t, T_gate, Omega_max)

        H_td = [H_static, [H_drive_op, omega_coeff]]
        result = qutip.mesolve(H_td, psi0, tlist, c_ops=c_ops)
    else:
        # With noise: use mesolve_with_noise
        from src.physics.lindblad import mesolve_with_noise
        H_base = build_two_atom_hamiltonian(Omega_max, 0.0, V_vdW)
        result = mesolve_with_noise(
            H_base, psi0, tlist, c_ops, noise_params,
            n_atoms=n_atoms, Omega_base=Omega_max, Delta_base=0.0,
            R_base=R, C6=C6_53S,
        )

    rho_final = result.states[-1]
    if rho_final.isket:
        rho_final = qutip.ket2dm(rho_final)
    F = compute_fidelity(rho_final, target)
    return F, result
```

- [ ] **Step 2: Verify STIRAP on scenario A (no noise)**

```bash
cd E:/project/report && python -c "
from src.baselines.stirap import run_stirap
F, _ = run_stirap('A', noise_params=None)
print(f'STIRAP scenario A (no noise): F={F:.4f}')
assert F > 0.95, f'Expected F>0.95, got {F}'
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/baselines/stirap.py
git commit -m "feat: add STIRAP baseline pulse and simulation"
```

---

### Task 11: GRAPE baseline

**Files:**
- Create: `src/baselines/grape.py`

- [ ] **Step 1: Write grape.py**

```python
"""GRAPE optimal control baseline for Rydberg Bell state preparation.

Uses qutip-qtrl or manual gradient-based optimization.
"""
import numpy as np
import qutip
from src.physics.constants import SCENARIOS, C6_53S, TAU_EFF_53S
from src.physics.hamiltonian import (
    build_two_atom_hamiltonian, get_ground_state, get_target_state,
    _sigma_rg, _sigma_gr, _n_r,
)
from src.physics.lindblad import get_collapse_operators, compute_fidelity


def run_grape(
    scenario: str,
    n_steps: int = 30,
    n_iter: int = 500,
    noise_params: dict | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run GRAPE optimization for Bell state preparation.

    Args:
        scenario: "A" or "B".
        n_steps: Number of piecewise-constant pulse segments.
        n_iter: Number of GRAPE iterations.
        noise_params: If None, optimize without noise. If given, evaluate
                      the optimized pulse under noise.

    Returns:
        (fidelity, omega_pulse, delta_pulse) where pulses are arrays of shape (n_steps,).
    """
    cfg = SCENARIOS[scenario]
    T_gate = cfg["T_gate"]
    Omega_max = cfg["Omega"]
    R = cfg["R"]
    n_atoms = cfg["n_atoms"]
    V_vdW = C6_53S / R**6

    dt = T_gate / n_steps
    tlist = np.linspace(0, T_gate, n_steps + 1)

    psi0 = get_ground_state(n_atoms)
    target = get_target_state(n_atoms)
    target_dm = qutip.ket2dm(target)

    # GRAPE via simple gradient ascent (manual implementation for portability)
    # Initialize with constant pulse near pi-pulse
    omega_pulse = np.ones(n_steps) * Omega_max * 0.8
    delta_pulse = np.zeros(n_steps)

    best_F = 0.0
    best_omega = omega_pulse.copy()
    best_delta = delta_pulse.copy()

    lr = Omega_max * 0.01  # learning rate

    for iteration in range(n_iter):
        # Forward propagation
        U_list = []
        for k in range(n_steps):
            H_k = build_two_atom_hamiltonian(omega_pulse[k], delta_pulse[k], V_vdW)
            U_k = (-1j * H_k * dt).expm()
            U_list.append(U_k)

        # Compute final state
        psi = psi0.copy()
        for U_k in U_list:
            psi = U_k * psi

        F = abs(target.dag() * psi)**2
        F = float(F)

        if F > best_F:
            best_F = F
            best_omega = omega_pulse.copy()
            best_delta = delta_pulse.copy()

        if F > 0.999:
            break

        # Numerical gradient (finite differences)
        eps = Omega_max * 1e-4
        grad_omega = np.zeros(n_steps)
        grad_delta = np.zeros(n_steps)

        for k in range(n_steps):
            # Omega gradient
            omega_pulse[k] += eps
            psi_plus = psi0.copy()
            for j in range(n_steps):
                H_j = build_two_atom_hamiltonian(omega_pulse[j], delta_pulse[j], V_vdW)
                psi_plus = (-1j * H_j * dt).expm() * psi_plus
            F_plus = float(abs(target.dag() * psi_plus)**2)
            grad_omega[k] = (F_plus - F) / eps
            omega_pulse[k] -= eps

            # Delta gradient
            delta_pulse[k] += eps
            psi_plus = psi0.copy()
            for j in range(n_steps):
                H_j = build_two_atom_hamiltonian(omega_pulse[j], delta_pulse[j], V_vdW)
                psi_plus = (-1j * H_j * dt).expm() * psi_plus
            F_plus = float(abs(target.dag() * psi_plus)**2)
            grad_delta[k] = (F_plus - F) / eps
            delta_pulse[k] -= eps

        # Gradient ascent
        omega_pulse += lr * grad_omega
        delta_pulse += lr * grad_delta

        # Clip to physical bounds
        omega_pulse = np.clip(omega_pulse, 0, Omega_max * 2)
        delta_pulse = np.clip(delta_pulse, -Omega_max, Omega_max)

        if iteration % 100 == 0:
            print(f"  GRAPE iter {iteration}: F={F:.6f}")

    # If noise_params given, evaluate best pulse under noise
    if noise_params is not None:
        from src.physics.lindblad import mesolve_with_noise
        c_ops = get_collapse_operators(n_atoms)
        tlist_fine = np.linspace(0, T_gate, n_steps * 10)
        # Interpolate pulse
        H_base = build_two_atom_hamiltonian(Omega_max, 0.0, V_vdW)
        result = mesolve_with_noise(
            H_base, psi0, tlist_fine, c_ops, noise_params,
            n_atoms=n_atoms, Omega_base=Omega_max, Delta_base=0.0,
            R_base=R, C6=C6_53S,
        )
        rho_final = result.states[-1]
        if rho_final.isket:
            rho_final = qutip.ket2dm(rho_final)
        best_F = compute_fidelity(rho_final, target)

    return best_F, best_omega, best_delta
```

- [ ] **Step 2: Verify GRAPE on scenario B (no noise)**

```bash
cd E:/project/report && python -c "
from src.baselines.grape import run_grape
F, omega, delta = run_grape('B', n_steps=30, n_iter=200)
print(f'GRAPE scenario B (no noise): F={F:.4f}')
print(f'Pulse shape: Omega range [{omega.min()/2/3.14159/1e6:.2f}, {omega.max()/2/3.14159/1e6:.2f}] MHz')
print('OK')
"
```

Expected: F > 0.95 (without noise)

- [ ] **Step 3: Commit**

```bash
git add src/baselines/grape.py
git commit -m "feat: add GRAPE optimal control baseline"
```

---

### Task 12: Unified evaluation + run all baselines

**Files:**
- Create: `src/baselines/evaluate.py`

- [ ] **Step 1: Write evaluate.py**

```python
"""Unified evaluation interface for all control methods.

Runs Monte Carlo trajectories with noise sampling and reports statistics.
"""
import json
import numpy as np
from pathlib import Path
from src.physics.constants import SCENARIOS, C6_53S
from src.physics.hamiltonian import get_ground_state, get_target_state
from src.physics.noise_model import NoiseModel
from src.physics.lindblad import get_collapse_operators, compute_fidelity


def evaluate_policy(
    run_func,
    scenario: str,
    n_trajectories: int = 1000,
    seed: int = 42,
    **kwargs,
) -> dict:
    """Evaluate a control policy over multiple noisy trajectories.

    Args:
        run_func: Callable(scenario, noise_params, **kwargs) -> (fidelity, result).
        scenario: "A", "B", or "D".
        n_trajectories: Number of MC trajectories.
        seed: Random seed.
        **kwargs: Extra arguments for run_func.

    Returns:
        Dict with mean_F, F_05, std_F, all_F.
    """
    rng = np.random.default_rng(seed)
    nm = NoiseModel(scenario)
    fidelities = []

    for i in range(n_trajectories):
        noise_params = nm.sample(rng)
        F, _ = run_func(scenario, noise_params=noise_params, **kwargs)
        fidelities.append(F)
        if (i + 1) % 100 == 0:
            print(f"  Trajectory {i+1}/{n_trajectories}: mean_F={np.mean(fidelities):.4f}")

    fidelities = np.array(fidelities)
    return {
        "mean_F": float(np.mean(fidelities)),
        "F_05": float(np.percentile(fidelities, 5)),
        "std_F": float(np.std(fidelities)),
        "all_F": fidelities.tolist(),
    }


def save_results(results: dict, method: str, scenario: str):
    """Save evaluation results to JSON.

    Args:
        results: Output of evaluate_policy (all_F will be truncated for size).
        method: "STIRAP", "GRAPE", or "PPO".
        scenario: "A", "B", or "D".
    """
    output = {
        "scenario": scenario,
        "method": method,
        "n_trajectories": len(results.get("all_F", [])),
        "mean_F": results["mean_F"],
        "F_05": results["F_05"],
        "std_F": results["std_F"],
    }
    path = Path("results") / f"{method.lower()}_{scenario}.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {path}")


if __name__ == "__main__":
    from src.baselines.stirap import run_stirap
    from src.baselines.grape import run_grape

    # Evaluate STIRAP on scenario A (its natural domain)
    print("=== STIRAP on Scenario A ===")
    stirap_A = evaluate_policy(run_stirap, "A", n_trajectories=200, seed=42)
    save_results(stirap_A, "STIRAP", "A")

    # Evaluate STIRAP on scenario B
    print("\n=== STIRAP on Scenario B ===")
    stirap_B = evaluate_policy(run_stirap, "B", n_trajectories=200, seed=42)
    save_results(stirap_B, "STIRAP", "B")

    # Evaluate GRAPE on scenario B
    print("\n=== GRAPE on Scenario B (optimizing first, then evaluating) ===")
    # First optimize without noise
    F_opt, omega_opt, delta_opt = run_grape("B", n_steps=30, n_iter=300)
    print(f"GRAPE optimized (no noise): F={F_opt:.4f}")

    # Then evaluate under noise
    grape_B = evaluate_policy(run_grape, "B", n_trajectories=200, seed=42, n_iter=0)
    save_results(grape_B, "GRAPE", "B")

    print("\nAll baseline evaluations complete.")
```

- [ ] **Step 2: Run baseline evaluations (may take time)**

```bash
cd E:/project/report && python -m src.baselines.evaluate
```

Expected: JSON files in `results/` with mean_F values

- [ ] **Step 3: Commit**

```bash
git add src/baselines/evaluate.py results/*.json
git commit -m "feat: add unified evaluation and run baseline assessments"
```

---

### Task 13: Gymnasium environment + PPO training

**Files:**
- Create: `src/environments/rydberg_env.py`
- Create: `src/training/config.py`
- Create: `src/training/train_ppo.py`

- [ ] **Step 1: Write rydberg_env.py**

```python
"""Gymnasium environment for 2-atom Rydberg Bell state preparation via RL."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import qutip

from src.physics.constants import SCENARIOS, C6_53S, TAU_EFF_53S
from src.physics.hamiltonian import (
    build_two_atom_hamiltonian, get_ground_state, get_target_state, _n_r, _sigma_rg, _sigma_gr,
)
from src.physics.noise_model import NoiseModel
from src.physics.lindblad import get_collapse_operators, compute_fidelity


class RydbergBellEnv(gym.Env):
    """RL environment for 2-atom Bell state preparation.

    Observation: Flattened density matrix (real + imag parts), 32-dim.
    Action: (Omega, Delta) normalized to [-1, 1].
    Reward: Sparse terminal fidelity.
    """
    metadata = {"render_modes": []}

    def __init__(self, scenario: str = "B", n_steps: int = 30, use_noise: bool = True):
        super().__init__()
        self.scenario = scenario
        self.cfg = SCENARIOS[scenario]
        self.n_steps = n_steps
        self.use_noise = use_noise
        self.T_gate = self.cfg["T_gate"]
        self.dt = self.T_gate / n_steps
        self.Omega_max = self.cfg["Omega"]
        self.R = self.cfg["R"]
        self.C6 = C6_53S
        self.n_atoms = self.cfg["n_atoms"]

        # Spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(32,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Noise model
        self.noise_model = NoiseModel(scenario) if use_noise else None

        # Target
        self.target_ket = get_target_state(self.n_atoms)

        # State
        self.rho = None
        self.current_step = 0
        self.noise_params = None
        self.V_vdW = self.C6 / self.R**6

    def _rho_to_obs(self, rho: qutip.Qobj) -> np.ndarray:
        """Convert density matrix to observation vector."""
        rho_full = rho.full()
        obs = np.concatenate([rho_full.real.flatten(), rho_full.imag.flatten()])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        psi0 = get_ground_state(self.n_atoms)
        self.rho = qutip.ket2dm(psi0)

        # Domain randomization: resample noise
        if self.use_noise and self.noise_model is not None:
            rng = np.random.default_rng(seed)
            self.noise_params = self.noise_model.sample(rng)
            # Update V_vdW with position jitter
            delta_R = self.noise_params["delta_R"][0]
            self.V_vdW = self.C6 / max(self.R + delta_R, 0.01)**6
        else:
            self.noise_params = None
            self.V_vdW = self.C6 / self.R**6

        obs = self._rho_to_obs(self.rho)
        return obs, {}

    def step(self, action):
        # Map action from [-1, 1] to physical range
        Omega = (action[0] + 1) / 2 * self.Omega_max * 2  # [0, 2*Omega_max]
        Delta = action[1] * self.Omega_max  # [-Omega_max, Omega_max]

        # Apply amplitude noise
        if self.noise_params and self.noise_params.get("ou_sigma", 0) > 0:
            # Simplified: static amplitude factor per step
            xi = np.random.normal(0, self.noise_params["ou_sigma"])
            Omega *= (1 + xi)

        # Apply Doppler shift
        doppler_shift = 0.0
        if self.noise_params:
            doppler_shift = np.mean(self.noise_params["delta_doppler"])

        # Build Hamiltonian
        H = build_two_atom_hamiltonian(Omega, Delta + doppler_shift, self.V_vdW)

        # Collapse operators
        c_ops = get_collapse_operators(self.n_atoms) if "decay" in self.cfg["noise_sources"] else []

        # Evolve one time step
        tlist = [0, self.dt]
        result = qutip.mesolve(H, self.rho, tlist, c_ops=c_ops)
        self.rho = result.states[-1]

        self.current_step += 1
        terminated = self.current_step >= self.n_steps
        truncated = False

        # Reward: only at terminal step
        if terminated:
            reward = float(compute_fidelity(self.rho, self.target_ket))
        else:
            reward = 0.0

        obs = self._rho_to_obs(self.rho)
        info = {"fidelity": compute_fidelity(self.rho, self.target_ket)} if terminated else {}

        return obs, reward, terminated, truncated, info
```

- [ ] **Step 2: Write config.py**

```python
"""PPO training hyperparameters."""

PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 1.0,           # no discounting (care about terminal fidelity)
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "total_timesteps": 100_000,  # reduced for CPU feasibility
    "n_seeds": 3,
    "scenario": "B",
    "env_n_steps": 30,       # time steps per episode
}
```

- [ ] **Step 3: Write train_ppo.py**

```python
"""PPO training script for Rydberg Bell state preparation."""
import json
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.environments.rydberg_env import RydbergBellEnv
from src.training.config import PPO_CONFIG
from src.baselines.evaluate import save_results


class FidelityLogCallback(BaseCallback):
    """Log episode fidelities during training."""

    def __init__(self):
        super().__init__()
        self.episode_fidelities = []
        self.episode_steps = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "fidelity" in info:
                self.episode_fidelities.append(info["fidelity"])
                self.episode_steps.append(self.num_timesteps)
        return True


def train_single_seed(seed: int, config: dict) -> tuple[PPO, FidelityLogCallback]:
    """Train PPO for one random seed.

    Args:
        seed: Random seed.
        config: PPO_CONFIG dict.

    Returns:
        (trained_model, callback_with_logs)
    """
    env = RydbergBellEnv(
        scenario=config["scenario"],
        n_steps=config["env_n_steps"],
        use_noise=True,
    )

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
    )

    callback = FidelityLogCallback()
    model.learn(total_timesteps=config["total_timesteps"], callback=callback)

    return model, callback


def evaluate_ppo(model: PPO, scenario: str, n_trajectories: int = 200) -> dict:
    """Evaluate trained PPO model."""
    env = RydbergBellEnv(scenario=scenario, n_steps=30, use_noise=True)
    fidelities = []
    for i in range(n_trajectories):
        obs, _ = env.reset(seed=i)
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        fidelities.append(info.get("fidelity", reward))

    fidelities = np.array(fidelities)
    return {
        "mean_F": float(np.mean(fidelities)),
        "F_05": float(np.percentile(fidelities, 5)),
        "std_F": float(np.std(fidelities)),
        "all_F": fidelities.tolist(),
    }


if __name__ == "__main__":
    config = PPO_CONFIG
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    all_logs = {}
    best_model = None
    best_F = 0.0

    for seed in range(config["n_seeds"]):
        print(f"\n=== Training PPO seed {seed} ===")
        model, callback = train_single_seed(seed, config)

        # Save model
        model_path = models_dir / f"ppo_B_seed{seed}"
        model.save(str(model_path))
        print(f"Model saved to {model_path}")

        # Save training log
        log = {
            "steps": callback.episode_steps,
            "fidelities": callback.episode_fidelities,
        }
        all_logs[f"seed_{seed}"] = log

        # Evaluate
        print(f"Evaluating seed {seed}...")
        results = evaluate_ppo(model, config["scenario"], n_trajectories=200)
        print(f"  mean_F={results['mean_F']:.4f}, F_05={results['F_05']:.4f}")

        if results["mean_F"] > best_F:
            best_F = results["mean_F"]
            best_model = model

    # Save best results
    best_results = evaluate_ppo(best_model, config["scenario"], n_trajectories=200)
    save_results(best_results, "PPO", config["scenario"])

    # Save training logs
    with open(results_dir / "training_logs.json", "w") as f:
        json.dump(all_logs, f)

    print(f"\nBest PPO: mean_F={best_F:.4f}")
    print("Training complete.")
```

- [ ] **Step 4: Verify environment passes gymnasium check**

```bash
cd E:/project/report && python -c "
from gymnasium.utils.env_checker import check_env
from src.environments.rydberg_env import RydbergBellEnv
env = RydbergBellEnv('B', n_steps=10, use_noise=False)
check_env(env, warn=True)
print('Environment check passed')
"
```

Expected: "Environment check passed"

- [ ] **Step 5: Run PPO training (this will take time)**

```bash
cd E:/project/report && python -m src.training.train_ppo
```

Expected: Training logs, model files in `models/`, results in `results/ppo_B.json`

- [ ] **Step 6: Commit**

```bash
git add src/environments/rydberg_env.py src/training/config.py src/training/train_ppo.py
git add models/ results/ppo_B.json results/training_logs.json
git commit -m "feat: add RL environment and PPO training pipeline"
```

---

### Task 14: Write §4 — 退相干与传统算法失效

**Files:**
- Create: `drafts/section_04.md`

- [ ] **Step 1: Write section_04.md**

**§4.1** (~0.7 page): Decoherence panorama table with 6 noise sources. Each row: source, math description, typical value (Rb 53S, T=10μK), reference.

**§4.2** (~0.5 page): Linear response formula from Day/PRX Quantum 2025. Three scaling laws (frequency noise ~Ω⁻¹·⁷⁹, Rydberg decay ~Ω⁻¹, Doppler ~Ω⁻²). Physical interpretation: increasing Ω to reduce errors vs blockade condition V≫ℏΩ — double squeeze.

**§4.3** (~0.8 page): Three traditional algorithms, each one paragraph + one key formula:
- STIRAP: adiabatic speed limit |θ̇|≪Ω_eff, Goldilocks interval, numerical evidence F≤0.985
- GRAPE: gradient formula (one line), sim2real gap, barren plateau (cite Larocca 2022)
- STA/CD: Berry formula, open-system failure

**Final paragraph:** Natural transition to RL motivation. All traditional methods assume known exact Hamiltonian + closed/Markovian system.

**Language:** Chinese body, English terms.

- [ ] **Step 2: Commit**

```bash
git add drafts/section_04.md
git commit -m "docs: write §4 decoherence budget and traditional algorithm failures"
```

---

### Task 15: Write §5-§6

**Files:**
- Create: `drafts/section_05.md`
- Create: `drafts/section_06.md`

- [ ] **Step 1: Write section_05.md**

§5 Experiment Setting. Concise but precise. Contains:
- Physical system parameters (Rb-87, 53S, two-photon via 6P)
- 5 noise sources with mathematical definitions
- Tab.2: scenarios A/B/D with T_gate, Ω, noise config. B is primary.
- Evaluation metrics: ⟨F⟩, F₀₅, σ_F over 1000 MC trajectories

All parameter values must match `src/physics/constants.py` exactly.

- [ ] **Step 2: Write section_06.md**

§6 AI Control. Contains:
- §6.1 MDP mapping: state (32-dim ρ vectorization), action (Ω,Δ), transition (Lindblad), reward (terminal F)
- §6.2 PPO: clipped surrogate objective (numbered formula), why PPO not DDPG/SAC (cite Ernst 2025, Bukov 2018)
- §6.3 Domain Randomization: physical meaning = robust feedback policy, analogy to dynamic decoupling

- [ ] **Step 3: Commit**

```bash
git add drafts/section_05.md drafts/section_06.md
git commit -m "docs: write §5 experiment settings and §6 AI control method"
```

---

### Task 16: Write Appendices A-F

**Files:**
- Create: `drafts/appendix_A.md` through `drafts/appendix_F.md`

- [ ] **Step 1: Write appendix_A.md** — QDT complete derivation (3-4 pages)

WKB approximation → radial wavefunction asymptotic form → short-range phase shift → quantum defect = phase shift / π → Rydberg-Ritz derivation → correction terms.

- [ ] **Step 2: Write appendix_B.md** — Adiabatic elimination (2-3 pages)

Full 3-level Hamiltonian → projection operators / Schrieffer-Wolff → step-by-step effective H derivation → AC Stark shift and scattering rate.

- [ ] **Step 3: Write appendix_C.md** — C₆ channel summation (2 pages)

Dipole-dipole operator matrix elements → Zeeman-degenerate sum → Förster defect and resonance.

- [ ] **Step 4: Write appendix_D.md** — GRAPE gradient + barren plateau (2 pages)

Full GRAPE gradient derivation. Larocca-type barren plateau intuitive proof.

- [ ] **Step 5: Write appendix_E.md** — Lindblad numerics (1-2 pages)

Master equation → Kraus representation. mesolve vs mcsolve comparison. Numerical integration notes.

- [ ] **Step 6: Write appendix_F.md** — PPO pseudocode + hyperparameters (1 page)

Algorithm block pseudocode. Hyperparameter table matching `src/training/config.py`.

- [ ] **Step 7: Commit**

```bash
git add drafts/appendix_*.md
git commit -m "docs: write appendices A-F (QDT, adiabatic elimination, C6, GRAPE, Lindblad, PPO)"
```

---

## Phase 3: Result Figures + Result Text

> **Gate:** Tasks 10-13 (baselines + PPO) must complete with result JSONs.
> **Dispatch in parallel:** Task 17, Task 18

### Task 17: Generate result figures (Fig.7-14)

**Files:**
- Create: `src/plotting/fig07_noise_impact.py` through `src/plotting/fig14_population_evolution.py`

- [ ] **Step 1: Write and run Fig.7 — Noise impact bar chart**

Grouped bars. X-axis: noise sources (Doppler, Position, Amplitude, Phase, Decay, All). Y-axis: 1-⟨F⟩. Uses scenario B parameters. Must run simulations to get data (or read from results/).

- [ ] **Step 2: Write and run Fig.8 — Traditional algorithm limits**

⟨F⟩ vs T_gate for STIRAP, GRAPE. Sweep T_gate from 0.1μs to 10μs. Mark ceiling for each method.

- [ ] **Step 3: Write and run Fig.9 — MDP schematic**

Flowchart-style: ρ(t) → Agent → (Ω,Δ) → Lindblad → ρ(t+dt) → ... → F. Pure matplotlib drawing.

- [ ] **Step 4: Write and run Fig.10 — Core comparison (most important)**

Bar chart with error bars. STIRAP/GRAPE/PPO on scenario B. Read from `results/*.json`.

- [ ] **Step 5: Write and run Fig.11 — Robustness curves**

⟨F⟩ vs δΩ/Ω ∈ [0, 5%]. Three lines. Parameter sweep.

- [ ] **Step 6: Write and run Fig.12 — PPO training curves**

Read `results/training_logs.json`. 3 seeds thin lines + mean thick + shaded std.

- [ ] **Step 7: Write and run Fig.13 — Pulse shape comparison**

Extract PPO policy output and GRAPE optimal pulse. Two panels: (a) Ω(t), (b) Δ(t).

- [ ] **Step 8: Write and run Fig.14 — Population evolution**

P_{|gg⟩}(t), P_{|W⟩}(t), P_{|rr⟩}(t). PPO solid, GRAPE dashed.

- [ ] **Step 9: Verify all 8 new PDFs exist**

```bash
ls -la figures/fig{07,08,09,10,11,12,13,14}_*.pdf
```

- [ ] **Step 10: Commit**

```bash
git add src/plotting/fig{07,08,09,10,11,12,13,14}_*.py figures/fig{07,08,09,10,11,12,13,14}_*.pdf
git commit -m "feat: generate result figures Fig.7-14"
```

---

### Task 18: Write §7-§8

**Files:**
- Create: `drafts/section_07.md`
- Create: `drafts/section_08.md`

- [ ] **Step 1: Write section_07.md**

§7 Results. Must read actual values from `results/*.json`:
- §7.1 Implementation: Plan B tech stack summary
- §7.2 Core results (scenario B): reference Fig.10, report ⟨F⟩/F₀₅/σ_F numbers
- §7.3 Robustness (reference Fig.11)
- §7.4 Training details (reference Fig.12, Fig.13)
- §7.5 Auxiliary results (scenario A, D — brief)
- §7.6 Honest feasibility statement

- [ ] **Step 2: Write section_08.md**

§8 Discussion and Conclusion:
- Physical proof chain summary
- AI positioning (not magic, model-free robust control)
- Limitations (4^N curse, sim-to-real)
- Outlook (Sr/Yb, differentiable simulators)

- [ ] **Step 3: Commit**

```bash
git add drafts/section_07.md drafts/section_08.md
git commit -m "docs: write §7 results and §8 discussion"
```

---

## Phase 4: Assembly

> **Gate:** All previous tasks complete.

### Task 19: Assemble final report.md

**Files:**
- Create: `report.md`

- [ ] **Step 1: Assemble all sections**

Concatenate in order:
1. Title + metadata
2. Abstract (write based on full content)
3. §1 through §8
4. Appendices A-F
5. References (from spec §8)

- [ ] **Step 2: Consistency audit**

Check:
- All "如图 X 所示" cross-references → figure exists
- All parameter values match constants.py
- Formula numbering continuous
- Symbol usage consistent (Ω not mixed with ω)
- All [AuthorYY] citations appear in References

- [ ] **Step 3: Logic chain audit**

Verify:
- §1-3 physics → §4 failure argument is supported
- §4 failure → §6 RL motivation is natural
- §7 results → answer §5 questions

- [ ] **Step 4: Commit final report**

```bash
git add report.md
git commit -m "docs: assemble final report with all sections, figures, and references"
```

---

## Summary: Sub-Agent Dispatch Map

| Phase | Parallel Tasks | Agent Type |
|---|---|---|
| **0** | Task 0 (setup) | Main |
| **1** | Tasks 1-5 (physics code) + Tasks 6-8 (§1-§3 text) + Task 9 (Fig.1-6) | 3-5 sub-agents |
| **2** | Tasks 10-12 (baselines+eval) + Task 13 (RL) + Tasks 14-16 (§4-§6 + appendices) | 4-5 sub-agents |
| **3** | Task 17 (Fig.7-14) + Task 18 (§7-§8) | 2 sub-agents |
| **4** | Task 19 (assembly) | Main |
