"""Physical constants and Rb-87 Rydberg parameters.

All values sourced from literature. Units: SI unless noted.
Frequencies in rad/s for computation; display values noted in comments.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Fundamental constants
# ---------------------------------------------------------------------------
HBAR = 1.054571817e-34       # J·s
K_B = 1.380649e-23           # J/K
C_LIGHT = 2.99792458e8       # m/s
E_CHARGE = 1.602176634e-19   # C
A0 = 5.29177210903e-11       # m, Bohr radius
RY_INF = 10973731.568160     # m^-1, Rydberg constant
ALPHA_FS = 7.2973525693e-3   # fine-structure constant

# ---------------------------------------------------------------------------
# Rb-87 atomic properties
# ---------------------------------------------------------------------------
RB87_MASS = 1.4431607e-25    # kg (86.909180527 amu)
RY_RB = RY_INF * (1 - 5.48579909065e-4 / 86.909180527)  # reduced-mass Rydberg (m^-1)
RY_RB_HZ = RY_RB * C_LIGHT   # Hz

# ---------------------------------------------------------------------------
# Quantum defects (Lorenzen & Niemax / Li et al.)
# ---------------------------------------------------------------------------
DELTA_0_S = 3.1311804        # nS_{1/2}
DELTA_2_S = 0.1784           # nS_{1/2} second-order
DELTA_0_P12 = 2.6548849      # nP_{1/2}
DELTA_0_P32 = 2.6416737      # nP_{3/2}
DELTA_0_D32 = 1.3480917      # nD_{3/2}
DELTA_0_D52 = 1.3462730      # nD_{5/2}

# ---------------------------------------------------------------------------
# 5P_{3/2} decay rate (intermediate state for two-photon excitation)
# ---------------------------------------------------------------------------
GAMMA_5P = 2 * np.pi * 6.065e6  # rad/s (linewidth 6.065 MHz)

# ---------------------------------------------------------------------------
# Rydberg state: 53S_{1/2}
# ---------------------------------------------------------------------------
N_RYD = 53
N_STAR_53S = N_RYD - DELTA_0_S          # effective quantum number ~49.87
TAU_RAD_53S = 135e-6                      # s, 0 K radiative lifetime (Beterov 2009)
GAMMA_BBR_53S = 1 / (200e-6)             # s^-1, BBR rate at 300 K (Beterov 2009)
TAU_EFF_53S = 1 / (1 / TAU_RAD_53S + GAMMA_BBR_53S)  # ~80.6 μs

# van der Waals coefficient for 53S + 53S
C6_53S = 2 * np.pi * 15.4e9  # rad/s · μm^6

# ---------------------------------------------------------------------------
# Scenario B baseline parameters (Evered 2023)
# ---------------------------------------------------------------------------
OMEGA_BASELINE = 2 * np.pi * 4.6e6   # rad/s (4.6 MHz)
R_ATOM = 2.0                          # μm (atom separation)
V_VDW_BASELINE = C6_53S / R_ATOM**6  # rad/s

# ---------------------------------------------------------------------------
# Two-photon excitation parameters (de Léséleuc 2018)
# ---------------------------------------------------------------------------
DELTA_INTERMEDIATE = 2 * np.pi * 7.8e9  # rad/s (7.8 GHz detuning from 6P)

# ---------------------------------------------------------------------------
# Noise parameters
# ---------------------------------------------------------------------------
SIGMA_DOPPLER = 2 * np.pi * 50e3     # rad/s (50 kHz at T = 10 μK)
SIGMA_POSITION = 0.1                   # μm (100 nm position jitter)
OU_CORRELATION_TIME = 10e-6            # s (amplitude noise)
OU_SIGMA = 0.02                        # relative amplitude noise (2%)
SERVO_BUMP_HEIGHT = 1e-8               # W/Hz (-80 dBc/Hz)
TEMPERATURE = 10e-6                    # K (10 μK)

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
SCENARIOS = {
    "A": {
        "description": "Long-time / STIRAP-compatible",
        "T_gate": 5e-6,
        "Omega": 2 * np.pi * 0.8e6,
        "R": 2.0,
        "noise_sources": ["doppler", "decay"],
        "n_atoms": 2,
    },
    "B": {
        "description": "Short-time / full noise (primary scenario)",
        "T_gate": 0.3e-6,
        "Omega": 2 * np.pi * 4.6e6,
        "R": 2.0,
        "noise_sources": ["doppler", "position", "amplitude", "phase", "decay"],
        "n_atoms": 2,
    },
    "D": {
        "description": "3-atom W-state preparation",
        "T_gate": 0.5e-6,
        "Omega": 2 * np.pi * 4.6e6,
        "R": 2.0,
        "noise_sources": ["doppler", "position", "amplitude", "phase", "decay"],
        "n_atoms": 3,
    },
}

# ---------------------------------------------------------------------------
# Blockade radius reference (Rb 70S at Omega = 1 MHz, for Table 1)
# ---------------------------------------------------------------------------
C6_70S = 2 * np.pi * 862e9   # rad/s · μm^6
TAU_RAD_70S = 150e-6          # s, 0 K
TAU_EFF_70S = 80e-6           # s, 300 K
