"""Physics simulation core for Rydberg atom quantum control.

Submodules
----------
constants     : Physical constants and Rb-87 Rydberg parameters.
hamiltonian   : Two-atom and three-atom Rydberg Hamiltonian builders.
noise_model   : Noise sampler (Doppler, position, amplitude, phase, decay).
lindblad      : Collapse operators and mesolve wrapper.
"""

from src.physics.constants import (
    C6_53S,
    HBAR,
    OMEGA_BASELINE,
    SCENARIOS,
    TAU_EFF_53S,
    V_VDW_BASELINE,
)
from src.physics.hamiltonian import (
    build_three_atom_hamiltonian,
    build_two_atom_hamiltonian,
    get_ground_state,
    get_target_state,
)
from src.physics.lindblad import (
    compute_fidelity,
    get_collapse_operators,
    mesolve_with_noise,
)
from src.physics.noise_model import NoiseModel

__all__ = [
    # constants
    "C6_53S",
    "HBAR",
    "OMEGA_BASELINE",
    "SCENARIOS",
    "TAU_EFF_53S",
    "V_VDW_BASELINE",
    # hamiltonian
    "build_two_atom_hamiltonian",
    "build_three_atom_hamiltonian",
    "get_ground_state",
    "get_target_state",
    # lindblad
    "compute_fidelity",
    "get_collapse_operators",
    "mesolve_with_noise",
    # noise_model
    "NoiseModel",
]
