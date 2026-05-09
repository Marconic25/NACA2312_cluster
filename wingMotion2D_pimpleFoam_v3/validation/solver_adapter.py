"""
solver_adapter.py — Import bridge for cosim_driver structural functions.

Adds the parent directory (wingMotion2D_pimpleFoam/) to sys.path so that
cosim_driver.py can be imported without modification.
"""

import sys
from pathlib import Path

# Insert cosim_driver parent dir at front of path
COSIM_DIR = Path(__file__).parent.parent.resolve()
if str(COSIM_DIR) not in sys.path:
    sys.path.insert(0, str(COSIM_DIR))

import cosim_driver  # noqa: E402 — needed for monkey-patching in tests

from cosim_driver import (  # noqa: E402
    integrate_structural,
    structural_rhs,
    # Structural parameters
    M_WING,
    I_WING,
    K_H,
    D_H,
    K_ALPHA,
    D_ALPHA,
    # Flap parameters
    M_FLAP,
    I_FLAP_EA,
    I_FLAP_HINGE,
    _D_X,
    _D_Y,
    _D2,
)

import numpy as np

# Derived augmented mass matrix (frozen flap, delta=0)
M_hh = M_WING + M_FLAP
M_aa = I_WING + I_FLAP_EA
M_ha = M_FLAP * _D_X      # off-diagonal (symmetric)

M_mat = np.array([[M_hh, M_ha],
                  [M_ha, M_aa]])

K_mat = np.array([[K_H,     0.0    ],
                  [0.0,     K_ALPHA]])

C_mat = np.array([[D_H,     0.0    ],
                  [0.0,     D_ALPHA]])

__all__ = [
    "cosim_driver",
    "integrate_structural",
    "structural_rhs",
    "COSIM_DIR",
    "M_WING", "I_WING", "K_H", "D_H", "K_ALPHA", "D_ALPHA",
    "M_FLAP", "I_FLAP_EA", "I_FLAP_HINGE", "_D_X", "_D_Y", "_D2",
    "M_hh", "M_aa", "M_ha", "M_mat", "K_mat", "C_mat",
]
