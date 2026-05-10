"""
coupling_analysis.py — Fixed-point FSI coupling iteration engine.

Implements surrogate-fluid strong coupling for one FSI window:
  k=0 : d^(0) = single-pass result (exactly what cosim_driver does)
  k≥1 : re-integrate structure with *same* forces (forces held fixed at f^{n-1})

Rationale for surrogate fluid
------------------------------
Real strong coupling requires re-running pimpleFoam per iteration — prohibitively
expensive.  By holding the aerodynamic force fixed at f^{n-1} (previous window),
we isolate the *structural sub-iteration convergence* from fluid re-computation.
The resulting fixed-point is the exact solution of the structural integrator
for a given force, so the residual measures how many iterations the structure
needs to reach self-consistency.

Interface displacement:  d = [h_end, alpha_end]
Residual:                r^(k) = ||d^(k) - d^(k-1)||_2

Optional Aitken Δ² relaxation accelerates convergence when ρ is close to 1.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from cosim_driver import integrate_structural


def fsi_fixed_point(
    h0, hd0, a0, ad0,
    t_win, Fy_arr, Mz_arr,
    tol=1e-6, max_iter=20, omega=1.0
):
    """
    Fixed-point iteration for one FSI window with surrogate (frozen) fluid.

    The structural integrator is called with the same Fy_arr / Mz_arr every
    iteration.  k=0 is just the first call (no previous d to compare against),
    k=1 is the first residual measurement.

    Parameters
    ----------
    h0, hd0 : float  — heave position and velocity at window start [m, m/s]
    a0, ad0 : float  — pitch angle and rate at window start [rad, rad/s]
    t_win   : ndarray, shape (N,) — time grid for this window [s]
    Fy_arr  : ndarray, shape (N,) — aerodynamic heave force [N]  (f^{n-1})
    Mz_arr  : ndarray, shape (N,) — aerodynamic pitch moment [N·m] (f^{n-1})
    tol     : float  — convergence tolerance on ||d^(k) - d^(k-1)||_2
    max_iter: int    — maximum iterations
    omega   : float  — constant under-relaxation factor (1.0 = no relaxation)

    Returns
    -------
    d_history  : list of ndarray([h_end, alpha_end])  — interface displacement per iter
    r_history  : list of float  — residuals (len = max(0, n_iter - 1))
    converged  : bool
    n_iter     : int  — number of integrator calls made
    """
    d_history = []
    r_history = []

    d_prev = None
    for k in range(max_iter):
        h_f, hd_f, a_f, ad_f, h_arr, alpha_arr = integrate_structural(
            h0, hd0, a0, ad0, t_win, Fy_arr, Mz_arr
        )
        d_curr = np.array([h_f, a_f])

        if d_prev is not None:
            # Apply under-relaxation
            d_curr_rlx = d_prev + omega * (d_curr - d_prev)
            r = np.linalg.norm(d_curr_rlx - d_prev)
            r_history.append(float(r))
            d_curr = d_curr_rlx
        else:
            d_curr_rlx = d_curr

        d_history.append(d_curr_rlx.copy())
        d_prev = d_curr_rlx

        # Check convergence after first residual
        if r_history and r_history[-1] < tol:
            return d_history, r_history, True, k + 1

    # Did not converge
    return d_history, r_history, False, max_iter


def aitken_update(d_prev, d_curr, d_prev_prev, omega_prev):
    """
    Aitken Δ² acceleration: update relaxation factor.

    ω^(k+1) = -ω^(k) * (Δd^(k-1))·(Δd^(k) - Δd^(k-1)) / ||Δd^(k) - Δd^(k-1)||²

    Parameters
    ----------
    d_prev      : ndarray  — d^(k-1)
    d_curr      : ndarray  — d^(k)   (un-relaxed)
    d_prev_prev : ndarray  — d^(k-2)
    omega_prev  : float    — ω^(k-1)

    Returns
    -------
    omega_new : float  (clamped to [0.1, 1.0])
    """
    delta_prev = d_prev      - d_prev_prev   # Δd^(k-1)
    delta_curr = d_curr      - d_prev        # Δd^(k)
    delta_diff = delta_curr  - delta_prev
    denom = np.dot(delta_diff, delta_diff)
    if denom < 1e-30:
        return omega_prev
    omega_new = -omega_prev * np.dot(delta_prev, delta_diff) / denom
    # Clamp to (0.1, 1.0] for stability
    return float(np.clip(omega_new, 0.1, 1.0))


def fsi_fixed_point_aitken(
    h0, hd0, a0, ad0,
    t_win, Fy_arr, Mz_arr,
    tol=1e-6, max_iter=20, omega0=0.5
):
    """
    Fixed-point iteration with Aitken Δ² adaptive relaxation.

    Same interface as fsi_fixed_point, with omega0 as starting relaxation factor.
    Aitken update kicks in from k=2 onward.

    Returns
    -------
    d_history, r_history, converged, n_iter  (same as fsi_fixed_point)
    """
    d_history = []
    r_history = []

    omega = omega0
    d_list = []

    for k in range(max_iter):
        h_f, hd_f, a_f, ad_f, h_arr, alpha_arr = integrate_structural(
            h0, hd0, a0, ad0, t_win, Fy_arr, Mz_arr
        )
        d_raw = np.array([h_f, a_f])

        if len(d_list) == 0:
            d_rlx = d_raw.copy()
        else:
            d_rlx = d_list[-1] + omega * (d_raw - d_list[-1])

        if len(d_list) >= 1:
            r = np.linalg.norm(d_rlx - d_list[-1])
            r_history.append(float(r))

        # Aitken update from k=2
        if len(d_list) >= 2:
            omega = aitken_update(d_list[-1], d_raw, d_list[-2], omega)

        d_list.append(d_rlx.copy())
        d_history.append(d_rlx.copy())

        if r_history and r_history[-1] < tol:
            return d_history, r_history, True, k + 1

    return d_history, r_history, False, max_iter
