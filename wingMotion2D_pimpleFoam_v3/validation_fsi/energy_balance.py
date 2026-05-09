"""
energy_balance.py — Interface power and energy balance analysis.

Interface power:
    P(t) = Fy(t) * h_dot(t) + Mz(t) * alpha_dot(t)

Velocities h_dot, alpha_dot are estimated via np.gradient (2nd-order central differences).

Energy balance check (per window):
    W_fluid_in = ∫ P(t) dt          (work done by fluid on structure)
    ΔE_str     = E_k(T) + E_p(T) - E_k(0) - E_p(0)   (change in mechanical energy)
    W_damp     = ∫ D_H*h_dot^2 + D_ALPHA*alpha_dot^2 dt  (energy dissipated by dampers)

Balance: W_fluid = ΔE_str + W_damp   (should hold exactly for the integrator)

Imbalance = |W_fluid - ΔE_str - W_damp| / max(|W_fluid|, eps)
"""

import sys
from pathlib import Path

import numpy as np

VALIDATION_DIR = Path(__file__).parent.parent / "validation"
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(VALIDATION_DIR))
from cosim_driver import K_H, D_H, K_ALPHA, D_ALPHA
from solver_adapter import M_hh, M_aa, M_ha
import plotting


# ── Physical sub-functions ──────────────────────────────────────────────────

def interface_power(t_win, Fy_arr, Mz_arr, h_arr, alpha_arr):
    """
    Instantaneous interface power P(t).

    P(t) = Fy(t) * h_dot(t) + Mz(t) * alpha_dot(t)

    Velocities estimated via np.gradient (central differences, O(dt²)).

    Parameters
    ----------
    t_win     : ndarray (N,)
    Fy_arr    : ndarray (N,)  — aerodynamic heave force [N]
    Mz_arr    : ndarray (N,)  — aerodynamic pitch moment [N·m]
    h_arr     : ndarray (N,)  — heave displacement [m]
    alpha_arr : ndarray (N,)  — pitch angle [rad]

    Returns
    -------
    P_arr : ndarray (N,)  — interface power [W]
    hd_arr  : ndarray (N,)  — heave velocity [m/s]
    ad_arr  : ndarray (N,)  — pitch rate [rad/s]
    """
    hd_arr = np.gradient(h_arr,     t_win)
    ad_arr = np.gradient(alpha_arr, t_win)
    P_arr  = Fy_arr * hd_arr + Mz_arr * ad_arr
    return P_arr, hd_arr, ad_arr


def cumulative_work(t_win, P_arr):
    """
    Cumulative work W(t) = ∫_t0^t P dt  (via trapezoidal quadrature).

    Returns
    -------
    W_arr : ndarray (N,)  — cumulative work [J], W[0] = 0
    """
    W_arr = np.zeros_like(P_arr)
    for i in range(1, len(t_win)):
        W_arr[i] = W_arr[i - 1] + 0.5 * (P_arr[i] + P_arr[i - 1]) * (t_win[i] - t_win[i - 1])
    return W_arr


def mechanical_energy(h_arr, hd_arr, alpha_arr, ad_arr):
    """
    Total mechanical energy: KE + PE.

    KE = 0.5*(M_hh*hd^2 + 2*M_ha*hd*ad + M_aa*ad^2)
    PE = 0.5*(K_H*h^2 + K_ALPHA*alpha^2)
    """
    KE = 0.5 * (M_hh * hd_arr**2
                + 2.0 * M_ha * hd_arr * ad_arr
                + M_aa * ad_arr**2)
    PE = 0.5 * (K_H * h_arr**2 + K_ALPHA * alpha_arr**2)
    return KE + PE


def damping_power(hd_arr, ad_arr):
    """
    Damping dissipation rate (always positive):
        P_damp = D_H * h_dot^2 + D_ALPHA * alpha_dot^2
    """
    return D_H * hd_arr**2 + D_ALPHA * ad_arr**2


def energy_imbalance(t_win, W_fluid_in, h_arr, alpha_arr, hd_arr, ad_arr):
    """
    Per-window relative energy imbalance.

    Balance: W_fluid = ΔE_str + W_damp
    Imbalance = (W_fluid - ΔE_str - W_damp) / max(|W_fluid|, eps)

    Parameters
    ----------
    W_fluid_in : float — total work done by fluid over the window [J]
    h_arr, alpha_arr : ndarray (N,)
    hd_arr, ad_arr   : ndarray (N,)

    Returns
    -------
    imbalance : float
    W_fluid   : float
    delta_E   : float  — change in mechanical energy [J]
    W_damp    : float  — energy dissipated by damping [J]
    """
    E   = mechanical_energy(h_arr, hd_arr, alpha_arr, ad_arr)
    delta_E = E[-1] - E[0]

    P_damp = damping_power(hd_arr, ad_arr)
    W_damp_arr = cumulative_work(t_win, P_damp)
    W_damp = W_damp_arr[-1]

    eps = max(abs(W_fluid_in), 1e-30)
    imbalance = (W_fluid_in - delta_E - W_damp) / eps
    return float(imbalance), float(W_fluid_in), float(delta_E), float(W_damp)


# ── Window-level analysis ───────────────────────────────────────────────────

def run_energy_analysis(
    h_traj_full, alpha_traj_full,
    Fy_traj, Mz_traj, t_windows
):
    """
    Compute interface power, cumulative work, and energy imbalance for
    the full simulation trajectory.

    Parameters
    ----------
    h_traj_full    : ndarray (T_total,) — full heave trajectory [m]
    alpha_traj_full: ndarray (T_total,) — full pitch trajectory [rad]
    Fy_traj        : list of ndarray   — Fy per window
    Mz_traj        : list of ndarray   — Mz per window
    t_windows      : list of ndarray   — time grids per window

    Returns
    -------
    summary : dict with keys:
        't_full', 'P_full', 'W_full',
        't_win_starts', 'imbalance_per_win',
        'W_fluid_total', 'delta_E_total', 'W_damp_total'
    """
    # Concatenate full time series (with possible small overlaps at boundaries)
    t_full    = np.concatenate(t_windows)
    Fy_full   = np.concatenate(Fy_traj)
    Mz_full   = np.concatenate(Mz_traj)

    P_full, hd_full, ad_full = interface_power(
        t_full, Fy_full, Mz_full, h_traj_full, alpha_traj_full
    )
    W_full = cumulative_work(t_full, P_full)

    # Per-window imbalance
    imbalance_list = []
    t_win_starts   = []
    offset = 0
    for i, t_win in enumerate(t_windows):
        n = len(t_win)
        sl = slice(offset, offset + n)
        h_w    = h_traj_full[sl]
        a_w    = alpha_traj_full[sl]
        hd_w   = hd_full[sl]
        ad_w   = ad_full[sl]
        W_in   = W_full[offset + n - 1] - W_full[offset]
        imb, W_f, dE, Wd = energy_imbalance(t_win, W_in, h_w, a_w, hd_w, ad_w)
        imbalance_list.append(imb)
        t_win_starts.append(float(t_win[0]))
        offset += n

    imbalance_arr = np.array(imbalance_list)
    t_win_starts  = np.array(t_win_starts)

    # NOTE: balance holds only for frozen-flap (t > 0.2 s).  During the ramp
    # the flap inertial coupling terms (Q_h_flap, Q_a_flap) in structural_rhs
    # contribute additional work not captured by Fy*h_dot + Mz*alpha_dot alone.
    print(f"  W_fluid total    : {W_full[-1]:.4e} J")
    E_total = mechanical_energy(h_traj_full, hd_full, alpha_traj_full, ad_full)
    dE_total = E_total[-1] - E_total[0]
    P_damp_full = damping_power(hd_full, ad_full)
    W_damp_total = cumulative_work(t_full, P_damp_full)[-1]
    print(f"  ΔE_structural    : {dE_total:.4e} J")
    print(f"  W_damping total  : {W_damp_total:.4e} J")
    imb_global = (W_full[-1] - dE_total - W_damp_total) / max(abs(W_full[-1]), 1e-30)
    print(f"  Global imbalance : {imb_global:.4e}  ({imb_global*100:.2f}%)")
    print(f"  Per-window imbalance — median={np.nanmedian(np.abs(imbalance_arr)):.4e}  "
          f"max={np.nanmax(np.abs(imbalance_arr)):.4e}")

    # ── Figures ──────────────────────────────────────────────────────────
    plotting.plot_energy(t_full, P_full, W_full, figname="interface_power_work")
    plotting.plot_energy_imbalance(t_win_starts, imbalance_arr, figname="energy_imbalance")

    return {
        "t_full"          : t_full,
        "P_full"          : P_full,
        "W_full"          : W_full,
        "t_win_starts"    : t_win_starts,
        "imbalance_per_win": imbalance_arr,
        "W_fluid_total"   : float(W_full[-1]),
        "delta_E_total"   : float(dE_total),
        "W_damp_total"    : float(W_damp_total),
        "imb_global"      : float(imb_global),
    }
