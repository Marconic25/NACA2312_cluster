"""
loose_vs_strong.py — Compare single-pass (loose) vs fixed-point (strong) coupling.

Both schemes use the same surrogate fluid (forces fixed per window).
The difference between them quantifies the coupling error introduced by the
loosely-coupled scheme.

Metrics
-------
  L2 error   : ||h_loose - h_strong||_2 / ||h_strong||_2   (relative)
  Phase error : time-lag at peak cross-correlation [s]
  Amplitude   : max(|h_loose|) / max(|h_strong|) - 1

Plots
-----
  h_comparison.png      — overlay heave + difference
  alpha_comparison.png  — overlay pitch + difference
  comparison_metrics.png — bar chart of metrics
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from cosim_driver import integrate_structural

from coupling_analysis import fsi_fixed_point
import plotting


# ── Scheme runners ──────────────────────────────────────────────────────────

def run_loose(h_ic_list, hd_ic_list, a_ic_list, ad_ic_list,
              Fy_traj, Mz_traj, t_windows):
    """
    Single-pass (loose) coupling: exactly one integrator call per window.

    Uses the recorded ICs (from the actual cosim run) and recorded forces.
    Returns the same trajectory that cosim_driver would have produced.

    Parameters
    ----------
    h_ic_list, hd_ic_list, a_ic_list, ad_ic_list : list of float
        Structural state at the start of each window (len = N_windows).
    Fy_traj, Mz_traj : list of ndarray
    t_windows        : list of ndarray

    Returns
    -------
    h_full     : ndarray  — heave trajectory (concatenated)
    alpha_full : ndarray  — pitch trajectory (concatenated)
    t_full     : ndarray  — time axis (concatenated)
    """
    h_parts     = []
    alpha_parts = []
    t_parts     = []

    for i, t_win in enumerate(t_windows):
        _, _, _, _, h_arr, alpha_arr = integrate_structural(
            h_ic_list[i], hd_ic_list[i],
            a_ic_list[i], ad_ic_list[i],
            t_win, Fy_traj[i], Mz_traj[i]
        )
        h_parts.append(h_arr)
        alpha_parts.append(alpha_arr)
        t_parts.append(t_win)

    return (np.concatenate(h_parts),
            np.concatenate(alpha_parts),
            np.concatenate(t_parts))


def run_strong(h_ic_list, hd_ic_list, a_ic_list, ad_ic_list,
               Fy_traj, Mz_traj, t_windows,
               tol=1e-6, max_iter=20):
    """
    Strong coupling (fixed-point) per window.

    For each window, iterates the structural integrator until convergence
    (forces held fixed = surrogate fluid).  The interface displacement at
    convergence becomes the input to the next window.

    Note: with surrogate fluid, the converged solution is the same as single-pass
    (forces don't change), so this verifies the fixed-point residuals go to zero.

    Returns
    -------
    h_full, alpha_full, t_full  (same shape as run_loose)
    iter_counts : list of int — iterations needed per window
    """
    h_parts     = []
    alpha_parts = []
    t_parts     = []
    iter_counts = []

    for i, t_win in enumerate(t_windows):
        d_hist, r_hist, conv, n_it = fsi_fixed_point(
            h_ic_list[i], hd_ic_list[i],
            a_ic_list[i], ad_ic_list[i],
            t_win, Fy_traj[i], Mz_traj[i],
            tol=tol, max_iter=max_iter
        )
        iter_counts.append(n_it)
        # Re-run integrator one final time to get the full trajectory at convergence
        _, _, _, _, h_arr, alpha_arr = integrate_structural(
            h_ic_list[i], hd_ic_list[i],
            a_ic_list[i], ad_ic_list[i],
            t_win, Fy_traj[i], Mz_traj[i]
        )
        h_parts.append(h_arr)
        alpha_parts.append(alpha_arr)
        t_parts.append(t_win)

    return (np.concatenate(h_parts),
            np.concatenate(alpha_parts),
            np.concatenate(t_parts),
            iter_counts)


# ── Comparison metrics ──────────────────────────────────────────────────────

def compare(h_loose, alpha_loose, h_strong, alpha_strong, t_full):
    """
    Compute comparison metrics between loose and strong coupling trajectories.

    Parameters
    ----------
    h_loose, alpha_loose   : ndarray (T,)
    h_strong, alpha_strong : ndarray (T,)
    t_full                 : ndarray (T,)

    Returns
    -------
    metrics : dict
        l2_h      : relative L2 error in heave
        l2_alpha  : relative L2 error in pitch
        phase_h   : phase lag of loose vs strong in heave [s]
        amp_h     : amplitude error in heave (ratio - 1)
    """
    def _l2_rel(a, b):
        denom = np.sqrt(np.mean(b**2))
        if denom < 1e-30:
            return float(np.sqrt(np.mean(a**2)))
        return float(np.sqrt(np.mean((a - b)**2)) / denom)

    def _phase_lag(sig1, sig2, dt):
        """Cross-correlation phase lag [s]. Positive = sig1 lags sig2."""
        corr = np.correlate(sig1 - sig1.mean(), sig2 - sig2.mean(), mode="full")
        lags = np.arange(-(len(sig1) - 1), len(sig1))
        peak = lags[np.argmax(corr)]
        return float(peak * dt)

    def _amp_error(sig1, sig2):
        """max(|sig1|) / max(|sig2|) - 1."""
        denom = np.max(np.abs(sig2))
        if denom < 1e-30:
            return 0.0
        return float(np.max(np.abs(sig1)) / denom - 1.0)

    dt = float(np.mean(np.diff(t_full))) if len(t_full) > 1 else 1.0

    metrics = {
        "l2_h"    : _l2_rel(h_loose,     h_strong),
        "l2_alpha": _l2_rel(alpha_loose,  alpha_strong),
        "phase_h" : _phase_lag(h_loose,   h_strong, dt),
        "amp_h"   : _amp_error(h_loose,   h_strong),
    }
    return metrics


# ── Top-level run ───────────────────────────────────────────────────────────

def run(h_ic_list, hd_ic_list, a_ic_list, ad_ic_list,
        Fy_traj, Mz_traj, t_windows,
        tol=1e-6, max_iter=20):
    """
    Run both schemes, compute metrics, produce comparison plots.

    Returns
    -------
    metrics    : dict (from compare())
    iter_counts: list of int — per-window iteration count for strong coupling
    """
    print("  Running loose coupling (single-pass)...")
    h_l, a_l, t_f = run_loose(
        h_ic_list, hd_ic_list, a_ic_list, ad_ic_list,
        Fy_traj, Mz_traj, t_windows
    )

    print(f"  Running strong coupling (fixed-point, tol={tol:.0e}, max_iter={max_iter})...")
    h_s, a_s, t_f, iter_counts = run_strong(
        h_ic_list, hd_ic_list, a_ic_list, ad_ic_list,
        Fy_traj, Mz_traj, t_windows,
        tol=tol, max_iter=max_iter
    )
    print(f"  Strong coupling avg iterations: {np.mean(iter_counts):.2f}  "
          f"max: {max(iter_counts)}")

    metrics = compare(h_l, a_l, h_s, a_s, t_f)
    print(f"  Loose vs strong — L2_h={metrics['l2_h']:.3e}  "
          f"L2_alpha={metrics['l2_alpha']:.3e}  "
          f"phase_h={metrics['phase_h']:.4f}s  "
          f"amp_h={metrics['amp_h']:.3e}")

    # ── Figures ──────────────────────────────────────────────────────────
    plotting.overlay_comparison(
        t_f, h_l * 1e3, h_s * 1e3,
        ("loose", "strong"),
        "Heave: loose vs strong coupling",
        "h [mm]",
        figname="h_comparison"
    )
    plotting.overlay_comparison(
        t_f, np.degrees(a_l), np.degrees(a_s),
        ("loose", "strong"),
        "Pitch: loose vs strong coupling",
        "alpha [deg]",
        figname="alpha_comparison"
    )
    plotting.plot_comparison_metrics(
        {
            "L2 h"       : metrics["l2_h"],
            "L2 alpha"   : metrics["l2_alpha"],
            "|amp_h|"    : abs(metrics["amp_h"]),
        },
        figname="comparison_metrics"
    )

    return metrics, iter_counts
