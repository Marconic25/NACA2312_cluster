"""
residuals.py — Coupling residual study over all recorded FSI windows.

For each window:
  1. Run fsi_fixed_point with surrogate fluid (forces fixed at f^{n-1}).
  2. Record residual history r^(0), r^(1), ..., and convergence.
  3. Estimate spectral radius ρ = r^(k)/r^(k-1).

Plots
-----
  residual_vs_iter.png       — r^(k)/r^(0) semilog per iteration for sampled windows
  spectral_radius_vs_time.png — ρ across simulation time
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from coupling_analysis import fsi_fixed_point
import plotting


# ── Public helpers ──────────────────────────────────────────────────────────

def compute_residual(d_new, d_old, eps=1e-30):
    """
    Relative displacement residual.

    r = ||d_new - d_old||_2 / max(||d_old||_2, eps)
    """
    return np.linalg.norm(d_new - d_old) / max(np.linalg.norm(d_old), eps)


def estimate_spectral_radius(r_history):
    """
    Estimate coupling convergence rate from residual history.

    ρ = geometric mean of r^(k)/r^(k-1) ratios (k = 1, 2, ...).

    Returns np.nan if fewer than 2 residuals available.
    """
    r = np.asarray(r_history, dtype=float)
    if len(r) < 2:
        return np.nan
    ratios = r[1:] / np.maximum(r[:-1], 1e-30)
    # Geometric mean of ratios
    log_ratios = np.log(np.maximum(ratios, 1e-30))
    return float(np.exp(np.mean(log_ratios)))


# ── Main study ──────────────────────────────────────────────────────────────

def run_residual_study(
    h_traj, hd_traj, a_traj, ad_traj,
    Fy_traj, Mz_traj, t_windows,
    tol=1e-6, max_iter=20,
    sample_windows=None
):
    """
    Loop over all recorded FSI windows and run fixed-point per window.

    Parameters
    ----------
    h_traj, hd_traj, a_traj, ad_traj : list/ndarray, len = N_windows
        Structural state at the START of each window (IC for integration).
    Fy_traj, Mz_traj : list of ndarray
        Aerodynamic force arrays for each window (one per window, each shape (M,)).
    t_windows : list of ndarray
        Time grid for each window (one per window, each shape (M,)).
    tol       : float  — fixed-point convergence tolerance
    max_iter  : int    — max iterations per window
    sample_windows : list of int or None
        Indices of windows to show in residual_vs_iter plot.
        Default: 5 evenly-spaced windows excluding window 0.

    Returns
    -------
    results : list of dict, one per window:
        {
          'i_win'     : int,
          't_start'   : float,
          'd_history' : list of ndarray,
          'r_history' : list of float,
          'converged' : bool,
          'n_iter'    : int,
          'rho'       : float,
        }
    """
    N = len(t_windows)
    results = []

    print(f"  Running residual study over {N} windows (tol={tol:.0e}, max_iter={max_iter})...")
    for i in range(N):
        t_win   = t_windows[i]
        Fy_arr  = Fy_traj[i]
        Mz_arr  = Mz_traj[i]
        h0  = float(h_traj[i])
        hd0 = float(hd_traj[i])
        a0  = float(a_traj[i])
        ad0 = float(ad_traj[i])

        d_hist, r_hist, conv, n_it = fsi_fixed_point(
            h0, hd0, a0, ad0, t_win, Fy_arr, Mz_arr,
            tol=tol, max_iter=max_iter
        )
        rho = estimate_spectral_radius(r_hist)
        results.append({
            "i_win"    : i,
            "t_start"  : float(t_win[0]),
            "d_history": d_hist,
            "r_history": r_hist,
            "converged": conv,
            "n_iter"   : n_it,
            "rho"      : rho,
        })

    n_conv   = sum(r["converged"] for r in results)
    rho_vals = [r["rho"] for r in results if not np.isnan(r["rho"])]
    rho_med  = float(np.median(rho_vals)) if rho_vals else np.nan
    print(f"  Converged windows: {n_conv}/{N}")
    if rho_vals:
        print(f"  Spectral radius — median={rho_med:.4f}  "
              f"max={max(rho_vals):.4f}  min={min(rho_vals):.4f}")
    else:
        print("  Spectral radius: ρ ≈ 0 (all windows converge in 1 iteration — "
              "expected with surrogate fluid: deterministic integrator)")
    n_iters = [r["n_iter"] for r in results]
    print(f"  Iterations per window — avg={np.mean(n_iters):.2f}  max={max(n_iters)}")

    # ── Plots ──────────────────────────────────────────────────────────────
    _plot_residuals(results, sample_windows)
    _plot_spectral_radius(results)

    return results


# ── Private plot helpers ────────────────────────────────────────────────────

def _plot_residuals(results, sample_windows=None):
    """Plot r^(k)/r^(0) semilog for selected windows."""
    N = len(results)
    if sample_windows is None:
        # 5 evenly-spaced windows, skip first (no prior forces)
        idx = np.linspace(1, N - 1, min(5, N - 1), dtype=int)
    else:
        idx = list(sample_windows)

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 0.85, len(idx)))

    for color, iw in zip(colors, idx):
        r_hist = results[iw]["r_history"]
        if not r_hist:
            continue
        r_arr = np.asarray(r_hist)
        r0 = max(r_arr[0], 1e-30)
        t_lbl = results[iw]["t_start"]
        ax.semilogy(
            np.arange(len(r_arr)), r_arr / r0,
            "o-", lw=1.4, ms=4, color=color,
            label=f"win {iw} (t={t_lbl:.3f}s)"
        )

    ax.set_xlabel("Iteration k")
    ax.set_ylabel(r"$r^{(k)} / r^{(0)}$")
    ax.set_title("Fixed-point residual per iteration (selected windows)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    plotting.save_fig(fig, "residual_vs_iter")


def _plot_spectral_radius(results):
    """Plot spectral radius vs simulation time."""
    t_arr   = np.array([r["t_start"] for r in results])
    rho_arr = np.array([r["rho"]     for r in results])

    # Replace nan with nan (keep gaps in plot)
    plotting.plot_spectral_radius(t_arr, rho_arr, figname="spectral_radius_vs_time")
