"""
plotting.py — Shared figure utilities for FSI coupling validation.

All figures saved to validation_fsi/figures/.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURE_DIR = Path(__file__).parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def save_fig(fig, name: str) -> Path:
    """Save figure as PNG and close it."""
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def semilog_residuals(iter_arr, res_arr, title: str, figname: str):
    """
    Semilog plot of r^(k)/r^(0) vs iteration number.

    Parameters
    ----------
    iter_arr : array_like, shape (N,)   — iteration indices (0, 1, 2, ...)
    res_arr  : array_like, shape (N,)   — residuals at each iteration
    title    : str
    figname  : str
    """
    res_arr = np.asarray(res_arr, dtype=float)
    iter_arr = np.asarray(iter_arr, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    r0 = max(res_arr[0], 1e-30)
    ax.semilogy(iter_arr, res_arr / r0, "o-", lw=1.5, ms=5)
    ax.set_xlabel("Iteration k")
    ax.set_ylabel(r"$r^{(k)} / r^{(0)}$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, figname)


def overlay_comparison(t, a1, a2, labels, title: str, ylabel: str, figname: str):
    """
    Two-panel figure: overlay + difference.

    Parameters
    ----------
    t      : array_like  — common time axis
    a1, a2 : array_like  — two signals to compare
    labels : (str, str)  — legend labels for a1, a2
    """
    t  = np.asarray(t)
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t, a1, lw=1.2, label=labels[0], color="navy")
    ax1.plot(t, a2, lw=1.2, label=labels[1], color="crimson", ls="--")
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, a1 - a2, lw=1.0, color="darkorange")
    ax2.axhline(0, color="k", lw=0.7, ls="--")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel(f"Δ{ylabel}")
    ax2.set_title("Difference (signal1 − signal2)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, figname)


def plot_spectral_radius(t_windows, rho_arr, figname: str = "spectral_radius_vs_time"):
    """
    Spectral radius ρ vs simulation time.  Red dashed line at ρ=1 (stability boundary).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_windows, rho_arr, lw=1.2, color="darkgreen")
    ax.axhline(1.0, color="red", lw=1.0, ls="--", label="ρ=1 (stability boundary)")
    ax.set_xlabel("Window start time [s]")
    ax.set_ylabel("Spectral radius ρ")
    ax.set_title("Coupling spectral radius vs simulation time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_fig(fig, figname)


def plot_energy(t, P_arr, W_arr, figname: str = "interface_energy"):
    """
    Two-panel: instantaneous power P(t) and cumulative work W(t).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t, P_arr, lw=1.0, color="steelblue")
    ax1.axhline(0, color="k", lw=0.7, ls="--")
    ax1.set_ylabel("Power P(t) [W]")
    ax1.set_title("Interface power: P = Fy·ḣ + Mz·α̇")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, W_arr, lw=1.2, color="darkorange")
    ax2.axhline(0, color="k", lw=0.7, ls="--")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Cumulative work W(t) [J]")
    ax2.set_title("Interface cumulative work")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, figname)


def plot_energy_imbalance(t_windows, imbalance_arr, figname: str = "energy_imbalance"):
    """
    Per-window relative energy imbalance |W_fluid - ΔE_str - W_damp| / |W_fluid|.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(t_windows, np.abs(imbalance_arr) + 1e-30, lw=1.0, color="purple")
    ax.set_xlabel("Window start time [s]")
    ax.set_ylabel("Relative energy imbalance")
    ax.set_title("Per-window interface energy imbalance")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, figname)


def plot_comparison_metrics(metrics: dict, figname: str = "comparison_metrics"):
    """
    Bar chart of loose-vs-strong comparison metrics.

    Parameters
    ----------
    metrics : dict  — keys: metric names, values: floats
    """
    names  = list(metrics.keys())
    values = [metrics[k] for k in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=["steelblue", "darkorange", "green"])
    ax.set_ylabel("Value")
    ax.set_title("Loose vs strong coupling — comparison metrics")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{v:.2e}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, figname)
