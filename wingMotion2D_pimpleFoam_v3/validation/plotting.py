"""
plotting.py — Shared plotting utilities for the validation framework.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path(__file__).parent / "figures"


def save_fig(fig, name):
    """Save figure to validation/figures/<name>.png."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure → {path}")
    return path


def plot_overlay(t, arr1, arr2, labels, title, ylabel, figname=None):
    """
    Two-panel figure: top = overlay of arr1 and arr2, bottom = residual.

    Parameters
    ----------
    t : array_like (N,)
    arr1, arr2 : array_like (N,) — signals to compare
    labels : tuple(str, str) — legend labels
    title, ylabel : str
    figname : str or None — if given, save and return path
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, arr1, label=labels[0], lw=1.5)
    axes[0].plot(t, arr2, label=labels[1], lw=1.0, ls="--")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    residual = arr1 - arr2
    axes[1].plot(t, residual, color="red", lw=1.0)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel(f"Residual ({ylabel})")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if figname:
        save_fig(fig, figname)
    return fig


def plot_energy(t, E, title="Energy conservation", figname=None):
    """Semilogy plot of energy vs time."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(t, E / E[0], color="navy", lw=1.5)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("E(t) / E(0)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    if figname:
        save_fig(fig, figname)
    return fig


def plot_loglog(x, y, slope, xlabel, ylabel, title, figname=None):
    """
    Log-log plot with fitted slope line.

    Parameters
    ----------
    x, y : array_like (N,) — data points
    slope : float — measured slope from np.polyfit
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(x, y, "o-", color="steelblue", lw=1.5, ms=6, label="Error")
    # Fitted line through last point
    x_fit = np.array([min(x), max(x)])
    y_fit = y[-1] * (x_fit / x[-1]) ** slope
    ax.loglog(x_fit, y_fit, "k--", lw=1.0,
              label=f"Slope = {slope:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    if figname:
        save_fig(fig, figname)
    return fig


def plot_phase_portrait(q, qd, xlabel, ylabel, title, figname=None):
    """Phase portrait: q vs qd."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(q, qd, lw=1.2, color="darkgreen")
    ax.plot(q[0], qd[0], "go", ms=8, label="start")
    ax.plot(q[-1], qd[-1], "rs", ms=8, label="end")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", "datalim")
    fig.tight_layout()
    if figname:
        save_fig(fig, figname)
    return fig


def plot_root_locus(Re_mat, Im_mat, U_arr, flutter_U, figname=None):
    """
    Root locus: Re(lambda) vs Im(lambda) parametrised by U.

    Re_mat, Im_mat : shape (n_modes, n_U)
    U_arr          : shape (n_U,)
    flutter_U      : float or None — flutter onset speed
    """
    n_modes = Re_mat.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_modes))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Re vs Im (Argand diagram)
    ax = axes[0]
    for i in range(n_modes):
        sc = ax.scatter(Re_mat[i], Im_mat[i],
                        c=U_arr, cmap="plasma", s=8, label=f"Mode {i+1}")
    ax.axvline(0, color="k", lw=1.0, ls="--")
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ) [rad/s]")
    ax.set_title("Root locus (Argand)")
    if flutter_U is not None:
        ax.set_title(f"Root locus — flutter at U={flutter_U:.1f} m/s")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="U [m/s]")

    # Right: Re(lambda) vs U — growth/decay plot
    ax2 = axes[1]
    for i in range(n_modes):
        ax2.plot(U_arr, Re_mat[i], lw=1.5, label=f"Mode {i+1}")
    ax2.axhline(0, color="k", lw=1.0, ls="--")
    if flutter_U is not None:
        ax2.axvline(flutter_U, color="red", lw=1.5, ls=":",
                    label=f"Flutter U={flutter_U:.1f} m/s")
    ax2.set_xlabel("U [m/s]")
    ax2.set_ylabel("Re(λ)")
    ax2.set_title("Damping vs airspeed")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if figname:
        save_fig(fig, figname)
    return fig
