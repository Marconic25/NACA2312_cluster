#!/usr/bin/env python3
"""
plot_response.py — reconstruct and plot wing structural response from cosim output.

Reads aerodynamic forces from postProcessing/forces/*/forces.dat, replays the
structural integrator window-by-window, and produces a 4-panel plot:
  - heave h(t) [mm]
  - pitch α(t) [deg]
  - lift Fy(t) [N]
  - moment Mz(t) [N·m]

Gust window and flap-deploy window are annotated on all panels.

Usage (from wingMotion2D_pimpleFoam/):
    python3 plot_response.py [--t-end 2.0]
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cosim_driver import (
    integrate_structural, read_forces,
    GUST_T_START, GUST_T_END, GUST_W0, U_INF,
    DELTA_TIMES, DELTA_ANGLES,
)

CASE_DIR   = Path(__file__).parent
FIG_DIR    = CASE_DIR / "figures"
DT         = 7e-5
WINDOW     = 286
WINDOW_DT  = WINDOW * DT

# Gust arrival time at wing leading edge (inlet at x=-10 m)
T_GUST_ARRIVE = GUST_T_START + 10.0 / U_INF   # ≈ 0.125 s

# Flap ramp window
T_FLAP_START = DELTA_TIMES[1]   # 0.8 s
T_FLAP_END   = DELTA_TIMES[2]   # 1.0 s


def reconstruct_structural(t_end):
    """Read forces and replay structural integration to get h(t) and α(t)."""
    print(f"Reading forces up to t={t_end:.3f} s ...")
    t_f, Fy_f, Mz_f = read_forces(0.0, t_end)
    if t_f is None:
        raise RuntimeError("No force data found in postProcessing/forces/")

    # Remove duplicate timestamps (can occur at window boundaries)
    _, idx = np.unique(t_f, return_index=True)
    t_f, Fy_f, Mz_f = t_f[idx], Fy_f[idx], Mz_f[idx]
    print(f"  Loaded {len(t_f)} force samples, t=[{t_f[0]:.5f}, {t_f[-1]:.5f}] s")

    h, hd, a, ad = 0.0, 0.0, 0.0, 0.0
    t_cur = 0.0
    t_hist, h_hist, a_hist = [], [], []

    n_windows = 0
    while t_cur < t_f[-1] - 1e-12:
        t_win_end = min(t_cur + WINDOW_DT, t_f[-1])
        t_win = np.arange(t_cur, t_win_end + DT * 0.5, DT)
        if len(t_win) < 2:
            break

        Fy_win = np.interp(t_win, t_f, Fy_f, left=0.0, right=0.0)
        Mz_win = np.interp(t_win, t_f, Mz_f, left=0.0, right=0.0)

        h_f, hd_f, a_f, ad_f, h_arr, a_arr = integrate_structural(
            h, hd, a, ad, t_win, Fy_win, Mz_win
        )

        t_hist.extend(t_win.tolist())
        h_hist.extend(h_arr.tolist())
        a_hist.extend(a_arr.tolist())

        h, hd, a, ad = h_f, hd_f, a_f, ad_f
        t_cur = t_win_end
        n_windows += 1

    print(f"  Replayed {n_windows} windows, final: h={h*1000:.2f} mm  α={np.degrees(a):.2f}°")

    return (
        t_f, Fy_f, Mz_f,
        np.array(t_hist),
        np.array(h_hist) * 1000.0,        # → mm
        np.degrees(np.array(a_hist)),      # → deg
    )


def annotate_events(ax):
    """Add gust and flap shading + gust-arrival line to an axes."""
    # Gust active at inlet
    ax.axvspan(GUST_T_START, GUST_T_END, color="steelblue", alpha=0.12,
               label=f"Gust at inlet (Wg0={GUST_W0:.0f} m/s)")
    # Gust arrival at wing
    ax.axvline(T_GUST_ARRIVE, color="steelblue", lw=1.2, ls="--",
               label=f"Gust hits wing (t={T_GUST_ARRIVE:.3f} s)")
    # Flap ramp
    ax.axvspan(T_FLAP_START, T_FLAP_END, color="darkorange", alpha=0.18,
               label=f"Flap ramp 0°→15°")


RHO  = 1.225   # air density [kg/m³] (from controlDict rhoInf)
AREF = 0.25    # reference area [m²] (from controlDict Aref = chord * span)
Q_INF = 0.5 * RHO * U_INF**2   # dynamic pressure [Pa]


def plot(t_f, Fy_f, Mz_f, t_s, h_s, a_s, t_end):
    # Skip first two windows: large initialisation transient from mapFields
    T_SKIP = 2 * WINDOW_DT
    mask = t_f >= T_SKIP
    t_fm, Fy_fm, Mz_fm = t_f[mask], Fy_f[mask], Mz_f[mask]
    Cl = Fy_fm / (Q_INF * AREF)

    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    fig.suptitle(
        f"Wing gust response  –  Wg0={GUST_W0:.0f} m/s, U∞={U_INF:.0f} m/s",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: heave ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t_s, h_s, "C0", lw=0.9)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    annotate_events(ax)
    ax.set_ylabel("Heave  h  [mm]")
    ax.legend(fontsize=7, loc="upper left", ncol=3)
    ax.grid(True, lw=0.4, alpha=0.5)

    # ── Panel 2: pitch ───────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t_s, a_s, "C1", lw=0.9)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    annotate_events(ax)
    ax.set_ylabel("Pitch  α  [deg]")
    ax.grid(True, lw=0.4, alpha=0.5)

    # ── Panel 3: lift coefficient ─────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t_fm, Cl, "C4", lw=0.7)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    annotate_events(ax)
    ax.set_ylabel("Lift coeff.  Cl  [-]")
    ax.grid(True, lw=0.4, alpha=0.5)

    # ── Panel 4: lift ────────────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(t_fm, Fy_fm, "C2", lw=0.7)
    annotate_events(ax)
    ax.set_ylabel("Lift  Fy  [N]")
    ax.grid(True, lw=0.4, alpha=0.5)

    # ── Panel 5: moment ──────────────────────────────────────────────────────
    ax = axes[4]
    ax.plot(t_fm, Mz_fm, "C3", lw=0.7)
    annotate_events(ax)
    ax.set_ylabel("Moment  Mz  [N·m]")
    ax.set_xlabel("Time  [s]")
    ax.set_xlim(0, t_end)
    ax.grid(True, lw=0.4, alpha=0.5)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "gust_response.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-end", type=float, default=2.0,
                        help="Time horizon for plot [s] (default: 2.0)")
    args = parser.parse_args()

    t_f, Fy_f, Mz_f, t_s, h_s, a_s = reconstruct_structural(args.t_end)
    plot(t_f, Fy_f, Mz_f, t_s, h_s, a_s, args.t_end)


if __name__ == "__main__":
    main()
