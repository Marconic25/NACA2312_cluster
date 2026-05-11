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
    integrate_structural,
    GUST_T_START, GUST_T_END, GUST_W0, U_INF,
    DELTA_TIMES, DELTA_ANGLES,
    delta_schedule,
    CASE_DIR as _CASE_DIR,
)
import re as _re


def read_all_forces(t_end):
    """Read all forces.dat files ignoring window boundaries."""
    forces_base = _CASE_DIR / "postProcessing" / "forces"
    t_list, Fy_list, Mz_list = [], [], []
    for forces_file in sorted(forces_base.glob("*/forces.dat"),
                              key=lambda p: float(p.parent.name)):
        samples_in_window = 0
        with open(forces_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                nums = _re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                if len(nums) < 19:
                    continue
                samples_in_window += 1
                n_skip = 10 if float(forces_file.parent.name) == 0.0 else 2
                if samples_in_window <= n_skip:
                    continue
                t = float(nums[0])
                if t > t_end + 1e-12:
                    continue
                t_list.append(t)
                Fy_list.append(float(nums[2]) + float(nums[5]))
                Mz_list.append(float(nums[12]) + float(nums[15]))
    if not t_list:
        raise RuntimeError("No force data found in postProcessing/forces/")
    t = np.array(t_list)
    Fy = np.array(Fy_list)
    Mz = np.array(Mz_list)
    idx = np.argsort(t)
    t, Fy, Mz = t[idx], Fy[idx], Mz[idx]
    _, ui = np.unique(t, return_index=True)
    return t[ui], Fy[ui], Mz[ui]

CASE_DIR   = Path(__file__).parent
FIG_DIR    = CASE_DIR / "figures"
DT         = 7e-5
WINDOW     = 200   # must match --window argument used in cosim_driver
WINDOW_DT  = WINDOW * DT

# Gust arrival time at wing leading edge (inlet at x=-10 m)
T_GUST_ARRIVE = GUST_T_START + 10.0 / U_INF   # ≈ 0.125 s

# Flap ramp window (only meaningful when flap schedule is active)
T_FLAP_START = DELTA_TIMES[-1]
T_FLAP_END   = DELTA_TIMES[-1]


def reconstruct_structural(t_end):
    """Read forces and replay structural integration to get h(t) and α(t)."""
    print(f"Reading forces up to t={t_end:.3f} s ...")
    t_f, Fy_f, Mz_f = read_all_forces(t_end)

    # Remove duplicate timestamps (can occur at window boundaries)
    _, idx = np.unique(t_f, return_index=True)
    t_f, Fy_f, Mz_f = t_f[idx], Fy_f[idx], Mz_f[idx]
    print(f"  Loaded {len(t_f)} force samples, t=[{t_f[0]:.5f}, {t_f[-1]:.5f}] s")

    # Read initial structural state from cosim_state.json (set by driver at t=0)
    import json
    state_file = CASE_DIR / "cosim_state.json"
    h0, hd0, a0, ad0 = 0.0, 0.0, 0.0, 0.0
    if state_file.exists():
        with open(state_file) as f:
            s = json.load(f)
        # window_idx=0 entry has the initial equilibrium state
        h0  = s.get("h",  0.0)
        hd0 = s.get("hd", 0.0)
        a0  = s.get("a",  0.0)
        ad0 = s.get("ad", 0.0)
        print(f"  Initial state from cosim_state.json: h={h0*1000:.3f}mm  α={np.degrees(a0):.4f}°")

    # Integrate over the full time range in one shot (no windowing artifacts)
    h_f, hd_f, a_f, ad_f, h_arr_full, a_arr_full = integrate_structural(
        h0, hd0, a0, ad0, t_f, Fy_f, Mz_f
    )
    t_arr = t_f
    hd_arr = np.gradient(h_arr_full, t_arr)
    ad_arr = np.gradient(a_arr_full, t_arr)

    print(f"  Integrated full trajectory: h_final={h_f*1000:.2f} mm  α_final={np.degrees(a_f):.2f}°")

    return (
        t_f, Fy_f, Mz_f,
        t_arr,
        h_arr_full * 1000.0,        # → mm
        np.degrees(a_arr_full),     # → deg
        hd_arr * 1000.0,            # → mm/s
        np.degrees(ad_arr),         # → deg/s
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


def plot(t_f, Fy_f, Mz_f, t_s, h_s, a_s, hd_s, ad_s, t_end):
    t_fm, Fy_fm, Mz_fm = t_f, Fy_f, Mz_f
    Cl = Fy_fm / (Q_INF * AREF)
    Cm = Mz_fm / (Q_INF * AREF * 1.0)   # chord = 1.0 m

    # Gust and flap schedules on structural time grid
    gust_s  = np.array([
        (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * (t - GUST_T_START) / (GUST_T_END - GUST_T_START)))
        if GUST_T_START <= t <= GUST_T_END else 0.0
        for t in t_s
    ])
    delta_s = np.array([delta_schedule(t) for t in t_s])

    fig, axes = plt.subplots(9, 1, figsize=(13, 22), sharex=True)
    fig.suptitle(
        f"Wing aeroelastic response  –  Wg0={GUST_W0:.0f} m/s, U∞={U_INF:.0f} m/s",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: heave ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t_s, h_s, "C0", lw=0.9)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    annotate_events(ax)
    ax.set_ylabel("h  [mm]")
    ax.legend(fontsize=7, loc="upper left", ncol=3)
    ax.grid(True, lw=0.4, alpha=0.5)

    # ── Panel 2: heave rate ──────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t_s, hd_s, "C0", lw=0.9, ls="--")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    annotate_events(ax)
    ax.set_ylabel("ḣ  [mm/s]")
    ax.grid(True, lw=0.4, alpha=0.5)

    # ── Panel 3: pitch ───────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t_s, a_s, "C1", lw=0.9)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    annotate_events(ax)
    ax.set_ylabel("α  [deg]")
    ax.grid(True, lw=0.4, alpha=0.5)

    # ── Panel 4: pitch rate ──────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(t_s, ad_s, "C1", lw=0.9, ls="--")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    annotate_events(ax)
    ax.set_ylabel("α̇  [deg/s]")
    ax.grid(True, lw=0.4, alpha=0.5)

    # ── Panel 5: lift coefficient ─────────────────────────────────────────────
    ax = axes[4]
    ax.plot(t_fm, Cl, "C4", lw=0.7)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    annotate_events(ax)
    ax.set_ylabel("Cl  [-]")
    ax.grid(True, lw=0.4, alpha=0.5)

    # ── Panel 6: lift ────────────────────────────────────────────────────────
    ax = axes[5]
    ax.plot(t_fm, Fy_fm, "C2", lw=0.7)
    annotate_events(ax)
    ax.set_ylabel("Fy  [N]")
    ax.grid(True, lw=0.4, alpha=0.5)

    # ── Panel 7: moment coefficient ──────────────────────────────────────────
    ax = axes[6]
    ax.plot(t_fm, Cm, "C3", lw=0.7)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    annotate_events(ax)
    ax.set_ylabel("Cm  [-]")
    ax.grid(True, lw=0.4, alpha=0.5)

    # ── Panel 8: gust profile ────────────────────────────────────────────────
    ax = axes[7]
    ax.plot(t_s, gust_s, "C5", lw=0.9)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_ylabel("W_gust  [m/s]")
    ax.grid(True, lw=0.4, alpha=0.5)

    # ── Panel 9: flap angle ──────────────────────────────────────────────────
    ax = axes[8]
    ax.plot(t_s, delta_s, "C6", lw=0.9)
    ax.set_ylabel("δ  [deg]")
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

    t_f, Fy_f, Mz_f, t_s, h_s, a_s, hd_s, ad_s = reconstruct_structural(args.t_end)
    plot(t_f, Fy_f, Mz_f, t_s, h_s, a_s, hd_s, ad_s, args.t_end)


if __name__ == "__main__":
    main()
