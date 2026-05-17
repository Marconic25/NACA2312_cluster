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
    K_H, K_ALPHA,
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
    """Read structural trajectory from CSV saved by driver, fallback to force integration."""
    traj_file = CASE_DIR / "structural_trajectory.csv"

    if traj_file.exists():
        print(f"Reading structural trajectory from {traj_file.name} ...")
        data = np.genfromtxt(traj_file, delimiter=",", skip_header=1)
        mask = data[:, 0] <= t_end + 1e-9
        data = data[mask]
        t_arr = data[:, 0]
        h_arr = data[:, 1] * 1000.0       # → mm
        hd_arr = data[:, 2] * 1000.0      # → mm/s
        a_arr = np.degrees(data[:, 3])    # → deg
        ad_arr = np.degrees(data[:, 4])   # → deg/s
        Fy_f = data[:, 5]
        Mz_f = data[:, 6]
        # Remove spike outliers (restart transient) using median filter threshold
        from scipy.signal import medfilt
        Fy_med = medfilt(Fy_f, kernel_size=5)
        Mz_med = medfilt(Mz_f, kernel_size=5)
        spike_mask = (np.abs(Fy_f - Fy_med) < 3 * np.std(Fy_f - Fy_med)) & \
                     (np.abs(Mz_f - Mz_med) < 3 * np.std(Mz_f - Mz_med))
        t_arr = t_arr[spike_mask]
        h_arr = h_arr[spike_mask]
        hd_arr = hd_arr[spike_mask]
        a_arr = a_arr[spike_mask]
        ad_arr = ad_arr[spike_mask]
        Fy_f = Fy_med[spike_mask]
        Mz_f = Mz_med[spike_mask]
        t_f = t_arr
        # Read W_gust and delta if present (columns 7 and 8)
        if data.shape[1] >= 9:
            W_gust_arr = data[:, 7][spike_mask]
            delta_arr  = data[:, 8][spike_mask]
        else:
            W_gust_arr = None
            delta_arr  = None
        print(f"  Loaded {len(t_arr)} samples, t=[{t_arr[0]:.5f}, {t_arr[-1]:.5f}] s")
        return t_f, Fy_f, Mz_f, t_arr, h_arr, a_arr, hd_arr, ad_arr, W_gust_arr, delta_arr

    # Fallback: read forces and integrate
    print(f"Reading forces up to t={t_end:.3f} s ...")
    t_f, Fy_f, Mz_f = read_all_forces(t_end)
    _, idx = np.unique(t_f, return_index=True)
    t_f, Fy_f, Mz_f = t_f[idx], Fy_f[idx], Mz_f[idx]
    print(f"  Loaded {len(t_f)} force samples")

    import json
    t0_file = CASE_DIR / "cosim_state_t0.json"
    h0, hd0, a0, ad0 = 0.0, 0.0, 0.0, 0.0
    if t0_file.exists():
        with open(t0_file) as f:
            s = json.load(f)
        h0, hd0, a0, ad0 = s["h"], s["hd"], s["a"], s["ad"]

    h_f, hd_f, a_f, ad_f, h_arr, hd_arr, a_arr, ad_arr = integrate_structural(
        h0, hd0, a0, ad0, t_f, Fy_f, Mz_f
    )
    return (
        t_f, Fy_f, Mz_f, t_f,
        h_arr * 1000.0, np.degrees(a_arr),
        hd_arr * 1000.0, np.degrees(ad_arr),
        None, None,  # W_gust and delta not available from fallback
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
AREF = 0.05    # reference area [m²] (chord=1m × span=0.05m, matches controlDict Aref)
Q_INF = 0.5 * RHO * U_INF**2   # dynamic pressure [Pa]


def plot(t_f, Fy_f, Mz_f, t_s, h_s, a_s, hd_s, ad_s, t_end, t_start=0.0,
         W_gust_s=None, delta_s_csv=None):
    t_fm, Fy_fm, Mz_fm = t_f, Fy_f, Mz_f
    Cl = Fy_fm / (Q_INF * AREF)
    Cm = Mz_fm / (Q_INF * AREF * 1.0)   # chord = 1.0 m

    # Use CSV columns if available, otherwise recompute from schedule
    if W_gust_s is not None:
        gust_s = W_gust_s
    else:
        gust_s = np.array([
            (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * (t - GUST_T_START) / (GUST_T_END - GUST_T_START)))
            if GUST_T_START <= t <= GUST_T_END else 0.0
            for t in t_s
        ])
    if delta_s_csv is not None:
        delta_s = delta_s_csv
    else:
        delta_s = np.array([delta_schedule(t) for t in t_s])

    fig, axes = plt.subplots(9, 1, figsize=(13, 22), sharex=True)
    axes[0].set_xlim(t_start, t_end)
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
    ax.set_xlim(t_start, t_end)
    ax.grid(True, lw=0.4, alpha=0.5)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "gust_response.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-end",        type=float, default=2.0)
    parser.add_argument("--t-start",      type=float, default=0.0)
    parser.add_argument("--gust-t-start", type=float, default=None,
                        help="Override gust start time for plot annotations [s]")
    parser.add_argument("--gust-t-end",   type=float, default=None,
                        help="Override gust end time for plot annotations [s]")
    parser.add_argument("--gust-w0",      type=float, default=None,
                        help="Override peak gust velocity [m/s]")
    parser.add_argument("--delta-times",  type=float, nargs="+", default=None,
                        help="Flap schedule time knots [s]")
    parser.add_argument("--delta-angles", type=float, nargs="+", default=None,
                        help="Flap schedule angle knots [deg]")
    args = parser.parse_args()

    # Override gust and flap parameters for annotations if provided
    global GUST_T_START, GUST_T_END, GUST_W0, T_GUST_ARRIVE, T_FLAP_START, T_FLAP_END
    global DELTA_TIMES, DELTA_ANGLES
    if args.gust_w0 is not None:
        GUST_W0 = args.gust_w0
    if args.gust_t_start is not None:
        GUST_T_START  = args.gust_t_start
        T_GUST_ARRIVE = GUST_T_START + 10.0 / U_INF
    if args.gust_t_end is not None:
        GUST_T_END = args.gust_t_end
    if args.delta_times is not None:
        DELTA_TIMES = args.delta_times
    if args.delta_angles is not None:
        DELTA_ANGLES = args.delta_angles

    t_f, Fy_f, Mz_f, t_s, h_s, a_s, hd_s, ad_s, W_gust_s, delta_s = reconstruct_structural(args.t_end)
    plot(t_f, Fy_f, Mz_f, t_s, h_s, a_s, hd_s, ad_s, args.t_end, t_start=args.t_start,
         W_gust_s=W_gust_s, delta_s_csv=delta_s)


if __name__ == "__main__":
    main()
