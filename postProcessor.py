#!/usr/bin/env python3
"""
postProcessor.py — Plot Cp, Cl/Cd/Cm, and residuals from an OpenFOAM case.

Usage:
    python3 postProcessor.py <case_dir>
    python3 postProcessor.py wingMotion2D_simpleFoam

Cp plot requires running sample first:
    cd <case_dir> && postProcess -func sample
"""

import sys
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# ── Reference values ──────────────────────────────────────────────────
RHO   = 1.225                       # kg/m3
UINF  = 100.0                       # m/s
PINF  = 0.0                         # Pa
QINF  = 0.5 * RHO * UINF**2        # dynamic pressure
CHORD = 1.0                         # m


# ── Readers ───────────────────────────────────────────────────────────

def read_force_coeffs(case_dir):
    """Read forceCoeffs.dat.
       OF7 forceCoeffs writes one total line per timestep:
           Time  Cm  Cd  Cl  Cl(f)  Cl(r)
       Returns time, Cd, Cl, Cm as numpy arrays."""
    path = os.path.join(case_dir, "postProcessing", "forceCoeffs", "0", "forceCoeffs.dat")

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()])

    rows = np.array(rows)
    time = rows[:, 0]
    Cm   = rows[:, 1]
    Cd   = rows[:, 2]
    Cl   = rows[:, 3]

    return time, Cd, Cl, Cm


def read_residuals(case_dir):
    """Parse log.<solver> for Final residual of each variable.
       Returns dict {varname: [values]}."""
    candidates = glob.glob(os.path.join(case_dir, "log.*Foam"))
    if not candidates:
        return None
    log_path = sorted(candidates)[-1]

    residuals = {}
    pattern = re.compile(r"Solving for\s+(\S+),.*Final residual\s*[=:]\s*([\d.eE+-]+)")

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                var = m.group(1).rstrip(",")
                val = float(m.group(2))
                residuals.setdefault(var, []).append(val)

    return residuals if residuals else None


def read_cp_sample(case_dir):
    """Read pressure on wing patches from postProcessing/surfaces/<latestTime>/p_*.raw.
       Combines wing_main and flap, deduplicates the 2-D slab (keeps one z-plane),
       and returns x/c, Cp arrays sorted by x. None if no data."""
    surf_base = os.path.join(case_dir, "postProcessing", "surfaces")
    if not os.path.isdir(surf_base):
        return None, None

    # Find latest time directory
    time_dirs = []
    for d in os.listdir(surf_base):
        try:
            time_dirs.append((float(d), d))
        except ValueError:
            pass
    if not time_dirs:
        return None, None
    time_dirs.sort()
    latest = os.path.join(surf_base, time_dirs[-1][1])

    # Read all p_*.raw files (p_wing_main.raw, p_flap.raw)
    p_files = glob.glob(os.path.join(latest, "p_*.raw"))
    if not p_files:
        return None, None

    all_x, all_y, all_p = [], [], []
    for fpath in p_files:
        data = np.loadtxt(fpath, comments="#")   # skip # header lines
        # Columns: x  y  z  p
        all_x.append(data[:, 0])
        all_y.append(data[:, 1])
        all_p.append(data[:, 3])

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)

    # Deduplicate: 2-D slab has two z-planes; keep unique (x, y) pairs
    _, idx = np.unique(np.column_stack((x, y)), axis=0, return_index=True)
    x, y, p = x[idx], y[idx], p[idx]

    Cp = (p - PINF) / QINF
    xc = x / CHORD

    # Split upper / lower by y relative to median (chord line)
    y_mid = np.median(y)
    upper = y >= y_mid
    lower = y < y_mid

    return xc, Cp, upper, lower


# ── Plotting ──────────────────────────────────────────────────────────

def plot(case_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    case_name = os.path.basename(os.path.abspath(case_dir))
    fig.suptitle(case_name, fontsize=13, fontweight="bold")

    # ── 1. Cp distribution ──────────────────────────────────────────
    ax = axes[0]
    result = read_cp_sample(case_dir)
    if result[0] is not None:
        xc, Cp, upper, lower = result
        idx_u = np.argsort(xc[upper])
        idx_l = np.argsort(xc[lower])
        ax.plot(xc[upper][idx_u], Cp[upper][idx_u], "b-", linewidth=1.5, label="upper")
        ax.plot(xc[lower][idx_l], Cp[lower][idx_l], "r-", linewidth=1.5, label="lower")
        ax.set_xlabel("x / c")
        ax.set_ylabel("Cp")
        ax.invert_yaxis()
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5,
                "No surface data.\nRun:\n  cd <case>\n  postProcess -func surfaces",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
    ax.set_title("Cp distribution")

    # ── 2. Cl / Cd / Cm vs time ─────────────────────────────────────
    ax = axes[1]
    try:
        time, Cd, Cl, Cm = read_force_coeffs(case_dir)
        ax.plot(time, Cl, label="Cl", color="blue",   linewidth=1.2)
        ax.plot(time, Cd, label="Cd", color="red",    linewidth=1.2)
        ax.plot(time, Cm, label="Cm", color="green",  linewidth=1.2)
        ax.set_xlabel("Time / Iteration")
        ax.set_ylabel("Coefficient")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f"No forceCoeffs data\n{e}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
    ax.set_title("Force coefficients")

    # ── 3. Residuals ────────────────────────────────────────────────
    ax = axes[2]
    residuals = read_residuals(case_dir)
    if residuals:
        color_map = {
            "p": "blue", "Ux": "red", "Uy": "orange", "Uz": "brown",
            "k": "green", "omega": "purple"
        }
        for var in sorted(residuals.keys()):
            ax.semilogy(residuals[var], label=var,
                        color=color_map.get(var, "black"), linewidth=1.2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Final residual")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, which="both")
    else:
        ax.text(0.5, 0.5, "No residual data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
    ax.set_title("Residuals")

    plt.tight_layout()
    out_path = os.path.join(case_dir, "postProcessing_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    case_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    plot(case_dir)
