"""
plot_temporal_comparison.py — Compare CL(t) between two timestep runs.

Usage:
    python3 plot_temporal_comparison.py \
        --ref  /path/to/wingMotion2D_pimpleFoam \
        --comp /path/to/temporal_dt2e5 \
        --out  temporal_comparison.png

The script reads postProcessing/forceCoeffs from each case and overlays
CL(t), CD(t) on the same axes, plus a difference plot.
"""

import argparse
import re
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "figure.dpi":     150,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.2,
})


def load_force_coeffs(case_dir: Path):
    """
    Load CL, CD, CM from postProcessing/forceCoeffs or forces windows.
    Returns (t, CL, CD, CM) arrays sorted by time.
    """
    # Try single forceCoeffs file first
    candidates = sorted(case_dir.glob("postProcessing/forceCoeffs/*/forceCoeffs.dat"))
    
    # If multiple windows (cosim style), concatenate all
    if not candidates:
        candidates = sorted(case_dir.glob("postProcessing/forces/*/forces.dat"))
        if not candidates:
            raise FileNotFoundError(f"No forceCoeffs/forces data in {case_dir}")
        return _load_from_forces(candidates)

    t_all, cm_all, cd_all, cl_all = [], [], [], []
    for f in candidates:
        for line in f.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            cols = line.split()
            if len(cols) >= 4:
                try:
                    t_all.append(float(cols[0]))
                    cm_all.append(float(cols[1]))
                    cd_all.append(float(cols[2]))
                    cl_all.append(float(cols[3]))
                except ValueError:
                    pass

    idx = np.argsort(t_all)
    return (np.array(t_all)[idx], np.array(cl_all)[idx],
            np.array(cd_all)[idx], np.array(cm_all)[idx])


def _load_from_forces(force_files):
    """Load from raw forces.dat files (cosim window format)."""
    t_all, cl_all, cd_all, cm_all = [], [], [], []
    rho   = 1.225
    U_inf = 80.0
    A_ref = 0.05
    c_ref = 1.0
    q     = 0.5 * rho * U_inf**2

    for fpath in force_files:
        for line in fpath.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if len(nums) < 19:
                continue
            try:
                t   = float(nums[0])
                Fy  = float(nums[2])  + float(nums[5])   # pressure + viscous
                Fx  = float(nums[1])  + float(nums[4])
                Mz  = float(nums[12]) + float(nums[15])
                cl  = Fy / (q * A_ref)
                cd  = Fx / (q * A_ref)
                cm  = Mz / (q * A_ref * c_ref)
                t_all.append(t)
                cl_all.append(cl)
                cd_all.append(cd)
                cm_all.append(cm)
            except (ValueError, IndexError):
                pass

    idx = np.argsort(t_all)
    return (np.array(t_all)[idx], np.array(cl_all)[idx],
            np.array(cd_all)[idx], np.array(cm_all)[idx])


def interp_to_common(t1, y1, t2, y2):
    """Interpolate both signals onto a common time axis."""
    t_min = max(t1[0],  t2[0])
    t_max = min(t1[-1], t2[-1])
    if t_max <= t_min:
        raise ValueError("No overlapping time range between the two runs.")
    t_common = np.linspace(t_min, t_max, min(len(t1), len(t2), 5000))
    y1_i = np.interp(t_common, t1, y1)
    y2_i = np.interp(t_common, t2, y2)
    return t_common, y1_i, y2_i


def plot_comparison(ref_dir, comp_dir, out_path, label_ref, label_comp):
    t1, cl1, cd1, cm1 = load_force_coeffs(ref_dir)
    t2, cl2, cd2, cm2 = load_force_coeffs(comp_dir)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # ── CL ────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t1, cl1, color="#378ADD", lw=1.2, label=label_ref,  alpha=0.9)
    ax.plot(t2, cl2, color="#D85A30", lw=1.2, label=label_comp, alpha=0.9, ls="--")
    ax.set_ylabel("CL")
    ax.set_title("Lift coefficient CL(t)")
    ax.legend()

    # ── CD ────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t1, cd1, color="#378ADD", lw=1.2, label=label_ref,  alpha=0.9)
    ax.plot(t2, cd2, color="#D85A30", lw=1.2, label=label_comp, alpha=0.9, ls="--")
    ax.set_ylabel("CD")
    ax.set_title("Drag coefficient CD(t)")
    ax.legend()

    # ── CL difference ─────────────────────────────────────────────────────
    ax = axes[2]
    try:
        t_c, cl1_i, cl2_i = interp_to_common(t1, cl1, t2, cl2)
        diff = cl1_i - cl2_i
        rel_err = np.abs(diff) / (np.abs(cl1_i) + 1e-10) * 100
        ax.plot(t_c, diff,    color="#7F77DD", lw=1.0, label="|ΔCL|")
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_ylabel("ΔCL")
        ax.set_title(f"CL difference ({label_ref} − {label_comp})   "
                     f"max|ΔCL|={np.max(np.abs(diff)):.4f}  "
                     f"RMS={np.sqrt(np.mean(diff**2)):.4f}")
        ax.legend()
    except ValueError as e:
        ax.text(0.5, 0.5, str(e), transform=ax.transAxes, ha="center")

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Temporal discretization comparison\n"
                 f"{label_ref} vs {label_comp}",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Print summary stats
    print(f"\nSummary:")
    print(f"  {label_ref:20s}: t=[{t1[0]:.3f}, {t1[-1]:.3f}]s  N={len(t1)}")
    print(f"  {label_comp:20s}: t=[{t2[0]:.3f}, {t2[-1]:.3f}]s  N={len(t2)}")
    try:
        t_c, cl1_i, cl2_i = interp_to_common(t1, cl1, t2, cl2)
        diff = cl1_i - cl2_i
        print(f"  max|ΔCL| = {np.max(np.abs(diff)):.4f}")
        print(f"  RMS ΔCL  = {np.sqrt(np.mean(diff**2)):.4f}")
        print(f"  Relative error (mean) = {np.mean(np.abs(diff)/(np.abs(cl1_i)+1e-10))*100:.2f}%")
    except ValueError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",  required=True, help="Reference case directory (dt=1e-5)")
    parser.add_argument("--comp", required=True, help="Comparison case directory (dt=2e-5)")
    parser.add_argument("--out",  default="temporal_comparison.png")
    parser.add_argument("--label-ref",  default="dt=1e-5s")
    parser.add_argument("--label-comp", default="dt=2e-5s")
    args = parser.parse_args()

    plot_comparison(
        Path(args.ref),
        Path(args.comp),
        Path(args.out),
        args.label_ref,
        args.label_comp,
    )
