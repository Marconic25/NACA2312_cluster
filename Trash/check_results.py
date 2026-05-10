"""
check_results.py
Quick validation of the imposed-motion test case results.
Reads forces/forceCoeffs from postProcessing and checks for divergence.
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "testCase_imposed")

# ── Locate force files ─────────────────────────────────────────────────────────
def find_data_file(subdir, filename_pattern):
    """Search postProcessing/<subdir>/<startTime>/<filename_pattern>"""
    pattern = os.path.join(CASE_DIR, "postProcessing", subdir,
                           "*", filename_pattern)
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]   # use latest start time


def load_forces(filepath):
    """Parse OpenFOAM forces.dat — columns: t Fx Fy Fz Mx My Mz (pressure+viscous)."""
    rows = []
    with open(filepath) as fp:
        for line in fp:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # Remove parentheses and split
            s = s.replace("(", " ").replace(")", " ")
            parts = s.split()
            if len(parts) >= 7:
                try:
                    rows.append([float(p) for p in parts[:7]])
                except ValueError:
                    continue
    if not rows:
        return None
    arr = np.array(rows)
    # columns: t Fx Fy Fz Mx My Mz  (total = pressure + viscous summed by OF)
    return arr


def load_force_coeffs(filepath):
    """Parse OpenFOAM forceCoeffs.dat — columns: t Cd Cs Cl CmRoll CmPitch CmYaw."""
    rows = []
    with open(filepath) as fp:
        for line in fp:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 7:
                try:
                    rows.append([float(p) for p in parts[:7]])
                except ValueError:
                    continue
    if not rows:
        return None
    return np.array(rows)


# ── Check for divergence ───────────────────────────────────────────────────────
def check_divergence(arr, col, name, threshold=1e6):
    vals = arr[:, col]
    if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
        print(f"  FAILED: {name} contains NaN/Inf")
        return False
    if np.any(np.abs(vals) > threshold):
        print(f"  FAILED: {name} exceeds {threshold:.0e}  "
              f"(max |val| = {np.max(np.abs(vals)):.3e})")
        return False
    return True


# ── Main ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("check_results.py — imposed-motion test case")
print("=" * 60)

ok = True

# --- Forces ---
forces_file = find_data_file("forces", "force.dat")
if forces_file is None:
    forces_file = find_data_file("forces", "forces.dat")

if forces_file is None:
    print("WARNING: no forces file found in postProcessing/forces/")
    ok = False
else:
    print(f"\nForces file: {forces_file}")
    fdata = load_forces(forces_file)
    if fdata is None or len(fdata) == 0:
        print("  WARNING: file empty or could not be parsed")
        ok = False
    else:
        t    = fdata[:, 0]
        Fy   = fdata[:, 2]   # lift direction (y)
        Fx   = fdata[:, 1]   # drag direction (x)
        My   = fdata[:, 5]   # pitch moment

        ok_Fx = check_divergence(fdata, 1, "Fx (drag)")
        ok_Fy = check_divergence(fdata, 2, "Fy (lift)")
        ok_My = check_divergence(fdata, 5, "My (pitch moment)")

        if ok_Fx and ok_Fy and ok_My:
            print(f"  CONVERGED  ({len(t)} time steps, "
                  f"t = {t[0]:.3f} .. {t[-1]:.3f} s)")
            print(f"  Max |Fy|  = {np.max(np.abs(Fy)):.2f} N")
            print(f"  Max |Fx|  = {np.max(np.abs(Fx)):.2f} N")
            print(f"  Max |My|  = {np.max(np.abs(My)):.4f} N·m")
        else:
            ok = False

        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t, Fy, "b")
        axes[0].set_ylabel("Fy — lift [N]")
        axes[0].axvline(0.5, color="r", ls="--", alpha=0.5, label="gust onset")
        axes[0].legend(fontsize=8)
        axes[0].grid(True)

        axes[1].plot(t, Fx, "g")
        axes[1].set_ylabel("Fx — drag [N]")
        axes[1].axvline(0.5, color="r", ls="--", alpha=0.5)
        axes[1].grid(True)

        axes[2].plot(t, My, "m")
        axes[2].set_ylabel("My — pitch [N·m]")
        axes[2].set_xlabel("Time [s]")
        axes[2].axvline(0.5, color="r", ls="--", alpha=0.5)
        axes[2].grid(True)

        fig.suptitle("Aerodynamic Forces — imposed-motion test case")
        plt.tight_layout()
        out = os.path.join(CASE_DIR, "forces_timeseries.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Plot saved: {out}")

# --- Force coefficients ---
coeff_file = find_data_file("forceCoeffs", "forceCoeffs.dat")
if coeff_file is None:
    coeff_file = find_data_file("forceCoeffs", "coefficient.dat")

if coeff_file is not None:
    print(f"\nForceCoeffs file: {coeff_file}")
    cdata = load_force_coeffs(coeff_file)
    if cdata is not None and len(cdata) > 0:
        t   = cdata[:, 0]
        Cd  = cdata[:, 1]
        Cl  = cdata[:, 3]
        print(f"  t = {t[0]:.3f} .. {t[-1]:.3f} s")
        print(f"  Cl range: [{np.min(Cl):.3f}, {np.max(Cl):.3f}]")
        print(f"  Cd range: [{np.min(Cd):.4f}, {np.max(Cd):.4f}]")

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t, Cl, "b")
        axes[0].set_ylabel("CL")
        axes[0].axvline(0.5, color="r", ls="--", alpha=0.5, label="gust onset")
        axes[0].legend(fontsize=8)
        axes[0].grid(True)

        axes[1].plot(t, Cd, "g")
        axes[1].set_ylabel("CD")
        axes[1].set_xlabel("Time [s]")
        axes[1].axvline(0.5, color="r", ls="--", alpha=0.5)
        axes[1].grid(True)

        fig.suptitle("Force Coefficients — imposed-motion test case")
        plt.tight_layout()
        out = os.path.join(CASE_DIR, "coefficients_timeseries.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Plot saved: {out}")

# --- Check log for divergence keywords ---
log_file = os.path.join(CASE_DIR, "log.pimpleFoam")
if os.path.exists(log_file):
    print(f"\nLog file: {log_file}")
    with open(log_file) as fp:
        lines = fp.readlines()

    diverge_lines = [l.strip() for l in lines
                     if any(kw in l.lower()
                            for kw in ["diverge", "nan", "inf", "floating"])]
    if diverge_lines:
        print("  WARNING — divergence indicators found in log:")
        for dl in diverge_lines[:5]:
            print(f"    {dl}")
        ok = False
    else:
        # Report last time step reached
        time_lines = [l for l in lines if l.startswith("Time =")]
        if time_lines:
            print(f"  Last time step: {time_lines[-1].strip()}")

print()
if ok:
    print("RESULT: PASSED — simulation appears stable")
else:
    print("RESULT: FAILED — check warnings above")
    sys.exit(1)
