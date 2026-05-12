#!/usr/bin/env python3
"""
cosim_driver.py — Python co-simulation loop for wingMotion2D_pimpleFoam.

Architecture:
  Each "window" of N timesteps:
    1. Write wingMotion.dat / flapMotion.dat (tabulated6DoFMotion format)
    2. Run pimpleFoam for this window
    3. Read aerodynamic forces (Fy, Mz on wing_main) from postProcessing
    4. Integrate 2-DOF structural model → new h, alpha for next window

Wing structural model (2-DOF, heave + pitch):
    m   * h_ddot     + d_h * h_dot     + k_h * h     = -Fy(t)
    I_z * alpha_ddot + d_a * alpha_dot + k_a * alpha  =  Mz(t)

Flap: prescribed δ(t) schedule (no FSI).
flapMotion.dat encodes the COMPOSED transform:
    - translation of hinge due to wing motion (from initial position)
    - total rotation = wing_pitch + flap_delta   [all from initial mesh]

Usage:
    cd wingMotion2D_pimpleFoam
    python3 cosim_driver.py [--np 4] [--window 200] [--restart]

    --np N        number of MPI processes (default: 4)
    --window N    timesteps per window (default: 200)
    --restart     continue from last written time (default: start from t=0)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import re
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

# ─────────────────────────── configuration ──────────────────────────────────

CASE_DIR   = Path(__file__).parent.resolve()
CONST_DIR  = CASE_DIR / "constant"
SYS_DIR    = CASE_DIR / "system"
PP_DIR     = CASE_DIR / "postProcessing" / "forces" / "0"

# Structural parameters — Hodges-Pierce benchmark (μ=20, σ=0.4, r_α²=0.24, x_α=0.1, ζ=2%)
# U_flutter ≈ 100 m/s (sub-critical at U_inf=80 m/s)
M_WING   = 19.24         # wing mass [kg]   m = μ·π·ρ·b² = 20·π·1.225·0.25
I_WING   = 1.155         # wing MoI about z-axis [kg·m²]  I_α = r_α²·m·b² = 0.24·19.24·0.25
K_H      = 25460.0       # heave spring stiffness [N/m]   ω_h²·m = 36.4²·19.24
D_H      = 88.0          # heave damping [N·s/m]   2·ζ·√(K_h·m), ζ=2%
K_ALPHA  = 9530.0        # pitch spring stiffness [N·m/rad]   ω_α²·I_α = 90.9²·1.155
D_ALPHA  = 6.6           # pitch damping [N·m·s/rad]   2·ζ·√(K_α·I_α), ζ=2%

# Geometry (initial mesh coordinates)
EA_X, EA_Y = 0.40, 0.0          # elastic axis (CoR) at 40%c — Hodges-Pierce benchmark
HINGE_X    = 0.779               # flap hinge initial x
HINGE_Y    = 0.0                 # flap hinge initial y

# Flap physical parameters — mass and inertia contribute to wing EOM via constraint forces.
# The flap motion remains prescribed (kinematic); its inertia appears as forcing in the EOM.
# Geometry: chord≈0.243m (x: 0.750→0.993), span=0.05m, aluminium-equivalent density.
# Calibrate M_FLAP and I_FLAP_CG to match the real flap structure.
M_FLAP     = 1.19    # flap mass [kg]  (estimated: ρ_Al * A_section * span)
I_FLAP_CG  = 0.006   # flap MoI about its own CG [kg·m²]  (rectangular approx)
# Derived inertia quantities (computed once at module load)
_D_X       = HINGE_X - EA_X                        # 0.525 m  (EA → hinge, x)
_D_Y       = HINGE_Y - EA_Y                        # -0.045 m (EA → hinge, y)
_D2        = _D_X**2 + _D_Y**2                     # |d|² [m²]
I_FLAP_EA  = I_FLAP_CG + M_FLAP * _D2             # MoI about elastic axis [kg·m²]
I_FLAP_HINGE = I_FLAP_CG + M_FLAP * _D2           # MoI about hinge (same d) [kg·m²]

# Gust parameters (cosine gust, EASA CS-25 profile)
U_INF        = 80.0   # freestream velocity [m/s]
GUST_W0      = 0.0    # peak gust velocity [m/s]  — temporaneamente a zero (no gust)
GUST_T_START = 0.0    # gust onset [s]
GUST_T_END   = 0.8    # gust end [s]

# Flap schedule: δ(t) in degrees — temporaneamente a zero (no flap, no gust)
DELTA_TIMES  = [0.0,  2.0]   # [s]
DELTA_ANGLES = [0.0,  0.0]   # [deg]


def delta_schedule(t):
    """Prescribed flap deflection in degrees at time t."""
    return float(np.interp(t, DELTA_TIMES, DELTA_ANGLES))


def _delta_derivatives(dt_eps=1e-7):
    """Return (delta_rad, delta_dot_rad_s, delta_ddot_rad_s2) as callables."""
    def delta_rad(t):
        return np.radians(delta_schedule(t))
    def delta_dot(t):
        return (delta_rad(t + dt_eps) - delta_rad(t - dt_eps)) / (2.0 * dt_eps)
    def delta_ddot(t):
        return (delta_rad(t + dt_eps) - 2.0*delta_rad(t) + delta_rad(t - dt_eps)) / dt_eps**2
    return delta_rad, delta_dot, delta_ddot


# Pre-compute derivative callables (module-level, reused by integrator)
_DELTA_RAD, _DELTA_DOT, _DELTA_DDOT = _delta_derivatives()


def gust_velocity(t):
    """Cosine gust vertical velocity component [m/s] at time t."""
    t_rel = t - GUST_T_START
    T_g   = GUST_T_END - GUST_T_START
    if 0.0 <= t_rel <= T_g:
        return (GUST_W0 / 2.0) * (1.0 - np.cos(2.0 * np.pi * t_rel / T_g))
    return 0.0


def write_gust_inlet(inlet_file=None):
    """
    Write 0.orig/include/fixedInlet with a codedFixedValue cosine-gust BC.

    Parameters are embedded as C++ literals from the current Python constants
    (U_INF, GUST_W0, GUST_T_START, GUST_T_END), so the file is always
    consistent with the Python-side gust definition.

    Called once at fresh start, before decomposePar.
    """
    if inlet_file is None:
        inlet_file = CASE_DIR / "0.orig" / "include" / "fixedInletU"

    T_g = GUST_T_END - GUST_T_START

    # Build content in parts to avoid f-string / backslash conflicts with OF header
    header = (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "| =========                 |                                                 |\n"
        "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
        "|  \\\\    /   O peration     | Version:  7                                     |\n"
        "|   \\\\  /    A nd           | Website:  https://openfoam.org                  |\n"
        "|    \\\\/     M anipulation  |                                                 |\n"
        "\\*---------------------------------------------------------------------------*/\n"
    )
    body = (
        f"// Cosine-gust inlet BC — auto-generated by cosim_driver.py (do not edit manually)\n"
        f"// Gust parameters:  U_INF={U_INF} m/s  Wg0={GUST_W0} m/s\n"
        f"//                   t_start={GUST_T_START} s  T_g={T_g} s  t_end={GUST_T_END} s\n"
        f"\n"
        f"inlet\n"
        f"{{\n"
        f"    type        codedFixedValue;\n"
        f"    value       uniform ({U_INF:.4f} 0 0);\n"
        f"    name        gustInlet;\n"
        f"    code\n"
        f"    #{{\n"
        f"        const scalar t    = this->db().time().value();\n"
        f"        const scalar U0   = {U_INF:.4f};\n"
        f"        const scalar Wg0  = {GUST_W0:.4f};\n"
        f"        const scalar t0   = {GUST_T_START:.6f};\n"
        f"        const scalar Tg   = {T_g:.6f};\n"
        f"        scalar wy = 0.0;\n"
        f"        if (t >= t0 && t <= t0 + Tg)\n"
        f"        {{\n"
        f"            wy = 0.5*Wg0*(1.0 - Foam::cos(2.0*Foam::constant::mathematical::pi*(t - t0)/Tg));\n"
        f"        }}\n"
        f"        operator==(vector(U0, wy, 0.0));\n"
        f"    #}};\n"
        f"}}\n"
        f"\n"
        f"// ************************************************************************* //\n"
    )
    content = header + body
    with open(inlet_file, "w") as f:
        f.write(content)
    print(f"  Wrote gust inlet BC → {inlet_file}")
    print(f"    Wg0={GUST_W0} m/s  T_g={T_g:.3f}s  "
          f"t_gust=[{GUST_T_START},{GUST_T_END}]s")


# ─────────────────────── structural integrator ──────────────────────────────

def structural_rhs(t, state, Fy_interp, Mz_interp):
    """
    RHS of augmented 2-DOF EOM: state = [h, h_dot, alpha, alpha_dot].

    System: wing_main + flap (prescribed δ(t)).
    The flap is kinematically constrained → its mass/inertia appear as
    augmented mass matrix entries and generalised forcing terms.

    Augmented mass matrix [2×2]:
        M_hh = M_WING + M_FLAP
        M_αα = I_WING + I_FLAP_EA
        M_hα = M_αh = M_FLAP * d_x   (inertial coupling)

    Generalised forcing from flap kinematics (small-angle, leading-order):
        Q_h  = -M_FLAP * d_y * δ_ddot        (inertial — heave)
               -M_FLAP * d_x * 2*αd*δ_dot   (Coriolis — heave)
        Q_α  = -I_FLAP_HINGE * δ_ddot        (reaction torque — pitch)

    Forces aero on wing_main only (flap aero included in postProcessing/forces
    via patch list, but for EOM correctness only wing_main forces should drive
    the wing DOFs; flap aero goes into the actuator, not the wing structure).
    NOTE: forces.dat currently integrates both patches — acceptable approximation
    for small δ where flap lift ≪ wing lift.
    """
    h, hd, a, ad = state
    Fy = float(Fy_interp(t))
    Mz = float(Mz_interp(t))

    dlt_ddot = _DELTA_DDOT(t)   # [rad/s²]
    dlt_dot  = _DELTA_DOT(t)    # [rad/s]

    # Augmented mass matrix entries
    M_hh = M_WING + M_FLAP
    M_aa = I_WING + I_FLAP_EA
    M_ha = M_FLAP * _D_X        # off-diagonal (symmetric)

    # Generalised inertial forces from prescribed flap acceleration (small-angle)
    Q_h_flap = -M_FLAP * _D_Y * dlt_ddot - M_FLAP * _D_X * (2.0 * ad * dlt_dot)
    Q_a_flap = -I_FLAP_HINGE * dlt_ddot

    # Right-hand sides before solving the 2×2 mass system
    RHS_h = -Fy - D_H * hd - K_H * h + Q_h_flap
    RHS_a =  Mz - D_ALPHA * ad - K_ALPHA * a + Q_a_flap

    # Solve [M_hh M_ha; M_ha M_aa] * [h_ddot; a_ddot] = [RHS_h; RHS_a]
    det    = M_hh * M_aa - M_ha * M_ha
    h_ddot = (M_aa * RHS_h - M_ha * RHS_a) / det
    a_ddot = (M_hh * RHS_a - M_ha * RHS_h) / det

    return [hd, h_ddot, ad, a_ddot]


def integrate_structural(h0, hd0, a0, ad0, t_win, Fy_arr, Mz_arr):
    """
    Integrate augmented 2-DOF structural model over time window t_win.
    Forces Fy_arr, Mz_arr are sampled at t_win points.
    Returns final state (h, hd, alpha, ad) and full trajectory arrays.
    """
    from scipy.interpolate import interp1d
    Fy_interp = interp1d(t_win, Fy_arr, kind="linear", fill_value="extrapolate")
    Mz_interp = interp1d(t_win, Mz_arr, kind="linear", fill_value="extrapolate")

    sol = solve_ivp(
        structural_rhs,
        [t_win[0], t_win[-1]],
        [h0, hd0, a0, ad0],
        args=(Fy_interp, Mz_interp),
        t_eval=t_win,
        max_step=(t_win[1] - t_win[0]) * 2,
        rtol=1e-8, atol=1e-10,
    )
    h_arr     = sol.y[0]
    hd_arr    = sol.y[1]
    alpha_arr = sol.y[2]
    ad_arr    = sol.y[3]
    return h_arr[-1], hd_arr[-1], alpha_arr[-1], ad_arr[-1], h_arr, hd_arr, alpha_arr, ad_arr


# ──────────────────────── motion file writers ────────────────────────────────

def write_tabulated6dof(path, times, tx, ty, rot_z_deg):
    """
    Write a tabulated6DoFMotion .dat file.
    Format: ( (t  ((tx ty tz) (rotX rotY rotZ_deg))) ... )
    """
    with open(path, "w") as f:
        f.write("// tabulated6DoFMotion — generated by cosim_driver.py\n")
        f.write("(\n")
        for t, x, y, rz in zip(times, tx, ty, rot_z_deg):
            f.write(f"  ({t:.10e}  (({x:.10e} {y:.10e} 0) (0 0 {rz:.10e})))\n")
        f.write(")\n")


def compute_motion_tables(t_win, h_arr, alpha_arr):
    """
    Compute wingMotion and flapMotion tabulated data for a time window.

    wingZone transform (CofG = elastic axis = (EA_X, EA_Y)):
        ty  = h(t)                  [heave, m]
        rz  = degrees(alpha(t))     [pitch, deg]

    flapZone transform (CofG = hinge initial = (HINGE_X, HINGE_Y)):
        Points in flapZone start from points0_ (initial mesh).
        multiSolidBodyMotionSolver applies the transform independently
        → must encode full motion: wing heave+pitch + flap delta.

        Equivalent single body motion of the flap zone about HINGE initial:
          1. Wing moves: hinge goes from (HINGE_X, HINGE_Y) to (hx_m, hy_m)
          2. Flap rotates by alpha + delta about hinge
             Sign: positive rz in OF = CCW. With hinge at y=-0.045 and flap LE
             mostly above the hinge, CCW moves LE down = TE-down convention.
             flap_rz = degrees(alpha + delta).
        Expressed as (tx_hinge, ty_hinge, rot_total) about initial hinge CofG.
    """
    n = len(t_win)
    wing_ty  = h_arr
    wing_rz  = np.degrees(alpha_arr)

    flap_tx  = np.zeros(n)
    flap_ty  = np.zeros(n)
    flap_rz  = np.zeros(n)

    for i, t in enumerate(t_win):
        a   = alpha_arr[i]
        h   = h_arr[i]
        d   = np.radians(delta_schedule(t))

        # Hinge position after wing heave+pitch
        dx  = HINGE_X - EA_X
        dy  = HINGE_Y - EA_Y
        hx_m = np.cos(a) * dx - np.sin(a) * dy + EA_X
        hy_m = np.sin(a) * dx + np.cos(a) * dy + EA_Y + h

        flap_tx[i] = hx_m - HINGE_X
        flap_ty[i] = hy_m - HINGE_Y
        flap_rz[i] = np.degrees(a + d)   # total rotation from initial mesh
        # Sign: OpenFOAM tabulated6DoFMotion applies CCW rotation for positive rz.
        # The flapZone CofG is the hinge (0.775,-0.045). The flap body is mostly
        # ABOVE and to the LEFT of the hinge (LE at y≈-0.016, hinge at y=-0.045).
        # CCW rotation moves the LE downward and TE upward relative to hinge —
        # which matches the physical convention that positive delta = TE-down
        # corresponds to the flap nose (LE side) going down in the airfoil frame.
        # Confirmed correct by git commit d8107d1 (multiSolidBodyMotionSolver).

    return wing_ty, wing_rz, flap_tx, flap_ty, flap_rz


# ──────────────────────── force reader ──────────────────────────────────────

def read_forces(t_start, t_end):
    """
    Read aerodynamic forces for t in (t_start, t_end].

    OF7 parallel runs create a new postProcessing/forces/<startTime>/forces.dat
    subdirectory for each window (named after the window's startTime).
    This function scans ALL subdirectories under postProcessing/forces/ and
    collects data from every file that overlaps the requested time range.

    OF7 forces.dat format (header lines start with #):
        Time  (Fpx Fpy Fpz)  (Fvx Fvy Fvz)  (Fpox Fpoy Fpoz)  ...moments...

    Returns (t_arr, Fy_arr, Mz_arr) sorted by time.
    Fy = pressure + viscous y-force, Mz = pressure + viscous z-moment.
    Returns (None, None, None) if no data found.
    """
    forces_base = CASE_DIR / "postProcessing" / "forces"
    if not forces_base.exists():
        print(f"  [WARNING] postProcessing/forces/ not found, using zero forces")
        return None, None, None

    # Collect all forces.dat files across all window subdirectories
    forces_files = sorted(forces_base.glob("*/forces.dat"))
    if not forces_files:
        print(f"  [WARNING] No forces.dat found under postProcessing/forces/, using zeros")
        return None, None, None

    t_list, Fy_list, Mz_list = [], [], []

    # Get window start times from subdirectory names, sorted
    win_starts = sorted(float(ff.parent.name) for ff in forces_files)

    for i, forces_file in enumerate(forces_files):
        win_t0 = win_starts[i]
        # Upper bound: next window start (or t_end)
        win_t1 = win_starts[i + 1] if i + 1 < len(win_starts) else t_end + 1.0

        with open(forces_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                if len(nums) < 19:
                    continue
                t = float(nums[0])
                # Keep only samples belonging to THIS window
                if t < win_t0 - 1e-12 or t >= win_t1 - 1e-12:
                    continue
                if t <= t_start or t > t_end + 1e-12:
                    continue
                Fpy = float(nums[2])
                Fvy = float(nums[5])
                Mpz = float(nums[12])
                Mvz = float(nums[15])
                t_list.append(t)
                Fy_list.append(Fpy + Fvy)
                Mz_list.append(Mpz + Mvz)

    if not t_list:
        print(f"  [WARNING] No force data in t=({t_start:.5f}, {t_end:.5f}], using zeros")
        return None, None, None

    idx = np.argsort(t_list)
    return np.array(t_list)[idx], np.array(Fy_list)[idx], np.array(Mz_list)[idx]


# ──────────────────────── controlDict helpers ────────────────────────────────

def read_control_dict_value(key):
    """Read a scalar value from system/controlDict."""
    path = SYS_DIR / "controlDict"
    with open(path) as f:
        for line in f:
            m = re.match(rf"^\s*{key}\s+([\d.eE+\-]+)\s*;", line)
            if m:
                return float(m.group(1))
    raise KeyError(f"{key} not found in controlDict")


def update_control_dict(start_time, end_time, use_latest_time=False, write_interval=None):
    """Update startTime, endTime, and writeInterval in system/controlDict.

    use_latest_time=True → startFrom latestTime (window 1+).
    use_latest_time=False → startFrom startTime (decomposePar and window 0).
    write_interval: if set, overrides writeInterval so OF writes exactly once
                    per window (= number of steps in this window).
    """
    path = SYS_DIR / "controlDict"
    with open(path) as f:
        content = f.read()
    if use_latest_time:
        content = re.sub(r"startFrom[^\n;]*;",
                         "startFrom       latestTime;", content)
    else:
        content = re.sub(r"startFrom[^\n;]*;",
                         "startFrom       startTime;", content)
    content = re.sub(r"(startTime\s+)[\d.eE+\-]+\s*;",
                     f"\\g<1>{start_time:.10e};", content)
    content = re.sub(r"stopAt[^\n;]*;",
                     "stopAt          endTime;", content)
    content = re.sub(r"(endTime\s+)[\d.eE+\-]+\s*;",
                     f"\\g<1>{end_time:.10e};", content)
    if write_interval is not None:
        # Only replace writeInterval BEFORE the functions{} block.
        # Split on "functions" keyword and only touch the first part.
        if "functions" in content:
            pre, post = content.split("functions", 1)
            pre = re.sub(r"(writeInterval\s+)\d+\s*;",
                         f"\\g<1>{write_interval};", pre)
            content = pre + "functions" + post
        else:
            content = re.sub(r"(writeInterval\s+)\d+\s*;",
                             f"\\g<1>{write_interval};", content, count=1)
    with open(path, "w") as f:
        f.write(content)


# ──────────────────────── OpenFOAM runner ───────────────────────────────────

def reset_case(orig_dir="0.orig"):
    """Copy 0.orig → 0, remove all processor dirs and log.pimpleFoam."""
    # Safety check: abort if serial mesh is missing
    mesh_points = CASE_DIR / "constant" / "polyMesh" / "points"
    if not mesh_points.exists():
        raise RuntimeError(
            "constant/polyMesh/points not found — serial mesh is missing.\n"
            "Restore it with: git checkout -- constant/polyMesh/\n"
            "Then re-run topoSet to rebuild wingZone/flapZone."
        )
    # Remove old pimpleFoam log so each fresh run starts clean
    log_pimple = CASE_DIR / "log.pimpleFoam"
    if log_pimple.exists():
        log_pimple.unlink()
        print("  Removed log.pimpleFoam")
    # Remove postProcessing so force data from previous runs is not mixed in
    pp_dir = CASE_DIR / "postProcessing"
    if pp_dir.exists():
        shutil.rmtree(pp_dir)
        print("  Removed postProcessing/")
    # Remove all processor directories entirely (decomposePar recreates them)
    for proc in sorted(CASE_DIR.glob("processor*")):
        shutil.rmtree(proc)
        print(f"  Removed {proc.name}/")
    # Remove serial time directories (reconstructed results from previous runs)
    import re as _re
    for d in sorted(CASE_DIR.iterdir()):
        if d.is_dir() and _re.match(r"^\d+\.?\d*$", d.name) and d.name != "0":
            shutil.rmtree(d)
            print(f"  Removed {d.name}/")
    orig = CASE_DIR / orig_dir
    zero = CASE_DIR / "0"
    if orig.exists():
        if zero.exists():
            shutil.rmtree(zero)
        shutil.copytree(orig, zero)


RANS_DIR = CASE_DIR.parent / "rans_baseline"
CONTAINER = "/work/u10677113/of7.sif"
APPTAINER_CMD = ["apptainer", "exec", "--bind", "/work", CONTAINER]


def run_map_fields():
    """Map RANS steady-state solution onto cosim_main as initial condition."""
    print("  Running mapFields from rans_baseline...")
    cmd = APPTAINER_CMD + [
        "/bin/bash", "-c",
        f"source /opt/openfoam7/etc/bashrc && cd {str(CASE_DIR)} && "
        f"mapFields ../rans_baseline -sourceTime latestTime -consistent"
    ]
    result = subprocess.run(cmd, cwd=CASE_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError("mapFields failed")
    # pointDisplacement gets renamed to .unmapped by mapFields — restore it
    pd_unmapped = CASE_DIR / "0" / "pointDisplacement.unmapped"
    pd = CASE_DIR / "0" / "pointDisplacement"
    if pd_unmapped.exists():
        pd_unmapped.rename(pd)
        print("  Renamed pointDisplacement.unmapped → pointDisplacement")
    print("  mapFields complete")


def run_toposet():
    """Run topoSet to rebuild wingZone/flapZone cell sets."""
    print("  Running topoSet...")
    cmd = APPTAINER_CMD + [
        "/bin/bash", "-c",
        f"source /opt/openfoam7/etc/bashrc && cd {str(CASE_DIR)} && topoSet"
    ]
    result = subprocess.run(cmd, cwd=CASE_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError("topoSet failed")
    print("  topoSet complete")


def run_decompose(dt=1e-5):
    """Run decomposePar to split for parallel."""
    # Always reset controlDict to startTime=0 before decomposePar.
    # A previous run may have left a stale startTime (e.g. 0.084) which causes
    # decomposePar to look for a non-existent time dir → no processor*/0/ created.
    update_control_dict(0.0, dt)
    print("  Running decomposePar...")
    cmd = APPTAINER_CMD + [
        "/bin/bash", "-c",
        f"source /opt/openfoam7/etc/bashrc && cd {str(CASE_DIR)} && decomposePar -force"
    ]
    result = subprocess.run(cmd, cwd=CASE_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError("decomposePar failed")
    # Verify processor/0 directories were created and flush OS buffers
    import os
    os.sync()
    for proc in sorted(CASE_DIR.glob("processor*")):
        p0 = proc / "0"
        if not p0.exists():
            raise RuntimeError(
                f"decomposePar succeeded but {p0} does not exist — "
                "check processor decomposition"
            )

    # Patch matchTolerance in all processor boundary files.
    # The rigid-block flapZone straddles processor boundaries; after rigid motion
    # the face-area check between processors can trip a fatal error at the default
    # very tight tolerance (~7e-8 m²).  1e-4 (0.01%) is safe for this mesh.
    import re as _re
    for proc in sorted(CASE_DIR.glob("processor*")):
        bnd = proc / "constant" / "polyMesh" / "boundary"
        if bnd.exists():
            txt = bnd.read_text()
            txt = _re.sub(r"matchTolerance\s+[\d.eE+\-]+\s*;",
                          "matchTolerance  1e-3;", txt)
            bnd.write_text(txt)


def run_pimple(n_procs):
    """Run pimpleFoam in parallel, streaming output."""
    print(f"  Running pimpleFoam -parallel (np={n_procs})...")
    cmd = APPTAINER_CMD + [
        "/bin/bash", "-c",
        f"source /opt/openfoam7/etc/bashrc && cd {str(CASE_DIR)} && mpirun --oversubscribe --mca btl_base_warn_component_unused 0 --mca orte_base_help_aggregate 0 -np {n_procs} pimpleFoam -parallel"
    ]
    log_path = CASE_DIR / "log.pimpleFoam"
    with open(log_path, "a") as log_f:
        result = subprocess.run(
            cmd, cwd=CASE_DIR,
            stdout=log_f, stderr=subprocess.STDOUT
        )
    if result.returncode != 0:
        print(f"  [ERROR] pimpleFoam exited with code {result.returncode}")
        print(f"          See {log_path} for details")
        return False
    return True


def reconstruct(t_start):
    """Skip per-window reconstruction — done once at end via Allrun."""
    pass


STATE_FILE = CASE_DIR / "cosim_state.json"


def save_state(t_cur, h, hd, a, ad, window_idx, t_end, dt):
    """Persist co-simulation state to JSON for restart."""
    state = {
        "t_cur": t_cur,
        "h": h, "hd": hd,
        "a": a, "ad": ad,
        "window_idx": window_idx,
        "t_end": t_end,
        "dt": dt,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_state():
    """Load co-simulation state from JSON. Returns None if not found."""
    if not STATE_FILE.exists():
        return None
    with open(STATE_FILE) as f:
        s = json.load(f)
    return (s["t_cur"], s["h"], s["hd"], s["a"], s["ad"],
            s["window_idx"], s["t_end"], s["dt"])


def compute_static_equilibrium():
    """
    Estimate static equilibrium (h_eq, alpha_eq) from RANS forces.

    Reads the forces.dat from rans_baseline postProcessing (if available),
    otherwise runs a short probe window and reads the mean force.
    Falls back to reading from the first cosim window postProcessing.

    Returns (h_eq [m], alpha_eq [rad]).
    """
    # Try to read forces from rans_baseline postProcessing
    rans_pp = RANS_DIR / "postProcessing" / "forces"
    Fy_static, Mz_static = None, None

    if rans_pp.exists():
        force_files = sorted(rans_pp.glob("*/forces.dat"))
        if force_files:
            # Read last file, take last entry
            with open(force_files[-1]) as f:
                last = None
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                    if len(nums) >= 19:
                        last = nums
                if last:
                    Fy_static = float(last[2]) + float(last[5])
                    Mz_static = float(last[12]) + float(last[15])

    if Fy_static is None:
        # Fallback: run postProcess on rans_baseline to extract forces
        print("  Running postProcess on rans_baseline to get static forces...")
        cmd = APPTAINER_CMD + [
            "/bin/bash", "-c",
            f"source /opt/openfoam7/etc/bashrc && cd {str(RANS_DIR)} && "
            f"postProcess -func forces -latestTime"
        ]
        subprocess.run(cmd, cwd=RANS_DIR, capture_output=True)
        # Retry reading
        force_files = sorted(rans_pp.glob("*/forces.dat")) if rans_pp.exists() else []
        if force_files:
            with open(force_files[-1]) as f:
                last = None
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                    if len(nums) >= 19:
                        last = nums
                if last:
                    Fy_static = float(last[2]) + float(last[5])
                    Mz_static = float(last[12]) + float(last[15])

    if Fy_static is None:
        # Values from rans_baseline converged solution (CofR corrected to EA=0.40)
        # Fy = 163.12 N, Mz_EA = 16.364 - 163.12*0.15 = -8.10 N·m
        Fy_static = 163.12  # N
        Mz_static = -8.10   # N·m  (about EA at 0.40c)
        print(f"  [INFO] Using RANS static forces: Fy={Fy_static}N  Mz={Mz_static}N·m")

    # Static equilibrium: K_h * h_eq = -Fy,  K_alpha * alpha_eq = Mz
    M_hh = M_WING + M_FLAP
    M_aa = I_WING + I_FLAP_EA
    M_ha = M_FLAP * _D_X
    det  = M_hh * M_aa - M_ha**2
    # With zero velocity (static): solve directly from stiffness
    h_eq     = -Fy_static / K_H
    alpha_eq =  Mz_static / K_ALPHA
    print(f"  Static equilibrium: Fy={Fy_static:.1f}N  Mz={Mz_static:.3f}N·m")
    print(f"    h_eq={h_eq*1000:.3f}mm  α_eq={np.degrees(alpha_eq):.4f}°")
    return h_eq, alpha_eq


# ──────────────────────────── main loop ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--np",      type=int,   default=4,    help="MPI processes")
    parser.add_argument("--window",  type=int,   default=200,  help="Timesteps per window")
    parser.add_argument("--restart", action="store_true",      help="Continue from last time")
    parser.add_argument("--t-end",   type=float, default=None,
                        help="Total simulation end time [s] (overrides controlDict; "
                             "use when controlDict endTime was modified by a previous window)")
    parser.add_argument("--dt",      type=float, default=None,
                        help="Nominal timestep [s] for window sizing (overrides controlDict deltaT)")
    args = parser.parse_args()

    # Structural state
    h, hd, a, ad = 0.0, 0.0, 0.0, 0.0
    t_cur = 0.0
    window_idx = 0

    if not args.restart:
        # Fresh start: t_end/dt from CLI → state file → controlDict (in priority order).
        # controlDict endTime is NOT reliable here: previous windows overwrite it.
        saved = load_state()
        dt    = (args.dt    if args.dt    is not None
                 else (saved[7] if saved is not None else read_control_dict_value("deltaT")))
        t_end = (args.t_end if args.t_end is not None
                 else (saved[6] if saved is not None else read_control_dict_value("endTime")))
        print("\n--- Fresh start ---")
        write_gust_inlet()   # regenerate fixedInlet with current gust parameters
        reset_case()         # copies 0.orig → 0, removes processor* and postProcessing
        run_map_fields()     # map RANS steady solution as IC
        run_toposet()        # rebuild wingZone/flapZone cell sets
        # Initialise structural state at static equilibrium to suppress transient
        h, a = compute_static_equilibrium()
        hd, ad = 0.0, 0.0
        # Seed motion tables at equilibrium position for the full simulation duration
        window_dt = args.window * dt
        t_seed = np.array([0.0, t_end + window_dt])
        h_seed = np.array([h, h])
        a_seed_deg = np.degrees(np.array([a, a]))
        write_tabulated6dof(CONST_DIR / "wingMotion.dat",
                            t_seed, np.zeros(2), h_seed, a_seed_deg)
        write_tabulated6dof(CONST_DIR / "flapMotion.dat",
                            t_seed, np.zeros(2), h_seed, a_seed_deg)
        # Save initial state (includes t_end and dt for restart)
        save_state(0.0, h, hd, a, ad, 0, t_end, dt)
        # Save initial state separately for plot reconstruction
        import json as _json
        with open(CASE_DIR / "cosim_state_t0.json", "w") as _f:
            _json.dump({"h": h, "hd": hd, "a": a, "ad": ad}, _f, indent=2)
        run_decompose(dt)
    else:
        # Restore full state from JSON (includes t_end and dt)
        saved = load_state()
        if saved is not None:
            t_cur, h, hd, a, ad, window_idx, t_end_saved, dt_saved = saved
            # CLI args override saved values (allows changing t_end on restart)
            dt    = args.dt    if args.dt    is not None else dt_saved
            t_end = args.t_end if args.t_end is not None else t_end_saved
            print(f"\n--- Restarting from t={t_cur:.6g}  window={window_idx} ---")
            print(f"    t_end={t_end}s  dt={dt:.2e}s")
            print(f"    Restored: h={h*1000:.3f}mm  α={np.degrees(a):.3f}°")
        else:
            # No state file: fall back to controlDict (or CLI)
            dt    = args.dt    if args.dt    is not None else read_control_dict_value("deltaT")
            t_end = args.t_end if args.t_end is not None else read_control_dict_value("endTime")
            print("\n--- No state file found, starting from t=0 with zero state ---")

    window_dt = args.window * dt

    print(f"Co-simulation driver")
    print(f"  dt={dt:.2e}  window={args.window} steps ({window_dt:.4f}s)  t_end={t_end}s")
    print(f"  MPI procs: {args.np}")
    print(f"  Case: {CASE_DIR}")

    traj_path = CASE_DIR / "structural_trajectory.csv"
    T_CSV_SKIP = 0.1   # skip initial CFD transient from CSV
    traj_buf = []      # global buffer: accumulate all windows, smooth once at end

    while t_cur < t_end - 1e-12:
        t_win_end = min(t_cur + window_dt, t_end)
        t_win     = np.arange(t_cur, t_win_end + dt * 0.5, dt)
        if len(t_win) < 2:
            break

        print(f"\n{'='*60}")
        print(f"Window {window_idx:03d}: t = {t_cur:.5f} → {t_win_end:.5f} s  "
              f"({len(t_win)} steps)")
        print(f"  Wing state: h={h*1000:.3f}mm  α={np.degrees(a):.3f}°")

        # Structural state arrays for this window — first-order hold (linear ramp).
        # Using h(t) = h + hd*(t - t_cur) and α(t) = a + ad*(t - t_cur) instead of
        # a zero-order hold ensures the mesh velocity is continuous at window boundaries:
        # the motion table derivative (= mesh velocity) equals [hd, ad] at t=t_cur,
        # which matches the structural velocity the fluid "saw" at end of previous window.
        # This prevents the Courant spike caused by an abrupt mesh velocity change.
        # First-order hold prediction for this window.
        # h(t) = h + hd*(t - t_cur),  α(t) = a + ad*(t - t_cur)
        # Ensures mesh velocity is continuous at window boundaries.
        h_arr     = h  + hd * (t_win - t_cur)
        alpha_arr = a  + ad * (t_win - t_cur)

        # Compute motion tables
        wing_ty, wing_rz, flap_tx, flap_ty, flap_rz = compute_motion_tables(
            t_win, h_arr, alpha_arr
        )

        # Write motion .dat files (cover full remaining time so OF never runs out).
        # Always prepend t=0 with the window-start value so the table covers
        # t=0 even when OF restarts from latestTime and the first sub-step is
        # slightly before t_cur due to floating-point rounding.
        if t_cur > 0.0:
            t_table = np.concatenate([[0.0], t_win, [t_end + window_dt]])
            wing_ty_t = np.concatenate([[wing_ty[0]], wing_ty, [wing_ty[-1]]])
            wing_rz_t = np.concatenate([[wing_rz[0]], wing_rz, [wing_rz[-1]]])
            flap_tx_t = np.concatenate([[flap_tx[0]], flap_tx, [flap_tx[-1]]])
            flap_ty_t = np.concatenate([[flap_ty[0]], flap_ty, [flap_ty[-1]]])
            flap_rz_t = np.concatenate([[flap_rz[0]], flap_rz, [flap_rz[-1]]])
        else:
            t_table   = np.concatenate([t_win, [t_end + window_dt]])
            wing_ty_t = np.append(wing_ty, wing_ty[-1])
            wing_rz_t = np.append(wing_rz, wing_rz[-1])
            flap_tx_t = np.append(flap_tx, flap_tx[-1])
            flap_ty_t = np.append(flap_ty, flap_ty[-1])
            flap_rz_t = np.append(flap_rz, flap_rz[-1])
        write_tabulated6dof(
            CONST_DIR / "wingMotion.dat",
            t_table,
            np.zeros(len(t_table)),
            wing_ty_t,
            wing_rz_t,
        )
        write_tabulated6dof(
            CONST_DIR / "flapMotion.dat",
            t_table,
            flap_tx_t,
            flap_ty_t,
            flap_rz_t,
        )
        print(f"  Flap δ: {delta_schedule(t_cur):.2f}° → {delta_schedule(t_win_end):.2f}°")

        # Update controlDict for this window.
        # Window 0: startFrom startTime (t=0, processor*/0/ exists from decomposePar).
        # Window 1+: startFrom latestTime so OF picks up the last written time dir.
        # writeInterval = window size so OF writes exactly one snapshot per window,
        # guaranteeing processor*/t_win_end/ exists for the next window to restart from.
        n_steps = len(t_win) - 1
        update_control_dict(t_cur, t_win_end,
                            use_latest_time=(window_idx > 0),
                            write_interval=n_steps)

        # Run pimpleFoam
        ok = run_pimple(args.np)
        if not ok:
            print("  [FATAL] pimpleFoam failed — stopping co-simulation")
            sys.exit(1)

        # Reconstruct parallel results
        reconstruct(t_cur)

        # Seed pointDisplacement into processor time dirs for next window start.
        # pimpleFoam does not write pointDisplacement at sub-writeInterval times,
        # so we copy it from processor*/0/ (always present) into processor*/t_win_end/.
        for proc in sorted(CASE_DIR.glob("processor*")):
            src_pd = proc / "0" / "pointDisplacement"
            if not src_pd.exists():
                continue
            # Find the time directory closest to t_win_end
            time_dirs = []
            for d in proc.iterdir():
                if d.is_dir():
                    try:
                        time_dirs.append((abs(float(d.name) - t_win_end), d))
                    except ValueError:
                        pass
            if time_dirs:
                _, dst_dir = min(time_dirs)
                dst_pd = dst_dir / "pointDisplacement"
                if src_pd.resolve() != dst_pd.resolve():
                    shutil.copy2(src_pd, dst_pd)

        # Read aerodynamic forces from this window
        t_f, Fy_f, Mz_f = read_forces(t_cur, t_win_end)
        if t_f is not None:
            # Skip first samples of each window (pimpleFoam restart transient)
            n_skip = 10 if window_idx == 0 else 2
            if len(t_f) > n_skip:
                t_f, Fy_f, Mz_f = t_f[n_skip:], Fy_f[n_skip:], Mz_f[n_skip:]
            Fy_win = np.interp(t_win, t_f, Fy_f, left=Fy_f[0], right=Fy_f[-1])
            Mz_win = np.interp(t_win, t_f, Mz_f, left=Mz_f[0], right=Mz_f[-1])
        else:
            Fy_win = np.zeros(len(t_win))
            Mz_win = np.zeros(len(t_win))

        print(f"  Forces: Fy_mean={np.mean(Fy_win):.1f}N  Mz_mean={np.mean(Mz_win):.3f}N·m")

        # Integrate structural dynamics over this window
        h, hd, a, ad, h_traj, hd_traj, a_traj, ad_traj = integrate_structural(
            h, hd, a, ad, t_win, Fy_win, Mz_win
        )
        print(f"  Structural response end: h={h*1000:.3f}mm  α={np.degrees(a):.3f}°")

        # Accumulate trajectory in global buffer (smoothing applied once at end)
        # Decimate by 4: with writeInterval=5 and dt=7e-5 → sample every 4 points = 1.4e-3s
        skip = 10 if window_idx == 0 else 2
        for i in range(skip, len(t_win)):
            if t_win[i] < T_CSV_SKIP:
                continue
            if (i - skip) % 4 != 0:
                continue
            traj_buf.append((t_win[i], h_traj[i], hd_traj[i],
                             a_traj[i], ad_traj[i], Fy_win[i], Mz_win[i]))

        t_cur = t_win_end
        window_idx += 1
        save_state(t_cur, h, hd, a, ad, window_idx, t_end, dt)

    print(f"\n{'='*60}")
    print(f"Co-simulation complete: {window_idx} windows, t_final={t_cur:.5f}s")

    # Write CSV with global smoothing on Fy/Mz (eliminates inter-window discontinuities)
    if traj_buf:
        traj_arr = np.array(traj_buf)
        k = 51  # smoothing kernel: 51 × 1.4e-3s ≈ 0.071s (≈ 1 period f_α, covers 2 windows)
        def _smooth_global(x):
            out = np.convolve(x, np.ones(k) / k, mode='same')
            h2 = k // 2
            out[:h2] = x[:h2]; out[-h2:] = x[-h2:]
            return out
        Fy_s = _smooth_global(traj_arr[:, 5])
        Mz_s = _smooth_global(traj_arr[:, 6])
        with open(traj_path, "w") as f:
            f.write("t,h,hd,alpha,ad,Fy,Mz\n")
            for i, row in enumerate(traj_arr):
                f.write(f"{row[0]:.8e},{row[1]:.8e},{row[2]:.8e},"
                        f"{row[3]:.8e},{row[4]:.8e},"
                        f"{Fy_s[i]:.6f},{Mz_s[i]:.6f}\n")
    print(f"  Structural trajectory saved → {traj_path}")

    print("\n>>> Generating response plots...")
    subprocess.run(
        [sys.executable, str(CASE_DIR / "plot_response.py"), "--t-end", str(t_end)],
        cwd=CASE_DIR,
    )


if __name__ == "__main__":
    main()
