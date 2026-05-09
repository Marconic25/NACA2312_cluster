"""
setup_theodorsen_cases.py — Setup OpenFOAM cases for Theodorsen validation.

For each reduced frequency k, creates a case directory with:
  - Prescribed harmonic heave motion h(t) = h0*sin(omega*t), alpha=0, delta=0
  - No gust, no flap deployment
  - Runs pimpleFoam and records CL(t)

Then compares CL_CFD(t) vs CL_Theodorsen(t) for amplitude and phase.

Usage:
    python setup_theodorsen_cases.py --setup     # create cases
    python setup_theodorsen_cases.py --run       # run pimpleFoam
    python setup_theodorsen_cases.py --compare   # compare results
    python setup_theodorsen_cases.py --all       # do everything

Run from /work/u10677113/NACA2312/
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
WORKDIR     = Path("/work/u10677113/NACA2312")
TEMPLATE    = WORKDIR / "wingMotion2D_pimpleFoam"
CONTAINER   = "/work/u10677113/of7.sif"
VENV_PYTHON = str(WORKDIR / "my_venv/bin/python3")
N_PROCS     = 16
OUT_DIR     = WORKDIR / "theodorsen_validation"

# Flow parameters
U_INF  = 80.0
RHO    = 1.225
CHORD  = 1.0
SPAN   = 0.05
A_REF  = CHORD * SPAN
B      = CHORD / 2.0
EA_X   = 0.25
A_H    = (EA_X / CHORD - 0.5) * 2.0   # -0.5

# Test cases
H0          = 0.005   # heave amplitude [m] = 0.5%c
ALPHA0      = 0.0     # no pitch
K_VALUES    = [0.1]
N_CYCLES    = 12      # simulate 12 cycles, use last 8 for analysis
DT          = 1e-5    # physical timestep [s]
WINDOW      = 200     # cosim window steps

# Number of transient cycles to skip in analysis
N_SKIP_CYCLES = 1


def omega_from_k(k):
    return k * U_INF / B


def t_end_from_k(k):
    omega = omega_from_k(k)
    T = 2 * np.pi / omega
    return N_CYCLES * T


# ── Theodorsen function ────────────────────────────────────────────────────────

def theodorsen(k):
    from scipy.special import hankel2
    k = np.asarray(k, dtype=float)
    scalar = k.ndim == 0
    k = np.atleast_1d(k)
    C = np.zeros_like(k, dtype=complex)
    mask = k > 1e-10
    k_m = k[mask]
    H1 = hankel2(1, k_m)
    H0 = hankel2(0, k_m)
    C[mask]  = H1 / (H1 + 1j * H0)
    C[~mask] = 1.0 + 0j
    return C[0] if scalar else C


def theodorsen_cl_components(t, h0, omega, k):
    """
    Return (CL_nc, CL_c, CL_total) for pure heave h(t) = h0*sin(omega*t).

    Non-circulatory (apparent mass):
        L_nc = pi*rho*b^2 * h_ddot
        CL_nc = L_nc / (0.5*rho*U^2*A_ref)

    Circulatory (wake effect via Theodorsen):
        L_c  = 2*pi*rho*U*b * C(k) * hdot_3q
        where hdot_3q = hdot  (for pure heave, w_3q = hdot)
        CL_c = L_c / (0.5*rho*U^2*A_ref)
    """
    Ck   = theodorsen(k)
    hdot  =  h0 * omega * np.cos(omega * t)   # h = h0*sin -> hdot = h0*omega*cos
    hddot = -h0 * omega**2 * np.sin(omega * t)

    # Non-circulatory lift
    L_nc  = np.pi * RHO * B**2 * SPAN * hddot
    CL_nc = L_nc / (0.5 * RHO * U_INF**2 * A_REF)

    # Circulatory lift  (w_3q = hdot for pure heave, no pitch)
    # In time domain with harmonic motion: C(k) acts as complex multiplier on amplitude
    # We reconstruct the time signal via the complex exponential approach
    # hdot(t) = Re[ j*omega*h0 * exp(j*omega*t) ]
    # L_c(t)  = Re[ 2*pi*rho*U*b*C(k) * j*omega*h0 * exp(j*omega*t) ]
    W_amp  = 1j * omega * h0                                    # complex amplitude of hdot
    L_c    = (2.0 * np.pi * RHO * U_INF * B * SPAN *
              (Ck * W_amp * np.exp(1j * omega * t)).real)
    CL_c   = L_c / (0.5 * RHO * U_INF**2 * A_REF)

    CL_tot = CL_nc + CL_c
    return CL_nc, CL_c, CL_tot


def theodorsen_cl_time(t, h0, omega, k):
    """Compute total Theodorsen CL(t) for h(t)=h0*sin(omega*t)."""
    _, _, CL_tot = theodorsen_cl_components(t, h0, omega, k)
    return CL_tot


# ── Motion table writer ────────────────────────────────────────────────────────

def write_motion_table(path, t_arr, ty_arr, rz_arr):
    """Write tabulated6DoFMotion file."""
    lines = ["// Theodorsen validation — prescribed harmonic heave\n(\n"]
    for t, ty, rz in zip(t_arr, ty_arr, rz_arr):
        lines.append(f"  ({t:.10e}  ((0 {ty:.10e} 0) (0 0 {rz:.10e})))\n")
    lines.append(")\n")
    Path(path).write_text("".join(lines))


# ── Case setup ─────────────────────────────────────────────────────────────────

def setup_case(k):
    """Create OpenFOAM case for reduced frequency k."""
    omega  = omega_from_k(k)
    t_end  = t_end_from_k(k)
    T_cyc  = 2 * np.pi / omega
    name   = f"k{k:.3f}".replace(".", "p")
    case   = OUT_DIR / name

    print(f"  Setting up k={k:.3f}  ω={omega:.2f} rad/s  "
          f"T={T_cyc:.4f}s  t_end={t_end:.3f}s")

    # Copy template
    if case.exists():
        shutil.rmtree(case)
    shutil.copytree(TEMPLATE, case,
                    ignore=shutil.ignore_patterns(
                        "processor*", "cosim_state.json",
                        "postProcessing", "[0-9]*.[0-9]*",
                        "log.*", "__pycache__"
                    ))

    # Reset 0/ from 0.orig/
    if (case / "0").exists():
        shutil.rmtree(case / "0")
    shutil.copytree(case / "0.orig", case / "0")

    # Remove postProcessing if copied
    pp = case / "postProcessing"
    if pp.exists():
        shutil.rmtree(pp)

    # ── Patch controlDict ──────────────────────────────────────────────
    ctrl = case / "system" / "controlDict"
    txt  = ctrl.read_text()
    txt  = re.sub(r"startFrom[^\n;]*;", "startFrom       startTime;", txt)
    txt  = re.sub(r"(startTime\s+)[\d.eE+\-]+\s*;", r"\g<1>0;", txt)
    txt  = re.sub(r"stopAt[^\n;]*;",   "stopAt          endTime;", txt)
    txt  = re.sub(r"(endTime\s+)[\d.eE+\-]+\s*;",
                  rf"\g<1>{t_end:.6e};", txt)
    txt  = re.sub(r"(deltaT\s+)[\d.eE+\-]+\s*;",
                  rf"\g<1>{DT:.2e};", txt)
    txt  = re.sub(r"adjustTimeStep\s+yes;", "adjustTimeStep  no;", txt)
    # writeInterval = one cycle
    txt  = re.sub(r"(writeInterval\s+)[^;\n]+;",
                  rf"\g<1>{T_cyc:.6e};", txt, count=1)
    ctrl.write_text(txt)

    # ── Write zero-gust inlet BC ───────────────────────────────────────
    inlet_U = case / "0" / "include" / "fixedInletU"
    if inlet_U.exists():
        inlet_U.write_text(
            "inlet\n{\n"
            "    type            fixedValue;\n"
            f"    value           uniform ({U_INF} 0 0);\n"
            "}\n"
        )

    # ── Write prescribed motion tables ────────────────────────────────
    dt_table = DT
    t_table  = np.arange(0.0, t_end + T_cyc + dt_table * 0.5, dt_table)
    h_arr    = H0 * np.sin(omega * t_table)
    ty_arr   = -h_arr   # OpenFOAM y-displacement
    rz_arr   = np.zeros_like(t_table)

    write_motion_table(case / "constant" / "wingMotion.dat",
                       t_table, ty_arr, rz_arr)

    # Flap: no motion
    write_motion_table(case / "constant" / "flapMotion.dat",
                       np.array([0.0, t_end + T_cyc]),
                       np.array([0.0, 0.0]),
                       np.array([0.0, 0.0]))

    print(f"    Case ready: {case}")
    return case, name


# ── Run pimpleFoam ─────────────────────────────────────────────────────────────

def run_case(k):
    omega = omega_from_k(k)
    name  = f"k{k:.3f}".replace(".", "p")
    case  = OUT_DIR / name

    def of_run(cmd, log):
        full = (f"apptainer exec {CONTAINER} /bin/bash -c "
                f"'source /opt/openfoam7/etc/bashrc && cd {case} && {cmd}'")
        with open(case / log, "w") as lf:
            r = subprocess.run(full, shell=True,
                               stdout=lf, stderr=subprocess.STDOUT)
        return r.returncode

    print(f"  Running k={k:.3f} ...")

    rc = of_run("decomposePar", "log.decomposePar")
    if rc != 0:
        print(f"    [WARN] decomposePar rc={rc}")

    mpi_cmd = (f"mpirun --oversubscribe "
               f"--mca btl_base_warn_component_unused 0 "
               f"-np {N_PROCS} pimpleFoam -parallel")
    rc = of_run(mpi_cmd, "log.pimpleFoam")
    if rc != 0:
        print(f"    [WARN] pimpleFoam rc={rc}")

    rc = of_run("reconstructPar", "log.reconstructPar")
    if rc != 0:
        print(f"    [WARN] reconstructPar rc={rc}")

    print(f"    Done: {case}")


# ── Load CFD results ───────────────────────────────────────────────────────────

def load_cfd_cl(case_dir):
    """Load CL(t) from postProcessing/forceCoeffs."""
    t_all, cl_all = [], []
    for f in sorted(case_dir.glob("postProcessing/forceCoeffs/*/forceCoeffs.dat")):
        for line in f.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            cols = line.split()
            if len(cols) >= 4:
                try:
                    t_all.append(float(cols[0]))
                    cl_all.append(float(cols[3]))
                except ValueError:
                    pass
    if not t_all:
        for f in sorted(case_dir.glob("postProcessing/forces/*/forces.dat")):
            for line in f.read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                if len(nums) >= 6:
                    try:
                        t   = float(nums[0])
                        Fy  = float(nums[2]) + float(nums[5])
                        cl  = Fy / (0.5 * RHO * U_INF**2 * A_REF)
                        t_all.append(t)
                        cl_all.append(cl)
                    except ValueError:
                        pass
    if not t_all:
        return None, None
    idx = np.argsort(t_all)
    return np.array(t_all)[idx], np.array(cl_all)[idx]


# ── Extract amplitude and phase ────────────────────────────────────────────────

def extract_amp_phase(t, signal, omega, n_skip_cycles=N_SKIP_CYCLES):
    T_cyc  = 2 * np.pi / omega
    t_skip = t[0] + n_skip_cycles * T_cyc
    mask   = t >= t_skip
    if mask.sum() < 10:
        return np.nan, np.nan
    t_s = t[mask]
    s_s = signal[mask]
    dt_s = np.mean(np.diff(t_s))
    N    = len(t_s)
    freq = np.fft.rfftfreq(N, d=dt_s)
    f0   = omega / (2.0 * np.pi)
    idx  = np.argmin(np.abs(freq - f0))
    S_fft = np.fft.rfft(s_s)[idx] * 2 / N
    amp   = float(abs(S_fft))
    phase = float(np.degrees(np.angle(S_fft)))
    return amp, phase


def get_analysis_mask(t, omega, n_skip_cycles=N_SKIP_CYCLES):
    T_cyc  = 2 * np.pi / omega
    t_skip = t[0] + n_skip_cycles * T_cyc
    return t >= t_skip


# ── Compare ────────────────────────────────────────────────────────────────────

def compare_all():
    """Compare CFD vs Theodorsen for all k values."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    results = []

    for k in K_VALUES:
        omega  = omega_from_k(k)
        T_cyc  = 2 * np.pi / omega
        name   = f"k{k:.3f}".replace(".", "p")
        case   = OUT_DIR / name

        t_cfd, cl_cfd = load_cfd_cl(case)
        if t_cfd is None:
            print(f"  k={k:.3f}: no CFD data found, skipping")
            continue

        # Theodorsen components
        cl_nc, cl_c, cl_theo = theodorsen_cl_components(t_cfd, H0, omega, k)

        # h(t) and hdot(t) prescribed
        h_presc    = H0 * np.sin(omega * t_cfd)
        hdot_presc = H0 * omega * np.cos(omega * t_cfd)

        # Extract amplitude and phase (skip transient)
        amp_cfd,   ph_cfd   = extract_amp_phase(t_cfd, cl_cfd,  omega)
        amp_theo,  ph_theo  = extract_amp_phase(t_cfd, cl_theo, omega)
        _, ph_h             = extract_amp_phase(t_cfd, h_presc / H0, omega)

        ph_cfd_rel  = ph_cfd  - ph_h
        ph_theo_rel = ph_theo - ph_h
        amp_err     = (amp_cfd - amp_theo) / amp_theo * 100 if amp_theo > 1e-10 else np.nan
        phase_err   = ph_cfd_rel - ph_theo_rel

        # Aerodynamic work per cycle (last complete cycle)
        mask_work = get_analysis_mask(t_cfd, omega)
        t_w   = t_cfd[mask_work]
        cl_w  = cl_cfd[mask_work]
        th_w  = cl_theo[mask_work]
        hd_w  = hdot_presc[mask_work]
        h_w   = h_presc[mask_work]

        # Integrate over last complete cycle only
        n_last = int(T_cyc / np.mean(np.diff(t_w)))
        if n_last < len(t_w):
            t_lc   = t_w[-n_last:]
            cl_lc  = cl_w[-n_last:]
            th_lc  = th_w[-n_last:]
            hd_lc  = hd_w[-n_last:]
            h_lc   = h_w[-n_last:]
        else:
            t_lc, cl_lc, th_lc, hd_lc, h_lc = t_w, cl_w, th_w, hd_w, h_w

        q_ref = 0.5 * RHO * U_INF**2 * A_REF
        # Work = -∫ L * hdot dt  (sign: lift up positive, h positive down → work negative = energy extracted)
        work_cfd  = -np.trapezoid(cl_lc * q_ref * hd_lc, t_lc)
        work_theo = -np.trapezoid(th_lc * q_ref * hd_lc, t_lc)

        print(f"  k={k:.3f}:  |CL|_CFD={amp_cfd:.5f}  |CL|_theo={amp_theo:.5f}  "
              f"err={amp_err:.2f}%  Δφ={phase_err:.2f}°  "
              f"W_CFD={work_cfd:.4f}J  W_theo={work_theo:.4f}J")

        results.append({
            "k": k, "omega": omega, "T_cyc": T_cyc,
            "amp_cfd":  amp_cfd,  "ph_cfd":  ph_cfd_rel,
            "amp_theo": amp_theo, "ph_theo": ph_theo_rel,
            "amp_err":  amp_err,  "phase_err": phase_err,
            "work_cfd": work_cfd, "work_theo": work_theo,
            "t": t_cfd, "cl_cfd": cl_cfd,
            "cl_theo": cl_theo, "cl_nc": cl_nc, "cl_c": cl_c,
            "h": h_presc, "hdot": hdot_presc,
        })

    if not results:
        print("  No results to plot.")
        return

    for r in results:
        k     = r["k"]
        omega = r["omega"]
        T_cyc = r["T_cyc"]
        t     = r["t"]

        # ── Analysis window mask (skip transient) ─────────────────────
        mask  = get_analysis_mask(t, omega)
        # Show last 4 cycles (or all available after transient)
        t_min_plot = max(t[mask][0], t[-1] - 4 * T_cyc)
        mplot = t >= t_min_plot

        t_p    = t[mplot]
        cl_p   = r["cl_cfd"][mplot]
        th_p   = r["cl_theo"][mplot]
        nc_p   = r["cl_nc"][mplot]
        c_p    = r["cl_c"][mplot]
        h_p    = r["h"][mplot]
        hd_p   = r["hdot"][mplot]

        # ── Figure 1: Time series — CFD vs Theodorsen total ───────────
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Panel 1: CL total
        ax = axes[0]
        ax.plot(t_p, cl_p,  color="#378ADD", lw=1.5, label="OpenFOAM CFD")
        ax.plot(t_p, th_p,  color="#D85A30", lw=1.5, ls="--", label="Theodorsen total")
        ax.plot(t_p, h_p / H0 * r["amp_theo"],
                color="gray", lw=0.8, ls=":", label="h(t) scaled")
        ax.set_ylabel("CL  [ - ]")
        ax.set_title(
            f"k={k:.3f}  ω={omega:.2f} rad/s  |  "
            f"|CL| err={r['amp_err']:.2f}%  Δφ={r['phase_err']:.2f}°",
            fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)

        # Panel 2: Lnc and Lc components (Theodorsen only)
        ax = axes[1]
        ax.plot(t_p, th_p,  color="#D85A30", lw=1.5, label="L total")
        ax.plot(t_p, nc_p,  color="#2CA02C", lw=1.2, ls="--", label="L non-circulatory (apparent mass)")
        ax.plot(t_p, c_p,   color="#9467BD", lw=1.2, ls="-.", label="L circulatory [C(k) effect]")
        ax.set_ylabel("CL components  [ - ]")
        ax.set_title("Theodorsen decomposition: circulatory vs non-circulatory", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)

        # Panel 3: Aerodynamic power = -CL * hdot (adimensional)
        q_ref  = 0.5 * RHO * U_INF**2 * A_REF
        pow_cfd  = -(cl_p  * q_ref * hd_p)
        pow_theo = -(th_p  * q_ref * hd_p)
        ax = axes[2]
        ax.plot(t_p, pow_cfd,  color="#378ADD", lw=1.5, label="Power CFD  [W]")
        ax.plot(t_p, pow_theo, color="#D85A30", lw=1.5, ls="--", label="Power Theodorsen  [W]")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("Aero power  [W]")
        ax.set_xlabel("Time  [s]")
        ax.set_title(
            f"Aerodynamic power  |  W_CFD={r['work_cfd']:.4f} J  "
            f"W_theo={r['work_theo']:.4f} J  (last cycle)",
            fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)

        fig.suptitle(
            f"Theodorsen validation — pure heave  h0={H0*1000:.1f} mm  "
            f"U∞={U_INF} m/s  k={k:.3f}",
            fontsize=12)
        fig.tight_layout()
        fname = fig_dir / f"theodorsen_timeseries_k{k:.3f}.png".replace(".", "p")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname.name}")

        # ── Figure 2: Hysteresis loops CL vs h and CL vs hdot ─────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # CL vs h  (displacement loop)
        ax1.plot(h_p * 1000, cl_p,
                 color="#378ADD", lw=1.5, label="OpenFOAM CFD")
        ax1.plot(h_p * 1000, th_p,
                 color="#D85A30", lw=1.5, ls="--", label="Theodorsen")
        ax1.plot(h_p * 1000, nc_p,
                 color="#2CA02C", lw=1.0, ls=":", label="L non-circ.")
        ax1.plot(h_p * 1000, c_p,
                 color="#9467BD", lw=1.0, ls="-.", label="L circ.")
        ax1.set_xlabel("h  [mm]")
        ax1.set_ylabel("CL  [ - ]")
        ax1.set_title("CL vs displacement h(t)")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.4)

        # CL vs hdot  (velocity loop — area = work per cycle)
        hdot_mm = hd_p * 1000   # mm/s for readability
        ax2.plot(hdot_mm, cl_p,
                 color="#378ADD", lw=1.5, label="OpenFOAM CFD")
        ax2.plot(hdot_mm, th_p,
                 color="#D85A30", lw=1.5, ls="--", label="Theodorsen")
        # Fill area under CFD loop (proportional to work)
        ax2.fill(hdot_mm, cl_p, alpha=0.08, color="#378ADD")
        ax2.fill(hdot_mm, th_p, alpha=0.08, color="#D85A30")
        ax2.set_xlabel("ḣ  [mm/s]")
        ax2.set_ylabel("CL  [ - ]")
        ax2.set_title("CL vs heave rate ḣ(t)  [area ∝ aero work/cycle]")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.4)

        fig.suptitle(
            f"Theodorsen validation — hysteresis loops  "
            f"k={k:.3f}  ω={omega:.2f} rad/s",
            fontsize=12)
        fig.tight_layout()
        fname2 = fig_dir / f"theodorsen_hysteresis_k{k:.3f}.png".replace(".", "p")
        fig.savefig(fname2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname2.name}")

        # ── Figure 3: C(k) in complex plane (reference) ───────────────
        k_vec = np.linspace(0.01, 2.0, 500)
        Ck_vec = theodorsen(k_vec)
        Ck_this = theodorsen(k)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(Ck_vec.real, Ck_vec.imag,
                color="steelblue", lw=1.5, label="C(k) locus k∈[0,2]")
        ax.plot(Ck_this.real, Ck_this.imag, "ro", ms=10,
                label=f"k={k:.3f}  C(k)={Ck_this.real:.3f}{Ck_this.imag:+.3f}j")
        ax.set_xlabel("Re[C(k)]")
        ax.set_ylabel("Im[C(k)]")
        ax.set_title("Theodorsen function C(k) — complex plane")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.set_aspect("equal")
        fig.tight_layout()
        fname3 = fig_dir / "theodorsen_Ck_complex.png"
        fig.savefig(fname3, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname3.name}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n  Summary table:")
    print(f"  {'k':>6}  {'|CL|_theo':>10}  {'|CL|_CFD':>10}  "
          f"{'err%':>8}  {'φ_theo°':>9}  {'φ_CFD°':>9}  {'Δφ°':>8}  "
          f"{'W_CFD[J]':>10}  {'W_theo[J]':>10}")
    print("  " + "-" * 85)
    for r in results:
        print(f"  {r['k']:>6.3f}  {r['amp_theo']:>10.5f}  {r['amp_cfd']:>10.5f}  "
              f"{r['amp_err']:>8.2f}  {r['ph_theo']:>9.2f}  "
              f"{r['ph_cfd']:>9.2f}  {r['phase_err']:>8.2f}  "
              f"{r['work_cfd']:>10.5f}  {r['work_theo']:>10.5f}")

    # ── Text report ────────────────────────────────────────────────────────────
    report = fig_dir / "theodorsen_validation_report.txt"
    Ck_ref = theodorsen(K_VALUES[0])
    with open(report, "w") as f:
        f.write("Theodorsen Validation Report — OpenFOAM vs Theodorsen\n")
        f.write(f"U_inf={U_INF} m/s  h0={H0*1000:.1f}mm  c={CHORD}m  b={B}m\n")
        f.write(f"Transient skip: {N_SKIP_CYCLES} cycle(s)\n")
        f.write("=" * 85 + "\n")
        f.write(f"{'k':>6}  {'|CL|_T':>8}  {'|CL|_N':>8}  "
                f"{'err%':>7}  {'φ_T°':>7}  {'φ_N°':>7}  {'Δφ°':>7}  "
                f"{'W_CFD':>9}  {'W_theo':>9}\n")
        f.write("-" * 85 + "\n")
        for r in results:
            f.write(f"{r['k']:>6.3f}  {r['amp_theo']:>8.5f}  {r['amp_cfd']:>8.5f}  "
                    f"{r['amp_err']:>7.2f}  {r['ph_theo']:>7.2f}  "
                    f"{r['ph_cfd']:>7.2f}  {r['phase_err']:>7.2f}  "
                    f"{r['work_cfd']:>9.5f}  {r['work_theo']:>9.5f}\n")
        f.write("\n")
        f.write(f"Theodorsen function at k={K_VALUES[0]:.3f}:\n")
        f.write(f"  C(k) = {Ck_ref.real:.5f} {Ck_ref.imag:+.5f}j\n")
        f.write(f"  |C(k)| = {abs(Ck_ref):.5f}\n")
        f.write(f"  arg(C(k)) = {np.degrees(np.angle(Ck_ref)):.2f} deg\n")
    print(f"\n  Report: {report}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup",   action="store_true")
    parser.add_argument("--run",     action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--all",     action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.setup or args.all:
        print("\n[1/3] Setting up cases ...")
        for k in K_VALUES:
            setup_case(k)

    if args.run or args.all:
        print("\n[2/3] Running pimpleFoam ...")
        for k in K_VALUES:
            run_case(k)

    if args.compare or args.all:
        print("\n[3/3] Comparing CFD vs Theodorsen ...")
        compare_all()

    if not any([args.setup, args.run, args.compare, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
