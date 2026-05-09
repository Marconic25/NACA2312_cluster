"""
temporal_study.py
Temporal discretization study for NACA 2312 pimpleFoam co-simulation.
Runs 4 timestep levels and writes a summary report.

Usage:
    python3 temporal_study.py [--workdir /path/to/NACA2312] [--np 16]
"""

import argparse
import subprocess
import shutil
import re
import json
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
WORKDIR       = Path("/work/u10677113/NACA2312")
CONTAINER     = "/work/u10677113/of7.sif"
PIMPLE_CASE   = WORKDIR / "wingMotion2D_pimpleFoam"
STUDY_DIR     = WORKDIR / "temporal_study"
END_TIME      = 1.0        # s — enough for 5 oscillation cycles
WINDOW_SIZE   = 0.02       # pimple window size (keep same as production)
N_PROCS       = 16

DT_LEVELS = {
    "DT1": 50,    # window=50  Δt_coupling=5e-4s
    "DT2": 20,    # window=20  Δt_coupling=2e-4s
    "DT3": 10,    # window=10  Δt_coupling=1e-4s
    "DT4": 5,     # window=5   Δt_coupling=5e-5s
}


# ── Helpers ────────────────────────────────────────────────────────────────
def run(cmd: str, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    full = (
        f"apptainer exec {CONTAINER} /bin/bash -c "
        f"'source /opt/openfoam7/etc/bashrc && cd {cwd} && {cmd}'"
    )
    with open(log, "w") as lf:
        r = subprocess.run(full, shell=True, stdout=lf, stderr=subprocess.STDOUT)
    return r.returncode


def run_python(cmd: str, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as lf:
        r = subprocess.run(cmd, shell=True, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    return r.returncode


def set_controldict_value(case_dir: Path, key: str, value: str):
    ctrl = case_dir / "system" / "controlDict"
    txt  = ctrl.read_text()
    txt  = re.sub(rf"({key}\s+)[^\s;]+;", rf"\g<1>{value};", txt)
    ctrl.write_text(txt)


def parse_cl_history(case_dir: Path):
    """Return (time, CL) arrays from postProcessing forceCoeffs."""
    for f in (case_dir / "postProcessing").glob("**/forceCoeffs.dat"):
        times, cl = [], []
        for line in f.read_text().splitlines():
            if line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) >= 4:
                try:
                    times.append(float(cols[0]))
                    cl.append(float(cols[3]))   # col3 = Cl
                except ValueError:
                    pass
        if times:
            return np.array(times), np.array(cl)
    return None, None


def max_co_from_log(log: Path) -> float:
    """Parse maximum Courant number from pimpleFoam log."""
    vals = re.findall(r"Courant Number mean: [\d.eE+\-]+ max: ([\d.eE+\-]+)", log.read_text())
    return max(float(v) for v in vals) if vals else -1.0


def signal_amplitude_phase(t: np.ndarray, cl: np.ndarray):
    """Estimate CL amplitude and phase (deg) via FFT on last half of signal."""
    if t is None or len(t) < 20:
        return -1.0, -1.0
    # Use last 60% to avoid transient
    n  = int(0.6 * len(t))
    tc = t[-n:]
    yc = cl[-n:]
    dt = np.mean(np.diff(tc))
    N  = len(yc)
    fft_vals = np.fft.rfft(yc - yc.mean())
    freqs    = np.fft.rfftfreq(N, d=dt)
    idx      = np.argmax(np.abs(fft_vals[1:])) + 1   # skip DC
    amp      = 2 * np.abs(fft_vals[idx]) / N
    phase    = np.angle(fft_vals[idx], deg=True)
    return float(amp), float(phase)


def wall_clock_per_second(log: Path, sim_time: float) -> float:
    """Wall-clock time per simulated second."""
    matches = re.findall(r"ClockTime = ([\d.]+) s", log.read_text())
    if matches:
        total_wall = float(matches[-1])
        return total_wall / sim_time if sim_time > 0 else -1.0
    return -1.0


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=str(WORKDIR))
    parser.add_argument("--np",      type=int, default=N_PROCS)
    args = parser.parse_args()

    workdir   = Path(args.workdir)
    study_dir = workdir / "temporal_study"
    study_dir.mkdir(exist_ok=True)

    results = {}

    for name, dt in DT_LEVELS.items():
        case_dir = study_dir / name
        print(f"\n{'='*60}")
        print(f"  {name}  (dt = {dt:.1e} s)")
        print(f"{'='*60}")

        # ── Check if already done ──────────────────────────────────────
        result_file = study_dir / f"{name}_result.json"
        if result_file.exists():
            with open(result_file) as f:
                results[name] = json.load(f)
            print(f"  Already completed, loading results...")
            continue

        # ── Copy pimpleFoam case ───────────────────────────────────────
        if case_dir.exists():
            shutil.rmtree(case_dir)
        shutil.copytree(PIMPLE_CASE, case_dir,
                        ignore=shutil.ignore_patterns(
                            "processor*", "cosim_state.json",
                            "log.*", "[0-9]*.[0-9]*", "postProcessing"
                        ))

        # Keep only 0/ and 0.orig/ — remove all numeric timesteps and processor dirs
        for d in case_dir.iterdir():
            if not d.is_dir():
                continue
            if d.name in ("0", "0.orig", "constant", "system", "__pycache__"):
                continue
            shutil.rmtree(d)
        # Reset 0/ from 0.orig/
        if (case_dir / "0").exists():
            shutil.rmtree(case_dir / "0")
        shutil.copytree(case_dir / "0.orig", case_dir / "0")

        # ── Patch controlDict ──────────────────────────────────────────
        set_controldict_value(case_dir, "deltaT",    f"{dt:.2e}")
        set_controldict_value(case_dir, "startFrom", "startTime")
        set_controldict_value(case_dir, "startTime", "0")
        # Restore correct deltaT and disable adjustTimeStep for fixed dt
        set_controldict_value(case_dir, "deltaT", "1e-5")
        ctrl = case_dir / "system" / "controlDict"
        txt = ctrl.read_text()
        txt = re.sub(r"adjustTimeStep\s+yes;", "adjustTimeStep  no;", txt)
        # Set writeInterval = window duration
        win_dur = dt * 1e-5  # window_steps * physical_dt
        txt = re.sub(r"(writeInterval\s+)[^;\n]+;",
                     f"\\g<1>{win_dur:.6e};", txt, count=1)
        ctrl.write_text(txt)

        # ── Remove copied postProcessing ──────────────────────────────
        pp = case_dir / "postProcessing"
        if pp.exists():
            shutil.rmtree(pp)

        # ── Patch flap schedule to start immediately ──────────────────
        cosim_src = case_dir / "cosim_driver.py"
        if cosim_src.exists():
            txt = cosim_src.read_text()
            txt = txt.replace(
                "DELTA_TIMES  = [0.0,  0.8,  1.0,  2.0]",
                "DELTA_TIMES  = [0.0,  0.0,  0.2,  2.0]"
            )
            txt = txt.replace(
                "DELTA_ANGLES = [0.0,  0.0,  15.0, 15.0]",
                "DELTA_ANGLES = [0.0,  0.0,  15.0, 15.0]"
            )
            cosim_src.write_text(txt)

        # ── Reset motion tables to t=0 ────────────────────────────────
        motion_template = """// tabulated6DoFMotion — generated by cosim_driver.py
(
  (0.0000000000e+00  ((0.0000000000e+00 0.0000000000e+00 0) (0 0 0.0000000000e+00)))
  (9.9990000000e+03  ((0.0000000000e+00 0.0000000000e+00 0) (0 0 0.0000000000e+00)))
)
"""
        for dat in ["wingMotion.dat", "flapMotion.dat"]:
            dat_path = case_dir / "constant" / dat
            if dat_path.exists():
                dat_path.write_text(motion_template)

        # ── Run cosim_driver ───────────────────────────────────────────
        print(f"  cosim_driver (window={dt})...")
        venv_python = str(Path("/work/u10677113/NACA2312/my_venv/bin/python3"))
        cosim_log = case_dir / "log.cosim_driver"
        cosim_cmd = (
            f"export PATH=$HOME/bin_of7:$PATH && "
            f"cd {case_dir} && "
            f"{venv_python} cosim_driver.py --np {args.np} --window {dt} --dt 1e-5 --t-end {END_TIME}"
        )
        rc = run_python(cosim_cmd, case_dir, cosim_log)
        if rc != 0:
            print(f"  [WARN] cosim_driver rc={rc}")
        pimple_log = case_dir / "log.pimpleFoam"


        # ── Extract results ────────────────────────────────────────────
        t, cl = parse_cl_history(case_dir)
        amp, phase = signal_amplitude_phase(t, cl)
        co_max     = max_co_from_log(pimple_log) if pimple_log.exists() else -1.0
        wc_per_s   = wall_clock_per_second(pimple_log, END_TIME) if pimple_log.exists() else -1.0

        res = {
            "dt":          dt,
            "co_max":      co_max,
            "cl_amp":      amp,
            "cl_phase":    phase,
            "wc_per_s":    wc_per_s,
            "t":           t.tolist()  if t  is not None else [],
            "cl":          cl.tolist() if cl is not None else [],
        }
        results[name] = res
        with open(result_file, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  Co_max={co_max:.2f}  CL_amp={amp:.4f}  phase={phase:.1f}°  wc/s={wc_per_s:.1f}s")
        # Remove case directory to free disk space
       # print(f"  Removing {case_dir} to free disk...")
        #shutil.rmtree(case_dir)

    # ── Write report ───────────────────────────────────────────────────────
    report_path = study_dir / "temporal_study_report.txt"
    with open(report_path, "w") as f:
        f.write("NACA 2312 — Temporal Discretization Study\n")
        f.write(f"endTime = {END_TIME} s,  mesh = M3 (wingMotion2D_pimpleFoam)\n")
        f.write("=" * 70 + "\n\n")

        # Table
        header = f"{'Level':<6} {'window':<8}  {'dt (s)':<12} {'Co_max':<10} {'CL_amp':<10} {'phase (deg)':<14} {'wc/s (s)':<10}\n"
        f.write(header)
        f.write("-" * 70 + "\n")
        for name in DT_LEVELS:
            r = results.get(name, {})
            win = r.get('dt', 0)
            dt_s = win * 1e-5
            f.write(
                f"{name:<6} "
                f"{win:<8}  "
                f"{dt_s:<12.2e} "
                f"{r.get('co_max', -1):<10.2f} "
                f"{r.get('cl_amp', -1):<10.4f} "
                f"{r.get('cl_phase', -1):<14.2f} "
                f"{r.get('wc_per_s', -1):<10.1f}\n"
            )
        f.write("\n")

        # Convergence check
        f.write("Convergence check (relative change in CL amplitude):\n")
        names = list(DT_LEVELS.keys())
        for i in range(1, len(names)):
            a1 = results.get(names[i-1], {}).get("cl_amp", 0)
            a2 = results.get(names[i],   {}).get("cl_amp", 0)
            if a1 > 0:
                rel = abs(a2 - a1) / abs(a1) * 100
                f.write(f"  {names[i-1]} -> {names[i]}: {rel:.2f}%\n")
        f.write("\n")

        # Selected timestep
        selected = None
        names = list(DT_LEVELS.keys())
        for i in range(len(names) - 1):
            a1 = results.get(names[i],   {}).get("cl_amp", 0)
            a2 = results.get(names[i+1], {}).get("cl_amp", 0)
            if a1 > 0 and abs(a2 - a1) / abs(a1) * 100 < 2.0:
                selected = names[i]
                break
        if selected is None:
            selected = names[-1]
        f.write(f"Selected timestep: {selected}  (window={DT_LEVELS[selected]}, dt={DT_LEVELS[selected]*1e-5:.2e} s)\n")
        f.write("Criterion: largest dt for which |Delta CL_amp| < 2% vs next finer level.\n")

    print(f"\nReport written: {report_path}")

    # ── Save all CL time histories to CSV ─────────────────────────────────
    csv_path = study_dir / "cl_histories.csv"
    max_len = max(len(results[n].get("t", [])) for n in results)
    with open(csv_path, "w") as f:
        header_cols = []
        for name in DT_LEVELS:
            header_cols += [f"t_{name}", f"CL_{name}"]
        f.write(",".join(header_cols) + "\n")
        for i in range(max_len):
            row = []
            for name in DT_LEVELS:
                t_arr  = results[name].get("t",  [])
                cl_arr = results[name].get("cl", [])
                row.append(str(t_arr[i])  if i < len(t_arr)  else "")
                row.append(str(cl_arr[i]) if i < len(cl_arr) else "")
            f.write(",".join(row) + "\n")
    print(f"CL histories CSV: {csv_path}")


if __name__ == "__main__":
    main()
