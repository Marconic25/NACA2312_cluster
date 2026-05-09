#!/usr/bin/env python3
"""
setup_sims.py — Create simulation directories for GLA dataset.

For each row in metadata.csv:
  1. Copy the base case (wingMotion2D_pimpleFoam) to dataset/<sim_name>/
  2. Patch cosim_driver.py with the correct gust and flap parameters
  3. Generate the PBS job script

Usage:
    python setup_sims.py --base-case /work/u10677113/NACA2312/wingMotion2D_pimpleFoam \
                         --metadata metadata.csv \
                         --output-dir /work/u10677113/NACA2312/dataset
"""

import argparse
import csv
import math
import os
import shutil
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Flap schedule builders: convert metadata params → DELTA_TIMES / DELTA_ANGLES
# ──────────────────────────────────────────────────────────────────────────────

T_SIM = 3.0  # simulation end time [s]


def build_flap_law0():
    """Family A: no flap."""
    return [0.0, T_SIM], [0.0, 0.0]


def build_flap_law1(delta_max, dt_ramp, t_start):
    """Law 1: linear ramp + hold."""
    t_end_ramp = t_start + dt_ramp
    return (
        [0.0, t_start, t_end_ramp, T_SIM],
        [0.0, 0.0,     delta_max,  delta_max],
    )


def build_flap_law2(delta_max, dt_1, dt_2, t_start):
    """Law 2: two-phase segmented ramp + hold."""
    t1 = t_start + dt_1
    t2 = t1 + dt_2
    return (
        [0.0, t_start, t1,           t2,        T_SIM],
        [0.0, 0.0,     delta_max/2,  delta_max, delta_max],
    )


def build_flap_law3(delta_max, dt_up, dt_hold, dt_down, t_start):
    """Law 3: trapezoid (ramp up + hold + ramp down)."""
    t1 = t_start + dt_up
    t2 = t1 + dt_hold
    t3 = t2 + dt_down
    return (
        [0.0, t_start, t1,        t2,        t3,  T_SIM],
        [0.0, 0.0,     delta_max, delta_max, 0.0, 0.0],
    )


def build_flap_law4(delta_max, dt_ramp1, dt_hold1, dt_ramp2, dt_hold2, t_start):
    """Law 4: oscillating (ramp to +delta, hold, ramp to -delta, hold)."""
    t1 = t_start + dt_ramp1            # end of ramp up
    t2 = t1 + dt_hold1                 # end of hold at +delta
    t3 = t2 + dt_ramp2                 # end of ramp to -delta
    t4 = t3 + dt_hold2                 # end of hold at -delta
    return (
        [0.0, t_start, t1,        t2,         t3,         t4,         T_SIM],
        [0.0, 0.0,     delta_max, delta_max, -delta_max, -delta_max, -delta_max],
    )


def build_flap_schedule(row):
    """Dispatch to the correct law builder based on metadata row."""
    law = int(row["law"])
    if law == 0:
        return build_flap_law0()

    delta_max = float(row["delta_max"])
    t_start = float(row["t_start_delta"])

    if law == 1:
        return build_flap_law1(delta_max, float(row["dt_ramp"]), t_start)
    elif law == 2:
        return build_flap_law2(delta_max, float(row["dt_1"]), float(row["dt_2"]), t_start)
    elif law == 3:
        return build_flap_law3(
            delta_max,
            float(row["dt_up"]), float(row["dt_hold"]), float(row["dt_down"]),
            t_start,
        )
    elif law == 4:
        return build_flap_law4(
            delta_max,
            float(row["dt_ramp1"]), float(row["dt_hold1"]),
            float(row["dt_ramp2"]), float(row["dt_hold2"]),
            t_start,
        )
    else:
        raise ValueError(f"Unknown law: {law}")


# ──────────────────────────────────────────────────────────────────────────────
# Patch cosim_driver.py
# ──────────────────────────────────────────────────────────────────────────────

def patch_cosim_driver(driver_path, row, delta_times, delta_angles):
    """Replace gust and flap parameters in cosim_driver.py."""
    text = driver_path.read_text()

    R = float(row["R"])
    T_g = float(row["T_g"])
    W_g0 = float(row["W_g0"])
    gust_t_end = T_g  # gust starts at t=0

    # Patch gust parameters
    import re

    def replace_param(text, name, new_value):
        # Match: NAME = <value>  # optional comment
        pattern = rf'^({name}\s*=\s*).*$'
        replacement = rf'\g<1>{new_value}'
        text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if n == 0:
            raise RuntimeError(f"Could not find '{name}' in cosim_driver.py")
        return text

    text = replace_param(text, "GUST_W0", f"{W_g0:.4f}   # peak gust velocity [m/s] (R={R:.4f})")
    text = replace_param(text, "GUST_T_END", f"{gust_t_end:.6f}   # gust end [s] (T_g={T_g:.4f}s)")

    # Patch flap schedule
    dt_str = repr(delta_times)
    da_str = repr(delta_angles)
    text = replace_param(text, "DELTA_TIMES", f"{dt_str}   # [s]")
    text = replace_param(text, "DELTA_ANGLES", f"{da_str}   # [deg]")

    driver_path.write_text(text)


# ──────────────────────────────────────────────────────────────────────────────
# PBS job script template
# ──────────────────────────────────────────────────────────────────────────────

PBS_TEMPLATE = """\
#!/bin/bash
#PBS -N {sim_name}
#PBS -q cpu
#PBS -l select=1:ncpus=16:mpiprocs=16
#PBS -l walltime=12:00:00
#PBS -j oe

# ── Paths ──
WORK_SIM="{work_sim_dir}"
WORK_BASE="{work_base_dir}"
SCRATCH="/scratch_local/$USER/{sim_name}"
CONTAINER_ABS="/work/u10677113/of7.sif"
WRAPPER_DIR="$HOME/bin_of7"
BASE_SIMPLE="{base_simple_dir}"

# ── Setup wrappers (with bind mounts) ──
mkdir -p "$WRAPPER_DIR"
for cmd in decomposePar pimpleFoam reconstructPar mapFields topoSet; do
    cat > "$WRAPPER_DIR/$cmd" <<WRAPPER
#!/bin/bash
apptainer exec --bind /work --bind /scratch_local "$CONTAINER_ABS" /bin/bash -c "source /opt/openfoam7/etc/bashrc && $cmd \\$*"
WRAPPER
    chmod +x "$WRAPPER_DIR/$cmd"
done
export PATH="$WRAPPER_DIR:$PATH"

# ── Copy to scratch_local ──
echo "Copying case to scratch_local..."
mkdir -p "$SCRATCH"
rsync -a --exclude='processor*' --exclude='postProcessing' --exclude='log.*' \\
      --exclude='cosim_state.json' --exclude='figures' --exclude='__pycache__' \\
      "$WORK_SIM/" "$SCRATCH/"

cd "$SCRATCH"

# ── Activate Python env ──
source ~/cosim_env/bin/activate 2>/dev/null || \\
source /work/u10677113/NACA2312/my_venv/bin/activate 2>/dev/null || true

# ── Stage 1: IC from simpleFoam ──
cp -r 0.orig 0

echo "Running mapFields..."
apptainer exec --bind /work --bind /scratch_local "$CONTAINER_ABS" /bin/bash -c \\
    "source /opt/openfoam7/etc/bashrc && \\
     cd '$SCRATCH' && \\
     mapFields '$BASE_SIMPLE' -sourceTime latestTime -consistent" \\
    2>&1 | tee log.mapFields

# ── Stage 2: topoSet ──
apptainer exec --bind /work --bind /scratch_local "$CONTAINER_ABS" /bin/bash -c \\
    "source /opt/openfoam7/etc/bashrc && cd '$SCRATCH' && topoSet" \\
    2>&1 | tee log.topoSet

# Fix pointDisplacement
if [ -f 0/pointDisplacement.unmapped ]; then
    mv -f 0/pointDisplacement.unmapped 0/pointDisplacement
fi

# ── Stage 3: Co-simulation ──
echo "Starting co-simulation..."
python3 cosim_driver.py --np 16 --window 286 --dt 7e-5 --t-end {t_end} \\
    2>&1 | tee log.cosim_driver

# ── Stage 4: Reconstruct parallel results ──
echo "Running reconstructPar..."
reconstructPar 2>&1 | tee log.reconstructPar

# ── Stage 5: Extract dataset (timeseries + fields) ──
echo "Extracting dataset..."
python3 "$WORK_BASE/extract_data.py" \\
    --metadata "$WORK_BASE/metadata.csv" \\
    --sim-dir "$SCRATCH" \\
    --output-base "$WORK_BASE/data/GLA" \\
    --only {sim_name} \\
    --fields \\
    2>&1 | tee log.extract_data

# ── Stage 6: Copy logs and lightweight results to /work ──
echo "Copying logs to /work..."
mkdir -p "$WORK_SIM"
for f in log.* cosim_state.json cosim_driver.py sim_info.txt; do
    cp "$SCRATCH/$f" "$WORK_SIM/" 2>/dev/null || true
done
cp -r "$SCRATCH/postProcessing" "$WORK_SIM/" 2>/dev/null || true

# scratch_local NOT cleaned — full case available for debugging/ParaView
# Files auto-deleted after 30 days by cluster policy
echo "Full case remains on: $SCRATCH"

echo "=== Job complete: {sim_name} ==="
"""


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# Files/dirs to copy from base case (exclude runtime artifacts)
COPY_DIRS = ["0.orig", "constant", "system", "micro"]
COPY_FILES = ["cosim_driver.py"]


def setup_one_sim(row, base_case, output_dir, base_simple_dir):
    """Set up one simulation directory."""
    sim_name = row["sim_name"]
    sim_dir = output_dir / sim_name
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if sim_dir.exists():
        print(f"  [SKIP] {sim_name} — directory already exists")
        return sim_dir

    sim_dir.mkdir(parents=True)

    # Copy essential directories
    for d in COPY_DIRS:
        src = base_case / d
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, sim_dir / d)
            else:
                shutil.copy2(src, sim_dir / d)

    # Copy essential files
    for f in COPY_FILES:
        src = base_case / f
        if src.exists():
            shutil.copy2(src, sim_dir / f)

    # Build flap schedule and patch cosim_driver
    delta_times, delta_angles = build_flap_schedule(row)
    patch_cosim_driver(sim_dir / "cosim_driver.py", row, delta_times, delta_angles)

    # Write PBS job script
    pbs_content = PBS_TEMPLATE.format(
        sim_name=sim_name,
        work_sim_dir=str(sim_dir),
        work_base_dir=str(output_dir.parent),
        base_simple_dir=str(base_simple_dir),
        t_end=T_SIM,
    )
    pbs_file = sim_dir / "job.pbs"
    pbs_file.write_text(pbs_content)

    # Write a small info file for reference
    info = (
        f"# {sim_name}\n"
        f"family={row['family']}  law={row['law']}  split={row['split']}\n"
        f"R={row['R']}  T_g={row['T_g']}  W_g0={row['W_g0']}\n"
        f"delta_max={row.get('delta_max', 0)}\n"
        f"DELTA_TIMES={delta_times}\n"
        f"DELTA_ANGLES={delta_angles}\n"
    )
    (sim_dir / "sim_info.txt").write_text(info)

    print(f"  [OK] {sim_name} — {row['family']}, law {row['law']}, R={float(row['R']):.3f}")
    return sim_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-case", type=str, required=True,
                        help="Path to base wingMotion2D_pimpleFoam case")
    parser.add_argument("--base-simple", type=str, default=None,
                        help="Path to wingMotion2D_simpleFoam (for mapFields IC)")
    parser.add_argument("--metadata", type=str, default="metadata.csv")
    parser.add_argument("--output-dir", type=str, default="dataset")
    parser.add_argument("--submit", action="store_true",
                        help="Also submit PBS jobs after setup")
    args = parser.parse_args()

    base_case = Path(args.base_case).resolve()
    if args.base_simple:
        base_simple = Path(args.base_simple).resolve()
    else:
        base_simple = base_case.parent / "wingMotion2D_simpleFoam"
    output_dir = Path(args.output_dir).resolve()
    metadata_path = Path(args.metadata)

    print(f"Base case:    {base_case}")
    print(f"Base simple:  {base_simple}")
    print(f"Output dir:   {output_dir}")
    print(f"Metadata:     {metadata_path}")

    # Read metadata
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\nSetting up {len(rows)} simulations...\n")

    # Copy metadata to output dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metadata_path, output_dir / "metadata.csv")

    pbs_files = []
    for row in rows:
        sim_dir = setup_one_sim(row, base_case, output_dir, base_simple)
        pbs_files.append(sim_dir / "job.pbs")

    print(f"\n{'='*50}")
    print(f"Setup complete: {len(rows)} simulations in {output_dir}")
    print(f"PBS scripts ready in each sim directory.")

    # Write master submit script
    submit_script = output_dir / "submit_all.sh"
    with open(submit_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Submit all {len(rows)} GLA dataset jobs\n")
        f.write(f"# Generated by setup_sims.py\n\n")
        f.write("SUBMITTED=0\n")
        f.write("SKIPPED=0\n\n")
        for pbs in pbs_files:
            sim_dir = pbs.parent
            # Skip if simulation already completed
            f.write(f'if [ -f "{sim_dir}/cosim_state.json" ]; then\n')
            f.write(f'    echo "SKIP (done): {sim_dir.name}"\n')
            f.write(f'    SKIPPED=$((SKIPPED+1))\n')
            f.write(f'else\n')
            f.write(f'    qsub "{pbs}"\n')
            f.write(f'    SUBMITTED=$((SUBMITTED+1))\n')
            f.write(f'    sleep 0.5\n')
            f.write(f'fi\n\n')
        f.write('echo ""\n')
        f.write('echo "Submitted: $SUBMITTED jobs"\n')
        f.write('echo "Skipped (already done): $SKIPPED jobs"\n')
    submit_script.chmod(0o755)

    print(f"\nTo submit all jobs:")
    print(f"  cd {output_dir}")
    print(f"  bash submit_all.sh")

    if args.submit:
        print("\n--- Submitting jobs ---")
        import subprocess
        for pbs in pbs_files:
            sim_dir = pbs.parent
            if (sim_dir / "cosim_state.json").exists():
                continue
            result = subprocess.run(["qsub", str(pbs)], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Submitted: {pbs.parent.name} → {result.stdout.strip()}")
            else:
                print(f"  FAILED: {pbs.parent.name} → {result.stderr.strip()}")


if __name__ == "__main__":
    main()
