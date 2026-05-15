#!/usr/bin/env python3
"""
generate_weekend_subset.py — Genera il subset PCA del weekend (22 sim).

Famiglie:
  A    — 8 sim (tutte: 6 train + 2 test): solo gust, no flap
  B1   — 8 sim (prime 8 train): flap positivo, law 1
  B1n  — 6 sim (prime 6 train): flap negativo, law 1

Ogni sim parte da checkpoint_W0_baseline/ (warm-start, no mapFields).
I dati vengono scritti su /scratch_local/$USER/<sim_name> e i risultati
leggeri (structural_trajectory.csv, postProcessing/) copiati su /work.

Usage:
    python generate_weekend_subset.py --dry-run
    python generate_weekend_subset.py --output-dir /work/u10677113/NACA2312/dataset_weekend
    python generate_weekend_subset.py --output-dir /work/u10677113/NACA2312/dataset_weekend --submit
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import setup_sims  # riusa build_flap_schedule, patch_cosim_driver

# ─────────────────────── Configurazione cluster ──────────────────────────────

WORK_BASE       = "/work/u10677113/NACA2312"
CONTAINER_ABS   = "/work/u10677113/of7.sif"
CHECKPOINT_NAME = "checkpoint_W0_baseline"   # nella cosim_main sul cluster
T_SIM           = 3.0                        # durata relativa dal checkpoint [s]
WALLTIME        = "03:30:00"                 # 2h sim + 1h30 setup/extract/copia
N_PROCS         = 16
WINDOW          = 286
DT              = 7e-5

# ─────────────────────── Filtro subset weekend ───────────────────────────────

SUBSET = {
    "A":   {"split": None,    "max_count": None},   # tutte le 8 sim
    "B1":  {"split": "train", "max_count": 8},      # prime 8 train
    "B1n": {"split": "train", "max_count": 6},      # prime 6 train
}

# ─────────────────────── PBS template ────────────────────────────────────────

PBS_TEMPLATE = """\
#!/bin/bash
#PBS -N {sim_name}
#PBS -q cpu
#PBS -l select=1:ncpus={n_procs}:mpiprocs={n_procs}
#PBS -l walltime={walltime}
#PBS -j oe

# ── Paths ──
WORK_BASE="{work_base}"
WORK_SIM="{work_sim_dir}"
SCRATCH="/scratch_local/$USER/{sim_name}"
CONTAINER_ABS="{container}"
CHECKPOINT_SRC="$WORK_BASE/cosim_main/{checkpoint_name}"
WRAPPER_DIR="$HOME/bin_of7"

# ── Setup wrappers OpenFOAM ──
mkdir -p "$WRAPPER_DIR"
for cmd in decomposePar pimpleFoam reconstructPar mapFields topoSet; do
    cat > "$WRAPPER_DIR/$cmd" <<WRAPPER
#!/bin/bash
apptainer exec --bind /work --bind /scratch_local "$CONTAINER_ABS" /bin/bash -c "source /opt/openfoam7/etc/bashrc && $cmd \\$*"
WRAPPER
    chmod +x "$WRAPPER_DIR/$cmd"
done
export PATH="$WRAPPER_DIR:$PATH"

# ── Copia caso su scratch_local ──
echo "=== Copying case to scratch_local... ==="
mkdir -p "$SCRATCH"
rsync -a --exclude='processor*' --exclude='postProcessing' --exclude='postProcessing_*' \\
      --exclude='log.*' --exclude='cosim_state.json' --exclude='figures' \\
      --exclude='__pycache__' --exclude='checkpoint' --exclude='checkpoint_*' \\
      "$WORK_BASE/cosim_main/" "$SCRATCH/"

# ── Copia checkpoint come 'checkpoint/' nella scratch ──
echo "=== Copying checkpoint... ==="
cp -r "$CHECKPOINT_SRC" "$SCRATCH/checkpoint"

cd "$SCRATCH"

# ── Attiva Python env ──
source ~/cosim_env/bin/activate 2>/dev/null || \\
source "$WORK_BASE/my_venv/bin/activate" 2>/dev/null || true

# ── Co-simulazione con warm-start da checkpoint ──
# --from-checkpoint: legge checkpoint/, no mapFields, no decomposePar
# I tempi del gust sono relativi al run (cosim_driver li shifta in coordinate CFD)
echo "=== Starting co-simulation (from checkpoint)... ==="
python3 cosim_driver.py \\
    --np {n_procs} --window {window} --dt {dt} --t-end {t_end} \\
    --from-checkpoint \\
    --gust-w0 {W_g0:.4f} \\
    --gust-t-start 0.0 \\
    --gust-t-end {T_g:.6f} \\
    2>&1 | tee log.cosim_driver

echo "=== Co-simulation done ==="

# ── Estrazione timeseries (no fields per ora) ──
echo "=== Extracting timeseries... ==="
python3 "$WORK_BASE/extract_data.py" \\
    --metadata "$WORK_BASE/metadata_merged.csv" \\
    --sim-dir "$SCRATCH" \\
    --output-base "$WORK_BASE/data/GLA" \\
    --only {sim_name} \\
    --only-timeseries \\
    2>&1 | tee log.extract_data

# ── Copia risultati leggeri su /work ──
echo "=== Copying results to /work... ==="
mkdir -p "$WORK_SIM"
for f in log.cosim_driver log.extract_data cosim_state.json cosim_driver.py structural_trajectory.csv; do
    [ -f "$SCRATCH/$f" ] && cp "$SCRATCH/$f" "$WORK_SIM/" || true
done
[ -d "$SCRATCH/postProcessing" ] && cp -r "$SCRATCH/postProcessing" "$WORK_SIM/" || true

echo "=== Job complete: {sim_name} ==="
echo "=== Full case on: $SCRATCH (auto-deleted after 30 days) ==="
"""


# ─────────────────────── Filtro righe metadata ───────────────────────────────

def select_rows(all_rows):
    selected = []
    counts = {fam: 0 for fam in SUBSET}
    for row in all_rows:
        fam = row["family"]
        if fam not in SUBSET:
            continue
        cfg = SUBSET[fam]
        if cfg["split"] is not None and row["split"] != cfg["split"]:
            continue
        if cfg["max_count"] is not None and counts[fam] >= cfg["max_count"]:
            continue
        selected.append(row)
        counts[fam] += 1
    return selected


# ─────────────────────── Creazione una sim ───────────────────────────────────

def setup_one_sim(row, output_dir: Path, dry_run: bool):
    sim_name = row["sim_name"]
    sim_dir  = output_dir / sim_name

    if sim_dir.exists():
        traj = sim_dir / "structural_trajectory.csv"
        if traj.exists():
            return sim_dir, "done"
        return sim_dir, "exists"

    if dry_run:
        return sim_dir, "dry-run"

    sim_dir.mkdir(parents=True)

    # Scrivi solo cosim_driver.py (patchato) e sim_info.txt
    # Il resto viene copiato dal PBS script dalla cosim_main sul cluster
    driver_src = Path(__file__).parent / "cosim_main" / "cosim_driver.py"
    driver_dst = sim_dir / "cosim_driver.py"
    import shutil
    shutil.copy2(driver_src, driver_dst)

    # Patcha gust e flap nel driver
    delta_times, delta_angles = setup_sims.build_flap_schedule(row)
    setup_sims.patch_cosim_driver(driver_dst, row, delta_times, delta_angles)

    # Scrivi PBS
    W_g0 = float(row["W_g0"])
    T_g  = float(row["T_g"])
    pbs_content = PBS_TEMPLATE.format(
        sim_name      = sim_name,
        n_procs       = N_PROCS,
        walltime      = WALLTIME,
        work_base     = WORK_BASE,
        work_sim_dir  = str(output_dir / sim_name),
        container     = CONTAINER_ABS,
        checkpoint_name = CHECKPOINT_NAME,
        window        = WINDOW,
        dt            = DT,
        t_end         = T_SIM,
        W_g0          = W_g0,
        T_g           = T_g,
    )
    (sim_dir / "job.pbs").write_text(pbs_content)

    # Info file
    delta_times, delta_angles = setup_sims.build_flap_schedule(row)
    info = (
        f"# {sim_name}\n"
        f"family={row['family']}  law={row['law']}  split={row['split']}\n"
        f"R={row['R']}  T_g={T_g}  W_g0={W_g0}\n"
        f"delta_max={row.get('delta_max', 0)}\n"
        f"DELTA_TIMES={delta_times}\n"
        f"DELTA_ANGLES={delta_angles}\n"
        f"gust: t_start=0.0  t_end={T_g:.4f}s  W_g0={W_g0:.4f} m/s\n"
    )
    (sim_dir / "sim_info.txt").write_text(info)

    return sim_dir, "created"


# ─────────────────────── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata",   type=str, default="metadata_merged.csv")
    parser.add_argument("--output-dir", type=str,
                        default="/work/u10677113/NACA2312/dataset_weekend")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stampa senza creare nulla")
    parser.add_argument("--submit", action="store_true",
                        help="Invia i PBS job dopo il setup")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    output_dir    = Path(args.output_dir)

    if not metadata_path.exists():
        print(f"ERROR: {metadata_path} non trovato")
        sys.exit(1)

    with open(metadata_path) as f:
        all_rows = list(csv.DictReader(f))

    rows = select_rows(all_rows)

    # Conteggio per famiglia
    from collections import Counter
    fam_counts = Counter(r["family"] for r in rows)

    print(f"\nGLA Weekend Subset — PCA dataset")
    print(f"  Metadata:    {metadata_path}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Simulazioni: {len(rows)}")
    for fam, n in sorted(fam_counts.items()):
        print(f"    {fam}: {n} sim")
    print(f"  Walltime/sim: {WALLTIME}")
    print(f"  Stima totale: ~{len(rows)*2}h (con parallelismo)")
    if args.dry_run:
        print(f"\n  [DRY RUN — nessuna directory verrà creata]\n")

    print(f"\n{'Sim':<30} {'Fam':<5} {'Law':>3} {'Split':<6} {'W_g0':>6} {'T_g':>5} {'δ_max':>6} {'dt_ramp':>7} {'Stato'}")
    print(f"{'-'*82}")

    pbs_files = []
    n_created = 0; n_done = 0; n_exists = 0

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        sim_dir, status = setup_one_sim(row, output_dir, args.dry_run)
        W_g0     = float(row["W_g0"])
        T_g      = float(row["T_g"])
        try:    delta_max = float(row["delta_max"])
        except: delta_max = 0.0
        try:    dt_ramp = float(row["dt_ramp"]) if row.get("dt_ramp") not in ("", "nan", None) else float("nan")
        except: dt_ramp = float("nan")
        dt_str = f"{dt_ramp:>7.3f}" if dt_ramp == dt_ramp else "    ---"
        print(f"{row['sim_name']:<30} {row['family']:<5} {row['law']:>3} "
              f"{row['split']:<6} {W_g0:>6.1f} {T_g:>5.2f} {delta_max:>6.1f} {dt_str}  {status}")
        if status == "created":
            n_created += 1
            pbs_files.append(sim_dir / "job.pbs")
        elif status == "done":
            n_done += 1
        elif status == "exists":
            n_exists += 1
            pbs_files.append(sim_dir / "job.pbs")

    print(f"\n{'='*50}")
    print(f"Creati: {n_created}  Già pronti (no traj): {n_exists}  Completati: {n_done}")

    if args.dry_run:
        print("\n[DRY RUN] Rimuovi --dry-run per procedere.")
        return

    # Genera submit_weekend.sh
    import shutil
    shutil.copy2(metadata_path, output_dir / "metadata_merged.csv")

    submit_script = output_dir / "submit_weekend.sh"
    with open(submit_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Submit {len(pbs_files)} GLA weekend jobs\n")
        f.write(f"# Generato da generate_weekend_subset.py\n\n")
        f.write(". /etc/profile.d/pbs.sh 2>/dev/null || true\n\n")
        f.write("SUBMITTED=0\nSKIPPED=0\n\n")
        for pbs in pbs_files:
            sim_dir = pbs.parent
            traj    = sim_dir / "structural_trajectory.csv"
            f.write(f'if [ -f "{traj}" ]; then\n')
            f.write(f'    echo "SKIP (done): {sim_dir.name}"\n')
            f.write(f'    SKIPPED=$((SKIPPED+1))\n')
            f.write(f'else\n')
            f.write(f'    qsub "{pbs}" && SUBMITTED=$((SUBMITTED+1)) || echo "FAILED: {sim_dir.name}"\n')
            f.write(f'    sleep 0.3\n')
            f.write(f'fi\n\n')
        f.write('echo ""\n')
        f.write('echo "Submitted: $SUBMITTED  Skipped (done): $SKIPPED"\n')
    submit_script.chmod(0o755)

    print(f"\nScript di submit → {submit_script}")
    print(f"\nPer lanciare sul cluster:")
    print(f"  cd {output_dir}")
    print(f"  bash submit_weekend.sh")
    print(f"\nMonitoraggio: watch qstat -u u10677113")

    if args.submit:
        print("\n--- Invio PBS jobs ---")
        for pbs in pbs_files:
            sim_dir = pbs.parent
            if (sim_dir / "structural_trajectory.csv").exists():
                continue
            result = subprocess.run(["qsub", str(pbs)], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Submitted: {sim_dir.name} → {result.stdout.strip()}")
            else:
                print(f"  FAILED:    {sim_dir.name} → {result.stderr.strip()}")


if __name__ == "__main__":
    main()
