#!/usr/bin/env python3
"""
generate_weekend_subset.py — Genera il subset PCA del weekend.

Famiglie (nuova logica isolata):
  A   — gust puro (W_g ≠ 0, δ = 0):   6 train + 2 test = 8 sim
  B1  — flap puro (W_g = 0, δ ≠ 0):   8 train + 2 test = 10 sim
Totale: 18 sim × ~2h = ~36h

Ogni sim parte da checkpoint_W0_baseline/ (warm-start, no mapFields).
Dati scritti su /scratch_local/$USER/<sim_name>; risultati leggeri copiati su /work.

Usage:
    python generate_weekend_subset.py --dry-run
    python generate_weekend_subset.py --output-dir /work/u10677113/NACA2312/dataset_weekend
    python generate_weekend_subset.py --output-dir /work/u10677113/NACA2312/dataset_weekend --submit
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import setup_sims
import generate_dataset_params as gdp   # riusa generate_A, generate_B1

# ─────────────────────── Configurazione cluster ──────────────────────────────

WORK_BASE       = "/work/u10677113/NACA2312"
CONTAINER_ABS   = "/work/u10677113/of7.sif"
CHECKPOINT_NAME = "checkpoint_W0_baseline"
T_SIM           = 3.0       # durata relativa dal checkpoint [s]
WALLTIME        = "05:00:00"
N_PROCS         = 16
WINDOW          = 50
DT              = 7e-5

# Dimensioni subset weekend
N_A_TRAIN  = 6;  N_A_TEST  = 2
N_B1_TRAIN = 8;  N_B1_TEST = 2
N_C1_TRAIN = 4;  N_C1_TEST = 0
SEED       = 42

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

# ── Copia caso su scratch_local (identica struttura di submit_gust.pbs) ──
# 1. Copia cosim_main/ senza processor*, postProcessing, checkpoint
# 2. Copia i processor* dal checkpoint direttamente come processor* nella scratch
#    (identico a come submit_gust.pbs li aveva già in cosim_main/)
# 3. Copia checkpoint/ per cosim_state.json (--from-checkpoint lo legge da lì)
# In questo modo write_gust_inlet() scrive fixedInletU corretto PRIMA che
# OF legga i campi — esattamente come nella run con W=60 che funzionava.
echo "=== Copying case to scratch_local... ==="
mkdir -p "$SCRATCH"
rsync -a --exclude='processor*' --exclude='postProcessing' --exclude='postProcessing_*' \\
      --exclude='log.*' --exclude='cosim_state.json' --exclude='figures' \\
      --exclude='__pycache__' --exclude='checkpoint' --exclude='checkpoint_*' \\
      "$WORK_BASE/cosim_main/" "$SCRATCH/"

# Sostituisci cosim_driver.py con quello patchato per questa sim
cp "{work_sim_dir}/cosim_driver.py" "$SCRATCH/cosim_driver.py"

# Copia checkpoint/ con processor* dentro (come si aspetta load_from_checkpoint)
echo "=== Copying checkpoint... ==="
mkdir -p "$SCRATCH/checkpoint"
cp "$CHECKPOINT_SRC/cosim_state.json"    "$SCRATCH/checkpoint/"
cp "$CHECKPOINT_SRC/cosim_state_t0.json" "$SCRATCH/checkpoint/" 2>/dev/null || true
for proc in "$CHECKPOINT_SRC"/processor*/; do
    cp -r "$proc" "$SCRATCH/checkpoint/$(basename $proc)"
done

cd "$SCRATCH"

# ── Attiva Python env ──
source ~/cosim_env/bin/activate 2>/dev/null || \\
source "$WORK_BASE/my_venv/bin/activate" 2>/dev/null || true

# ── Co-simulazione con warm-start da checkpoint ──
echo "=== Starting co-simulation (from checkpoint)... ==="
python3 cosim_driver.py \\
    --np {n_procs} --window {window} --dt {dt} --t-end {t_end} \\
    --from-checkpoint \\
    --gust-w0 {W_g0:.4f} \\
    --gust-t-start 0.0 \\
    --gust-t-end {T_g:.6f} \\
    2>&1 | tee log.cosim_driver

echo "=== Co-simulation done ==="

# ── Estrazione timeseries ──
echo "=== Extracting timeseries... ==="
python3 "$WORK_BASE/extract_data.py" \\
    --metadata "{work_sim_dir}/sim_params.csv" \\
    --sim-dir "$SCRATCH" \\
    --output-base "$WORK_BASE/data/GLA" \\
    --only {sim_name} \\
    --only-timeseries \\
    2>&1 | tee log.extract_data

# ── Copia risultati leggeri su /work ──
echo "=== Copying results to /work... ==="
mkdir -p "$WORK_SIM"
for f in log.cosim_driver log.extract_data cosim_state.json structural_trajectory.csv; do
    [ -f "$SCRATCH/$f" ] && cp "$SCRATCH/$f" "$WORK_SIM/" || true
done
[ -d "$SCRATCH/postProcessing" ] && cp -r "$SCRATCH/postProcessing" "$WORK_SIM/" || true

echo "=== Job complete: {sim_name} ==="
echo "=== Full case on: $SCRATCH (auto-deleted after 30 days) ==="
"""


# ─────────────────────── Setup una sim ───────────────────────────────────────

def setup_one_sim(row, output_dir: Path, dry_run: bool):
    sim_name = row["sim_name"]
    sim_dir  = output_dir / sim_name

    if sim_dir.exists():
        if (sim_dir / "structural_trajectory.csv").exists():
            return sim_dir, "done"
        return sim_dir, "exists"

    if dry_run:
        return sim_dir, "dry-run"

    sim_dir.mkdir(parents=True)

    # Copia e patcha cosim_driver.py
    driver_src = Path(__file__).parent / "cosim_main" / "cosim_driver.py"
    driver_dst = sim_dir / "cosim_driver.py"
    shutil.copy2(driver_src, driver_dst)

    delta_times, delta_angles = setup_sims.build_flap_schedule(row)
    setup_sims.patch_cosim_driver(driver_dst, row, delta_times, delta_angles)

    # Scrivi sim_params.csv (una riga, usato da extract_data.py)
    with open(sim_dir / "sim_params.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=gdp.FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)

    # Scrivi PBS
    pbs = PBS_TEMPLATE.format(
        sim_name        = sim_name,
        n_procs         = N_PROCS,
        walltime        = WALLTIME,
        work_base       = WORK_BASE,
        work_sim_dir    = str(output_dir / sim_name),
        container       = CONTAINER_ABS,
        checkpoint_name = CHECKPOINT_NAME,
        window          = WINDOW,
        dt              = DT,
        t_end           = T_SIM,
        W_g0            = float(row["W_g0"]),
        T_g             = float(row["T_g"]),
    )
    (sim_dir / "job.pbs").write_text(pbs)

    # sim_info.txt
    (sim_dir / "sim_info.txt").write_text(
        f"# {sim_name}\n"
        f"family={row['family']}  law={row['law']}  split={row['split']}\n"
        f"R={row['R']}  T_g={row['T_g']}  W_g0={row['W_g0']}\n"
        f"delta_max={row['delta_max']}  dt_ramp={row['dt_ramp']}\n"
        f"DELTA_TIMES={delta_times}\n"
        f"DELTA_ANGLES={delta_angles}\n"
    )
    return sim_dir, "created"


# ─────────────────────── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=str,
                        default="/work/u10677113/NACA2312/dataset_weekend")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--submit",   action="store_true")
    parser.add_argument("--seed",     type=int, default=SEED)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Genera parametri con la nuova logica isolata
    rows_A  = gdp.generate_A( N_A_TRAIN,  N_A_TEST,  args.seed)
    rows_B1 = gdp.generate_B1(N_B1_TRAIN, N_B1_TEST, args.seed)
    rows_C1 = gdp.generate_C1(N_C1_TRAIN, N_C1_TEST, args.seed)

    # Assegna sim_name e global_index
    all_rows = []
    for i, row in enumerate(rows_A + rows_B1 + rows_C1):
        row["global_index"] = i
        row["sim_name"] = f"sim_{row['family']}_{row['index']:03d}_{row['split']}"
        all_rows.append(row)

    print(f"\nGLA Weekend Subset — PCA dataset")
    print(f"  Output dir:  {output_dir}")
    print(f"  A  (gust puro, δ=0):      {N_A_TRAIN} train + {N_A_TEST} test = {N_A_TRAIN+N_A_TEST}")
    print(f"  B1 (flap puro, W_g=0):   {N_B1_TRAIN} train + {N_B1_TEST} test = {N_B1_TRAIN+N_B1_TEST}")
    print(f"  C1 (gust + flap):         {N_C1_TRAIN} train + {N_C1_TEST} test = {N_C1_TRAIN+N_C1_TEST}")
    print(f"  Totale: {len(all_rows)} sim  (~{len(all_rows)*2}h stima con parallelismo)")
    print(f"  Walltime/sim: {WALLTIME}")
    if args.dry_run:
        print(f"\n  [DRY RUN]\n")

    print(f"\n{'Sim':<30} {'Fam':<4} {'Split':<6} {'W_g0':>6} {'T_g':>5} {'δ_max':>6} {'dt_ramp':>7}  Stato")
    print(f"{'-'*78}")

    pbs_files = []
    n_created = n_done = n_exists = 0

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for row in all_rows:
        sim_dir, status = setup_one_sim(row, output_dir, args.dry_run)
        print(f"{row['sim_name']:<30} {row['family']:<4} {row['split']:<6} "
              f"{float(row['W_g0']):>6.1f} {float(row['T_g']):>5.2f} "
              f"{float(row['delta_max']):>+6.1f} {float(row['dt_ramp']):>7.3f}  {status}")
        if status == "created":
            n_created += 1; pbs_files.append(sim_dir / "job.pbs")
        elif status == "done":
            n_done += 1
        elif status == "exists":
            n_exists += 1; pbs_files.append(sim_dir / "job.pbs")

    print(f"\n{'='*50}")
    print(f"Creati: {n_created}  Esistenti: {n_exists}  Completati: {n_done}")

    if args.dry_run:
        print("\n[DRY RUN] Rimuovi --dry-run per procedere.")
        return

    # Scrivi metadata globale del subset
    with open(output_dir / "metadata_weekend.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=gdp.FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # Genera submit_weekend.sh
    submit_script = output_dir / "submit_weekend.sh"
    with open(submit_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Submit {len(pbs_files)} GLA weekend jobs\n\n")
        f.write(". /etc/profile.d/pbs.sh 2>/dev/null || true\n\n")
        f.write("SUBMITTED=0\nSKIPPED=0\n\n")
        for pbs in pbs_files:
            traj = pbs.parent / "structural_trajectory.csv"
            f.write(f'if [ -f "{traj}" ]; then\n')
            f.write(f'    echo "SKIP (done): {pbs.parent.name}"\n')
            f.write(f'    SKIPPED=$((SKIPPED+1))\n')
            f.write(f'else\n')
            f.write(f'    qsub "{pbs}" && SUBMITTED=$((SUBMITTED+1)) || echo "FAILED: {pbs.parent.name}"\n')
            f.write(f'    sleep 0.3\n')
            f.write(f'fi\n\n')
        f.write('echo "Submitted: $SUBMITTED  Skipped: $SKIPPED"\n')
    submit_script.chmod(0o755)

    print(f"\nScript → {submit_script}")
    print(f"\nSul cluster:")
    print(f"  cd {output_dir} && bash submit_weekend.sh")
    print(f"  watch qstat -u u10677113")

    if args.submit:
        print("\n--- Invio PBS jobs ---")
        for pbs in pbs_files:
            if (pbs.parent / "structural_trajectory.csv").exists():
                continue
            r = subprocess.run(["qsub", str(pbs)], capture_output=True, text=True)
            if r.returncode == 0:
                print(f"  OK:     {pbs.parent.name} → {r.stdout.strip()}")
            else:
                print(f"  FAILED: {pbs.parent.name} → {r.stderr.strip()}")


if __name__ == "__main__":
    main()
