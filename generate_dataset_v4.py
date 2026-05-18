#!/usr/bin/env python3
"""
generate_dataset_v4.py — Crea le directory di simulazione per il dataset v4.

Legge metadata_v4.csv (generato da generate_dataset_params_v4.py) e produce:
  - Una directory per ogni sim con job.pbs e sim_params.csv
  - submit_v4.sh per lanciare tutti i job sul cluster

Famiglie:
  A  — gust puro (law 0): --delta-times 0 3 --delta-angles 0 0
  Cc — gust + controllo feed-forward (law 5): --law 5 --law5-k-eff --law5-tau --law5-rate-max

Logica identica a generate_weekend_subset.py, adattata per v4.

Usage:
    python generate_dataset_v4.py --dry-run
    python generate_dataset_v4.py --metadata metadata_v4.csv \\
        --output-dir /work/u10677113/NACA2312/dataset_v4
    python generate_dataset_v4.py --metadata metadata_v4.csv \\
        --output-dir /work/u10677113/NACA2312/dataset_v4 --submit
"""

import argparse
import csv
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import generate_dataset_params_v4 as gdp

# ─────────────────────── Configurazione cluster ───────────────────────────────

WORK_BASE     = "/work/u10677113/NACA2312"
CONTAINER_ABS = "/work/u10677113/of7.sif"
T_SIM         = 3.0
WALLTIME      = "06:00:00"
N_PROCS       = 16
WINDOW        = 50
DT            = 7e-5

# ─────────────────────── PBS templates ───────────────────────────────────────

_PBS_HEADER = """\
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
CHECKPOINT_SRC="$WORK_BASE/cosim_main/checkpoint"
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

# ── Copia checkpoint ──
echo "=== Copying checkpoint... ==="
mkdir -p "$SCRATCH/checkpoint"
cp "$CHECKPOINT_SRC/cosim_state.json"    "$SCRATCH/checkpoint/"
cp "$CHECKPOINT_SRC/cosim_state_t0.json" "$SCRATCH/checkpoint/" 2>/dev/null || true
for proc in "$CHECKPOINT_SRC"/processor*/; do
    dst="$SCRATCH/checkpoint/$(basename $proc)"
    mkdir -p "$dst"
    rsync -a "$proc" "$dst/"
done

cd "$SCRATCH"

# ── Attiva Python env ──
source ~/cosim_env/bin/activate 2>/dev/null || \\
source "$WORK_BASE/my_venv/bin/activate" 2>/dev/null || true

# ── Co-simulazione ──
echo "=== Starting co-simulation (from checkpoint)... ==="
"""

_PBS_COSIM_LAW0 = """\
python3 cosim_driver.py \\
    --np {n_procs} --window {window} --dt {dt} --t-end {t_end} \\
    --from-checkpoint \\
    --gust-w0 {W_g0:.4f} \\
    --gust-t-start 0.0 \\
    --gust-t-end {T_g:.6f} \\
    --delta-times 0.0 {t_end} \\
    --delta-angles 0.0 0.0 \\
    2>&1 | tee log.cosim_driver
"""

_PBS_COSIM_LAW5 = """\
python3 cosim_driver.py \\
    --np {n_procs} --window {window} --dt {dt} --t-end {t_end} \\
    --from-checkpoint \\
    --gust-w0 {W_g0:.4f} \\
    --gust-t-start 0.0 \\
    --gust-t-end {T_g:.6f} \\
    --law 5 \\
    --law5-k-eff {K_eff:.6f} \\
    --law5-tau {tau:.6f} \\
    --law5-rate-max {delta_rate_max:.2f} \\
    2>&1 | tee log.cosim_driver
"""

_PBS_FOOTER = """\

echo "=== Co-simulation done ==="

# ── Copia risultati leggeri su /work ──
echo "=== Copying results to /work... ==="
mkdir -p "$WORK_SIM"
for f in log.cosim_driver cosim_state.json structural_trajectory.csv; do
    [ -f "$SCRATCH/$f" ] && cp "$SCRATCH/$f" "$WORK_SIM/" || true
done
[ -d "$SCRATCH/postProcessing" ] && cp -r "$SCRATCH/postProcessing" "$WORK_SIM/" || true
[ -d "$SCRATCH/figures" ]        && cp -r "$SCRATCH/figures"        "$WORK_SIM/" || true

echo "=== Job complete: {sim_name} ==="
echo "=== Full case on: $SCRATCH (auto-deleted after 30 days) ==="
"""


# ─────────────────────── Helpers ─────────────────────────────────────────────

def _make_pbs(row: dict, output_dir: Path) -> str:
    common = dict(
        sim_name     = row["sim_name"],
        n_procs      = N_PROCS,
        walltime     = WALLTIME,
        work_base    = WORK_BASE,
        work_sim_dir = str(output_dir / row["sim_name"]),
        container    = CONTAINER_ABS,
        window       = WINDOW,
        dt           = DT,
        t_end        = T_SIM,
        W_g0         = float(row["W_g0"]),
        T_g          = float(row["T_g"]),
    )
    law = int(row["law"])
    if law == 0:
        cosim_block = _PBS_COSIM_LAW0.format(**common)
    elif law == 5:
        cosim_block = _PBS_COSIM_LAW5.format(
            **common,
            K_eff          = float(row["K_eff"]),
            tau            = float(row["tau"]),
            delta_rate_max = float(row["delta_rate_max"]),
        )
    else:
        raise ValueError(f"Law {law} non supportata")

    return _PBS_HEADER.format(**common) + cosim_block + _PBS_FOOTER.format(sim_name=row["sim_name"])


def _make_info(row: dict) -> str:
    law = int(row["law"])
    if law == 0:
        ctrl = "delta=0 (no flap)"
    else:
        ctrl = (
            f"K_eff={row['K_eff']} deg/(m/s)  tau={row['tau']}s  "
            f"rate_max={row['delta_rate_max']} deg/s  delta_max_eff≈{row['delta_max_eff']}°"
        )
    return (
        f"# {row['sim_name']}\n"
        f"family={row['family']}  law={law}  split={row['split']}\n"
        f"R={row['R']}  T_g={row['T_g']}  W_g0={row['W_g0']}\n"
        f"{ctrl}\n"
    )


def setup_one_sim(row: dict, output_dir: Path, dry_run: bool,
                  overwrite_pbs: bool = False):
    sim_dir = output_dir / row["sim_name"]

    if sim_dir.exists():
        if (sim_dir / "structural_trajectory.csv").exists():
            return sim_dir, "done"
        if overwrite_pbs and not dry_run:
            (sim_dir / "job.pbs").write_text(_make_pbs(row, output_dir))
            return sim_dir, "pbs-updated"
        return sim_dir, "exists"

    if dry_run:
        return sim_dir, "dry-run"

    sim_dir.mkdir(parents=True)

    with open(sim_dir / "sim_params.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=gdp.FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)

    (sim_dir / "job.pbs").write_text(_make_pbs(row, output_dir))
    (sim_dir / "sim_info.txt").write_text(_make_info(row))

    return sim_dir, "created"


# ─────────────────────── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata",      type=str, default="metadata_v4.csv")
    parser.add_argument("--output-dir",    type=str,
                        default="/work/u10677113/NACA2312/dataset_v4")
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--submit",        action="store_true")
    parser.add_argument("--overwrite-pbs", action="store_true",
                        help="Rigenera job.pbs per sim esistenti senza ricreare la dir")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    meta_path  = Path(args.metadata)

    if not meta_path.exists():
        print(f"[ERROR] metadata non trovato: {meta_path}")
        print("  Genera prima con: python generate_dataset_params_v4.py --output metadata_v4.csv")
        return

    with open(meta_path, newline="") as f:
        all_rows = list(csv.DictReader(f))

    print(f"\nGLA Dataset v4")
    print(f"  Metadata:   {meta_path}  ({len(all_rows)} sim)")
    print(f"  Output dir: {output_dir}")
    print(f"  Walltime:   {WALLTIME}/sim   N_procs: {N_PROCS}")
    if args.dry_run:
        print("  [DRY RUN]")

    counter = Counter((r["family"], r["split"]) for r in all_rows)
    print(f"\n  {'Famiglia':<8} {'Split':<8} {'N':>4}")
    print(f"  {'-'*22}")
    for (fam, split), n in sorted(counter.items()):
        print(f"  {fam:<8} {split:<8} {n:>4}")
    print()

    print(f"{'Sim':<32} {'Fam':<4} {'Split':<6} {'W_g0':>6} {'T_g':>5} "
          f"{'K_eff':>6} {'τ':>5} {'ṙ':>6}  Stato")
    print("-" * 82)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    pbs_files = []
    counts = {"created": 0, "done": 0, "exists": 0, "pbs-updated": 0, "dry-run": 0}

    for row in all_rows:
        sim_dir, status = setup_one_sim(row, output_dir, args.dry_run,
                                        overwrite_pbs=args.overwrite_pbs)
        law = int(row["law"])
        k_str    = f"{float(row.get('K_eff', 0)):>6.3f}"        if law == 5 else f"{'—':>6}"
        tau_str  = f"{float(row.get('tau', 0)):>5.2f}"          if law == 5 else f"{'—':>5}"
        rate_str = f"{float(row.get('delta_rate_max', 0)):>6.0f}" if law == 5 else f"{'—':>6}"

        print(f"{row['sim_name']:<32} {row['family']:<4} {row['split']:<6} "
              f"{float(row['W_g0']):>6.1f} {float(row['T_g']):>5.2f} "
              f"{k_str} {tau_str} {rate_str}  {status}")

        counts[status] += 1
        if status in ("created", "pbs-updated", "exists"):
            pbs_files.append(sim_dir / "job.pbs")

    print(f"\n{'='*55}")
    print(f"Creati: {counts['created']}  PBS aggiornati: {counts['pbs-updated']}  "
          f"Esistenti: {counts['exists']}  Completati: {counts['done']}")

    if args.dry_run:
        print("\n[DRY RUN] Rimuovi --dry-run per procedere.")
        return

    shutil.copy2(meta_path, output_dir / "metadata_v4.csv")

    # Genera submit_v4.sh
    submit_script = output_dir / "submit_v4.sh"
    with open(submit_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Submit {len(pbs_files)} GLA v4 jobs\n\n")
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
    print(f"  cd {output_dir} && bash submit_v4.sh")
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
