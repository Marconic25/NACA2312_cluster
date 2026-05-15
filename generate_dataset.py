#!/usr/bin/env python3
"""
generate_dataset.py — Orchestratore pipeline GLA dataset (closed-loop FSI).

Legge metadata_merged.csv e per ogni simulazione:
  1. Verifica se già completata (structural_trajectory.csv o postProcessing/)
  2. Crea la directory caso via setup_sims.setup_one_sim()
  3. Scrive il PBS job script

Poi genera submit_all.sh per lanciare tutto sull'HPC.

Usage:
    python generate_dataset.py --dry-run
    python generate_dataset.py --family A,B1 --output-dir /work/u10677113/NACA2312/dataset
    python generate_dataset.py --output-dir /work/u10677113/NACA2312/dataset --submit
    python generate_dataset.py --status --output-dir /work/u10677113/NACA2312/dataset
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

# Riusa le funzioni già testate di setup_sims.py
import setup_sims


def check_completed(sim_dir: Path) -> str:
    """
    Verifica stato di completamento di una simulazione.
    Returns: 'done', 'running', 'partial', 'not_started'
    """
    if not sim_dir.exists():
        return "not_started"

    has_traj    = (sim_dir / "structural_trajectory.csv").exists()
    has_state   = (sim_dir / "cosim_state.json").exists()
    has_forces  = (sim_dir / "postProcessing" / "forces").exists()
    has_log     = (sim_dir / "log.cosim_driver").exists()

    if has_traj and has_forces:
        return "done"
    if has_state and has_forces:
        return "running"   # started but traj not written yet (still in progress)
    if has_log:
        return "partial"   # started but no forces yet
    return "not_started"


def print_status(rows, output_dir: Path):
    """Stampa stato di tutte le simulazioni."""
    counts = {"done": 0, "running": 0, "partial": 0, "not_started": 0}
    families = {}

    for row in rows:
        sim_name = row["sim_name"]
        fam = row["family"]
        sim_dir = output_dir / sim_name
        status = check_completed(sim_dir)
        counts[status] += 1
        families.setdefault(fam, {"done": 0, "running": 0, "partial": 0, "not_started": 0})
        families[fam][status] += 1

    print(f"\n{'='*60}")
    print(f"Dataset status — {output_dir}")
    print(f"{'='*60}")
    print(f"{'Family':<8} {'Done':>6} {'Run':>6} {'Part':>6} {'Todo':>6} {'Total':>6}")
    print(f"{'-'*44}")
    for fam in sorted(families):
        d = families[fam]
        tot = sum(d.values())
        print(f"{fam:<8} {d['done']:>6} {d['running']:>6} {d['partial']:>6} {d['not_started']:>6} {tot:>6}")
    print(f"{'-'*44}")
    tot = sum(counts.values())
    print(f"{'TOTAL':<8} {counts['done']:>6} {counts['running']:>6} {counts['partial']:>6} {counts['not_started']:>6} {tot:>6}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metadata", type=str, default="metadata_merged.csv",
        help="CSV con parametri delle simulazioni (default: metadata_merged.csv)",
    )
    parser.add_argument(
        "--base-case", type=str, default=None,
        help="Path alla cartella cosim_main base (default: ./cosim_main)",
    )
    parser.add_argument(
        "--base-simple", type=str, default=None,
        help="Path alla cartella rans_baseline per mapFields IC",
    )
    parser.add_argument(
        "--output-dir", type=str, default="dataset",
        help="Directory dove creare le simulazioni (default: dataset)",
    )
    parser.add_argument(
        "--family", type=str, default=None,
        help="Filtra per famiglia, es. 'A,B1,B2' (default: tutte)",
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Filtra per split: 'train' o 'test' (default: entrambi)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Stampa cosa verrebbe fatto senza creare nulla",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Mostra solo lo stato delle simulazioni esistenti e termina",
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="Invia i PBS job dopo aver creato le directory",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ricrea le directory anche se già completate",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.resolve()
    base_case = Path(args.base_case).resolve() if args.base_case else repo_root / "cosim_main"
    base_simple = Path(args.base_simple).resolve() if args.base_simple else repo_root / "rans_baseline"
    output_dir = Path(args.output_dir).resolve()
    metadata_path = Path(args.metadata)

    if not metadata_path.exists():
        print(f"ERROR: metadata file not found: {metadata_path}")
        sys.exit(1)
    if not base_case.exists():
        print(f"ERROR: base case not found: {base_case}")
        sys.exit(1)

    with open(metadata_path) as f:
        all_rows = list(csv.DictReader(f))

    # Filtri
    rows = all_rows
    if args.family:
        families = {fam.strip() for fam in args.family.split(",")}
        rows = [r for r in rows if r["family"] in families]
        if not rows:
            print(f"ERROR: nessuna simulazione per le famiglie {families}")
            sys.exit(1)
    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
        if not rows:
            print(f"ERROR: nessuna simulazione per split='{args.split}'")
            sys.exit(1)

    # Solo status
    if args.status:
        print_status(rows, output_dir)
        return

    print(f"\nGLA Dataset Generator")
    print(f"  Metadata:    {metadata_path} ({len(rows)} simulazioni)")
    print(f"  Base case:   {base_case}")
    print(f"  Base simple: {base_simple}")
    print(f"  Output dir:  {output_dir}")
    if args.dry_run:
        print(f"  [DRY RUN — nessuna directory verrà creata]")
    print()

    # Intestazione tabella
    print(f"{'Sim':<30} {'Family':<6} {'Law':>3} {'Split':<6} {'Stato':<12} {'Azione'}")
    print(f"{'-'*75}")

    pbs_files = []
    n_setup = 0; n_skip = 0; n_done = 0

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        sim_name = row["sim_name"]
        fam = row["family"]
        law = row["law"]
        split = row["split"]
        sim_dir = output_dir / sim_name
        status = check_completed(sim_dir)

        if status == "done" and not args.force:
            print(f"{sim_name:<30} {fam:<6} {law:>3} {split:<6} {'done':<12} SKIP")
            n_done += 1
            pbs_files.append(sim_dir / "job.pbs")
            continue

        action = "DRY-RUN" if args.dry_run else "SETUP"
        print(f"{sim_name:<30} {fam:<6} {law:>3} {split:<6} {status:<12} {action}")

        if not args.dry_run:
            sim_dir_created = setup_sims.setup_one_sim(
                row, base_case, output_dir, base_simple
            )
            pbs_files.append(sim_dir_created / "job.pbs")
            n_setup += 1
        else:
            n_setup += 1

    print(f"\n{'='*50}")
    print(f"Setup: {n_setup}  Già completate (skip): {n_done}  Totale: {len(rows)}")

    if args.dry_run:
        print("\n[DRY RUN] Nessuna directory creata. Rimuovi --dry-run per procedere.")
        return

    # Copia metadata nell'output dir
    import shutil
    shutil.copy2(metadata_path, output_dir / "metadata.csv")

    # Genera submit_all.sh
    submit_script = output_dir / "submit_all.sh"
    with open(submit_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Submit {len(pbs_files)} GLA dataset jobs\n")
        f.write(f"# Generato da generate_dataset.py\n\n")
        f.write("SUBMITTED=0\nSKIPPED=0\n\n")
        for pbs in pbs_files:
            sim_dir = pbs.parent
            traj = sim_dir / "structural_trajectory.csv"
            f.write(f'if [ -f "{traj}" ]; then\n')
            f.write(f'    echo "SKIP (done): {sim_dir.name}"\n')
            f.write(f'    SKIPPED=$((SKIPPED+1))\n')
            f.write(f'else\n')
            f.write(f'    qsub "{pbs}"\n')
            f.write(f'    SUBMITTED=$((SUBMITTED+1))\n')
            f.write(f'    sleep 0.5\n')
            f.write(f'fi\n\n')
        f.write('echo ""\n')
        f.write('echo "Submitted: $SUBMITTED  Skipped (done): $SKIPPED"\n')
    submit_script.chmod(0o755)
    print(f"\nScript di submit → {submit_script}")
    print(f"  cd {output_dir} && bash submit_all.sh")

    if args.submit:
        print("\n--- Invio PBS jobs ---")
        for pbs in pbs_files:
            sim_dir = pbs.parent
            if check_completed(sim_dir) == "done":
                continue
            if not pbs.exists():
                continue
            result = subprocess.run(["qsub", str(pbs)], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Submitted: {sim_dir.name} → {result.stdout.strip()}")
            else:
                print(f"  FAILED:    {sim_dir.name} → {result.stderr.strip()}")


if __name__ == "__main__":
    main()
