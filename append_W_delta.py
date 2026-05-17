#!/usr/bin/env python3
"""
append_W_delta.py — Append W_gust and delta columns to existing structural_trajectory.csv files.

For each sim directory, reads sim_info.txt to get the flap schedule and gust parameters,
then rewrites the CSV adding W_gust and delta columns.

Usage:
    python3 append_W_delta.py --dataset-dir /work/u10677113/NACA2312/dataset_weekend
    python3 append_W_delta.py --dataset-dir /work/u10677113/NACA2312/dataset_weekend --dry-run
"""

import argparse
import ast
import math
import numpy as np
from pathlib import Path


U_INF = 80.0  # m/s


def parse_sim_info(sim_info_path):
    """Parse sim_info.txt and return gust + flap parameters."""
    params = {}
    delta_times = None
    delta_angles = None

    for line in sim_info_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if "=" in line:
            # Handle multi-key lines like "family=A  law=0  split=train"
            for token in line.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    params[k.strip()] = v.strip()
        if line.startswith("DELTA_TIMES="):
            delta_times = ast.literal_eval(line.split("=", 1)[1])
        if line.startswith("DELTA_ANGLES="):
            delta_angles = ast.literal_eval(line.split("=", 1)[1])

    W_g0     = float(params.get("W_g0", 0.0))
    T_g      = float(params.get("T_g",  0.0))
    return W_g0, T_g, delta_times, delta_angles


def gust_velocity(t, W_g0, T_g):
    """1-cosine gust starting at t=0."""
    if W_g0 == 0.0 or T_g == 0.0:
        return 0.0
    if 0.0 <= t <= T_g:
        return 0.5 * W_g0 * (1.0 - math.cos(2.0 * math.pi * t / T_g))
    return 0.0


def delta_schedule(t, delta_times, delta_angles):
    """Interpolate flap angle at time t."""
    return float(np.interp(t, delta_times, delta_angles))


def process_sim(sim_dir: Path, dry_run: bool) -> str:
    csv_path  = sim_dir / "structural_trajectory.csv"
    info_path = sim_dir / "sim_info.txt"

    if not csv_path.exists():
        return "no-csv"
    if not info_path.exists():
        return "no-info"

    # Check if already has W_gust and delta columns
    header = csv_path.open().readline().strip()
    if "W_gust" in header:
        return "already-done"

    W_g0, T_g, delta_times, delta_angles = parse_sim_info(info_path)

    if dry_run:
        return f"dry-run (W_g0={W_g0:.1f} T_g={T_g:.2f} δ_times={delta_times})"

    # Read existing CSV
    lines = csv_path.read_text().splitlines()
    new_lines = [lines[0] + ",W_gust,delta"]  # new header

    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split(",")
        t = float(cols[0])
        W = gust_velocity(t, W_g0, T_g)
        d = delta_schedule(t, delta_times, delta_angles)
        new_lines.append(f"{line},{W:.6f},{d:.6f}")

    csv_path.write_text("\n".join(new_lines) + "\n")
    return "updated"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=str,
                        default="/work/u10677113/NACA2312/dataset_weekend")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    sim_dirs = sorted(d for d in dataset_dir.iterdir()
                      if d.is_dir() and d.name.startswith("sim_"))

    print(f"Dataset dir: {dataset_dir}")
    print(f"Found {len(sim_dirs)} sim directories\n")

    n_updated = n_done = n_skip = 0
    for sim_dir in sim_dirs:
        status = process_sim(sim_dir, args.dry_run)
        print(f"  {sim_dir.name:<30}  {status}")
        if status == "updated":
            n_updated += 1
        elif status == "already-done":
            n_done += 1
        else:
            n_skip += 1

    print(f"\nUpdated: {n_updated}  Already done: {n_done}  Skipped: {n_skip}")
    if args.dry_run:
        print("\n[DRY RUN] Rimuovi --dry-run per procedere.")


if __name__ == "__main__":
    main()
