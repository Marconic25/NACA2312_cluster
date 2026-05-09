#!/usr/bin/env python3
"""
generate_dataset_params_v3.py — LHS sampling for extended GLA dataset.

Adds to v2 (64 sims):
  - B1n/B2n/B3n/Cn: negative flap deflection (mirror of B1/B2/B3/C)
  - D: oscillating flap (positive → negative → hold)

Generates metadata_ext.csv with ONLY the new simulations (to append to existing dataset).
The global_index continues from 64.

Usage:
    python generate_dataset_params_v3.py [--seed 123] [--output metadata_ext.csv]
"""

import argparse
import numpy as np
import csv
from pathlib import Path

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

T_SIM = 3.0
T_CONSTRAINT = T_SIM - 0.5  # 2.5 s
GLOBAL_INDEX_START = 64  # continue from existing dataset


def lhs_sample(n_samples, n_dims, l_bounds, u_bounds, seed=42, max_attempts=50):
    if HAS_SCIPY:
        sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
        raw = sampler.random(n=n_samples * max_attempts)
        scaled = qmc.scale(raw, l_bounds, u_bounds)
    else:
        rng = np.random.default_rng(seed)
        raw = np.zeros((n_samples * max_attempts, n_dims))
        for j in range(n_dims):
            perm = np.tile(np.arange(n_samples * max_attempts), 1)
            rng.shuffle(perm)
            raw[:, j] = (perm + rng.random(n_samples * max_attempts)) / (n_samples * max_attempts)
        scaled = raw * (np.array(u_bounds) - np.array(l_bounds)) + np.array(l_bounds)
    return scaled


def sample_with_constraints(n_samples, n_dims, l_bounds, u_bounds,
                            constraint_fn=None, seed=42):
    candidates = lhs_sample(n_samples * 20, n_dims, l_bounds, u_bounds, seed=seed)
    if constraint_fn is not None:
        mask = np.array([constraint_fn(row) for row in candidates])
        candidates = candidates[mask]
    if len(candidates) < n_samples:
        raise RuntimeError(f"Only {len(candidates)} valid samples from {n_samples*20}")
    return candidates[:n_samples]


# ---------------------------------------------------------------------------
# New family definitions (negative flap + oscillating)
# ---------------------------------------------------------------------------

# Negative flap families: same structure as positive, but delta_max is negative
# We sample |delta_max| in [2, 20] and negate it

NEW_FAMILIES = {
    # ── Negative flap (mirror of B1/B2/B3/C) ──
    "B1n": {
        "description": "Informed, Law 1 neg (ramp down + hold)",
        "law": 1,
        "n_train": 10,
        "n_test": 3,
        "params": ["R", "T_g", "delta_max", "dt_ramp", "t_start_delta"],
        "l_bounds": [0.10, 0.30, -20.0, 0.05, 0.20],
        "u_bounds": [0.60, 1.20, -2.0,  0.50, 0.80],
        "constraint": lambda x: x[4] + x[3] < T_CONSTRAINT,
    },
    "B2n": {
        "description": "Informed, Law 2 neg (two-phase ramp down)",
        "law": 2,
        "n_train": 8,
        "n_test": 2,
        "params": ["R", "T_g", "delta_max", "dt_1", "dt_2", "t_start_delta"],
        "l_bounds": [0.10, 0.30, -20.0, 0.05, 0.10, 0.20],
        "u_bounds": [0.60, 1.20, -2.0,  0.20, 0.40, 0.80],
        "constraint": lambda x: x[5] + x[3] + x[4] < T_CONSTRAINT,
    },
    "B3n": {
        "description": "Informed, Law 3 neg (trapezoid down)",
        "law": 3,
        "n_train": 6,
        "n_test": 2,
        "params": ["R", "T_g", "delta_max", "dt_up", "dt_hold", "dt_down", "t_start_delta"],
        "l_bounds": [0.10, 0.30, -20.0, 0.05, 0.10, 0.05, 0.20],
        "u_bounds": [0.60, 1.20, -2.0,  0.50, 0.50, 0.50, 0.80],
        "constraint": lambda x: x[6] + x[3] + x[4] + x[5] < T_CONSTRAINT,
    },
    "Cn": {
        "description": "Uninformed, Law 1 neg (ramp down + hold)",
        "law": 1,
        "n_train": 4,
        "n_test": 1,
        "params": ["R", "T_g", "delta_max", "dt_ramp", "t_start_delta"],
        "l_bounds": [0.10, 0.30, -20.0, 0.05, 0.00],
        "u_bounds": [0.60, 1.20, -2.0,  0.50, 0.15],
        "constraint": lambda x: x[4] + x[3] < T_CONSTRAINT,
    },

    # ── Oscillating flap (Law 4) ──
    # Ramp to +delta_max, hold, ramp to -delta_max, hold
    # Parameters: delta_max (positive), dt_ramp1, dt_hold1, dt_ramp2, dt_hold2, t_start_delta
    "D": {
        "description": "Oscillating flap (positive → negative → hold)",
        "law": 4,
        "n_train": 10,
        "n_test": 3,
        "params": ["R", "T_g", "delta_max", "dt_ramp1", "dt_hold1", "dt_ramp2", "dt_hold2", "t_start_delta"],
        "l_bounds": [0.10, 0.30, 2.0,  0.05, 0.05, 0.05, 0.05, 0.20],
        "u_bounds": [0.60, 1.20, 20.0, 0.30, 0.30, 0.30, 0.30, 0.60],
        "constraint": lambda x: x[7] + x[3] + x[4] + x[5] + x[6] < T_CONSTRAINT,
    },
}

ALL_PARAM_COLS = [
    "R", "T_g", "delta_max",
    "dt_ramp",
    "dt_1", "dt_2",
    "dt_up", "dt_hold", "dt_down",
    "dt_ramp1", "dt_hold1", "dt_ramp2", "dt_hold2",
    "t_start_delta",
]


def generate_family(family_name, fam_def, seed_base=123):
    n_total = fam_def["n_train"] + fam_def["n_test"]
    params = fam_def["params"]

    samples = sample_with_constraints(
        n_total, len(params),
        fam_def["l_bounds"], fam_def["u_bounds"],
        constraint_fn=fam_def["constraint"],
        seed=seed_base + hash(family_name) % 1000,
    )

    rows = []
    for i, sample in enumerate(samples):
        split = "train" if i < fam_def["n_train"] else "test"
        sim_name = f"sim_{family_name}_{i:03d}_{split}"

        row = {
            "sim_name": sim_name,
            "family": family_name,
            "law": fam_def["law"],
            "split": split,
            "index": i,
        }

        for j, p in enumerate(params):
            row[p] = round(float(sample[j]), 6)

        row["W_g0"] = round(row["R"] * 80.0, 4)

        for col in ALL_PARAM_COLS:
            if col not in row:
                row[col] = 0.0 if fam_def["law"] == 0 else np.nan

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=str, default="metadata_ext.csv")
    args = parser.parse_args()

    all_rows = []
    global_idx = GLOBAL_INDEX_START

    print(f"Generating EXTENDED dataset parameters (seed={args.seed})...")
    print(f"Starting from global_index={GLOBAL_INDEX_START}")
    print(f"\n{'Family':<8} {'Train':>5} {'Test':>5} {'Total':>5} {'LHS dim':>7}")
    print("-" * 40)

    for fam_name, fam_def in NEW_FAMILIES.items():
        rows = generate_family(fam_name, fam_def, seed_base=args.seed)
        for row in rows:
            row["global_index"] = global_idx
            global_idx += 1
        all_rows.extend(rows)
        n_tr = sum(1 for r in rows if r["split"] == "train")
        n_te = sum(1 for r in rows if r["split"] == "test")
        print(f"{fam_name:<8} {n_tr:>5} {n_te:>5} {len(rows):>5} {len(fam_def['params']):>7}")

    print("-" * 40)
    n_train = sum(1 for r in all_rows if r["split"] == "train")
    n_test = sum(1 for r in all_rows if r["split"] == "test")
    print(f"{'NEW':<8} {n_train:>5} {n_test:>5} {len(all_rows):>5}")
    print(f"{'TOTAL':<8} {n_train+50:>5} {n_test+14:>5} {len(all_rows)+64:>5}")

    fieldnames = [
        "global_index", "sim_name", "family", "law", "split", "index",
        "R", "T_g", "W_g0",
        "delta_max", "t_start_delta",
        "dt_ramp", "dt_1", "dt_2", "dt_up", "dt_hold", "dt_down",
        "dt_ramp1", "dt_hold1", "dt_ramp2", "dt_hold2",
    ]

    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved → {out_path} ({len(all_rows)} new simulations)")

    # Also create merged metadata
    merged_path = Path("metadata_merged.csv")
    orig_rows = []
    if Path("metadata.csv").exists():
        with open("metadata.csv") as f:
            reader = csv.DictReader(f)
            orig_rows = list(reader)
        print(f"Loaded {len(orig_rows)} existing simulations from metadata.csv")

    with open(merged_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in orig_rows:
            # Add missing columns with nan
            for col in fieldnames:
                if col not in row:
                    row[col] = np.nan
            writer.writerow(row)
        writer.writerows(all_rows)

    print(f"Saved → {merged_path} ({len(orig_rows) + len(all_rows)} total simulations)")

    # Print samples
    print("\nSample entries:")
    for row in all_rows[:3]:
        print(f"  {row['sim_name']}: R={row['R']:.3f}, T_g={row['T_g']:.3f}, "
              f"δ_max={row.get('delta_max', 0):.1f}°, law={row['law']}")
    print("  ...")
    for row in all_rows[-2:]:
        print(f"  {row['sim_name']}: R={row['R']:.3f}, T_g={row['T_g']:.3f}, "
              f"δ_max={row.get('delta_max', 0):.1f}°, law={row['law']}")


if __name__ == "__main__":
    main()
