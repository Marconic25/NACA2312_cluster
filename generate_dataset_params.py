#!/usr/bin/env python3
"""
generate_dataset_params.py — LHS sampling for GLA dataset v2.

Generates metadata.csv with parameters for all 64 simulations across 5 families.
Uses Latin Hypercube Sampling with constraint rejection.

Usage:
    python generate_dataset_params.py [--seed 42] [--output metadata.csv]
"""

import argparse
import numpy as np
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Try scipy LHS; fall back to simple stratified sampling if unavailable
# ---------------------------------------------------------------------------
try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not found — using simple stratified sampling instead of LHS")


def lhs_sample(n_samples, n_dims, l_bounds, u_bounds, seed=42, max_attempts=50):
    """Generate LHS samples within bounds, returns (n_samples, n_dims) array."""
    if HAS_SCIPY:
        sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
        raw = sampler.random(n=n_samples * max_attempts)  # oversample for rejection
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
    """Sample n_samples valid points using LHS + rejection."""
    candidates = lhs_sample(n_samples * 20, n_dims, l_bounds, u_bounds, seed=seed)

    if constraint_fn is not None:
        mask = np.array([constraint_fn(row) for row in candidates])
        candidates = candidates[mask]

    if len(candidates) < n_samples:
        raise RuntimeError(
            f"Only {len(candidates)} valid samples from {n_samples*20} candidates. "
            f"Relax constraints or increase oversampling."
        )

    # Take first n_samples (LHS property is approximate after rejection,
    # but coverage is still good with large oversampling)
    return candidates[:n_samples]


# ---------------------------------------------------------------------------
# Family definitions
# ---------------------------------------------------------------------------

T_SIM = 3.0
T_CONSTRAINT = T_SIM - 0.5  # 2.5 s — flap must finish before this

FAMILIES = {
    "A": {
        "description": "No flap (baseline)",
        "law": 0,
        "n_train": 6,
        "n_test": 2,
        "params": ["R", "T_g"],
        "l_bounds": [0.10, 0.30],
        "u_bounds": [0.60, 1.20],
        "constraint": None,
    },
    "B1": {
        "description": "Informed, Law 1 (ramp + hold)",
        "law": 1,
        "n_train": 16,
        "n_test": 4,
        "params": ["R", "T_g", "delta_max", "dt_ramp", "t_start_delta"],
        "l_bounds": [0.10, 0.30, 2.0, 0.05, 0.20],
        "u_bounds": [0.60, 1.20, 20.0, 0.50, 0.80],
        "constraint": lambda x: x[4] + x[3] < T_CONSTRAINT,  # t_start + dt_ramp < 2.5
    },
    "B2": {
        "description": "Informed, Law 2 (two-phase ramp)",
        "law": 2,
        "n_train": 12,
        "n_test": 3,
        "params": ["R", "T_g", "delta_max", "dt_1", "dt_2", "t_start_delta"],
        "l_bounds": [0.10, 0.30, 2.0, 0.05, 0.10, 0.20],
        "u_bounds": [0.60, 1.20, 20.0, 0.20, 0.40, 0.80],
        "constraint": lambda x: x[5] + x[3] + x[4] < T_CONSTRAINT,  # t_start + dt1 + dt2
    },
    "B3": {
        "description": "Informed, Law 3 (trapezoid)",
        "law": 3,
        "n_train": 10,
        "n_test": 3,
        "params": ["R", "T_g", "delta_max", "dt_up", "dt_hold", "dt_down", "t_start_delta"],
        "l_bounds": [0.10, 0.30, 2.0, 0.05, 0.10, 0.05, 0.20],
        "u_bounds": [0.60, 1.20, 20.0, 0.50, 0.50, 0.50, 0.80],
        "constraint": lambda x: x[6] + x[3] + x[4] + x[5] < T_CONSTRAINT,
    },
    "C": {
        "description": "Uninformed, Law 1 (ramp + hold)",
        "law": 1,
        "n_train": 6,
        "n_test": 2,
        "params": ["R", "T_g", "delta_max", "dt_ramp", "t_start_delta"],
        "l_bounds": [0.10, 0.30, 2.0, 0.05, 0.00],
        "u_bounds": [0.60, 1.20, 20.0, 0.50, 0.15],
        "constraint": lambda x: x[4] + x[3] < T_CONSTRAINT,
    },
}

# All possible column names (superset across families)
ALL_PARAM_COLS = [
    "R", "T_g", "delta_max",
    "dt_ramp",                   # Law 1 (B1, C)
    "dt_1", "dt_2",              # Law 2 (B2)
    "dt_up", "dt_hold", "dt_down",  # Law 3 (B3)
    "t_start_delta",
]


def generate_family(family_name, fam_def, seed_base=42):
    """Generate train + test samples for one family."""
    n_total = fam_def["n_train"] + fam_def["n_test"]
    params = fam_def["params"]

    samples = sample_with_constraints(
        n_total,
        len(params),
        fam_def["l_bounds"],
        fam_def["u_bounds"],
        constraint_fn=fam_def["constraint"],
        seed=seed_base + hash(family_name) % 1000,
    )

    rows = []
    for i, sample in enumerate(samples):
        split = "train" if i < fam_def["n_train"] else "test"
        idx_in_family = i

        sim_name = f"sim_{family_name}_{idx_in_family:03d}_{split}"

        row = {
            "sim_name": sim_name,
            "family": family_name,
            "law": fam_def["law"],
            "split": split,
            "index": idx_in_family,
        }

        # Fill family-specific params
        for j, p in enumerate(params):
            row[p] = round(float(sample[j]), 6)

        # Derived quantities
        row["W_g0"] = round(row["R"] * 80.0, 4)  # U_inf = 80

        # Fill missing params with 0 or NaN
        for col in ALL_PARAM_COLS:
            if col not in row:
                row[col] = 0.0 if fam_def["law"] == 0 else np.nan

        # For family A: ensure all flap params are 0
        if family_name == "A":
            for col in ALL_PARAM_COLS:
                if col not in ["R", "T_g"]:
                    row[col] = 0.0

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="metadata.csv")
    args = parser.parse_args()

    all_rows = []
    global_idx = 0

    print(f"Generating dataset parameters (seed={args.seed})...")
    print(f"{'Family':<8} {'Train':>5} {'Test':>5} {'Total':>5} {'LHS dim':>7}")
    print("-" * 40)

    for fam_name, fam_def in FAMILIES.items():
        rows = generate_family(fam_name, fam_def, seed_base=args.seed)

        # Add global index
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
    print(f"{'TOTAL':<8} {n_train:>5} {n_test:>5} {len(all_rows):>5}")

    # Write CSV
    fieldnames = [
        "global_index", "sim_name", "family", "law", "split", "index",
        "R", "T_g", "W_g0",
        "delta_max", "t_start_delta",
        "dt_ramp", "dt_1", "dt_2", "dt_up", "dt_hold", "dt_down",
    ]

    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved → {out_path} ({len(all_rows)} simulations)")

    # Print a few samples for verification
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
