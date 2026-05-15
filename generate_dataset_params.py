#!/usr/bin/env python3
"""
generate_dataset_params.py — LHS sampling for GLA dataset v3 (redesigned).

Famiglia | Descrizione                        | W_g | delta | Law
---------|------------------------------------|----|-------|----
A        | Gust puro, no flap                 | LHS | 0     | 0
B1       | Flap puro, no gust, law 1          | 0   | LHS   | 1
C1       | Gust + flap combinati, law 1       | LHS | LHS   | 1

Famiglie A e B1 isolano gli effetti; C1 cattura l'interazione.
delta_max ∈ [-20, -2] ∪ [2, 20] deg (unifico positivo e negativo in B1/C1).
Le dimensioni delle famiglie vengono scelte dopo la PCA — questo script
accetta --n-train/--n-test per famiglia come argomenti.

Usage:
    python generate_dataset_params.py --dry-run
    python generate_dataset_params.py --output metadata.csv
    python generate_dataset_params.py --n-A-train 12 --n-B1-train 25 --n-C1-train 25
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
    print("[WARNING] scipy not found — using stratified sampling")

# ─────────────────────── Parametri fissi ─────────────────────────────────────

U_INF        = 80.0   # m/s
T_SIM        = 3.0    # s
T_CONSTRAINT = 2.5    # flap deve finire prima di questo istante [s]

# ─────────────────────── LHS + rejection ─────────────────────────────────────

def lhs_sample(n, d, l_bounds, u_bounds, seed):
    if HAS_SCIPY:
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        raw = sampler.random(n=n)
        return qmc.scale(raw, l_bounds, u_bounds)
    else:
        rng = np.random.default_rng(seed)
        raw = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            raw[:, j] = (perm + rng.random(n)) / n
        return raw * (np.array(u_bounds) - np.array(l_bounds)) + np.array(l_bounds)


def sample_with_constraints(n_total, d, l_bounds, u_bounds, constraint_fn, seed,
                            oversample=20):
    candidates = lhs_sample(n_total * oversample, d, l_bounds, u_bounds, seed)
    if constraint_fn is not None:
        mask = np.array([constraint_fn(row) for row in candidates])
        candidates = candidates[mask]
    if len(candidates) < n_total:
        raise RuntimeError(
            f"Solo {len(candidates)} campioni validi su {n_total*oversample}. "
            f"Rilassa i vincoli o aumenta l'oversampling."
        )
    return candidates[:n_total]


# ─────────────────────── Campionamento delta_max con gap ─────────────────────

def sample_delta_max(n, seed):
    """
    Campiona delta_max su [-20,-2] ∪ [2,20] evitando la banda morta attorno a 0.
    Strategia: campiona u ∈ [0,1] con LHS, mappa su [-20,-2] (prima metà) o
    [2,20] (seconda metà) a seconda del segno.
    """
    rng = np.random.default_rng(seed + 77)
    # LHS 1D su [0, 1]
    if HAS_SCIPY:
        sampler = qmc.LatinHypercube(d=1, seed=seed + 77)
        u = sampler.random(n=n).ravel()
    else:
        perm = rng.permutation(n)
        u = (perm + rng.random(n)) / n

    delta = np.where(u < 0.5,
                     -20.0 + (u / 0.5) * 18.0,        # [-20, -2]
                      2.0  + ((u - 0.5) / 0.5) * 18.0) # [2, 20]
    return delta


# ─────────────────────── Generatori per famiglia ─────────────────────────────

def generate_A(n_train, n_test, seed):
    """
    Famiglia A — gust puro, no flap.
    Parametri LHS: R ∈ [0.10, 0.60], T_g ∈ [0.30, 1.20]
    W_g0 = R * U_inf; delta = 0.
    """
    n_total = n_train + n_test
    samples = sample_with_constraints(
        n_total, 2,
        l_bounds=[0.10, 0.30],
        u_bounds=[0.60, 1.20],
        constraint_fn=None,
        seed=seed,
    )
    rows = []
    for i, s in enumerate(samples):
        R, T_g = s
        rows.append({
            "family":        "A",
            "law":           0,
            "split":         "train" if i < n_train else "test",
            "index":         i,
            "R":             round(R, 6),
            "T_g":           round(T_g, 6),
            "W_g0":          round(R * U_INF, 4),
            "delta_max":     0.0,
            "t_start_delta": 0.0,
            "dt_ramp":       0.0,
        })
    return rows


def generate_B1(n_train, n_test, seed):
    """
    Famiglia B1 — flap puro, no gust, law 1 (ramp + hold).
    W_g = 0 (R = 0).
    Parametri LHS: delta_max ∈ [-20,-2]∪[2,20], dt_ramp ∈ [0.05,0.50],
                   t_start_delta ∈ [0.10, 0.80].
    Vincolo: t_start + dt_ramp < T_CONSTRAINT.
    """
    n_total = n_train + n_test

    # Campiona t_start e dt_ramp con LHS 2D + vincolo
    timing = sample_with_constraints(
        n_total, 2,
        l_bounds=[0.10, 0.05],
        u_bounds=[0.80, 0.50],
        constraint_fn=lambda x: x[0] + x[1] < T_CONSTRAINT,  # t_start + dt_ramp
        seed=seed,
    )
    delta_vals = sample_delta_max(n_total, seed)

    rows = []
    for i in range(n_total):
        t_start, dt_ramp = timing[i]
        delta_max = delta_vals[i]
        rows.append({
            "family":        "B1",
            "law":           1,
            "split":         "train" if i < n_train else "test",
            "index":         i,
            "R":             0.0,
            "T_g":           0.0,
            "W_g0":          0.0,
            "delta_max":     round(float(delta_max), 6),
            "t_start_delta": round(float(t_start), 6),
            "dt_ramp":       round(float(dt_ramp), 6),
        })
    return rows


def generate_C1(n_train, n_test, seed):
    """
    Famiglia C1 — gust + flap combinati, law 1.
    Parametri LHS: R ∈ [0.10,0.60], T_g ∈ [0.30,1.20],
                   delta_max ∈ [-20,-2]∪[2,20],
                   dt_ramp ∈ [0.05,0.50], t_start_delta ∈ [0.10,0.80].
    Vincolo: t_start + dt_ramp < T_CONSTRAINT.
    """
    n_total = n_train + n_test

    # LHS 4D per i parametri continui (escluso delta_max che ha gap)
    samples = sample_with_constraints(
        n_total, 4,
        l_bounds=[0.10, 0.30, 0.10, 0.05],
        u_bounds=[0.60, 1.20, 0.80, 0.50],
        constraint_fn=lambda x: x[2] + x[3] < T_CONSTRAINT,  # t_start + dt_ramp
        seed=seed,
    )
    delta_vals = sample_delta_max(n_total, seed)

    rows = []
    for i in range(n_total):
        R, T_g, t_start, dt_ramp = samples[i]
        delta_max = delta_vals[i]
        rows.append({
            "family":        "C1",
            "law":           1,
            "split":         "train" if i < n_train else "test",
            "index":         i,
            "R":             round(float(R), 6),
            "T_g":           round(float(T_g), 6),
            "W_g0":          round(float(R * U_INF), 4),
            "delta_max":     round(float(delta_max), 6),
            "t_start_delta": round(float(t_start), 6),
            "dt_ramp":       round(float(dt_ramp), 6),
        })
    return rows


# ─────────────────────── Assemblaggio metadata ───────────────────────────────

FIELDNAMES = [
    "global_index", "sim_name", "family", "law", "split", "index",
    "R", "T_g", "W_g0",
    "delta_max", "t_start_delta", "dt_ramp",
]

GENERATORS = {
    "A":  generate_A,
    "B1": generate_B1,
    "C1": generate_C1,
}


def build_metadata(counts, seed=42):
    """
    counts: dict {family: (n_train, n_test)}
    Returns list of row dicts with global_index and sim_name.
    """
    all_rows = []
    global_idx = 0
    for fam, (n_tr, n_te) in counts.items():
        fam_seed = seed + hash(fam) % 10000
        rows = GENERATORS[fam](n_tr, n_te, fam_seed)
        for row in rows:
            row["global_index"] = global_idx
            row["sim_name"] = f"sim_{fam}_{row['index']:03d}_{row['split']}"
            global_idx += 1
        all_rows.extend(rows)
    return all_rows


# ─────────────────────── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output",      type=str, default="metadata.csv")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Stampa il sommario senza scrivere il CSV")
    # Dimensioni per famiglia (da decidere dopo PCA)
    parser.add_argument("--n-A-train",   type=int, default=10)
    parser.add_argument("--n-A-test",    type=int, default=3)
    parser.add_argument("--n-B1-train",  type=int, default=20)
    parser.add_argument("--n-B1-test",   type=int, default=5)
    parser.add_argument("--n-C1-train",  type=int, default=20)
    parser.add_argument("--n-C1-test",   type=int, default=5)
    args = parser.parse_args()

    counts = {
        "A":  (args.n_A_train,  args.n_A_test),
        "B1": (args.n_B1_train, args.n_B1_test),
        "C1": (args.n_C1_train, args.n_C1_test),
    }

    all_rows = build_metadata(counts, seed=args.seed)

    # Sommario
    print(f"\nGLA Dataset — parametri")
    print(f"{'Famiglia':<8} {'Descrizione':<35} {'Train':>6} {'Test':>5} {'Totale':>7}")
    print("-" * 60)
    descriptions = {
        "A":  "Gust puro (W_g≠0, δ=0)",
        "B1": "Flap puro (W_g=0, δ≠0, law 1)",
        "C1": "Combinata (W_g≠0, δ≠0, law 1)",
    }
    for fam, (n_tr, n_te) in counts.items():
        print(f"{fam:<8} {descriptions[fam]:<35} {n_tr:>6} {n_te:>5} {n_tr+n_te:>7}")
    print("-" * 60)
    n_tr_tot = sum(v[0] for v in counts.values())
    n_te_tot = sum(v[1] for v in counts.values())
    print(f"{'TOTALE':<44} {n_tr_tot:>6} {n_te_tot:>5} {n_tr_tot+n_te_tot:>7}")

    # Anteprima campioni
    print(f"\nAnteprima (prime 3 righe per famiglia):")
    for fam in counts:
        fam_rows = [r for r in all_rows if r["family"] == fam][:3]
        for r in fam_rows:
            print(f"  {r['sim_name']:<30}  R={r['R']:.3f}  T_g={r['T_g']:.3f}  "
                  f"δ_max={r['delta_max']:+.1f}°  dt_ramp={r['dt_ramp']:.3f}s")

    if args.dry_run:
        print(f"\n[DRY RUN] Nessun file scritto. Rimuovi --dry-run per salvare.")
        return

    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSalvato → {out_path} ({len(all_rows)} simulazioni)")


if __name__ == "__main__":
    main()
