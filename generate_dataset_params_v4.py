#!/usr/bin/env python3
"""
generate_dataset_params_v4.py — Dataset redesign post-PCA.

Motivazione (da analisi PCA):
  - La dimensionalità effettiva di (C_L, C_M) è 2, guidata da W(t) e δ(t).
  - La famiglia B1 (solo flap, no raffica) non è fisicamente realistica per
    addestrare una LDNet usata in un loop di controllo GLA — il controllore
    aziona il flap sempre in risposta a una raffica, non nel vuoto.
  - delta_max, t_start, dt_ramp come parametri scalari sono una
    parametrizzazione povera: quello che conta è la forma di δ(t).

Nuovo design:

  Famiglia A   — Gust puro, delta=0 (baseline aerodinamica)
                 LHS 2D: R ∈ [0.10, 0.60], T_g ∈ [0.30, 1.20]

  Famiglia Cc  — Gust + controllo realistico (law 5: feed-forward su W_g)
                 δ(t) = clip(-K_eff * W_g(t - tau), -20, +20)
                 LHS 4D: R, T_g, K_eff ∈ [0.05, 0.20] deg/(m/s), tau ∈ [0, 0.3] s

Split: train / val / test  (3 split invece di 2).

Usage:
    python generate_dataset_params_v4.py --dry-run
    python generate_dataset_params_v4.py --output metadata_v4.csv
    python generate_dataset_params_v4.py \\
        --n-A-train 20 --n-A-val 5 --n-A-test 5 \\
        --n-Cc-train 50 --n-Cc-val 10 --n-Cc-test 10
"""

import argparse
import csv
import numpy as np
from pathlib import Path

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not found — using stratified sampling")

# ── Parametri fisici fissi ────────────────────────────────────────────────────

U_INF = 80.0   # m/s
T_SIM = 3.0    # s
DELTA_MAX_ABS = 20.0  # deg, saturazione fisica flap

# ── LHS sampling ─────────────────────────────────────────────────────────────

def lhs_sample(n, d, l_bounds, u_bounds, seed):
    if HAS_SCIPY:
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        raw = sampler.random(n=n)
        return qmc.scale(raw, l_bounds, u_bounds)
    rng = np.random.default_rng(seed)
    raw = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        raw[:, j] = (perm + rng.random(n)) / n
    return raw * (np.array(u_bounds) - np.array(l_bounds)) + np.array(l_bounds)


def sample_with_constraints(n_total, d, l_bounds, u_bounds,
                             constraint_fn, seed, oversample=20):
    candidates = lhs_sample(n_total * oversample, d, l_bounds, u_bounds, seed)
    if constraint_fn is not None:
        mask = np.array([constraint_fn(row) for row in candidates])
        candidates = candidates[mask]
    if len(candidates) < n_total:
        raise RuntimeError(
            f"Solo {len(candidates)} campioni validi su {n_total*oversample}. "
            "Rilassa i vincoli o aumenta l'oversampling."
        )
    return candidates[:n_total]


# ── Generatori per famiglia ───────────────────────────────────────────────────

def generate_A(n_train, n_val, n_test, seed):
    """
    Famiglia A — gust puro, delta=0.
    LHS 2D: R ∈ [0.10, 0.60], T_g ∈ [0.30, 1.20].
    """
    n_total = n_train + n_val + n_test
    samples = lhs_sample(n_total, 2,
                         l_bounds=[0.10, 0.30],
                         u_bounds=[0.60, 1.20],
                         seed=seed)
    splits = (["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    rows = []
    for i, (s, split) in enumerate(zip(samples, splits)):
        R, T_g = s
        rows.append({
            "family":        "A",
            "law":           0,
            "split":         split,
            "index":         i,
            "R":             round(float(R), 6),
            "T_g":           round(float(T_g), 6),
            "W_g0":          round(float(R * U_INF), 4),
            "K_eff":         0.0,
            "tau":           0.0,
            "delta_max_eff": 0.0,
        })
    return rows


def generate_Cc(n_train, n_val, n_test, seed):
    """
    Famiglia Cc — gust + controllo feed-forward su W_g, law 5.

    δ_ref(t) = clip(-K_eff * W_g(t - tau), -20, +20)
    δ(t)     = rate_limited(δ_ref, delta_rate_max)

    dove W_g(t) è la raffica 1-coseno analitica (nota a priori).

    LHS 5D:
        R              ∈ [0.10, 0.60]            — intensità raffica
        T_g            ∈ [0.30, 1.20] s          — durata raffica
        K_eff          ∈ [0.05, 0.20] deg/(m/s)  — guadagno feed-forward
        tau            ∈ [0.00, 0.30] s           — ritardo del controllore
        delta_rate_max ∈ [20,  200]  deg/s        — rate limit attuatore

    Motivazione rate limit (da analisi PCA):
        La PCA ha mostrato che la velocità di applicazione di δ è importante
        quanto δ_max nel determinare la risposta. Il rate limit cattura questa
        variabilità in modo fisicamente diretto, indipendentemente da K_eff e tau.
        Range 20-200 deg/s: 20 deg/s = attuatore lento (1s per full deflection),
        200 deg/s = attuatore rapido (0.1s per full deflection).

    Stima delta_max_eff = clip(K_eff * W_g0, 0, 20) [deg] — per reference.
    """
    n_total = n_train + n_val + n_test
    samples = lhs_sample(n_total, 5,
                         l_bounds=[0.10, 0.30, 0.05, 0.00,  20.0],
                         u_bounds=[0.60, 1.20, 0.20, 0.30, 200.0],
                         seed=seed)
    splits = (["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    rows = []
    for i, (s, split) in enumerate(zip(samples, splits)):
        R, T_g, K_eff, tau, delta_rate_max = s
        W_g0 = R * U_INF
        delta_max_eff = min(K_eff * W_g0, DELTA_MAX_ABS)
        rows.append({
            "family":          "Cc",
            "law":             5,
            "split":           split,
            "index":           i,
            "R":               round(float(R), 6),
            "T_g":             round(float(T_g), 6),
            "W_g0":            round(float(W_g0), 4),
            "K_eff":           round(float(K_eff), 6),
            "tau":             round(float(tau), 6),
            "delta_rate_max":  round(float(delta_rate_max), 2),
            "delta_max_eff":   round(float(delta_max_eff), 4),
        })
    return rows


# ── Assemblaggio metadata ─────────────────────────────────────────────────────

FIELDNAMES = [
    "global_index", "sim_name", "family", "law", "split", "index",
    "R", "T_g", "W_g0",
    "K_eff", "tau", "delta_rate_max", "delta_max_eff",
]

GENERATORS = {
    "A":  generate_A,
    "Cc": generate_Cc,
}

DESCRIPTIONS = {
    "A":  "Gust puro  (W_g≠0, δ=0)",
    "Cc": "Gust+ctrl  (δ=-K_eff·W_g(t-τ))",
}


def build_metadata(counts, seed=42):
    """
    counts: dict {family: (n_train, n_val, n_test)}
    """
    all_rows = []
    global_idx = 0
    for fam, (n_tr, n_va, n_te) in counts.items():
        fam_seed = seed + hash(fam) % 10000
        rows = GENERATORS[fam](n_tr, n_va, n_te, fam_seed)
        for row in rows:
            row["global_index"] = global_idx
            row["sim_name"] = f"sim_{fam}_{row['index']:03d}_{row['split']}"
            global_idx += 1
        all_rows.extend(rows)
    return all_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--output",       type=str, default="metadata_v4.csv")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--n-A-train",    type=int, default=20)
    parser.add_argument("--n-A-val",      type=int, default=5)
    parser.add_argument("--n-A-test",     type=int, default=5)
    parser.add_argument("--n-Cc-train",   type=int, default=50)
    parser.add_argument("--n-Cc-val",     type=int, default=10)
    parser.add_argument("--n-Cc-test",    type=int, default=10)
    args = parser.parse_args()

    counts = {
        "A":  (args.n_A_train,  args.n_A_val,  args.n_A_test),
        "Cc": (args.n_Cc_train, args.n_Cc_val, args.n_Cc_test),
    }

    all_rows = build_metadata(counts, seed=args.seed)

    # Sommario
    print(f"\nGLA Dataset v4 — parametri")
    print(f"{'Famiglia':<8} {'Descrizione':<35} {'Train':>6} {'Val':>5} {'Test':>5} {'Tot':>5}")
    print("-" * 65)
    for fam, (n_tr, n_va, n_te) in counts.items():
        print(f"{fam:<8} {DESCRIPTIONS[fam]:<35} {n_tr:>6} {n_va:>5} {n_te:>5} {n_tr+n_va+n_te:>5}")
    print("-" * 65)
    n_tr_tot = sum(v[0] for v in counts.values())
    n_va_tot = sum(v[1] for v in counts.values())
    n_te_tot = sum(v[2] for v in counts.values())
    print(f"{'TOTALE':<44} {n_tr_tot:>6} {n_va_tot:>5} {n_te_tot:>5} {n_tr_tot+n_va_tot+n_te_tot:>5}")

    # Anteprima
    print(f"\nAnteprima (prime 3 righe per famiglia):")
    for fam in counts:
        fam_rows = [r for r in all_rows if r["family"] == fam][:3]
        for r in fam_rows:
            rate_str = f"  ṙ={r.get('delta_rate_max', 0):.0f}°/s" if r["family"] == "Cc" else ""
            print(f"  {r['sim_name']:<30}  R={r['R']:.3f}  T_g={r['T_g']:.3f}  "
                  f"K_eff={r.get('K_eff', 0):.3f}  τ={r.get('tau', 0):.3f}s"
                  f"{rate_str}  δ_eff≈{r.get('delta_max_eff', 0):+.1f}°")

    if args.dry_run:
        print(f"\n[DRY RUN] Nessun file scritto.")
        return

    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSalvato → {out_path}  ({len(all_rows)} simulazioni)")


if __name__ == "__main__":
    main()
