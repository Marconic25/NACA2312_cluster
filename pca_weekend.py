"""
PCA on output trajectories from dataset_weekend simulations.

Runs two parallel PCA analyses:
  1. Structural outputs: h(t) and alpha(t)
  2. Aerodynamic outputs: C_L(t) and C_M(t)  ← direct LDNet targets

Goal: verify that both output spaces have effective dimensionality 2, driven by
the two input signals W_gust(t) and delta(t). If the first 2 PCs explain >95%
of variance and their scores correlate with W_g0 and delta_max, then the LDNet
only needs W and delta as inputs.

Usage:
    python pca_weekend.py \
        --data-dir /work/u10677113/NACA2312/dataset_weekend \
        --metadata /work/u10677113/NACA2312/dataset_weekend/metadata_weekend.csv \
        --out-dir ./pca_results
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PCA on h and alpha trajectories")
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Base directory containing sim_XXX folders")
    p.add_argument("--metadata", required=True, type=Path,
                   help="Path to metadata_weekend.csv")
    p.add_argument("--out-dir", default=Path("pca_results"), type=Path,
                   help="Directory for output plots (created if missing)")
    p.add_argument("--t-length", default=1500, type=int,
                   help="Expected number of timesteps per simulation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Aerodynamic reference quantities for coefficient normalisation
RHO = 1.225       # kg/m³
U_INF = 80.0      # m/s
A_REF = 0.25      # m²  (chord 1.0 m × span 0.25 m)
C_REF = 1.0       # m   (chord, for moment normalisation)
Q_INF = 0.5 * RHO * U_INF**2   # dynamic pressure [Pa]

# Column name aliases for each variable
COL_ALIASES = {
    "h":     ("h", "heave"),
    "alpha": ("alpha", "pitch", "alpha_deg"),
    "cl":    ("cl", "c_l", "lift_coeff", "cl_total"),
    "cm":    ("cm", "c_m", "moment_coeff", "cm_total"),
    # raw forces — converted to coefficients on load
    "fy":    ("fy", "f_y", "lift", "lift_force"),
    "mz":    ("mz", "m_z", "moment", "pitch_moment"),
}


def _find_col(df_cols, aliases):
    return next((c for c in df_cols if c in aliases), None)


def load_trajectories(meta: pd.DataFrame, data_dir: Path, T: int,
                      var1: str, var2: str):
    """
    Load two output variables from structural_trajectory.csv for each sim.

    Args:
        var1, var2: keys into COL_ALIASES (e.g. 'h'/'alpha' or 'cl'/'cm').
                    If var1='cl' and the file only has 'fy', converts automatically.
                    Same for var2='cm' / 'mz'.

    Returns:
        X      : ndarray (N, 2*T)  — rows are [var1 | var2] for each sim
        params : DataFrame (N, ...)  — metadata rows successfully loaded
    """
    rows_X = []
    loaded_idx = []

    # For cl/cm, also accept raw force columns and convert
    fallback = {"cl": "fy", "cm": "mz"}

    def extract(df, var):
        aliases = COL_ALIASES[var]
        col = _find_col(df.columns, aliases)
        if col is not None:
            return df[col].values
        # try force fallback
        fb = fallback.get(var)
        if fb:
            col = _find_col(df.columns, COL_ALIASES[fb])
            if col is not None:
                raw = df[col].values
                if var == "cl":
                    return raw / (Q_INF * A_REF)
                if var == "cm":
                    return raw / (Q_INF * A_REF * C_REF)
        return None

    for i, row in meta.iterrows():
        traj_path = data_dir / row["sim_name"] / "structural_trajectory.csv"
        if not traj_path.exists():
            print(f"  [skip] {row['sim_name']}: file not found")
            continue

        df = pd.read_csv(traj_path)
        df.columns = [c.strip().lower() for c in df.columns]

        v1 = extract(df, var1)
        v2 = extract(df, var2)

        if v1 is None or v2 is None:
            print(f"  [skip] {row['sim_name']}: missing {var1}/{var2} columns "
                  f"(found: {list(df.columns)})")
            continue

        v1 = v1[:T]
        v2 = v2[:T]
        if len(v1) < T:
            v1 = np.pad(v1, (0, T - len(v1)), mode="edge")
            v2 = np.pad(v2, (0, T - len(v2)), mode="edge")

        rows_X.append(np.concatenate([v1, v2]))
        loaded_idx.append(i)

    if not rows_X:
        sys.exit(f"No data loaded for ({var1}, {var2}). Check column names.")

    X = np.array(rows_X)
    params = meta.loc[loaded_idx].reset_index(drop=True)
    print(f"  Loaded {len(rows_X)} sims  →  X shape: {X.shape}")
    return X, params


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_pca(X: np.ndarray, n_components: int = None):
    """
    Returns scores U (N, k), explained variance ratio evr (k,),
    loadings V (2T, k), and the column-mean used for centring.
    """
    mean = X.mean(axis=0)
    Xc = X - mean

    # SVD — use economy form; X is (N, 2T) with N << 2T so svd on Xc directly
    # is fine for small N.
    U_full, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    evr = s**2 / (s**2).sum()

    if n_components is None:
        n_components = len(s)

    U = U_full[:, :n_components] * s[:n_components]   # scores (N, k)
    V = Vt[:n_components, :].T                         # loadings (2T, k)

    return U, evr, V, mean


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def correlation_table(scores: np.ndarray, params: pd.DataFrame, n_pc: int = 5):
    """Print Pearson correlation between each score and each input parameter."""
    input_cols = ["W_g0", "delta_max", "T_g", "R", "dt_ramp", "t_start_delta"]
    available = [c for c in input_cols if c in params.columns]

    print("\n--- Pearson correlation: PC scores vs input parameters ---")
    header = f"{'':>12s}" + "".join(f"  PC{k+1:>2d}" for k in range(min(n_pc, scores.shape[1])))
    print(header)
    print("-" * len(header))

    for col in available:
        vals = params[col].values.astype(float)
        if np.nanstd(vals) < 1e-12:
            continue
        line = f"{col:>12s}"
        for k in range(min(n_pc, scores.shape[1])):
            mask = np.isfinite(vals)
            r, _ = pearsonr(scores[mask, k], vals[mask])
            line += f"  {r:+.3f}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

MARKERS = {"A": "o", "B1": "s", "C1": "^"}
COLORS_FAM = {"A": "#e07b39", "B1": "#4c8fbd", "C1": "#5caa5c"}


def family_style(fam):
    return MARKERS.get(fam, "D"), COLORS_FAM.get(fam, "gray")


def plot_scree(evr: np.ndarray, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    k = np.arange(1, len(evr) + 1)
    ax.bar(k, evr * 100, color="#4c8fbd", alpha=0.8, label="Individual")
    ax.plot(k, np.cumsum(evr) * 100, "o-", color="#e07b39", lw=2, label="Cumulative")
    ax.axhline(95, ls="--", color="gray", lw=1)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Scree plot — h and α trajectories")
    ax.legend()
    ax.set_xticks(k)
    fig.tight_layout()
    fig.savefig(out_dir / "scree_plot.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'scree_plot.png'}")


def plot_scores_colored(scores: np.ndarray, params: pd.DataFrame,
                        color_col: str, label: str, fname: str, out_dir: Path):
    vals = params[color_col].values.astype(float)
    valid = np.isfinite(vals) & (np.abs(vals) > 1e-10)

    if valid.sum() < 2:
        print(f"  [skip] {fname}: not enough valid points for {color_col}")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot sims not in this coloring (e.g., no gust when coloring by W_g0)
    for i, row in params.iterrows():
        if not valid[i]:
            m, _ = family_style(row.get("family", "?"))
            ax.scatter(scores[i, 0], scores[i, 1],
                       marker=m, color="lightgray", s=60, zorder=2)

    sc = ax.scatter(scores[valid, 0], scores[valid, 1],
                    c=vals[valid], cmap="plasma", s=80,
                    marker="o", zorder=3)
    plt.colorbar(sc, ax=ax, label=label)

    # Family legend
    for fam, (m, c) in zip(MARKERS.keys(), [family_style(f) for f in MARKERS]):
        ax.scatter([], [], marker=m, color=c, label=f"Family {fam}", s=60)
    ax.legend(fontsize=8, loc="best")

    ax.set_xlabel("PC 1 score")
    ax.set_ylabel("PC 2 score")
    ax.set_title(f"PCA scores coloured by {label}")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / fname}")


def plot_loadings(V: np.ndarray, T: int, out_dir: Path,
                  label1: str, label2: str, fname: str):
    """Plot the temporal mode shapes for two variables for PC1 and PC2."""
    t = np.arange(T) * 0.002

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

    for pc in range(min(2, V.shape[1])):
        v = V[:, pc]
        axes[0, pc].plot(t, v[:T], color="#4c8fbd", lw=1.5)
        axes[0, pc].set_title(f"PC {pc+1} loading")
        axes[0, pc].set_ylabel(label1)
        axes[0, pc].axhline(0, color="gray", lw=0.8, ls="--")

        axes[1, pc].plot(t, v[T:2*T], color="#e07b39", lw=1.5)
        axes[1, pc].set_ylabel(label2)
        axes[1, pc].set_xlabel("Time (s)")
        axes[1, pc].axhline(0, color="gray", lw=0.8, ls="--")

    fig.suptitle(f"PCA loadings — {label1} and {label2}")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / fname}")


def plot_scores_by_family(scores: np.ndarray, params: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for fam, (m, c) in zip(MARKERS.keys(), [family_style(f) for f in MARKERS]):
        mask = params["family"].values == fam
        if mask.sum() == 0:
            continue
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   marker=m, color=c, s=80, label=f"Family {fam}", zorder=3)
    ax.set_xlabel("PC 1 score")
    ax.set_ylabel("PC 2 score")
    ax.set_title("PCA scores by simulation family")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "scores_family.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'scores_family.png'}")


def pca_per_family(X: np.ndarray, params: pd.DataFrame, T: int, out_dir: Path,
                   label1: str, label2: str, tag: str):
    """PCA separata per ogni famiglia."""
    for fam in sorted(params["family"].unique()):
        mask = params["family"].values == fam
        if mask.sum() < 3:
            print(f"  [skip per-family PCA] {fam}: only {mask.sum()} sims, need ≥3")
            continue

        X_fam = X[mask]
        params_fam = params[mask].reset_index(drop=True)
        n_comp = min(mask.sum(), 5)
        U, evr, V, _ = run_pca(X_fam, n_components=n_comp)

        print(f"\n--- Family {fam} [{tag}] — EVR ({mask.sum()} sims) ---")
        print(f"{'PC':>4s}  {'EVR (%)':>8s}  {'Cumulative (%)':>14s}")
        cumsum = 0.0
        for k in range(n_comp):
            cumsum += evr[k] * 100
            print(f"{k+1:>4d}  {evr[k]*100:>8.2f}  {cumsum:>14.2f}")
        correlation_table(U, params_fam, n_pc=min(3, n_comp))

        # Scree
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ks = np.arange(1, n_comp + 1)
        ax.bar(ks, evr[:n_comp] * 100, color=COLORS_FAM.get(fam, "gray"), alpha=0.8)
        ax.plot(ks, np.cumsum(evr[:n_comp]) * 100, "o-", color="black", lw=1.5)
        ax.axhline(95, ls="--", color="gray", lw=1)
        ax.set_xlabel("PC")
        ax.set_ylabel("Explained variance (%)")
        ax.set_title(f"Family {fam} [{tag}] — scree")
        ax.set_xticks(ks)
        fig.tight_layout()
        fig.savefig(out_dir / f"scree_{fam}_{tag}.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / f'scree_{fam}_{tag}.png'}")

        # Scores colorati
        color_col = "W_g0" if fam == "A" else "delta_max" if fam == "B1" else "W_g0"
        color_label = "W_g0 (m/s)" if color_col == "W_g0" else "δ_max (deg)"
        if color_col in params_fam.columns and U.shape[1] >= 2:
            vals = params_fam[color_col].values.astype(float)
            valid = np.isfinite(vals) & (np.abs(vals) > 1e-10)
            if valid.sum() >= 2:
                fig, ax = plt.subplots(figsize=(5, 4))
                sc = ax.scatter(U[valid, 0], U[valid, 1],
                                c=vals[valid], cmap="plasma", s=100, zorder=3)
                plt.colorbar(sc, ax=ax, label=color_label)
                ax.set_xlabel("PC 1 score")
                ax.set_ylabel("PC 2 score")
                ax.set_title(f"Family {fam} [{tag}] — scores vs {color_label}")
                fig.tight_layout()
                fig.savefig(out_dir / f"scores_{fam}_{color_col}_{tag}.png", dpi=150)
                plt.close(fig)
                print(f"Saved {out_dir / f'scores_{fam}_{color_col}_{tag}.png'}")

        # Loadings
        if U.shape[1] >= 2:
            plot_loadings(V, T, out_dir, label1, label2,
                          fname=f"loadings_{fam}_{tag}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ANALYSES = [
    # (var1,  var2,    label1,      label2,       tag)
    ("h",    "alpha", "h  (m)",    "α  (deg)",   "struct"),
    ("cl",   "cm",    "C_L  (-)",  "C_M  (-)",   "aero"),
]


def run_analysis(meta, data_dir, T, out_dir, var1, var2, label1, label2, tag):
    sub = out_dir / tag
    sub.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  ANALYSIS: {label1}  &  {label2}  [{tag}]")
    print(f"{'='*55}")

    X, params = load_trajectories(meta, data_dir, T, var1, var2)
    n_comp = min(len(X), 10)
    U, evr, V, _ = run_pca(X, n_components=n_comp)

    print(f"\n--- Explained variance ratio [{tag}] ---")
    print(f"{'PC':>4s}  {'EVR (%)':>8s}  {'Cumulative (%)':>14s}")
    cumsum = 0.0
    for k in range(n_comp):
        cumsum += evr[k] * 100
        print(f"{k+1:>4d}  {evr[k]*100:>8.2f}  {cumsum:>14.2f}")

    correlation_table(U, params, n_pc=min(5, n_comp))

    plot_scree(evr[:n_comp], sub)
    plot_scores_by_family(U, params, sub)
    if "W_g0" in params.columns:
        plot_scores_colored(U, params, "W_g0", "W_g0 (m/s)", "scores_W.png", sub)
    if "delta_max" in params.columns:
        plot_scores_colored(U, params, "delta_max", "δ_max (deg)", "scores_delta.png", sub)
    plot_loadings(V, T, sub, label1, label2, fname="loadings.png")

    print(f"\n--- Per-family PCA [{tag}] ---")
    pca_per_family(X, params, T, sub, label1, label2, tag)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata)
    print(f"Metadata: {len(meta)} simulations, columns: {list(meta.columns)}")

    for var1, var2, label1, label2, tag in ANALYSES:
        run_analysis(meta, args.data_dir, args.t_length,
                     args.out_dir, var1, var2, label1, label2, tag)

    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
