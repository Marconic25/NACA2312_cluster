"""
PCA on h(t) and alpha(t) trajectories from dataset_weekend simulations.

Goal: verify that the effective dimensionality of (h, alpha) is 2, driven by
the two input signals W_gust and delta. If the first 2 PCs explain >95% of
variance and their scores correlate with W_g0 and delta_max, then the LDNet
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

def load_trajectories(meta: pd.DataFrame, data_dir: Path, T: int):
    """
    Returns:
        X      : ndarray (N, 2*T)  — rows are [h | alpha] for each sim
        params : DataFrame (N, ...)  — metadata rows that were successfully loaded
    """
    rows_X = []
    loaded_idx = []

    for i, row in meta.iterrows():
        traj_path = data_dir / row["sim_name"] / "structural_trajectory.csv"
        if not traj_path.exists():
            print(f"  [skip] {row['sim_name']}: file not found")
            continue

        df = pd.read_csv(traj_path)

        # Normalise column names (strip spaces, lowercase)
        df.columns = [c.strip().lower() for c in df.columns]

        # Accept 'h' or 'heave', 'alpha' or 'pitch'
        h_col = next((c for c in df.columns if c in ("h", "heave")), None)
        a_col = next((c for c in df.columns if c in ("alpha", "pitch", "alpha_deg")), None)

        if h_col is None or a_col is None:
            print(f"  [skip] {row['sim_name']}: missing h or alpha column "
                  f"(found: {list(df.columns)})")
            continue

        h = df[h_col].values
        alpha = df[a_col].values

        if len(h) < T:
            print(f"  [warn] {row['sim_name']}: only {len(h)} timesteps, expected {T}; truncating")
        h = h[:T]
        alpha = alpha[:T]

        if len(h) < T:
            # Pad with last value if still short (should not happen normally)
            h = np.pad(h, (0, T - len(h)), mode="edge")
            alpha = np.pad(alpha, (0, T - len(alpha)), mode="edge")

        rows_X.append(np.concatenate([h, alpha]))
        loaded_idx.append(i)

    if not rows_X:
        sys.exit("No simulation data could be loaded. Check --data-dir and sim folder structure.")

    X = np.array(rows_X)          # (N, 2T)
    params = meta.loc[loaded_idx].reset_index(drop=True)
    print(f"\nLoaded {len(rows_X)} simulations  →  X shape: {X.shape}")
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


def plot_loadings(V: np.ndarray, T: int, out_dir: Path):
    """Plot the temporal mode shapes for h and alpha for PC1 and PC2."""
    t = np.arange(T) * 0.002   # dt_save = 0.002 s

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    titles = ["PC 1 loading", "PC 2 loading"]
    var_labels = ["h  (m)", "α  (deg)"]

    for pc in range(min(2, V.shape[1])):
        v = V[:, pc]
        h_mode = v[:T]
        a_mode = v[T:2*T]

        axes[0, pc].plot(t, h_mode, color="#4c8fbd", lw=1.5)
        axes[0, pc].set_title(titles[pc])
        axes[0, pc].set_ylabel(var_labels[0])
        axes[0, pc].axhline(0, color="gray", lw=0.8, ls="--")

        axes[1, pc].plot(t, a_mode, color="#e07b39", lw=1.5)
        axes[1, pc].set_ylabel(var_labels[1])
        axes[1, pc].set_xlabel("Time (s)")
        axes[1, pc].axhline(0, color="gray", lw=0.8, ls="--")

    fig.suptitle("PCA loadings — temporal mode shapes of h and α")
    fig.tight_layout()
    fig.savefig(out_dir / "loadings.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'loadings.png'}")


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta = pd.read_csv(args.metadata)
    print(f"Metadata: {len(meta)} simulations, columns: {list(meta.columns)}")

    # Load trajectory data
    X, params = load_trajectories(meta, args.data_dir, args.t_length)
    N, DIM = X.shape
    T = args.t_length

    # PCA
    n_comp = min(N, 10)
    U, evr, V, _ = run_pca(X, n_components=n_comp)

    # Print variance table
    print("\n--- Explained variance ratio ---")
    print(f"{'PC':>4s}  {'EVR (%)':>8s}  {'Cumulative (%)':>14s}")
    cumsum = 0.0
    for k in range(n_comp):
        cumsum += evr[k] * 100
        print(f"{k+1:>4d}  {evr[k]*100:>8.2f}  {cumsum:>14.2f}")

    # Correlation analysis
    correlation_table(U, params, n_pc=min(5, n_comp))

    # Plots
    plot_scree(evr[:n_comp], args.out_dir)
    plot_scores_by_family(U, params, args.out_dir)

    if "W_g0" in params.columns:
        plot_scores_colored(U, params, "W_g0", "W_g0 (m/s)", "scores_W.png", args.out_dir)

    if "delta_max" in params.columns:
        plot_scores_colored(U, params, "delta_max", "δ_max (deg)", "scores_delta.png", args.out_dir)

    plot_loadings(V, T, args.out_dir)

    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
