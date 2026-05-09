#!/usr/bin/env python3
"""
design_grid.py — Design strategic extraction grid for U,p fields.

The grid is denser near the airfoil and in the wake region,
sparser in the far-field. Accounts for airfoil motion envelope:
  - heave: ±70 mm (±0.07 m)
  - pitch: ±4° about EA at x=0.25

Grid regions:
  1. Near-field: tight around airfoil + motion envelope (-0.3 to 1.5c, -0.4 to 0.4c)
  2. Wake: downstream of TE (1.0 to 4.0c, -0.6 to 0.6c)
  3. Upstream: inflow region (-2.0 to -0.3c, -1.0 to 1.0c)
  4. Far-field: coarse outer region

Target: ~7500 points total.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def build_grid():
    """Build strategic non-uniform grid. Returns (N, 2) array of (x, y) points."""

    points = []

    # ── Region 1: Near-field around airfoil ──────────────────────────────────
    # Airfoil at x∈[0, 1], moves ±0.07m in y, ±4° pitch about EA(0.25, 0)
    # Dense grid: Δx≈0.02c, Δy≈0.02c
    x1 = np.linspace(-0.3, 1.5, 90)     # 90 points
    y1 = np.linspace(-0.35, 0.35, 35)   # 35 points
    X1, Y1 = np.meshgrid(x1, y1)
    points.append(np.column_stack([X1.ravel(), Y1.ravel()]))
    n1 = X1.size
    print(f"  Near-field:  {n1} points ({len(x1)}×{len(y1)}), "
          f"x=[{x1[0]:.1f}, {x1[-1]:.1f}], y=[{y1[0]:.2f}, {y1[-1]:.2f}]")

    # ── Region 2: Wake region ────────────────────────────────────────────────
    # Behind TE, wake spreads — medium density: Δx≈0.06c, Δy≈0.04c
    x2 = np.linspace(1.5, 4.0, 42)      # 42 points
    y2 = np.linspace(-0.5, 0.5, 25)     # 25 points
    X2, Y2 = np.meshgrid(x2, y2)
    points.append(np.column_stack([X2.ravel(), Y2.ravel()]))
    n2 = X2.size
    print(f"  Wake:        {n2} points ({len(x2)}×{len(y2)}), "
          f"x=[{x2[0]:.1f}, {x2[-1]:.1f}], y=[{y2[0]:.2f}, {y2[-1]:.2f}]")

    # ── Region 3: Upstream ───────────────────────────────────────────────────
    # Inflow — captures gust arrival, coarser: Δx≈0.1c, Δy≈0.08c
    x3 = np.linspace(-2.0, -0.3, 18)    # 18 points
    y3 = np.linspace(-0.8, 0.8, 20)     # 20 points
    X3, Y3 = np.meshgrid(x3, y3)
    points.append(np.column_stack([X3.ravel(), Y3.ravel()]))
    n3 = X3.size
    print(f"  Upstream:    {n3} points ({len(x3)}×{len(y3)}), "
          f"x=[{x3[0]:.1f}, {x3[-1]:.1f}], y=[{y3[0]:.2f}, {y3[-1]:.2f}]")

    # ── Region 4: Far-field (above/below near+wake) ─────────────────────────
    # Coarse: Δx≈0.15c, Δy≈0.1c
    x4 = np.linspace(-2.0, 4.0, 40)     # 40 points
    # Upper far-field
    y4_upper = np.linspace(0.35, 1.5, 12)
    X4u, Y4u = np.meshgrid(x4, y4_upper)
    points.append(np.column_stack([X4u.ravel(), Y4u.ravel()]))
    # Lower far-field
    y4_lower = np.linspace(-1.5, -0.35, 12)
    X4l, Y4l = np.meshgrid(x4, y4_lower)
    points.append(np.column_stack([X4l.ravel(), Y4l.ravel()]))
    n4 = X4u.size + X4l.size
    print(f"  Far-field:   {n4} points ({len(x4)}×{2*12}), "
          f"x=[{x4[0]:.1f}, {x4[-1]:.1f}], y=[{y4_lower[0]:.1f}, {y4_upper[-1]:.1f}]")

    # ── Combine and deduplicate ──────────────────────────────────────────────
    all_points = np.vstack(points)

    # Remove duplicates (points near region boundaries)
    _, unique_idx = np.unique(np.round(all_points, 6), axis=0, return_index=True)
    grid = all_points[unique_idx]

    print(f"\n  Total before dedup: {len(all_points)}")
    print(f"  Total after dedup:  {len(grid)}")
    print(f"  Domain: x=[{grid[:,0].min():.1f}, {grid[:,0].max():.1f}], "
          f"y=[{grid[:,1].min():.2f}, {grid[:,1].max():.2f}]")

    # Storage estimate
    n_sim, n_t, n_ch = 64, 1500, 3
    gb = n_sim * n_t * len(grid) * n_ch * 4 / 1e9
    print(f"\n  Storage estimate: {n_sim} sims × {n_t} timesteps × {len(grid)} pts × "
          f"{n_ch} channels × 4B = {gb:.1f} GB")

    return grid


def plot_grid(grid, output_path="figures/extraction_grid.png"):
    """Visualize the grid with the airfoil outline."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Full domain ──
    ax = axes[0]
    ax.scatter(grid[:, 0], grid[:, 1], s=0.3, c="steelblue", alpha=0.5)

    # Airfoil outline (NACA 2312 approx)
    x_af = np.linspace(0, 1, 100)
    t_af = 0.12  # max thickness
    yt = 5 * t_af * (0.2969*np.sqrt(x_af) - 0.1260*x_af - 0.3516*x_af**2
                      + 0.2843*x_af**3 - 0.1015*x_af**4)
    yc = np.where(x_af < 0.3,
                  0.02/0.09 * (0.06*x_af - x_af**2),
                  0.02/0.49 * (0.098 - 0.196*x_af + x_af**2 - x_af**2 + 0.06*x_af))
    ax.fill_between(x_af, yc - yt, yc + yt, color="gray", alpha=0.4, label="NACA 2312")
    ax.axvline(0.25, color="red", ls="--", lw=0.8, alpha=0.5, label="EA")
    ax.axvline(0.779, color="orange", ls="--", lw=0.8, alpha=0.5, label="Hinge")

    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_title(f"Extraction grid — {len(grid)} points (full domain)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.4)

    # ── Zoom near airfoil ──
    ax = axes[1]
    mask = (grid[:, 0] > -0.5) & (grid[:, 0] < 2.0) & \
           (grid[:, 1] > -0.5) & (grid[:, 1] < 0.5)
    ax.scatter(grid[mask, 0], grid[mask, 1], s=1, c="steelblue", alpha=0.6)
    ax.fill_between(x_af, yc - yt, yc + yt, color="gray", alpha=0.4)
    ax.axvline(0.25, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0.779, color="orange", ls="--", lw=0.8, alpha=0.5)

    # Motion envelope (±70mm heave, ±4° pitch)
    ax.axhspan(-0.07, 0.07, color="red", alpha=0.05, label="Heave envelope")

    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_title("Zoom: near-field")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(-0.5, 0.5)

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved grid plot → {output_path}")


def main():
    print("Designing extraction grid...\n")
    grid = build_grid()

    # Save grid
    Path("data/GLA").mkdir(parents=True, exist_ok=True)
    np.save("data/GLA/grid_points.npy", grid.astype(np.float32))
    print(f"  Saved → data/GLA/grid_points.npy")

    plot_grid(grid)


if __name__ == "__main__":
    main()
