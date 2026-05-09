#!/usr/bin/env python3
"""Final plot: last sample per window, light smoothing."""
import numpy as np, matplotlib, re
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cosim_driver import U_INF, GUST_T_START, GUST_T_END, GUST_W0

forces_base = Path("postProcessing/forces")

# Only take dirs that are valid floats
dirs = []
for d in forces_base.iterdir():
    try: float(d.name); dirs.append(d)
    except ValueError: pass
dirs.sort(key=lambda p: float(p.name))

t_all, Fy_all, Mz_all = [], [], []
for d in dirs:
    ff = d / "forces.dat"
    if not ff.exists(): continue
    # Read all non-comment lines, take last one
    lines = [l.strip() for l in open(ff) if l.strip() and not l.strip().startswith("#")]
    if not lines: continue
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lines[-1])
    if len(nums) < 19: continue
    t_all.append(float(nums[0]))
    Fy_all.append(float(nums[2]) + float(nums[5]))
    Mz_all.append(float(nums[12]) + float(nums[15]))

t = np.array(t_all); Fy = np.array(Fy_all); Mz = np.array(Mz_all)

# Verify no duplicates
_, ui = np.unique(t, return_index=True)
t, Fy, Mz = t[ui], Fy[ui], Mz[ui]

mask = t > 0.04
t, Fy, Mz = t[mask], Fy[mask], Mz[mask]

print(f"Points: {len(t)}, t=[{t[0]:.4f}, {t[-1]:.4f}]")
print(f"Fy range: [{Fy.min():.1f}, {Fy.max():.1f}]")

# Check for outliers: flag points > 3 sigma from rolling mean
def remove_outliers(y, window=15, sigma=3):
    y_clean = y.copy()
    for i in range(len(y)):
        lo = max(0, i - window)
        hi = min(len(y), i + window)
        local = np.concatenate([y[lo:max(0,i-1)], y[min(len(y),i+1):hi]])
        if len(local) < 3: continue
        if abs(y[i] - np.median(local)) > sigma * np.std(local):
            y_clean[i] = np.median(local)
    return y_clean

Fy = remove_outliers(Fy)
Mz = remove_outliers(Mz)

RHO = 1.225; AREF = 0.25
Cl = Fy / (0.5 * RHO * U_INF**2 * AREF)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"Cosim forces — Wg0={GUST_W0:.0f} m/s, U∞={U_INF:.0f} m/s", fontweight="bold")
for ax in axes:
    ax.axvspan(GUST_T_START, GUST_T_END, color="steelblue", alpha=0.12, label="Gust at inlet")
    ax.axvline(GUST_T_START + 10.0/U_INF, color="steelblue", ls="--", lw=1, label="Gust hits wing")
    ax.grid(True, lw=0.4, alpha=0.5)

axes[0].plot(t, Cl, "C0", lw=1); axes[0].set_ylabel("CL [-]")
axes[0].legend(fontsize=7, loc="upper left")
axes[1].plot(t, Fy, "C2", lw=1); axes[1].set_ylabel("Fy [N]")
axes[2].plot(t, Mz, "C3", lw=1); axes[2].set_ylabel("Mz [N·m]")
axes[2].set_xlabel("Time [s]"); axes[2].set_xlim(0, 5)

fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/gust_response_final4.png", dpi=150, bbox_inches="tight")
print("Saved → figures/gust_response_final4.png")
