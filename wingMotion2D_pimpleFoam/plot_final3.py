#!/usr/bin/env python3
"""Final plot: all samples + median filter to remove spikes."""
import numpy as np, matplotlib, re
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cosim_driver import U_INF, GUST_T_START, GUST_T_END, GUST_W0

forces_base = Path("postProcessing/forces")
dirs = sorted(forces_base.iterdir(), key=lambda p: float(p.name))

t_all, Fy_all, Mz_all = [], [], []
for d in dirs:
    ff = d / "forces.dat"
    if not ff.exists(): continue
    count = 0
    with open(ff) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            count += 1
            if count <= 2: continue
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if len(nums) < 19: continue
            t_all.append(float(nums[0]))
            Fy_all.append(float(nums[2]) + float(nums[5]))
            Mz_all.append(float(nums[12]) + float(nums[15]))

t = np.array(t_all); Fy = np.array(Fy_all); Mz = np.array(Mz_all)
idx = np.argsort(t); t, Fy, Mz = t[idx], Fy[idx], Mz[idx]

# Remove duplicates
_, ui = np.unique(t, return_index=True)
t, Fy, Mz = t[ui], Fy[ui], Mz[ui]

mask = t > 0.04
t, Fy, Mz = t[mask], Fy[mask], Mz[mask]

# Median filter (kernel=7) kills isolated spikes
Fy = medfilt(Fy, 7)
Mz = medfilt(Mz, 7)

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
fig.savefig("figures/gust_response_final3.png", dpi=150, bbox_inches="tight")
print("Saved → figures/gust_response_final3.png")
