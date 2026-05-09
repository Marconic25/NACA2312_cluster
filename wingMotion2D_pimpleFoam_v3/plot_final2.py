#!/usr/bin/env python3
"""Final plot: median of last 5 samples per window."""
import numpy as np, matplotlib, re
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    lines = []
    with open(ff) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            lines.append(line)
    if len(lines) < 3: continue
    # Take last 5 samples (or all if fewer)
    tail = lines[-min(5, len(lines)):]
    t_w, Fy_w, Mz_w = [], [], []
    for l in tail:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", l)
        if len(nums) < 19: continue
        t_w.append(float(nums[0]))
        Fy_w.append(float(nums[2]) + float(nums[5]))
        Mz_w.append(float(nums[12]) + float(nums[15]))
    t_all.append(np.median(t_w))
    Fy_all.append(np.median(Fy_w))
    Mz_all.append(np.median(Mz_w))

t = np.array(t_all); Fy = np.array(Fy_all); Mz = np.array(Mz_all)
idx = np.argsort(t); t, Fy, Mz = t[idx], Fy[idx], Mz[idx]
mask = t > 0.04
t, Fy, Mz = t[mask], Fy[mask], Mz[mask]

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
fig.savefig("figures/gust_response_final2.png", dpi=150, bbox_inches="tight")
print("Saved → figures/gust_response_final2.png")
