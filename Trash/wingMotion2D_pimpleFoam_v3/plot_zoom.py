#!/usr/bin/env python3
import numpy as np, matplotlib, re
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cosim_driver import read_forces, U_INF

t, Fy, Mz = read_forces(0.0, 5.0)
_, idx = np.unique(t, return_index=True)
t, Fy, Mz = t[idx], Fy[idx], Mz[idx]

RHO = 1.225; AREF = 0.25
Cl = Fy / (0.5 * RHO * U_INF**2 * AREF)

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Zoom 0.3-0.5s
mask = (t >= 0.3) & (t <= 0.5)
axes[0].plot(t[mask], Cl[mask], ".-", ms=3, lw=0.8)
axes[0].set_title(f"Zoom 0.3-0.5s ({mask.sum()} points)")
axes[0].set_ylabel("CL")
axes[0].grid(True)

# Zoom 2.0-2.2s
mask2 = (t >= 2.0) & (t <= 2.2)
axes[1].plot(t[mask2], Cl[mask2], ".-", ms=3, lw=0.8)
axes[1].set_title(f"Zoom 2.0-2.2s ({mask2.sum()} points)")
axes[1].set_ylabel("CL")
axes[1].set_xlabel("Time [s]")
axes[1].grid(True)

fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/zoom_Cl.png", dpi=150)
print("Saved → figures/zoom_Cl.png")
