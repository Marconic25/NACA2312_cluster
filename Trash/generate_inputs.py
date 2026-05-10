"""
generate_inputs.py
Generates input time series for the imposed-motion test case:
  - wing_motion.dat   : h(t) [m] and alpha(t) [rad] for wing_main patch
  - flap_control.dat  : delta(t) [deg] for flap patch
  - gust_profile.dat  : U_inlet(t) [(Ux Uy 0)] including step gust at t=0.5s
"""

import numpy as np
import os

# ── output directory (written into testCase_imposed/constant/) ────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "testCase_imposed", "constant")
os.makedirs(OUT_DIR, exist_ok=True)

# ── time axis ─────────────────────────────────────────────────────────────────
T_END = 5.0
DT    = 0.01
t     = np.arange(0.0, T_END + DT * 0.5, DT)   # inclusive of T_END

# ── a) Wing motion ─────────────────────────────────────────────────────────────
f_wing  = 0.8          # Hz
h       =  0.05 * np.sin(2 * np.pi * f_wing * t)      # [m]
alpha   =  0.10 * np.sin(2 * np.pi * f_wing * t)      # [rad]

wing_path = os.path.join(OUT_DIR, "wing_motion.dat")
with open(wing_path, "w") as fp:
    fp.write("// Wing motion table: (time  (h[m]  alpha[rad]  0))\n")
    fp.write("(\n")
    for ti, hi, ai in zip(t, h, alpha):
        fp.write(f"({ti:.4f}  ({hi:.6f} {ai:.6f} 0.0))\n")
    fp.write(")\n")
print(f"Written {len(t)} points -> {wing_path}")
print(f"  h:     amplitude {np.max(np.abs(h)):.4f} m,  freq {f_wing} Hz")
print(f"  alpha: amplitude {np.max(np.abs(alpha)):.4f} rad = "
      f"{np.degrees(np.max(np.abs(alpha))):.2f} deg")

# ── b) Flap control ────────────────────────────────────────────────────────────
f_flap  = 1.0          # Hz
delta   =  5.0 * np.sin(2 * np.pi * f_flap * t)       # [deg]

flap_path = os.path.join(OUT_DIR, "flap_control.dat")
with open(flap_path, "w") as fp:
    fp.write("// Flap deflection table: (time  delta[deg])\n")
    fp.write("(\n")
    for ti, di in zip(t, delta):
        fp.write(f"({ti:.4f}  {di:.6f})\n")
    fp.write(")\n")
print(f"\nWritten {len(t)} points -> {flap_path}")
print(f"  delta: amplitude {np.max(np.abs(delta)):.2f} deg,  freq {f_flap} Hz")

# ── c) Gust disturbance ────────────────────────────────────────────────────────
U_AXIAL       = 80.0   # m/s  freestream axial velocity (matches case)
W_GUST_FINAL  = 50.0   # m/s  transverse gust amplitude
T_GUST_START  = 0.5    # s    gust onset
T_GUST_RAMP   = 0.05   # s    (1-cos) ramp duration

# Build gust transverse velocity using (1-cos) ramp
W = np.zeros_like(t)
mask_ramp  = (t >= T_GUST_START) & (t < T_GUST_START + T_GUST_RAMP)
mask_full  = t >= T_GUST_START + T_GUST_RAMP
W[mask_ramp] = (W_GUST_FINAL / 2.0) * (
    1.0 - np.cos(np.pi * (t[mask_ramp] - T_GUST_START) / T_GUST_RAMP)
)
W[mask_full] = W_GUST_FINAL

gust_path = os.path.join(OUT_DIR, "gust_profile.dat")
with open(gust_path, "w") as fp:
    fp.write("// Inlet velocity table: (time  (Ux[m/s]  Uy[m/s]  0))\n")
    fp.write("// Gust step at t=0.5s with (1-cos) ramp over 0.05s\n")
    fp.write(f"{len(t)}\n")
    fp.write("(\n")
    for ti, wi in zip(t, W):
        fp.write(f"({ti:.4f}  ({U_AXIAL:.2f} {wi:.6f} 0.0))\n")
    fp.write(")\n")
print(f"\nWritten {len(t)} points -> {gust_path}")
print(f"  Gust: 0 -> {W_GUST_FINAL} m/s at t={T_GUST_START}s, "
      f"ramp over {T_GUST_RAMP}s")
print(f"  Effective AoA from gust at end: "
      f"{np.degrees(np.arctan2(W_GUST_FINAL, U_AXIAL)):.2f} deg")

print("\nDone. Files written to:", OUT_DIR)
