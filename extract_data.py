#!/usr/bin/env python3
"""
extract_data.py — Extract timeseries + fields from completed cosim runs.

Timeseries (1500 samples @ dt=0.002s):
  Forces from postProcessing, structural replay, analytical gust/flap.
  → timeseries/sim_XXX.csv

Fields (150 snapshots @ dt≈0.02s on OF mesh slice):
  Uses pyvista to read OF mesh with native connectivity.
  Slices at z=mid-plane, crops to near+wake region, saves point data.
  → fields/sim_XXX.npy  [150, N_pts, 3] float32

Mesh saved once:
  → mesh_points.npy    [N_pts, 2]
  → mesh_triangles.npy [N_tri, 3]

Usage:
    python extract_data.py --metadata metadata_merged.csv --sim-base /scratch_local/$USER --only-timeseries
    python extract_data.py --metadata metadata_merged.csv --sim-base /scratch_local/$USER --only sim_B1n_000_train --only-timeseries
    python extract_data.py --metadata metadata_merged.csv --sim-dir /scratch_local/$USER/sim_A_000_train \
        --only sim_A_000_train --fields --reconstruct
"""

import argparse, csv, re, subprocess, sys
import numpy as np
from pathlib import Path

# ─────────────────────── Fixed parameters ────────────────────────────────────

T_SIM   = 3.0
DT_SAVE = 0.002
N_SAVE  = int(round(T_SIM / DT_SAVE))  # 1500
U_INF   = 80.0; RHO = 1.225; AREF = 0.25
M_WING  = 19.24; I_WING = 1.155
K_H = 25460.0; D_H = 88.0; K_ALPHA = 9530.0; D_ALPHA = 6.6
Q_INF = 0.5 * RHO * U_INF**2

CROP_XMIN, CROP_XMAX = -1.0, 4.0
CROP_YMIN, CROP_YMAX = -1.0, 1.0

# ─────────────────────── Force reader ────────────────────────────────────────

def read_forces_all(sim_dir):
    forces_base = sim_dir / "postProcessing" / "forces"
    if not forces_base.exists():
        raise FileNotFoundError(f"No postProcessing/forces/ in {sim_dir}")
    dirs = []
    for d in forces_base.iterdir():
        try: float(d.name); dirs.append(d)
        except ValueError: pass
    dirs.sort(key=lambda p: float(p.name))
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
                if count <= 10: continue
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                if len(nums) < 19: continue
                t_all.append(float(nums[0]))
                Fy_all.append(float(nums[2]) + float(nums[5]))
                Mz_all.append(float(nums[12]) + float(nums[15]))
    t = np.array(t_all); Fy = np.array(Fy_all); Mz = np.array(Mz_all)
    idx = np.argsort(t); t, Fy, Mz = t[idx], Fy[idx], Mz[idx]
    _, ui = np.unique(t, return_index=True)
    return t[ui], Fy[ui], Mz[ui]

# ─────────────────────── Structural replay ───────────────────────────────────

def replay_structural(t_save, Fy_save, Mz_save):
    n = len(t_save)
    h = np.zeros(n); hd = np.zeros(n); a = np.zeros(n); ad = np.zeros(n)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    def deriv(s, Fy, Mz):
        return np.array([s[1], (-D_H*s[1] - K_H*s[0] - Fy)/M_WING,
                         s[3], (-D_ALPHA*s[3] - K_ALPHA*s[2] + Mz)/I_WING])
    h[0], hd[0], a[0], ad[0] = state
    for i in range(1, n):
        dt = t_save[i] - t_save[i-1]
        F0, M0 = Fy_save[i-1], Mz_save[i-1]
        F1, M1 = Fy_save[i], Mz_save[i]
        Fm, Mm = 0.5*(F0+F1), 0.5*(M0+M1)
        k1 = deriv(state, F0, M0)
        k2 = deriv(state + 0.5*dt*k1, Fm, Mm)
        k3 = deriv(state + 0.5*dt*k2, Fm, Mm)
        k4 = deriv(state + dt*k3, F1, M1)
        state += (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        h[i], hd[i], a[i], ad[i] = state
    return h, hd, a, ad

# ─────────────────────── Gust & flap models ──────────────────────────────────

def compute_gust(t, R, T_g):
    W_g0 = R * U_INF; W_g = np.zeros_like(t)
    mask = (t >= 0) & (t <= T_g)
    W_g[mask] = (W_g0/2) * (1 - np.cos(2*np.pi * t[mask] / T_g))
    return W_g

def _p(row, key, default=0.0):
    """Read a float param from a metadata row; returns default for missing/NaN values.
    Handles the sparse metadata layout where unused law params are filled with NaN."""
    v = row.get(key, "")
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except (ValueError, TypeError):
        return default

def compute_flap(t, row):
    law = int(row["law"])
    if law == 0: return np.zeros_like(t)
    dm = _p(row, "delta_max"); ts = _p(row, "t_start_delta")
    if law == 1:
        # Ramp + hold (positive or negative delta_max)
        dr = _p(row, "dt_ramp")
        times = [0, ts, ts+dr, T_SIM]; angles = [0, 0, dm, dm]
    elif law == 2:
        # Two-phase ramp (positive or negative delta_max)
        d1, d2 = _p(row, "dt_1"), _p(row, "dt_2")
        times = [0, ts, ts+d1, ts+d1+d2, T_SIM]; angles = [0, 0, dm/2, dm, dm]
    elif law == 3:
        # Trapezoid (positive or negative delta_max)
        du, dh, dd = _p(row, "dt_up"), _p(row, "dt_hold"), _p(row, "dt_down")
        t1=ts+du; t2=t1+dh; t3=t2+dd
        times = [0, ts, t1, t2, t3, T_SIM]; angles = [0, 0, dm, dm, 0, 0]
    elif law == 4:
        # Oscillating: ramp to +dm, hold, ramp to -dm, hold
        dr1=_p(row,"dt_ramp1"); dh1=_p(row,"dt_hold1")
        dr2=_p(row,"dt_ramp2"); dh2=_p(row,"dt_hold2")
        t1=ts+dr1; t2=t1+dh1; t3=t2+dr2; t4=t3+dh2
        times = [0, ts, t1, t2, t3, t4, T_SIM]
        angles = [0, 0, dm, dm, -dm, -dm, -dm]
    else: raise ValueError(f"Unknown law {law}")
    return np.interp(t, times, angles)

# ─────────────────────── Timeseries extraction ───────────────────────────────

def extract_timeseries(sim_dir, row, output_dir):
    sim_name = row["sim_name"]
    print(f"  [{sim_name}] Reading forces...")
    t_raw, Fy_raw, Mz_raw = read_forces_all(sim_dir)
    print(f"    {len(t_raw)} samples, t=[{t_raw[0]:.5f}, {t_raw[-1]:.5f}]")
    t_save = np.linspace(DT_SAVE, T_SIM, N_SAVE)
    Fy_save = np.interp(t_save, t_raw, Fy_raw, left=Fy_raw[0], right=Fy_raw[-1])
    Mz_save = np.interp(t_save, t_raw, Mz_raw, left=Mz_raw[0], right=Mz_raw[-1])
    print(f"    Replaying structural model...")
    h, hd, a, ad = replay_structural(t_save, Fy_save, Mz_save)
    R = float(row["R"]); T_g = float(row["T_g"])
    W_g = compute_gust(t_save, R, T_g); delta = compute_flap(t_save, row)
    Cl = Fy_save / (Q_INF * AREF); Cm = Mz_save / (Q_INF * AREF)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sim_name}.csv"
    header = "t,h,h_dot,alpha,alpha_dot,delta,W_g,C_L,C_M,F_y,M_z"
    data = np.column_stack([t_save, h, hd, a, ad, delta, W_g, Cl, Cm, Fy_save, Mz_save])
    np.savetxt(out_path, data, delimiter=",", header=header, comments="", fmt="%.8e")
    print(f"    Saved → {out_path}")
    print(f"    h: [{h.min()*1000:.1f}, {h.max()*1000:.1f}] mm, "
          f"α: [{np.degrees(a.min()):.2f}, {np.degrees(a.max()):.2f}]°, "
          f"δ: [{delta.min():.2f}, {delta.max():.2f}]°")

# ─────────────────────── Field extraction (OF mesh) ──────────────────────────

def extract_fields(sim_dir, row, output_base):
    """Extract U,p on OF mesh slice, cropped to near+wake."""
    import pyvista as pv
    sim_name = row["sim_name"]
    print(f"  [{sim_name}] Extracting fields...")

    foam_file = sim_dir / "extract.foam"
    created = not foam_file.exists()
    if created: foam_file.touch()

    try:
        reader = pv.OpenFOAMReader(str(foam_file))
        times = np.array(reader.time_values)
        # Skip t=0 (BC parsing issues)
        if times[0] == 0.0: times = times[1:]
        print(f"    {len(times)} field timesteps")

        # First timestep: get mesh topology + crop mask
        reader.set_active_time_value(times[0])
        mesh = reader.read()['internalMesh']
        slc = mesh.slice(normal='z', origin=(0, 0, 0.125))
        slc = slc.cell_data_to_point_data()
        surf = slc.triangulate()
        pts = surf.points[:, :2]

        crop = ((pts[:, 0] >= CROP_XMIN) & (pts[:, 0] <= CROP_XMAX) &
                (pts[:, 1] >= CROP_YMIN) & (pts[:, 1] <= CROP_YMAX))
        n_crop = crop.sum()

        # Save mesh (once)
        mesh_file = output_base / "mesh_points.npy"
        if not mesh_file.exists():
            old_to_new = np.full(len(pts), -1, dtype=int)
            old_to_new[crop] = np.arange(n_crop)
            faces = surf.faces.reshape(-1, 4)[:, 1:]
            tri_mask = np.all(crop[faces], axis=1)
            faces_crop = old_to_new[faces[tri_mask]]
            output_base.mkdir(parents=True, exist_ok=True)
            np.save(mesh_file, pts[crop].astype(np.float32))
            np.save(output_base / "mesh_triangles.npy", faces_crop.astype(np.int32))
            print(f"    Saved mesh: {n_crop} pts, {len(faces_crop)} tri")

        # Extract all timesteps
        fields = np.zeros((len(times), n_crop, 3), dtype=np.float32)
        for i, t_val in enumerate(times):
            if i % 20 == 0:
                print(f"    Snapshot {i}/{len(times)} (t={t_val:.4f}s)...")
            reader.set_active_time_value(t_val)
            mesh = reader.read()['internalMesh']
            slc = mesh.slice(normal='z', origin=(0, 0, 0.125))
            slc = slc.cell_data_to_point_data()
            surf = slc.triangulate()
            fields[i, :, 0] = surf.point_data["U"][crop, 0]
            fields[i, :, 1] = surf.point_data["U"][crop, 1]
            fields[i, :, 2] = surf.point_data["p"][crop]

        fields_dir = output_base / "fields"
        fields_dir.mkdir(parents=True, exist_ok=True)
        out_path = fields_dir / f"{sim_name}.npy"
        np.save(out_path, fields)
        print(f"    Saved → {out_path} ({fields.nbytes/1e6:.0f} MB, shape={fields.shape})")

        # Save field timestamps
        np.save(output_base / "field_times.npy", times.astype(np.float32))

    finally:
        if created: foam_file.unlink(missing_ok=True)

# ─────────────────────── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=str, default="metadata.csv")
    parser.add_argument("--sim-base", type=str, default="dataset")
    parser.add_argument("--sim-dir", type=str, default=None)
    parser.add_argument("--output-base", type=str, default="data/GLA")
    parser.add_argument("--only-timeseries", action="store_true")
    parser.add_argument("--fields", action="store_true")
    parser.add_argument("--reconstruct", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    output_base = Path(args.output_base)
    ts_dir = output_base / "timeseries"

    with open(args.metadata) as f:
        rows = list(csv.DictReader(f))
    if args.only:
        rows = [r for r in rows if r["sim_name"] == args.only]
        if not rows: print(f"ERROR: '{args.only}' not found"); sys.exit(1)

    print(f"\nExtracting {len(rows)} simulations...")
    if not args.only_timeseries and args.fields:
        print(f"  Fields → {output_base}/fields/")

    for row in rows:
        sim_name = row["sim_name"]
        sim_dir = Path(args.sim_dir) if args.sim_dir else Path(args.sim_base) / sim_name
        if not sim_dir.exists():
            print(f"  [{sim_name}] SKIP — not found"); continue
        if not (sim_dir / "postProcessing" / "forces").exists():
            print(f"  [{sim_name}] SKIP — no forces"); continue

        try: extract_timeseries(sim_dir, row, ts_dir)
        except Exception as e: print(f"  [{sim_name}] ERROR ts: {e}"); continue

        if args.fields and not args.only_timeseries:
            if args.reconstruct:
                print(f"  [{sim_name}] reconstructPar...")
                subprocess.run(["reconstructPar", "-case", str(sim_dir)],
                              capture_output=True, text=True)
            try: extract_fields(sim_dir, row, output_base)
            except Exception as e: print(f"  [{sim_name}] ERROR fields: {e}")

    import shutil
    shutil.copy2(args.metadata, output_base / "metadata.csv")
    print(f"\nDone. Output in {output_base}/")

if __name__ == "__main__":
    main()
