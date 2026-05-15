#!/usr/bin/env python3
"""
extract_data.py — Extract timeseries + fields from completed cosim runs.

Timeseries (1500 samples @ dt=0.002s):
  Primary source: structural_trajectory.csv written by cosim_driver (h, hd, alpha, ad,
  Fy, Mz already integrated with correct augmented mass matrix and flap inertia).
  Fallback: replay structural model from forces.dat if trajectory file missing.
  Analytical gust and flap schedule appended from metadata.
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
# Must match cosim_driver.py exactly (Hodges-Pierce benchmark parameters).

T_SIM   = 3.0
DT_SAVE = 0.002
N_SAVE  = int(round(T_SIM / DT_SAVE))  # 1500
U_INF   = 80.0
RHO     = 1.225
AREF    = 0.25   # reference area [m²] (chord × span = 1.0 × 0.25)
Q_INF   = 0.5 * RHO * U_INF**2

# Structural parameters — Hodges-Pierce benchmark (same as cosim_driver.py)
# Augmented mass accounts for flap inertia (M_WING + M_FLAP, I_WING + I_FLAP_EA).
M_WING  = 19.24 + 1.19   # augmented wing+flap mass [kg]
I_WING  = 1.155 + 0.006 + 1.19 * (0.525**2 + 0.045**2)  # augmented MoI [kg·m²]
K_H     = 25460.0
D_H     = 88.0
K_ALPHA = 9530.0
D_ALPHA = 6.6

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

# ─────────────────────── Structural replay (fallback) ────────────────────────

def replay_structural(t_save, Fy_save, Mz_save):
    """
    Fallback: integrate 2-DOF EOM from forces when structural_trajectory.csv
    is not available. Uses augmented mass matrix matching cosim_driver.py.
    Simple decoupled approximation (off-diagonal coupling neglected).
    """
    n = len(t_save)
    h = np.zeros(n); hd = np.zeros(n); a = np.zeros(n); ad = np.zeros(n)
    state = np.array([0.0, 0.0, 0.0, 0.0])

    def deriv(s, Fy, Mz):
        h_ddot = (-D_H * s[1] - K_H * s[0] - Fy) / M_WING
        a_ddot = (-D_ALPHA * s[3] - K_ALPHA * s[2] + Mz) / I_WING
        return np.array([s[1], h_ddot, s[3], a_ddot])

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
    W_g0 = R * U_INF
    W_g = np.zeros_like(t)
    mask = (t >= 0) & (t <= T_g)
    W_g[mask] = (W_g0 / 2) * (1 - np.cos(2 * np.pi * t[mask] / T_g))
    return W_g


def _p(row, key, default=0.0):
    """Read float param from metadata row; returns default for missing/NaN."""
    v = row.get(key, "")
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except (ValueError, TypeError):
        return default


def compute_flap(t, row):
    law = int(row["law"])
    if law == 0:
        return np.zeros_like(t)
    dm = _p(row, "delta_max")
    ts = _p(row, "t_start_delta")
    if law == 1:
        dr = _p(row, "dt_ramp")
        times = [0, ts, ts+dr, T_SIM]; angles = [0, 0, dm, dm]
    elif law == 2:
        d1, d2 = _p(row, "dt_1"), _p(row, "dt_2")
        times = [0, ts, ts+d1, ts+d1+d2, T_SIM]; angles = [0, 0, dm/2, dm, dm]
    elif law == 3:
        du, dh, dd = _p(row, "dt_up"), _p(row, "dt_hold"), _p(row, "dt_down")
        t1 = ts+du; t2 = t1+dh; t3 = t2+dd
        times = [0, ts, t1, t2, t3, T_SIM]; angles = [0, 0, dm, dm, 0, 0]
    elif law == 4:
        dr1 = _p(row, "dt_ramp1"); dh1 = _p(row, "dt_hold1")
        dr2 = _p(row, "dt_ramp2"); dh2 = _p(row, "dt_hold2")
        t1 = ts+dr1; t2 = t1+dh1; t3 = t2+dr2; t4 = t3+dh2
        times = [0, ts, t1, t2, t3, t4, T_SIM]
        angles = [0, 0, dm, dm, -dm, -dm, -dm]
    else:
        raise ValueError(f"Unknown law {law}")
    return np.interp(t, times, angles)

# ─────────────────────── Timeseries extraction ───────────────────────────────

def extract_timeseries(sim_dir, row, output_dir):
    sim_name = row["sim_name"]
    t_save = np.linspace(DT_SAVE, T_SIM, N_SAVE)

    # ── Primary: read structural_trajectory.csv from cosim_driver ──
    traj_file = sim_dir / "structural_trajectory.csv"
    if traj_file.exists():
        print(f"  [{sim_name}] Reading structural_trajectory.csv...")
        traj = np.loadtxt(traj_file, delimiter=",", skiprows=1)
        # columns: t, h, hd, alpha, ad, Fy, Mz
        t_traj = traj[:, 0]
        h_traj  = traj[:, 1]; hd_traj = traj[:, 2]
        a_traj  = traj[:, 3]; ad_traj = traj[:, 4]
        Fy_traj = traj[:, 5]; Mz_traj = traj[:, 6]
        print(f"    {len(t_traj)} samples, t=[{t_traj[0]:.5f}, {t_traj[-1]:.5f}]")

        # Interpolate onto uniform t_save grid
        kw = dict(left=None, right=None)  # extrapolate at edges using boundary value
        h   = np.interp(t_save, t_traj, h_traj,  left=h_traj[0],  right=h_traj[-1])
        hd  = np.interp(t_save, t_traj, hd_traj, left=hd_traj[0], right=hd_traj[-1])
        a   = np.interp(t_save, t_traj, a_traj,  left=a_traj[0],  right=a_traj[-1])
        ad  = np.interp(t_save, t_traj, ad_traj, left=ad_traj[0], right=ad_traj[-1])
        Fy  = np.interp(t_save, t_traj, Fy_traj, left=Fy_traj[0], right=Fy_traj[-1])
        Mz  = np.interp(t_save, t_traj, Mz_traj, left=Mz_traj[0], right=Mz_traj[-1])

    # ── Fallback: replay from forces.dat ──
    else:
        print(f"  [{sim_name}] structural_trajectory.csv not found — replaying from forces...")
        t_raw, Fy_raw, Mz_raw = read_forces_all(sim_dir)
        print(f"    {len(t_raw)} force samples, t=[{t_raw[0]:.5f}, {t_raw[-1]:.5f}]")
        Fy = np.interp(t_save, t_raw, Fy_raw, left=Fy_raw[0], right=Fy_raw[-1])
        Mz = np.interp(t_save, t_raw, Mz_raw, left=Mz_raw[0], right=Mz_raw[-1])
        print(f"    Replaying structural model (fallback, decoupled approximation)...")
        h, hd, a, ad = replay_structural(t_save, Fy, Mz)

    # ── Analytical inputs: gust and flap ──
    R = float(row["R"]); T_g = float(row["T_g"])
    W_g   = compute_gust(t_save, R, T_g)
    delta = compute_flap(t_save, row)

    # ── Aerodynamic coefficients ──
    CL = Fy / (Q_INF * AREF)
    CM = Mz / (Q_INF * AREF)

    # ── Convert angles to degrees (uniform units in CSV) ──
    a_deg  = np.degrees(a)
    ad_deg = np.degrees(ad)   # rad/s → deg/s

    # ── Save CSV ──
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sim_name}.csv"
    header = "t,h,h_dot,alpha,alpha_dot,delta,W_g,C_L,C_M,F_y,M_z"
    data = np.column_stack([t_save, h, hd, a_deg, ad_deg, delta, W_g, CL, CM, Fy, Mz])
    np.savetxt(out_path, data, delimiter=",", header=header, comments="", fmt="%.8e")
    print(f"    Saved → {out_path}")
    print(f"    h: [{h.min()*1000:.1f}, {h.max()*1000:.1f}] mm, "
          f"α: [{a_deg.min():.2f}, {a_deg.max():.2f}]°, "
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
        if times[0] == 0.0: times = times[1:]
        print(f"    {len(times)} field timesteps")

        reader.set_active_time_value(times[0])
        mesh = reader.read()['internalMesh']
        slc = mesh.slice(normal='z', origin=(0, 0, 0.125))
        slc = slc.cell_data_to_point_data()
        surf = slc.triangulate()
        pts = surf.points[:, :2]

        crop = ((pts[:, 0] >= CROP_XMIN) & (pts[:, 0] <= CROP_XMAX) &
                (pts[:, 1] >= CROP_YMIN) & (pts[:, 1] <= CROP_YMAX))
        n_crop = crop.sum()

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

        np.save(output_base / "field_times.npy", times.astype(np.float32))

    finally:
        if created: foam_file.unlink(missing_ok=True)

# ─────────────────────── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=str, default="metadata_merged.csv")
    parser.add_argument("--sim-base", type=str, default="dataset")
    parser.add_argument("--sim-dir", type=str, default=None)
    parser.add_argument("--output-base", type=str, default="data/GLA")
    parser.add_argument("--only-timeseries", action="store_true")
    parser.add_argument("--fields", action="store_true")
    parser.add_argument("--reconstruct", action="store_true")
    parser.add_argument("--only", type=str, default=None,
                        help="Process only this sim_name")
    parser.add_argument("--family", type=str, default=None,
                        help="Comma-separated family filter, e.g. A,B1")
    args = parser.parse_args()

    output_base = Path(args.output_base)
    ts_dir = output_base / "timeseries"

    with open(args.metadata) as f:
        rows = list(csv.DictReader(f))

    if args.only:
        rows = [r for r in rows if r["sim_name"] == args.only]
        if not rows:
            print(f"ERROR: '{args.only}' not found in {args.metadata}")
            sys.exit(1)
    elif args.family:
        families = {fam.strip() for fam in args.family.split(",")}
        rows = [r for r in rows if r["family"] in families]
        if not rows:
            print(f"ERROR: no simulations found for families {families}")
            sys.exit(1)

    print(f"\nExtracting {len(rows)} simulations → {output_base}/")
    if not args.only_timeseries and args.fields:
        print(f"  Fields → {output_base}/fields/")

    ok = 0; skipped = 0; errors = 0
    for row in rows:
        sim_name = row["sim_name"]
        sim_dir = Path(args.sim_dir) if args.sim_dir else Path(args.sim_base) / sim_name
        if not sim_dir.exists():
            print(f"  [{sim_name}] SKIP — directory not found: {sim_dir}")
            skipped += 1; continue

        has_traj = (sim_dir / "structural_trajectory.csv").exists()
        has_forces = (sim_dir / "postProcessing" / "forces").exists()
        if not has_traj and not has_forces:
            print(f"  [{sim_name}] SKIP — no structural_trajectory.csv nor forces/")
            skipped += 1; continue

        try:
            extract_timeseries(sim_dir, row, ts_dir)
            ok += 1
        except Exception as e:
            print(f"  [{sim_name}] ERROR timeseries: {e}")
            errors += 1; continue

        if args.fields and not args.only_timeseries:
            if args.reconstruct:
                print(f"  [{sim_name}] reconstructPar...")
                subprocess.run(["reconstructPar", "-case", str(sim_dir)],
                               capture_output=True, text=True)
            try:
                extract_fields(sim_dir, row, output_base)
            except Exception as e:
                print(f"  [{sim_name}] ERROR fields: {e}")

    import shutil
    output_base.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.metadata, output_base / "metadata.csv")

    print(f"\nDone. Output in {output_base}/")
    print(f"  Extracted: {ok}  Skipped: {skipped}  Errors: {errors}")


if __name__ == "__main__":
    main()
