"""
run_coupling_validation.py — Master script for FSI coupling convergence analysis.

Reads recorded aerodynamic forces from postProcessing/forces/ (648 windows,
t=0..1.293s) and replays the structural integrator forward to reconstruct the
full structural trajectory alongside the aerodynamic force history.  Then runs:

  1. Residuals study   — fixed-point convergence per window
  2. Energy balance    — interface power, cumulative work, per-window imbalance
  3. Loose vs strong   — compare single-pass vs iterated coupling

Usage
-----
    cd wingMotion2D_pimpleFoam
    python validation_fsi/run_coupling_validation.py
    python validation_fsi/run_coupling_validation.py --window-range 0 100
    python validation_fsi/run_coupling_validation.py --skip-residuals
"""

import sys
import re
import argparse
from pathlib import Path

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent   # wingMotion2D_pimpleFoam/
sys.path.insert(0, str(ROOT_DIR))

from cosim_driver import integrate_structural

import residuals    as res_mod
import energy_balance as eb_mod
import loose_vs_strong as lvs_mod


# ── Constants ────────────────────────────────────────────────────────────────
PP_FORCES = ROOT_DIR / "postProcessing" / "forces"


# ── Force loading ─────────────────────────────────────────────────────────────

def _parse_forces_file(path):
    """
    Parse one forces.dat file.  Returns arrays t, Fy, Mz.
    Replicates the logic in cosim_driver.read_forces.
    """
    t_list, Fy_list, Mz_list = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if len(nums) < 19:
                continue
            t_list.append(float(nums[0]))
            Fy_list.append(float(nums[2]) + float(nums[5]))    # Fpy + Fvy
            Mz_list.append(float(nums[12]) + float(nums[15]))  # Mpz + Mvz
    if not t_list:
        return None, None, None
    idx = np.argsort(t_list)
    return (np.array(t_list)[idx],
            np.array(Fy_list)[idx],
            np.array(Mz_list)[idx])


def load_all_forces(pp_forces: Path):
    """
    Scan all postProcessing/forces/<T>/forces.dat and return a dict:
        { t_window_start: (t_arr, Fy_arr, Mz_arr) }
    sorted by window start time.
    """
    force_files = sorted(pp_forces.glob("*/forces.dat"))
    if not force_files:
        raise FileNotFoundError(f"No forces.dat files under {pp_forces}")

    windows = {}
    for fpath in force_files:
        t_win_start = float(fpath.parent.name)
        t_arr, Fy_arr, Mz_arr = _parse_forces_file(fpath)
        if t_arr is not None and len(t_arr) > 0:
            windows[t_win_start] = (t_arr, Fy_arr, Mz_arr)

    # Sort by window start time
    sorted_keys = sorted(windows.keys())
    print(f"  Loaded {len(sorted_keys)} force windows from {pp_forces}")
    return sorted_keys, windows


# ── Trajectory replay ─────────────────────────────────────────────────────────

def replay_trajectory(sorted_keys, windows, win_range=None):
    """
    Replay the structural integrator forward from t=0 using recorded forces.

    This reconstructs the full h(t), alpha(t) trajectory consistent with the
    force history actually seen by the cosim.

    Parameters
    ----------
    sorted_keys : list of float — window start times
    windows     : dict  { t_start: (t_arr, Fy_arr, Mz_arr) }
    win_range   : (i_start, i_end) or None  — window index range to process

    Returns
    -------
    t_windows   : list of ndarray  — time grid per window
    Fy_traj     : list of ndarray  — Fy per window (on t_windows grid)
    Mz_traj     : list of ndarray  — Mz per window
    h_ic        : list of float    — heave IC per window
    hd_ic       : list of float
    a_ic        : list of float
    ad_ic       : list of float
    h_full      : ndarray — full heave trajectory
    alpha_full  : ndarray — full pitch trajectory
    """
    if win_range is not None:
        i0, i1 = win_range
        keys = sorted_keys[i0:i1]
    else:
        keys = sorted_keys

    # ICs: start at rest
    h0, hd0, a0, ad0 = 0.0, 0.0, 0.0, 0.0

    t_windows  = []
    Fy_traj    = []
    Mz_traj    = []
    h_ic_list  = []
    hd_ic_list = []
    a_ic_list  = []
    ad_ic_list = []
    h_parts    = []
    alpha_parts = []

    from scipy.interpolate import interp1d

    for t_start in keys:
        t_raw, Fy_raw, Mz_raw = windows[t_start]

        # Build uniform time grid for this window
        # Deduplicate/sort t_win (OF may write duplicate timestamps at boundaries)
        t_raw_s, unique_idx = np.unique(t_raw, return_index=True)
        if len(t_raw_s) < 2:
            # Skip degenerate windows
            continue
        t_win  = t_raw_s
        Fy_arr = Fy_raw[unique_idx]
        Mz_arr = Mz_raw[unique_idx]

        # Record ICs
        h_ic_list.append(h0)
        hd_ic_list.append(hd0)
        a_ic_list.append(a0)
        ad_ic_list.append(ad0)

        # Integrate
        h_f, hd_f, a_f, ad_f, h_arr, alpha_arr = integrate_structural(
            h0, hd0, a0, ad0, t_win, Fy_arr, Mz_arr
        )

        # Trim to actual integrator output length (solve_ivp may return < len(t_eval))
        n_out = len(h_arr)
        t_windows.append(t_win[:n_out])
        Fy_traj.append(Fy_arr[:n_out])
        Mz_traj.append(Mz_arr[:n_out])
        h_parts.append(h_arr)
        alpha_parts.append(alpha_arr)

        # Advance ICs for next window
        h0, hd0, a0, ad0 = h_f, hd_f, a_f, ad_f

    h_full    = np.concatenate(h_parts)
    alpha_full = np.concatenate(alpha_parts)

    print(f"  Replayed {len(keys)} windows: "
          f"t=[{float(keys[0]):.3f}, {float(keys[-1]):.3f}]s  "
          f"N_total={len(h_full)}")
    return (t_windows, Fy_traj, Mz_traj,
            h_ic_list, hd_ic_list, a_ic_list, ad_ic_list,
            h_full, alpha_full)


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(res_results, eb_summary, lv_metrics, lv_iters):
    sep = "─" * 60
    print()
    print(sep)
    print("  FSI COUPLING VALIDATION SUMMARY")
    print(sep)

    if res_results is not None:
        N = len(res_results)
        rho_all = [r["rho"] for r in res_results if not np.isnan(r["rho"])]
        n_conv  = sum(r["converged"] for r in res_results)
        print(f"  Residual study : {n_conv}/{N} windows converged")
        if rho_all:
            print(f"  Spectral radius: median={np.median(rho_all):.4f}  "
                  f"max={max(rho_all):.4f}")

    if eb_summary is not None:
        print(f"  W_fluid total  : {eb_summary['W_fluid_total']:.4e} J")
        print(f"  ΔE_structural  : {eb_summary['delta_E_total']:.4e} J")
        print(f"  W_damping      : {eb_summary['W_damp_total']:.4e} J")
        print(f"  Global imbalance: {eb_summary['imb_global']*100:.3f}%")

    if lv_metrics is not None:
        print(f"  Loose vs strong — L2_h={lv_metrics['l2_h']:.3e}  "
              f"L2_alpha={lv_metrics['l2_alpha']:.3e}  "
              f"phase_h={lv_metrics['phase_h']:.4f}s  "
              f"amp_h={lv_metrics['amp_h']:.3e}")
        if lv_iters:
            print(f"  Strong coupling avg iters: {np.mean(lv_iters):.2f}  "
                  f"max: {max(lv_iters)}")

    print(f"  Figures saved to: validation_fsi/figures/")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FSI coupling convergence validation"
    )
    parser.add_argument("--window-range", nargs=2, type=int, default=None,
                        metavar=("I_START", "I_END"),
                        help="Process windows [I_START, I_END) only")
    parser.add_argument("--skip-residuals", action="store_true",
                        help="Skip the (slow) fixed-point residual study")
    parser.add_argument("--skip-energy", action="store_true",
                        help="Skip energy balance analysis")
    parser.add_argument("--skip-lv", action="store_true",
                        help="Skip loose-vs-strong comparison")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Fixed-point convergence tolerance (default: 1e-6)")
    parser.add_argument("--max-iter", type=int, default=20,
                        help="Max fixed-point iterations per window (default: 20)")
    parser.add_argument("--sample-windows", nargs="+", type=int, default=None,
                        help="Window indices to highlight in residual plot")
    args = parser.parse_args()

    print("=" * 60)
    print("  FSI Coupling Convergence Validation")
    print("=" * 60)

    # ── 1. Load forces ─────────────────────────────────────────────────────
    print("\n[1/4] Loading force history from postProcessing/forces/ ...")
    sorted_keys, windows = load_all_forces(PP_FORCES)

    # ── 2. Replay trajectory ───────────────────────────────────────────────
    print("\n[2/4] Replaying structural trajectory ...")
    win_range = tuple(args.window_range) if args.window_range else None
    (t_windows, Fy_traj, Mz_traj,
     h_ic, hd_ic, a_ic, ad_ic,
     h_full, alpha_full) = replay_trajectory(sorted_keys, windows, win_range)

    N_win = len(t_windows)

    # ── 3. Residual study ──────────────────────────────────────────────────
    res_results = None
    if not args.skip_residuals:
        print(f"\n[3a/4] Residual study ({N_win} windows, tol={args.tol:.0e}) ...")
        res_results = res_mod.run_residual_study(
            h_ic, hd_ic, a_ic, ad_ic,
            Fy_traj, Mz_traj, t_windows,
            tol=args.tol, max_iter=args.max_iter,
            sample_windows=args.sample_windows
        )
    else:
        print("\n[3a/4] Residual study — SKIPPED")

    # ── 4. Energy balance ──────────────────────────────────────────────────
    eb_summary = None
    if not args.skip_energy:
        print("\n[3b/4] Energy balance analysis ...")
        eb_summary = eb_mod.run_energy_analysis(
            h_full, alpha_full,
            Fy_traj, Mz_traj, t_windows
        )
    else:
        print("\n[3b/4] Energy balance — SKIPPED")

    # ── 5. Loose vs strong ─────────────────────────────────────────────────
    lv_metrics = None
    lv_iters   = None
    if not args.skip_lv:
        print(f"\n[4/4] Loose vs strong coupling comparison ...")
        lv_metrics, lv_iters = lvs_mod.run(
            h_ic, hd_ic, a_ic, ad_ic,
            Fy_traj, Mz_traj, t_windows,
            tol=args.tol, max_iter=args.max_iter
        )
    else:
        print("\n[4/4] Loose vs strong — SKIPPED")

    # ── Summary ────────────────────────────────────────────────────────────
    _print_summary(res_results, eb_summary, lv_metrics, lv_iters)


if __name__ == "__main__":
    main()
