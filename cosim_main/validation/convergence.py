"""
convergence.py — Timestep convergence study for structural integrator.

Uses same harmonic forcing as forced_validation.py.
Reference solution: very tight tolerances (rtol=1e-12, atol=1e-14).
Test: L2-norm of error in h(t) vs dt on a log-log plot.
Expected slope ≈ 4 for RK45 (4th-order accurate in step size).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Patch path so we can call structural_rhs directly
sys.path.insert(0, str(Path(__file__).parent.parent))
import cosim_driver
from cosim_driver import structural_rhs

import analytical_solution as anal
import plotting


# Forcing parameters (match forced_validation.py)
F0_H    = 100.0
OMEGA_F = 2 * np.pi * 5.0    # 5 Hz
# Start at t=0.5s (flap stationary): structural_rhs = frozen-flap 2-DOF = matches analytical
T_START = 0.5
T_END   = T_START + 0.5      # 0.5s window starting at 0.5s

# Timesteps to sweep [s]
DT_LIST = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]

SLOPE_TOL = 1.5   # minimum acceptable convergence order


def _run_with_max_step(t_win, Fy_arr, Mz_arr, max_step, rtol=1e-8, atol=1e-10):
    """Run integrator with a given max_step, return h_arr at t_win points."""
    Fy_interp = interp1d(t_win, Fy_arr, kind="linear", fill_value="extrapolate")
    Mz_interp = interp1d(t_win, Mz_arr, kind="linear", fill_value="extrapolate")
    sol = solve_ivp(
        structural_rhs,
        [t_win[0], t_win[-1]],
        [0.0, 0.0, 0.0, 0.0],
        args=(Fy_interp, Mz_interp),
        t_eval=t_win,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )
    return sol.y[0], sol.y[2]   # h, alpha


def run():
    """
    Run convergence study.

    Returns
    -------
    passed : bool
    message : str
    """
    # Use analytical full_response as exact reference (no discretisation error)
    dt_fine = 2e-4
    t_ref   = np.arange(T_START, T_END + dt_fine * 0.5, dt_fine)
    print("  Computing analytical reference (full_response, t0=0.5s)...")
    q_ref = anal.full_response([0.0, 0.0], [0.0, 0.0], F0_H, 0.0, OMEGA_F, t_ref)
    h_ref = q_ref[0]
    h_ref_interp = interp1d(t_ref, h_ref, kind="cubic", fill_value="extrapolate")

    errors = []
    print(f"  {'dt':>10s}  {'L2_error_h':>14s}")
    for dt in DT_LIST:
        t_win  = np.arange(T_START, T_END + dt * 0.5, dt)
        Fy_arr = F0_H * np.sin(OMEGA_F * t_win)
        Mz_arr = np.zeros_like(t_win)
        h_num, _ = _run_with_max_step(t_win, Fy_arr, Mz_arr, max_step=dt)
        h_ref_on_grid = h_ref_interp(t_win)
        err = np.sqrt(np.mean((h_num - h_ref_on_grid)**2))
        errors.append(err)
        print(f"  {dt:>10.1e}  {err:>14.6e}")

    dt_arr  = np.array(DT_LIST)
    err_arr = np.array(errors)

    # Fit slope on log-log
    log_dt  = np.log10(dt_arr)
    log_err = np.log10(err_arr)
    slope, _ = np.polyfit(log_dt, log_err, 1)
    print(f"  Convergence slope: {slope:.3f}  (required > {SLOPE_TOL})")

    plotting.plot_loglog(
        dt_arr, err_arr, slope,
        xlabel="dt [s]",
        ylabel="L2 error in h(t)",
        title=f"Timestep convergence — slope={slope:.2f} (RK45 expected ≈ 4)",
        figname="convergence"
    )

    passed = slope > SLOPE_TOL
    msg = f"slope={slope:.3f} (min={SLOPE_TOL}) {'OK' if passed else 'FAIL'}"
    return passed, msg


if __name__ == "__main__":
    passed, msg = run()
    print(f"convergence: {'PASS' if passed else 'FAIL'} — {msg}")
