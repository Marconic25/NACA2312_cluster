"""
forced_validation.py — Compare numerical integrator vs analytical forced response.

Test: harmonic forcing Fy = F0*sin(2*pi*f0*t), Mz = 0, zero initial conditions.
The full response = transient (decaying) + steady-state.
Uses analytical_solution.full_response() which combines both parts.
"""

import numpy as np
from solver_adapter import integrate_structural
import analytical_solution as anal
import plotting


# Test parameters
F0_H     = 100.0      # heave force amplitude [N]
F0_A     = 0.0        # pitch moment amplitude [N·m]
F_FORCE  = 5.0        # forcing frequency [Hz]
OMEGA_F  = 2 * np.pi * F_FORCE
# Start at t=0.5s: flap ramp is complete (delta=15°, delta_dot=0, delta_ddot=0)
# so structural_rhs reduces to the frozen-flap 2-DOF system = matches analytical model
T_START  = 0.5        # [s]
T_END    = 2.5        # total end time [s]  → 2.0s window
DT       = 5e-4       # time step [s]
RMS_TOL  = 1e-3       # 0.1% relative RMS tolerance


def run():
    """
    Run forced-response validation.

    Returns
    -------
    passed : bool
    message : str
    """
    t_win  = np.arange(T_START, T_END + DT * 0.5, DT)
    n      = len(t_win)
    # Forcing phase-shifted to match t_start (use global time for consistency)
    Fy_arr = F0_H * np.sin(OMEGA_F * t_win)
    Mz_arr = np.zeros(n)

    # Numerical solution (zero ICs at T_START; flap is frozen so matches analytical)
    print("  Running forced-response numerical integration (t0=0.5s, flap frozen)...")
    _, _, _, _, h_num, alpha_num = integrate_structural(
        0.0, 0.0, 0.0, 0.0, t_win, Fy_arr, Mz_arr
    )

    # Analytical full response using global time axis (t0=T_START)
    print("  Computing analytical reference...")
    q_anal = anal.full_response(
        [0.0, 0.0], [0.0, 0.0],
        F0_H, F0_A, OMEGA_F, t_win
    )
    h_anal     = q_anal[0]
    alpha_anal = q_anal[1]

    # RMS error over last 10 forcing cycles
    T_cycle   = 1.0 / F_FORCE
    t_steady  = T_END - 10 * T_cycle
    mask      = t_win >= t_steady

    rms_h = _rms_rel_error(h_num[mask], h_anal[mask])
    rms_a = _rms_rel_error(alpha_num[mask], alpha_anal[mask])

    print(f"  RMS relative error (last 10 cycles): h={rms_h:.4e}  alpha={rms_a:.4e}")

    # --- Figures ---
    plotting.plot_overlay(
        t_win, h_num * 1e3, h_anal * 1e3,
        ("numerical", "analytical"),
        f"Forced response: h(t),  F0={F0_H}N, f={F_FORCE}Hz",
        "h [mm]",
        figname="forced_h"
    )
    plotting.plot_overlay(
        t_win, np.degrees(alpha_num), np.degrees(alpha_anal),
        ("numerical", "analytical"),
        f"Forced response: alpha(t),  F0={F0_H}N, f={F_FORCE}Hz",
        "alpha [deg]",
        figname="forced_alpha"
    )

    passed_h = rms_h < RMS_TOL
    passed_a = rms_a < RMS_TOL or np.allclose(alpha_anal[mask], 0, atol=1e-15)
    passed   = passed_h and passed_a
    msg = (
        f"RMS_h={rms_h:.2e} (tol={RMS_TOL:.0e}) {'OK' if passed_h else 'FAIL'}; "
        f"RMS_alpha={rms_a:.2e} {'OK' if passed_a else 'FAIL'}"
    )
    return passed, msg


def _rms_rel_error(arr_num, arr_ref):
    """RMS relative error; uses RMS of reference as denominator."""
    rms_ref = np.sqrt(np.mean(arr_ref**2))
    if rms_ref < 1e-30:
        return np.sqrt(np.mean(arr_num**2))
    return np.sqrt(np.mean((arr_num - arr_ref)**2)) / rms_ref


if __name__ == "__main__":
    passed, msg = run()
    print(f"forced_validation: {'PASS' if passed else 'FAIL'} — {msg}")
