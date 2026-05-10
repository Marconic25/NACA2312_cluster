"""
free_vibration.py — Free vibration energy conservation and log-decrement check.

Two sub-tests:
  1. Undamped (D_H=D_ALPHA=0): total mechanical energy must be conserved.
  2. Damped (physical D_H, D_ALPHA): log-decrement of heave must match theory.

Monkey-patching cosim_driver module-level globals is safe because Python
function closures look up module globals at call time.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cosim_driver
from solver_adapter import (
    integrate_structural, M_mat, K_mat, C_mat,
    K_H, D_H, K_ALPHA, D_ALPHA,
    M_hh, M_aa, M_ha
)
import analytical_solution as anal
import plotting


# Initial conditions
H0    = 0.01   # m  (10 mm heave)
HD0   = 0.0
A0    = 0.0
AD0   = 0.0
# Start at t=0.5s: flap fully deployed (delta=15°, delta_dot=0, delta_ddot=0)
# so structural_rhs is equivalent to the frozen-flap analytical model
T_START = 0.5   # [s]
DT    = 1e-4   # time step [s]
ENERGY_TOL = 1e-3   # 0.1% relative energy drift


def _mechanical_energy(h_arr, hd_arr, alpha_arr, ad_arr):
    """Total mechanical energy: KE + PE."""
    KE = 0.5 * (M_hh * hd_arr**2
                + 2 * M_ha * hd_arr * ad_arr
                + M_aa * ad_arr**2)
    PE = 0.5 * (K_H * h_arr**2 + K_ALPHA * alpha_arr**2)
    return KE + PE


def _log_decrement_from_signal(t, sig):
    """Estimate log-decrement from successive peaks of a time series."""
    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(sig)
    if len(peaks) < 3:
        return np.nan
    # Use first and last peak
    n_periods = len(peaks) - 1
    delta = np.log(sig[peaks[0]] / sig[peaks[-1]]) / n_periods
    return delta


def run():
    """
    Run free vibration validation.

    Returns
    -------
    passed : bool
    message : str
    """
    omega, _ = anal.undamped_modes()
    T1 = 2 * np.pi / omega[0]   # period of first (heave-dominated) mode

    # ── Sub-test 1: Undamped energy conservation ──────────────────────────
    print(f"  Mode frequencies: ω1={omega[0]:.3f} rad/s, ω2={omega[1]:.3f} rad/s")
    print(f"  Mode periods:     T1={T1:.4f} s, T2={2*np.pi/omega[1]:.4f} s")

    n_periods = 5
    T_total   = n_periods * T1
    # Use t_start=T_START where flap is stationary (delta_dot=delta_ddot=0)
    t_win     = np.arange(T_START, T_START + T_total + DT * 0.5, DT)
    n         = len(t_win)
    Fy_zero   = np.zeros(n)
    Mz_zero   = np.zeros(n)

    # Temporarily zero damping
    orig_DH    = cosim_driver.D_H
    orig_DA    = cosim_driver.D_ALPHA
    cosim_driver.D_H    = 0.0
    cosim_driver.D_ALPHA = 0.0
    print("  Running undamped free vibration (D_H=D_ALPHA=0, t0=0.5s)...")
    try:
        _, _, _, _, h_u, alpha_u = integrate_structural(
            H0, HD0, A0, AD0, t_win, Fy_zero, Mz_zero
        )
        # Velocities from analytical undamped free response (exact), shift tau=t-T_START
        C_zero = np.zeros((2, 2))
        tau = t_win - T_START
        _, qd_u = anal.free_response([H0, A0], [HD0, AD0], tau,
                                      M=M_mat, K=K_mat, C=C_zero)
        hd_u = qd_u[0]
        ad_u = qd_u[1]
    finally:
        cosim_driver.D_H    = orig_DH
        cosim_driver.D_ALPHA = orig_DA

    E_u = _mechanical_energy(h_u, hd_u, alpha_u, ad_u)
    max_drift = np.max(np.abs(E_u - E_u[0])) / E_u[0]
    print(f"  Undamped max energy drift: {max_drift:.2e}  (tol={ENERGY_TOL:.2e})")
    passed_energy = max_drift < ENERGY_TOL

    # ── Sub-test 2: Damped log-decrement ──────────────────────────────────
    # Theoretical damping ratios (uncoupled approximation)
    zeta_h = D_H  / (2.0 * np.sqrt(K_H     * M_hh))
    zeta_a = D_ALPHA / (2.0 * np.sqrt(K_ALPHA * M_aa))
    delta_h_theory = 2 * np.pi * zeta_h / np.sqrt(1 - zeta_h**2)
    print(f"  Theoretical ζ_h={zeta_h:.4f}, δ_h={delta_h_theory:.4f}")

    T_damp  = 10 * T1
    t_damp  = np.arange(T_START, T_START + T_damp + DT * 0.5, DT)
    n_d     = len(t_damp)
    print("  Running damped free vibration...")
    _, _, _, _, h_d, alpha_d = integrate_structural(
        H0, HD0, A0, AD0, t_damp,
        np.zeros(n_d), np.zeros(n_d)
    )

    delta_h_num = _log_decrement_from_signal(t_damp, h_d)
    if not np.isnan(delta_h_num):
        err_delta = abs(delta_h_num - delta_h_theory) / abs(delta_h_theory)
        print(f"  Numerical log-decrement: δ_h={delta_h_num:.4f}, "
              f"theory={delta_h_theory:.4f}, rel_err={err_delta:.4f}")
        passed_logdec = err_delta < 0.05   # 5% tolerance (coupling shifts it slightly)
    else:
        err_delta = np.nan
        passed_logdec = False
        print("  WARNING: could not find peaks for log-decrement")

    # ── Figures ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t_win, h_u * 1e3, lw=1.2, color="navy")
    axes[0].set_ylabel("h [mm]")
    axes[0].set_title("Undamped free vibration")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_win, np.degrees(alpha_u), lw=1.2, color="darkorange")
    axes[1].set_ylabel("alpha [deg]")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(t_win, E_u / E_u[0], lw=1.2, color="green")
    axes[2].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[2].set_ylabel("E(t)/E(0)")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_title(f"Energy conservation — max drift = {max_drift:.2e}")
    axes[2].grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    plotting.save_fig(fig, "free_vibration_undamped")

    # Phase portrait (undamped)
    plotting.plot_phase_portrait(
        h_u * 1e3, np.gradient(h_u, t_win) * 1e3,
        "h [mm]", "h_dot [mm/s]",
        "Phase portrait (undamped, heave)",
        figname="free_vibration_phase"
    )

    # Damped response
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t_damp, h_d * 1e3, lw=1.2, color="steelblue")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("h [mm]")
    ax2.set_title(f"Damped free vibration — ζ_h={zeta_h:.4f}, "
                  f"δ_num={delta_h_num:.4f} vs δ_theory={delta_h_theory:.4f}")
    ax2.grid(True, alpha=0.3)
    plotting.save_fig(fig2, "free_vibration_damped")

    passed = passed_energy and passed_logdec
    msg = (
        f"energy_drift={max_drift:.2e} {'OK' if passed_energy else 'FAIL'}; "
        f"log_decrement_err={err_delta:.4f} {'OK' if passed_logdec else 'FAIL'}"
    )
    return passed, msg


if __name__ == "__main__":
    passed, msg = run()
    print(f"free_vibration: {'PASS' if passed else 'FAIL'} — {msg}")
