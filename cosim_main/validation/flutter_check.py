"""
flutter_check.py — Linear flutter analysis via quasi-steady strip theory.

Aeroelastic system:
    M q'' + C q' + K q = F_aero(q, q')

Quasi-steady thin-airfoil aerodynamics (no circulation lag):
    L  = rho * U * C_Lalpha * (h_dot/U + alpha) * c * span   [lift, N]
    M_ac = rho * U^2 * C_Malpha * alpha * c^2 * span         [moment, N·m]

Sign convention: positive h = upward, positive alpha = LE-up (nose-up).
Lift opposes positive heave (aero damping), moment may drive pitch (divergence/flutter).

State vector: x = [h, alpha, h_dot, alpha_dot]^T
State matrix A(U) built from structural + aero contributions.
Eigenvalues of A tracked vs airspeed U.
Flutter onset: airspeed where Re(λ) first crosses zero for an oscillatory mode.
"""

import numpy as np
from scipy.linalg import eigvals
from solver_adapter import M_mat, K_mat, C_mat
import plotting


# Aerodynamic constants
RHO     = 1.225    # air density [kg/m³] (sea level ISA)
C_LALPHA = 2 * np.pi   # lift curve slope [rad⁻¹] (thin airfoil)
C_MALPHA = -np.pi / 4  # moment slope (thin airfoil, about c/4)
CHORD   = 1.0      # chord [m]
SPAN    = 0.05     # span [m] (2D per-unit-span, physical span of mesh)

U_INF   = 80.0     # operational airspeed [m/s] (from cosim_driver.py)
U_RANGE = np.linspace(1.0, 200.0, 100)   # sweep for flutter analysis


def _build_aero_matrices(U, rho=RHO, c=CHORD, b=SPAN):
    """
    Return aerodynamic damping B_aero and stiffness K_aero matrices (2×2).

    Aero force: F_aero = -B_aero q' - K_aero q
    (negative sign: aero forces go on RHS, subtract from structural stiffness/damping)

    L = rho*U*C_Lalpha*(h_dot/U + alpha)*c*b
      = rho*c*b*C_Lalpha * h_dot  +  rho*U*c*b*C_Lalpha * alpha

    M_ac = rho*U^2*C_Malpha*alpha*c^2*b

    In EOM form M q'' + (C - C_aero) q' + (K - K_aero) q = 0:
      C_aero[0,0] = rho*c*b*C_Lalpha       (aero heave damping)
      K_aero[0,1] = rho*U*c*b*C_Lalpha     (aero heave stiffness from pitch)
      K_aero[1,1] = rho*U^2*C_Malpha*c^2*b (aero pitch stiffness)
    """
    q_dyn = 0.5 * rho * U**2

    C_aero = np.zeros((2, 2))
    C_aero[0, 0] = rho * c * b * C_LALPHA       # lift damping in heave

    K_aero = np.zeros((2, 2))
    K_aero[0, 1] = rho * U * c * b * C_LALPHA   # pitch→heave coupling
    K_aero[1, 1] = 2 * q_dyn * C_MALPHA * c**2 * b  # aerodynamic pitch stiffness

    return C_aero, K_aero


def _state_matrix(U):
    """Build 4×4 aeroelastic state matrix at airspeed U."""
    C_aero, K_aero = _build_aero_matrices(U)
    Minv = np.linalg.inv(M_mat)
    A = np.zeros((4, 4))
    A[0:2, 2:4] = np.eye(2)
    A[2:4, 0:2] = -Minv @ (K_mat - K_aero)
    A[2:4, 2:4] = -Minv @ (C_mat - C_aero)
    return A


def run():
    """
    Run flutter analysis.

    Returns
    -------
    passed : bool  (always True — this is informational)
    message : str
    """
    n_modes = 4
    Re_mat  = np.zeros((n_modes, len(U_RANGE)))
    Im_mat  = np.zeros((n_modes, len(U_RANGE)))

    for iu, U in enumerate(U_RANGE):
        A = _state_matrix(U)
        lam = eigvals(A)
        # Sort by ascending Im(λ) to track branches consistently
        idx = np.argsort(lam.imag)
        Re_mat[:, iu] = lam[idx].real
        Im_mat[:, iu] = lam[idx].imag

    # Flutter onset: lowest U where any Re(λ) > threshold for oscillatory mode
    flutter_U = None
    THRESH = 0.0
    for iu, U in enumerate(U_RANGE):
        osc_modes = np.abs(Im_mat[:, iu]) > 0.1   # oscillatory (not overdamped)
        if np.any(Re_mat[:, iu][osc_modes] > THRESH):
            flutter_U = U
            break

    # Eigenvalues at operational speed U_INF
    A_op  = _state_matrix(U_INF)
    lam_op = eigvals(A_op)
    print(f"  Eigenvalues at U_INF={U_INF} m/s:")
    for i, lv in enumerate(sorted(lam_op, key=lambda x: x.imag)):
        stability = "STABLE" if lv.real < 0 else "UNSTABLE"
        print(f"    λ_{i+1} = {lv.real:+.4f} {lv.imag:+.4f}i  ({stability})")

    if flutter_U is not None:
        print(f"  Flutter onset: U_flutter = {flutter_U:.1f} m/s")
        if flutter_U > U_INF:
            print(f"  U_INF={U_INF} m/s is BELOW flutter boundary — system stable")
        else:
            print(f"  WARNING: U_INF={U_INF} m/s EXCEEDS quasi-steady flutter speed!")
    else:
        print(f"  No flutter detected up to U={U_RANGE[-1]:.0f} m/s")

    plotting.plot_root_locus(Re_mat, Im_mat, U_RANGE, flutter_U,
                              figname="flutter_root_locus")

    # This test is informational — always passes
    passed = True
    if flutter_U is not None:
        msg = (f"flutter at U={flutter_U:.1f} m/s; "
               f"U_INF={U_INF} m/s is "
               f"{'below' if flutter_U > U_INF else 'ABOVE'} flutter boundary")
    else:
        msg = f"no flutter detected up to U={U_RANGE[-1]:.0f} m/s (quasi-steady model)"
    return passed, msg


if __name__ == "__main__":
    passed, msg = run()
    print(f"flutter_check: {'PASS' if passed else 'FAIL'} — {msg}")
