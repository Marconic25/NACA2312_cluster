"""
analytical_solution.py — Closed-form 2-DOF structural dynamics solutions.

System (frozen flap, delta=0):
    M q'' + C q' + K q = F(t)
    q = [h, alpha]^T

Mass, stiffness, damping matrices imported from solver_adapter.
"""

import numpy as np
from scipy.linalg import eig, inv
from solver_adapter import M_mat, K_mat, C_mat


def undamped_modes():
    """
    Solve undamped eigenvalue problem: (K - omega^2 M) phi = 0.

    Returns
    -------
    omega : ndarray, shape (2,)
        Natural frequencies [rad/s], sorted ascending.
    V : ndarray, shape (2, 2)
        Mode shape columns, mass-normalised.
    """
    Minv = inv(M_mat)
    A = Minv @ K_mat
    eigenvalues, eigenvectors = eig(A)
    # Eigenvalues should be real and positive
    omega_sq = eigenvalues.real
    idx = np.argsort(omega_sq)
    omega = np.sqrt(omega_sq[idx])
    V = eigenvectors[:, idx].real
    # Mass-normalise: phi^T M phi = 1
    for i in range(2):
        scale = np.sqrt(V[:, i] @ M_mat @ V[:, i])
        V[:, i] /= scale
    return omega, V


def _build_state_matrix(M=M_mat, K=K_mat, C=C_mat):
    """Build 4x4 state-space matrix A: [0, I; -Minv K, -Minv C]."""
    Minv = inv(M)
    A = np.zeros((4, 4))
    A[0:2, 2:4] = np.eye(2)
    A[2:4, 0:2] = -Minv @ K
    A[2:4, 2:4] = -Minv @ C
    return A


def free_response(q0, qd0, t_arr, M=M_mat, K=K_mat, C=C_mat):
    """
    Analytical free response via eigendecomposition of the state matrix.

    Computes x(t) = V * diag(exp(lambda*(t-t0))) * V^{-1} * x0 for all t at once.
    O(N) cost after a one-time O(1) eigendecomposition — no per-step matrix expm.

    Parameters
    ----------
    q0 : array_like, shape (2,)  — initial displacement [h, alpha]
    qd0 : array_like, shape (2,) — initial velocity
    t_arr : array_like, shape (N,) — time points [s]
    M, K, C : optional override matrices (for undamped variants)

    Returns
    -------
    q_arr : ndarray, shape (2, N)
    qd_arr : ndarray, shape (2, N)
    """
    A = _build_state_matrix(M, K, C)
    x0 = np.concatenate([np.asarray(q0, dtype=complex),
                         np.asarray(qd0, dtype=complex)])
    t_arr = np.asarray(t_arr, dtype=float)
    t0    = t_arr[0]
    tau   = t_arr - t0          # shape (N,)

    # Eigendecomposition: A = V * diag(lam) * V^{-1}
    lam, V = eig(A)             # lam shape (4,), V shape (4,4)
    Vinv   = inv(V)

    # Coefficients in eigenbasis: c = V^{-1} x0, shape (4,)
    c = Vinv @ x0

    # exp(lam_i * tau): shape (4, N)
    exp_lam = np.exp(np.outer(lam, tau))   # (4, N)

    # x(t) = V * (c * exp_lam), where * is element-wise broadcast over N
    # V shape (4,4), c shape (4,), exp_lam shape (4,N)
    # x_mat shape (4, N)
    x_mat = V @ (c[:, None] * exp_lam)     # (4, N)

    q_arr  = x_mat[0:2].real
    qd_arr = x_mat[2:4].real
    return q_arr, qd_arr


def forced_harmonic(F0_h, F0_a, omega_force, t_arr, M=M_mat, K=K_mat, C=C_mat):
    """
    Steady-state harmonic response to Fy(t) = F0_h*sin(omega*t), Mz(t) = F0_a*sin(omega*t).

    Matches sign convention of structural_rhs:
        RHS_h =  -Fy  = -F0_h * sin(omega*t)
        RHS_a =  +Mz  = +F0_a * sin(omega*t)

    Particular solution: q_p(t) = Im[ Z^{-1} F_gen * exp(i*omega*t) ]
    where Z = K - omega^2*M + i*omega*C  and  F_gen = [-F0_h, +F0_a].

    Parameters
    ----------
    F0_h : float — Fy amplitude [N]  (sign flip applied internally)
    F0_a : float — Mz amplitude [N·m]
    omega_force : float — forcing frequency [rad/s]
    t_arr : array_like, shape (N,)

    Returns
    -------
    q_ss : ndarray, shape (2, N) — steady-state displacement [h, alpha]
    """
    # Generalised force vector matching structural_rhs sign convention
    F0 = np.array([-F0_h, F0_a], dtype=complex)
    omega = omega_force
    Z = (K - omega**2 * M + 1j * omega * C).astype(complex)
    q_amp = np.linalg.solve(Z, F0)    # shape (2,), complex
    t_arr = np.asarray(t_arr, dtype=float)
    # F(t) = F0 * sin(omega*t) = Im[F0 * exp(i*omega*t)]
    # q_ss(t) = Im[ q_amp * exp(i*omega*t) ]
    # Vectorised: q_amp[:,None] * exp(i*omega*t_arr)[None,:] → (2,N), take .imag
    q_ss = (q_amp[:, None] * np.exp(1j * omega * t_arr)[None, :]).imag
    return q_ss


def full_response(q0, qd0, F0_h, F0_a, omega_force, t_arr,
                  M=M_mat, K=K_mat, C=C_mat):
    """
    Full response = homogeneous (free) + particular (steady-state harmonic).

    The homogeneous part uses ICs adjusted so that at t=t_arr[0]:
        q_total = q_ss + q_hom = q0  =>  q_hom_0 = q0 - q_ss(t0)

    Returns
    -------
    q_total : ndarray, shape (2, N)
    """
    t_arr = np.asarray(t_arr)
    t0 = t_arr[0]
    q_ss = forced_harmonic(F0_h, F0_a, omega_force, t_arr, M, K, C)

    # Steady-state velocity at t0 (same sign convention as forced_harmonic)
    F0 = np.array([-F0_h, F0_a], dtype=complex)
    Z = (K - omega_force**2 * M + 1j * omega_force * C).astype(complex)
    q_amp = np.linalg.solve(Z, F0)
    q_ss_t0  = q_ss[:, 0]
    qd_ss_t0 = (q_amp * 1j * omega_force * np.exp(1j * omega_force * t0)).imag

    # Homogeneous ICs: subtract steady-state at t0
    q_hom_0  = np.asarray(q0)  - q_ss_t0
    qd_hom_0 = np.asarray(qd0) - qd_ss_t0

    q_hom, _ = free_response(q_hom_0, qd_hom_0, t_arr, M, K, C)
    return q_ss + q_hom
