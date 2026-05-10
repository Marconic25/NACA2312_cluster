#!/usr/bin/env python3
"""
3-DOF Flutter Stability Analysis — Typical Section with Control Surface
========================================================================

Based on: G. Bindolino, Politecnico di Milano
         "Dinamica strutturale e aeroelasticità — 3 dof flutter"

Equations of motion (Lagrange):
    [M]{q̈} + [D]{q̇} + ([K] - q_dyn·[KA]){q} = 0

    q = {h, θ, β}^T   (plunge, pitch, flap rotation)
    q_dyn = ½ρV²

Parameters extracted from:
    wingMotion2D_pimpleFoam/constant/dynamicMeshDict
"""

import numpy as np
from numpy import linalg as la
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETERS  (from dynamicMeshDict and geometry drawing)
# =============================================================================

# --- Geometry (positions in world frame, metres) ---
x_LE   = 0.015     # wing leading edge (estimated)
x_EA   = 0.25      # elastic axis = wing_main joint/pivot
x_CG_F = 0.4975    # wing_main CG  (0.25 + centreOfMass_x = 0.2474612746)
x_H    = 0.775     # flap hinge     (0.25 + flap_transform_x = 0.525)
x_CG_CS= 0.8875    # flap CG        (0.775 + flap_centreOfMass_x = 0.1125)

c_wing = 0.720     # wing chord  (720.3 mm)
c_flap = 0.223     # flap chord  (223 mm)
gap    = 0.030     # gap between wing TE and flap LE  (29.9 mm)
c      = c_wing + gap + c_flap   # total section chord ≈ 0.973 m
span   = 0.25      # 2-D slab thickness (z = 0..0.25)

x_AC   = x_LE + 0.25 * c         # aerodynamic centre at c/4

# Offset distances (PDF notation)
d_F  = x_CG_F  - x_EA    # wing CG offset from EA   ≈ 0.2475 m
d_CS = x_CG_CS - x_EA    # flap CG offset from EA   ≈ 0.6375 m
d_H  = x_CG_CS - x_H     # flap CG offset from hinge ≈ 0.1125 m
e    = x_EA - x_AC        # EA–AC distance (positive when EA aft of AC)

# --- Masses (kg) ---
m_F  = 22.9       # wing_main mass
m_CS = 4.6        # flap mass
m_TOT = m_F + m_CS

# --- Moments of inertia about own CG, Izz component (kg·m²) ---
I_F  = 2.057121362    # wing_main  (inertia tensor entry [5] in OF)
I_CS = 0.4            # flap       (inertia tensor entry [5] in OF)

# --- Spring stiffnesses (from dynamicMeshDict restraints) ---
K_h    = 30000.0   # vertical spring          [N/m]
K_theta= 3000.0    # wing pitch spring        [N·m/rad]
K_beta = 1800.0    # flap hinge spring        [N·m/rad]

# --- Structural damping (from dynamicMeshDict restraints) ---
D_h    = 500.0     # vertical spring damping   [N·s/m]
D_theta= 25.0      # wing pitch damping        [N·m·s/rad]
D_beta = 8.0       # flap hinge damping        [N·m·s/rad]

# --- Flow conditions ---
rho   = 1.225      # air density  [kg/m³]
V_inf = 80.0       # freestream velocity in OpenFOAM  [m/s]

# --- Aerodynamic coefficients (quasi-steady, thin-airfoil defaults) ---
CL_alpha  = 2.0 * np.pi          # lift curve slope
# CL_beta from thin-airfoil theory:  2*(arccos(1-2E) + 2*sqrt(E(1-E)))
E = c_flap / c                   # flap-to-chord ratio ≈ 0.229
CL_beta   = 2.0 * (np.arccos(1 - 2*E) + 2*np.sqrt(E*(1-E)))
CMAC_beta = -0.6                  # moment about AC due to flap
CH_alpha  = -0.25                 # hinge moment coeff w.r.t. AoA
CH_beta   = -0.6                  # hinge moment coeff w.r.t. flap

# =============================================================================
# 2. BUILD MATRICES
# =============================================================================

def build_matrices():
    """Return M, K, D, KA  (all 3×3 numpy arrays)."""

    # --- Derived inertial quantities (PDF page 5) ---
    S_EA  = m_F * d_F + m_CS * d_CS          # static moment about EA
    S_H   = m_CS * d_H                       # static moment of CS about hinge
    I_EA  = I_F + I_CS + m_F*d_F**2 + m_CS*d_CS**2   # total inertia about EA
    I_CCS = I_CS + S_H * d_CS                # centrifugal inertia of CS
    I_HCS = I_CS + m_CS * d_H**2             # CS inertia about hinge

    # Mass matrix  [M]  (symmetric, 3×3)
    M = np.array([
        [m_TOT,   S_EA,    S_H    ],
        [S_EA,    I_EA,    I_CCS  ],
        [S_H,     I_CCS,   I_HCS  ],
    ])

    # Stiffness matrix [K]
    K = np.diag([K_h, K_theta, K_beta])

    # Structural damping matrix [D]
    D = np.diag([D_h, D_theta, D_beta])

    # Aerodynamic stiffness matrix [KA]  (PDF page 7)
    # Reference area S_ref = c * span   (per-span for 2-D section)
    S_ref = c * span
    S_H_area = c_flap * span   # flap reference area

    KA = np.array([
        [0.0,  -S_ref * CL_alpha,                   -S_ref * CL_beta                              ],
        [0.0,   S_ref * e * CL_alpha,                 S_ref * (e * CL_beta + c * CMAC_beta)        ],
        [0.0,   S_H_area * c_flap * CH_alpha,         S_H_area * c_flap * CH_beta                  ],
    ])

    return M, K, D, KA


def print_matrices(M, K, D, KA):
    """Pretty-print the system matrices."""
    np.set_printoptions(precision=4, linewidth=120)
    print("=" * 70)
    print("SYSTEM MATRICES")
    print("=" * 70)
    print(f"\n[M]  (mass matrix):\n{M}")
    print(f"\n[K]  (structural stiffness):\n{K}")
    print(f"\n[D]  (structural damping):\n{D}")
    print(f"\n[KA] (aerodynamic stiffness, multiply by q_dyn):\n{KA}")

    # Derived quantities
    S_EA = m_F * d_F + m_CS * d_CS
    S_H  = m_CS * d_H
    print(f"\n--- Derived inertial quantities ---")
    print(f"  m_TOT  = {m_TOT:.2f} kg")
    print(f"  S_EA   = {S_EA:.4f} kg·m   (static moment about EA)")
    print(f"  S_H    = {S_H:.4f} kg·m    (static moment of CS about hinge)")
    print(f"  d_F    = {d_F:.4f} m        (wing CG offset from EA)")
    print(f"  d_CS   = {d_CS:.4f} m       (flap CG offset from EA)")
    print(f"  d_H    = {d_H:.4f} m        (flap CG offset from hinge)")
    print(f"  e      = {e:.4f} m          (EA – AC distance)")
    print(f"  c      = {c:.4f} m          (total chord)")
    print(f"  c_flap = {c_flap:.4f} m     (flap chord)")
    print(f"  CL_beta= {CL_beta:.4f}      (thin-airfoil, E={E:.3f})")

    # Natural frequencies (undamped, no aero)
    M_inv = la.inv(M)
    omega2 = la.eigvalsh(M_inv @ K)
    omega  = np.sqrt(np.maximum(omega2, 0))
    freq   = omega / (2 * np.pi)
    print(f"\n--- Uncoupled natural frequencies (V=0, no damping) ---")
    for i, f in enumerate(sorted(freq)):
        print(f"  Mode {i+1}:  {f:.2f} Hz  ({f*2*np.pi:.2f} rad/s)")


# =============================================================================
# 3. EIGENVALUE SWEEP  (V–f and V–g diagrams)
# =============================================================================

def eigenvalue_sweep(M, K, D, KA, V_range):
    """
    For each velocity V, form the 6×6 state-space matrix and solve
    for eigenvalues.  Return arrays of frequencies and damping ratios
    for every mode at every velocity.
    """
    n = M.shape[0]          # 3 DOFs
    I3 = np.eye(n)
    Z3 = np.zeros((n, n))
    M_inv = la.inv(M)

    n_V = len(V_range)
    # Store all eigenvalues (6 complex numbers per velocity)
    all_eigs = np.zeros((n_V, 2*n), dtype=complex)

    for i, V in enumerate(V_range):
        q_dyn = 0.5 * rho * V**2
        K_eff = K - q_dyn * KA

        # State-space matrix A  (6×6)
        # x = {q, qdot}^T,   xdot = A x
        #
        #  A = [   0       I     ]
        #      [ -M⁻¹K_eff  -M⁻¹D ]
        A = np.block([
            [Z3,             I3           ],
            [-M_inv @ K_eff, -M_inv @ D   ],
        ])

        eigs = la.eig(A)[0]
        # Sort by imaginary part (frequency) for consistent tracking
        idx = np.argsort(np.abs(eigs.imag))
        all_eigs[i] = eigs[idx]

    return all_eigs


def track_modes(all_eigs, V_range):
    """
    From raw eigenvalues, extract frequency (Hz) and damping (real part)
    for each of the 3 conjugate-pair modes.  Only keep eigenvalues with
    non-negative imaginary part (positive frequency branch).
    """
    n_V = len(V_range)
    n_modes = all_eigs.shape[1] // 2   # 3 modes

    freqs   = np.zeros((n_V, n_modes))
    dampings = np.zeros((n_V, n_modes))

    for i in range(n_V):
        eigs = all_eigs[i]
        # Keep only eigenvalues with Im >= 0  (positive frequency branch)
        pos = eigs[eigs.imag >= -1e-10]
        # Sort by ascending frequency
        pos = pos[np.argsort(pos.imag)]
        for j in range(min(n_modes, len(pos))):
            freqs[i, j]    = pos[j].imag / (2 * np.pi)   # Hz
            dampings[i, j] = pos[j].real                  # σ (positive = unstable)

    return freqs, dampings


# =============================================================================
# 4. TIME RESPONSE at a given velocity
# =============================================================================

def time_response(M, K, D, KA, V, t_end=0.5, h0=0.001):
    """
    Integrate the 3-DOF system in time at velocity V.
    Initial condition: small perturbation in plunge h.
    Returns t, q(t).
    """
    n = M.shape[0]
    M_inv = la.inv(M)
    q_dyn = 0.5 * rho * V**2
    K_eff = K - q_dyn * KA

    def rhs(t, x):
        q  = x[:n]
        qd = x[n:]
        qdd = -M_inv @ (D @ qd + K_eff @ q)
        return np.concatenate([qd, qdd])

    x0 = np.zeros(2 * n)
    x0[0] = h0          # initial plunge perturbation  [m]
    x0[1] = 0.01        # initial pitch perturbation   [rad]
    x0[2] = 0.01        # initial flap perturbation    [rad]

    sol = solve_ivp(rhs, [0, t_end], x0, max_step=1e-4, rtol=1e-9, atol=1e-12)
    return sol.t, sol.y[:n]


# =============================================================================
# 5. FIND FLUTTER AND DIVERGENCE SPEEDS
# =============================================================================

def find_flutter_speed(V_range, dampings):
    """Find the lowest velocity where any modal damping crosses zero → positive."""
    flutter_V = None
    flutter_mode = None
    for j in range(dampings.shape[1]):
        for i in range(1, len(V_range)):
            if dampings[i-1, j] < 0 and dampings[i, j] >= 0:
                # Linear interpolation
                V1, V2 = V_range[i-1], V_range[i]
                d1, d2 = dampings[i-1, j], dampings[i, j]
                V_cross = V1 + (V2 - V1) * (-d1) / (d2 - d1)
                if flutter_V is None or V_cross < flutter_V:
                    flutter_V = V_cross
                    flutter_mode = j + 1
    return flutter_V, flutter_mode


def find_divergence_speed(M, K, KA, V_range):
    """Find velocity where det(K - q*KA) = 0  (static divergence)."""
    for i in range(1, len(V_range)):
        q1 = 0.5 * rho * V_range[i-1]**2
        q2 = 0.5 * rho * V_range[i]**2
        d1 = la.det(K - q1 * KA)
        d2 = la.det(K - q2 * KA)
        if d1 * d2 < 0:
            # Linear interpolation
            V1, V2 = V_range[i-1], V_range[i]
            V_cross = V1 + (V2 - V1) * (-d1) / (d2 - d1) if (d2 - d1) != 0 else V2
            return V_cross
    return None


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    M, K, D, KA = build_matrices()
    print_matrices(M, K, D, KA)

    # --- Eigenvalue sweep ---
    V_max = 150.0
    V_range = np.linspace(0.1, V_max, 600)
    all_eigs = eigenvalue_sweep(M, K, D, KA, V_range)
    freqs, dampings = track_modes(all_eigs, V_range)

    # --- Flutter and divergence speeds ---
    flutter_V, flutter_mode = find_flutter_speed(V_range, dampings)
    div_V = find_divergence_speed(M, K, KA, V_range)

    print("\n" + "=" * 70)
    print("STABILITY RESULTS")
    print("=" * 70)
    if flutter_V is not None:
        print(f"  FLUTTER SPEED:     {flutter_V:.1f} m/s   (mode {flutter_mode})")
    else:
        print(f"  FLUTTER SPEED:     NOT FOUND in [0, {V_max}] m/s")
    if div_V is not None:
        print(f"  DIVERGENCE SPEED:  {div_V:.1f} m/s")
    else:
        print(f"  DIVERGENCE SPEED:  NOT FOUND in [0, {V_max}] m/s")

    # --- Eigenvalues at operating point V = V_inf ---
    idx_op = np.argmin(np.abs(V_range - V_inf))
    eigs_op = all_eigs[idx_op]
    print(f"\n  Eigenvalues at V = {V_inf} m/s:")
    for j, ev in enumerate(eigs_op):
        status = "UNSTABLE" if ev.real > 0 else "stable"
        print(f"    λ_{j+1} = {ev.real:+10.4f} ± {abs(ev.imag):10.4f}j   "
              f"(f={abs(ev.imag)/(2*np.pi):.2f} Hz)  [{status}]")

    max_real = max(ev.real for ev in eigs_op)
    if max_real > 0:
        print(f"\n  >>> SYSTEM IS UNSTABLE at V={V_inf} m/s  (max σ = {max_real:.4f})")
        print(f"  >>> This explains the OpenFOAM simulation blow-up!")
    else:
        print(f"\n  >>> System is STABLE at V={V_inf} m/s  (max σ = {max_real:.4f})")

    # --- Time response at operating velocity ---
    print(f"\n  Computing time response at V = {V_inf} m/s ...")
    t_resp, q_resp = time_response(M, K, D, KA, V_inf, t_end=0.5)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("3-DOF Flutter Stability Analysis\n"
                 f"(K_h={K_h}, K_θ={K_theta}, K_β={K_beta}, "
                 f"D_h={D_h}, D_θ={D_theta}, D_β={D_beta})",
                 fontsize=12)

    mode_labels = ["Mode 1", "Mode 2", "Mode 3"]
    colors = ["tab:blue", "tab:red", "tab:green"]

    # --- V-f diagram ---
    ax = axes[0, 0]
    for j in range(freqs.shape[1]):
        ax.plot(V_range, freqs[:, j], color=colors[j], label=mode_labels[j], linewidth=1.2)
    ax.axvline(V_inf, color="k", linestyle="--", alpha=0.5, label=f"V_op = {V_inf} m/s")
    if flutter_V:
        ax.axvline(flutter_V, color="r", linestyle=":", alpha=0.7,
                   label=f"Flutter = {flutter_V:.1f} m/s")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("V–f diagram (modal frequency)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, V_max)

    # --- V-g diagram ---
    ax = axes[0, 1]
    for j in range(dampings.shape[1]):
        ax.plot(V_range, dampings[:, j], color=colors[j], label=mode_labels[j], linewidth=1.2)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axvline(V_inf, color="k", linestyle="--", alpha=0.5, label=f"V_op = {V_inf} m/s")
    if flutter_V:
        ax.axvline(flutter_V, color="r", linestyle=":", alpha=0.7,
                   label=f"Flutter = {flutter_V:.1f} m/s")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Damping σ  (>0 = unstable)")
    ax.set_title("V–g diagram (modal damping)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, V_max)

    # --- Time response ---
    ax = axes[1, 0]
    labels_q = ["h [m]", "θ [rad]", "β [rad]"]
    for j in range(3):
        ax.plot(t_resp, q_resp[j], color=colors[j], label=labels_q[j], linewidth=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Response")
    ax.set_title(f"Time response at V = {V_inf} m/s  (h₀=1mm, θ₀=β₀=0.01rad)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Eigenvalue map at operating point ---
    ax = axes[1, 1]
    for ev in eigs_op:
        color = "red" if ev.real > 0 else "blue"
        ax.plot(ev.real, ev.imag, "o", color=color, markersize=8)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Real part σ")
    ax.set_ylabel("Imaginary part ω")
    ax.set_title(f"Eigenvalue map at V = {V_inf} m/s  (red = unstable)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "flutter_stability.png"
    fig.savefig(out_path, dpi=150)
    print(f"\n  Plots saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
