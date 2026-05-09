# FSI Dataset Specification v2 — GLA (Gust Load Alleviation)
## NACA 2312 + Flap, 2D, OpenFOAM cosim_driver

---

## 1. Fixed Parameters

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Freestream velocity | U_inf | 80 | m/s |
| Simulation duration | T_sim | 3.0 | s |
| Data saving timestep | dt_save | 0.002 | s |
| Samples per simulation | N_samples | 1500 | - |
| Gust start time | t_start_gust | 0.0 | s |
| Chord length | c | 1.0 | m |
| Wing mass | M_wing | 22.9 | kg |
| Wing MoI (z-axis) | I_wing | 2.057 | kg·m² |
| Heave stiffness | k_h | 4000.0 | N/m |
| Heave damping | c_h | 2.0 | N·s/m |
| Pitch stiffness | k_alpha | 700.0 | N·m/rad |
| Pitch damping | c_alpha | 0.5 | N·m·s/rad |
| Elastic axis | EA_x | 0.25 | m |
| Flap hinge | hinge_x | 0.779 | m |
| CFD timestep | dt_CFD | 7e-5 | s |
| Coupling window | window | 286 steps | - |
| Coupling dt | window_dt | 0.02002 | s |

---

## 2. Gust Model

1-cosine gust (CS-25 / EASA):

```
W_g(t) = (W_g0 / 2) * (1 - cos(2π(t - t_start) / T_g))   for t_start ≤ t ≤ t_start + T_g
W_g(t) = 0                                                   otherwise

W_g0 = R × U_inf
```

---

## 3. Flap Actuation Laws

### Law 1: Linear ramp + hold
```
δ(t) = 0                                      t < t_start_δ
δ(t) = δ_max × (t - t_start_δ) / dt_ramp     t ∈ [t_start_δ, t_start_δ + dt_ramp]
δ(t) = δ_max                                  t > t_start_δ + dt_ramp
```

### Law 2: Two-phase segmented ramp + hold
```
δ(t) = 0                                      t < t_start_δ
δ(t) = (δ_max/2) × (t-t_start_δ)/dt_1        Phase 1: [t_start_δ, t_start_δ + dt_1]
δ(t) = δ_max/2 + (δ_max/2)×(t-t1)/dt_2       Phase 2: [t_start_δ+dt_1, t_start_δ+dt_1+dt_2]
δ(t) = δ_max                                  t > t_start_δ + dt_1 + dt_2
```

### Law 3: Trapezoid (ramp up + hold + ramp down)
```
δ(t) = 0                                      t < t_start_δ
δ(t) = δ_max × (t-t_start_δ)/dt_up            Ramp up
δ(t) = δ_max                                  Hold
δ(t) = δ_max × (1 - (t-t2)/dt_down)           Ramp down
δ(t) = 0                                      t > t_end_δ
```

---

## 4. Dataset Families

### Family A — No flap (baseline)
- Actuation: δ(t) = 0
- LHS dimension: 2D
- Sims: 6 train + 2 test = **8**

| Parameter | Symbol | Range | Unit |
|-----------|--------|-------|------|
| Gust intensity | R | [0.10, 0.60] | - |
| Gust duration | T_g | [0.30, 1.20] | s |

### Family B1 — Informed, Law 1 (ramp + hold)
- LHS dimension: 5D
- Sims: 16 train + 4 test = **20**

| Parameter | Symbol | Range | Unit |
|-----------|--------|-------|------|
| Gust intensity | R | [0.10, 0.60] | - |
| Gust duration | T_g | [0.30, 1.20] | s |
| Max flap angle | δ_max | [2.0, 20.0] | deg |
| Ramp duration | dt_ramp | [0.05, 0.50] | s |
| Flap start time | t_start_δ | [0.20, 0.80] | s |

Constraint: t_start_δ + dt_ramp < 2.5 s

### Family B2 — Informed, Law 2 (two-phase ramp)
- LHS dimension: 6D
- Sims: 12 train + 3 test = **15**

| Parameter | Symbol | Range | Unit |
|-----------|--------|-------|------|
| Gust intensity | R | [0.10, 0.60] | - |
| Gust duration | T_g | [0.30, 1.20] | s |
| Max flap angle | δ_max | [2.0, 20.0] | deg |
| Phase 1 duration | dt_1 | [0.05, 0.20] | s |
| Phase 2 duration | dt_2 | [0.10, 0.40] | s |
| Flap start time | t_start_δ | [0.20, 0.80] | s |

Constraint: t_start_δ + dt_1 + dt_2 < 2.5 s

### Family B3 — Informed, Law 3 (trapezoid)
- LHS dimension: 7D
- Sims: 10 train + 3 test = **13**

| Parameter | Symbol | Range | Unit |
|-----------|--------|-------|------|
| Gust intensity | R | [0.10, 0.60] | - |
| Gust duration | T_g | [0.30, 1.20] | s |
| Max flap angle | δ_max | [2.0, 20.0] | deg |
| Ramp up | dt_up | [0.05, 0.50] | s |
| Hold | dt_hold | [0.10, 0.50] | s |
| Ramp down | dt_down | [0.05, 0.50] | s |
| Flap start time | t_start_δ | [0.20, 0.80] | s |

Constraint: t_start_δ + dt_up + dt_hold + dt_down < 2.5 s

### Family C — Uninformed, Law 1 (ramp + hold)
- LHS dimension: 5D
- Sims: 6 train + 2 test = **8**

| Parameter | Symbol | Range | Unit |
|-----------|--------|-------|------|
| Gust intensity | R | [0.10, 0.60] | - |
| Gust duration | T_g | [0.30, 1.20] | s |
| Max flap angle | δ_max | [2.0, 20.0] | deg |
| Ramp duration | dt_ramp | [0.05, 0.50] | s |
| Flap start time | t_start_δ | [0.00, 0.15] | s |

Constraint: t_start_δ + dt_ramp < 2.5 s

---

## 5. Summary

| Family | Description | Train | Test | Total | LHS dim |
|--------|-------------|-------|------|-------|---------|
| A | No flap | 6 | 2 | 8 | 2 |
| B1 | Informed, Law 1 | 16 | 4 | 20 | 5 |
| B2 | Informed, Law 2 | 12 | 3 | 15 | 6 |
| B3 | Informed, Law 3 | 10 | 3 | 13 | 7 |
| C | Uninformed, Law 1 | 6 | 2 | 8 | 5 |
| **Total** | | **50** | **14** | **64** | |

---

## 6. Output Variables per Simulation

### Timeseries (1500 rows × 11 columns)

| Col | Variable | Symbol | Unit | Source |
|-----|----------|--------|------|--------|
| 1 | Time | t | s | simulation |
| 2 | Heave | h | m | structural |
| 3 | Heave rate | ḣ | m/s | structural |
| 4 | Pitch angle | α | rad | structural |
| 5 | Pitch rate | α̇ | rad/s | structural |
| 6 | Flap deflection | δ | deg | prescribed |
| 7 | Gust velocity | W_g | m/s | analytical |
| 8 | Lift coefficient | C_L | - | OpenFOAM |
| 9 | Moment coefficient | C_M | - | OpenFOAM |
| 10 | Lift force | F_y | N | OpenFOAM |
| 11 | Pitching moment | M_z | N·m | OpenFOAM |

### Fields (1500 × N_grid × 3, float32)

| Channel | Variable | Unit |
|---------|----------|------|
| 0 | u_x | m/s |
| 1 | u_y | m/s |
| 2 | p | Pa |

Grid: cartesian fixed frame, x ∈ [-2, 6]c, y ∈ [-3, 3]c, ~200×150 = 30000 points.

### Metadata (1 row per simulation)

R, T_g, W_g0, δ_max, t_start_δ, law, family, split (train/test), plus all law-specific params.

---

## 7. Naming Convention

```
sim_{FAMILY}_{INDEX:03d}_{SPLIT}
```

Examples:
- `sim_A_000_train` — Family A, index 0, training set
- `sim_B1_005_test` — Family B1, index 5, test set
- `sim_C_002_train` — Family C, index 2, training set

Directory structure:
```
data/GLA/
├── dataset_spec_v2.md
├── metadata.csv
├── grid_points.npy
├── timeseries/
│   ├── sim_A_000_train.csv
│   └── ...
└── fields/
    ├── sim_A_000_train.npy
    └── ...
```

---

## 8. Constraints Summary

1. Flap trajectory ends before T_sim - 0.5s: `t_start_δ + actuation_duration < 2.5 s`
2. Gust ends before T_sim: `T_g < 3.0 s` (always satisfied, T_g ≤ 1.2)
3. Family B (informed): `t_start_δ ≥ 0.2 s`
4. Family C (uninformed): `t_start_δ ≤ 0.15 s`
5. Samples violating constraint 1 are rejected and redrawn.

---

## 9. Scalability

The design supports easy expansion:
- Increase N_train/N_test per family by regenerating LHS with more points
- Add new families (e.g., B4 with new actuation law) by appending to metadata.csv
- All scripts accept family/count as parameters
