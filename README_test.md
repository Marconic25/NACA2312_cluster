# Imposed-Motion Test Case

Quick test to verify numerical stability with time-varying wing/flap motion and a step gust.

## Files

| File | Purpose |
|------|---------|
| `generate_inputs.py` | Generates the three input `.dat` files |
| `testCase_imposed/` | OpenFOAM case (copy of `wingMotion2D_pimpleFoam`) |
| `run_test.sh` | Full pipeline: generate → mesh check → run → validate |
| `check_results.py` | Post-processing: plot forces, check for divergence |

## Input signals

| Signal | Definition | File |
|--------|-----------|------|
| Plunge h(t) | 0.05·sin(2π·0.8·t) m | `constant/wing_motion.dat` |
| Pitch α(t) | 0.1·sin(2π·0.8·t) rad (≈5.7°) | `constant/wing_motion.dat` |
| Flap δ(t) | 5·sin(2π·1.0·t) deg | `constant/flap_control.dat` |
| Gust W(t) | (1-cos) ramp 0→15 m/s at t=0.5s | `constant/gust_profile.dat` |

## How motion is applied

- **Wing (`wing_main`)**: `codedFixedValue` in `0/pointDisplacement`
  Each mesh point is displaced by rotation around the elastic axis `(0.25, 0.007)` by `α(t)` plus a pure plunge `h(t)`.

- **Flap**: `codedFixedValue` in `0/pointDisplacement`
  Each point rotated by `δ(t)` around the hinge `(0.775, -0.045)`.

- **Mesh interior**: `displacementLaplacian` with `quadratic inverseDistance` diffusivity propagates the boundary displacements into the mesh volume.

- **Inlet velocity**: `uniformFixedValue` with `tableFile` reads `gust_profile.dat` at each time step.

## Quick start

```bash
cd /home/marco/OpenFOAM/marco-7/run_python
bash run_test.sh
```

Or step by step:

```bash
source /opt/openfoam7/etc/bashrc
python3 generate_inputs.py

cd testCase_imposed
checkMesh -allGeometry -allTopology | tail -5
decomposePar -force
mpirun -np 4 pimpleFoam -parallel > log.pimpleFoam 2>&1
reconstructPar

cd ..
python3 check_results.py
```

## Stability settings

| Parameter | Value | Reason |
|-----------|-------|--------|
| `maxCo` | 0.5 | Conservative for moving mesh |
| `nOuterCorrectors` | 3 | Extra PIMPLE iterations |
| `nCorrectors` | 2 | Extra pressure corrections |
| `p` relaxation | 0.2 | Damped for stability |
| `U/k/ω` relaxation | 0.5 | Damped for stability |

## Troubleshooting

**Diverges early (t < 0.1 s)**
- Reduce `maxCo` to 0.3 in `system/controlDict`
- Increase `nOuterCorrectors` to 5

**Diverges at gust onset (t ≈ 0.5 s)**
- The 15 m/s gust adds ~10.6° effective AoA — may cause flow separation
- Reduce `W_GUST_FINAL` in `generate_inputs.py` to 5–8 m/s for the first test

**Mesh distortion (flap gap)**
- The 15 mm gap between wing TE and flap LE is the critical zone
- If `checkMesh` reports negative volumes, reduce motion amplitudes

**Check residuals**
```bash
grep "Solving for p" testCase_imposed/log.pimpleFoam | tail -20
```
