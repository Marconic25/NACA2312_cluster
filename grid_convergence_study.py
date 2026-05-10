#!/usr/bin/env python3
"""
Grid Convergence Study — NACA 2312
Generates 4 meshes (M1→M4) by progressively refining snappyHexMesh parameters,
runs simpleFoam on each, computes CL/CD/CM and GCI.

Usage:
    python3 grid_convergence_study.py
    python3 grid_convergence_study.py --workdir /work/u10677113/NACA2312
    python3 grid_convergence_study.py --submit   # submit via PBS instead of running directly

Mesh progression (M1=baseline from snappyHexMeshDict):
    M1: surface level 5, box level 2, gap level 4, layers 3  → y+ ~300
    M2: surface level 6, box level 3, gap level 5, layers 4  → y+ ~100
    M3: surface level 7, box level 3, gap level 6, layers 5  → y+ ~30
    M4: surface level 8, box level 4, gap level 7, layers 6  → y+ ~10
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKDIR        = Path("/work/u10677113/NACA2312")
CONTAINER      = "/work/u10677113/of7.sif"
SNAPPY_CASE    = WORKDIR / "mesh_baseline"
SIMPLE_CASE    = WORKDIR / "rans_baseline"
STUDY_DIR      = WORKDIR / "grid_convergence"

# Reference chord and span for force coefficients (must match simpleFoam forceCoeffs)
CHORD  = 1.0   # m  (lRef in forceCoeffs)
SPAN   = 0.05  # m  (extrusion thickness from extrudeMeshDict)
UINF   = 80.0  # m/s (magUInf in forceCoeffs)
RHO    = 1.225 # kg/m3
AREF   = 0.05  # m2 (Aref in forceCoeffs)

# Number of simpleFoam iterations to average for CL/CD/CM
AVG_LAST_N = 50

# Mesh levels: (surface_min, surface_max, box_level, gap_level, n_layers)
MESH_LEVELS = {
    "M1": dict(surf_min=4, surf_max=4, box=2, gap=3, layers=2,  first_layer=0.30, exp_ratio=1.20),
    "M2": dict(surf_min=5, surf_max=5, box=3, gap=4, layers=3,  first_layer=0.30, exp_ratio=1.20),
    "M3": dict(surf_min=5, surf_max=5, box=3, gap=5, layers=5,  first_layer=0.20, exp_ratio=1.20),
    "M4": dict(surf_min=5, surf_max=5, box=4, gap=5, layers=5, first_layer=0.20, exp_ratio=1.20),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_in_container(cmd: str, cwd: Path, log_path: Path) -> int:
    """Run an OpenFOAM command inside the Apptainer container."""
    full_cmd = (
        f"apptainer exec {CONTAINER} /bin/bash -c "
        f"'source /opt/openfoam7/etc/bashrc && cd {cwd} && {cmd}'"
    )
    with open(log_path, "w") as lf:
        result = subprocess.run(full_cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT)
    return result.returncode


def patch_snappy(src: Path, dst: Path, p: dict):
    """Write a modified snappyHexMeshDict with the given refinement parameters."""
    txt = src.read_text()

    sl_min      = p["surf_min"]
    sl_max      = p["surf_max"]
    box         = p["box"]
    gap         = p["gap"]
    layers      = p["layers"]
    first_layer = p["first_layer"]
    exp_ratio   = p["exp_ratio"]

    # Surface refinement levels
    txt = re.sub(
        r"(wing_main\s*\{\s*level\s*\()(\d+\s+\d+)(\s*\);)",
        rf"\g<1>{sl_min} {sl_max}\g<3>", txt
    )
    txt = re.sub(
        r"(flap\s*\{\s*level\s*\()(\d+\s+\d+)(\s*\);)",
        rf"\g<1>{sl_min} {sl_max}\g<3>", txt
    )

    # Feature edge levels
    txt = re.sub(
        r'(file\s+"wing_main\.eMesh";\s*level\s*)(\d+)(;)',
        rf"\g<1>{sl_max}\g<3>", txt
    )
    txt = re.sub(
        r'(file\s+"flap\.eMesh";\s*level\s*)(\d+)(;)',
        rf"\g<1>{sl_max}\g<3>", txt
    )

    # Refinement box level
    txt = re.sub(
        r"(refinementBox\s*\{\s*mode\s+inside;\s*levels\s*\(\(1e15\s*)(\d+)(\s*\)\))",
        rf"\g<1>{box}\g<3>", txt
    )

    # Gap region level
    txt = re.sub(
        r"(gapRegion\s*\{\s*mode\s+inside;\s*levels\s*\(\(1e15\s*)(\d+)(\s*\)\))",
        rf"\g<1>{gap}\g<3>", txt
    )

    # Number of surface layers
    txt = re.sub(
        r"(wing_main\s*\{\s*nSurfaceLayers\s*)(\d+)(;)",
        rf"\g<1>{layers}\g<3>", txt
    )
    txt = re.sub(
        r"(flap\s*\{\s*nSurfaceLayers\s*)(\d+)(;)",
        rf"\g<1>{layers}\g<3>", txt
    )

    # Layer thickness — use relativeSizes true with finalLayerThickness
    txt = re.sub(r"(relativeSizes\s+)(true|false)(;)", r"\g<1>true\g<3>", txt)
    txt = re.sub(r"(firstLayerThickness|finalLayerThickness)\s*[\d.eE+\-]+;", f"finalLayerThickness {first_layer};", txt)
    txt = re.sub(r"(expansionRatio\s*)([\d.]+)(;)", rf"\g<1>{exp_ratio}\g<3>", txt)

    # Scale maxGlobalCells with refinement
    scale = 2 ** (2 * (sl_max - 5))
    max_cells = max(4_000_000, 4_000_000 * scale)
    txt = re.sub(r"(maxGlobalCells\s*)(\d+)(;)", rf"\g<1>{max_cells}\g<3>", txt)

    # Robust layer controls
    txt = re.sub(r"(nLayerIter\s*)(\d+)(;)", r"\g<1>50\g<3>", txt)
    txt = re.sub(r"(minThickness\s*)([\d.eE+\-]+)(;)", rf"\g<1>{first_layer * 0.1}\g<3>", txt)
    txt = re.sub(r"(maxThicknessToMedialRatio\s*)([\d.]+)(;)", r"\g<1>0.6\g<3>", txt)
    txt = re.sub(r"(minMedianAxisAngle\s*)(\d+)(;)", r"\g<1>60\g<3>", txt)
    txt = re.sub(r"(nRelaxIter\s*)(\d+)(;)", r"\g<1>50\g<3>", txt)
    txt = re.sub(r"(nSmoothSurfaceNormals\s*)(\d+)(;)", r"\g<1>5\g<3>", txt)
    txt = re.sub(r"(nSmoothNormals\s*)(\d+)(;)", r"\g<1>10\g<3>", txt)
    txt = re.sub(r"(nBufferCellsNoExtrude\s*)(\d+)(;)", r"\g<1>1\g<3>", txt)

    dst.write_text(txt)


def count_cells(case_dir: Path) -> int:
    """Count cells from checkMesh log or polyMesh/owner."""
    log = case_dir / "log.checkMesh"
    if log.exists():
        m = re.search(r'cells:\s+(\d+)', log.read_text())
        if m:
            return int(m.group(1))
    # Fallback: count from owner file header
    owner = case_dir / "constant" / "polyMesh" / "owner"
    if owner.exists():
        for line in owner.read_text().splitlines():
            m = re.search(r'nCells\s+(\d+)', line)
            if m:
                return int(m.group(1))
    return -1


def parse_forces_log(log_path: Path, avg_n: int) -> dict:
    """
    Parse forces/forceCoeffs postProcessing output.
    Returns dict with keys: Cl, Cd, Cm (averaged over last avg_n entries).
    """
    # Try postProcessing first
    pp_dir = log_path.parent / "postProcessing"
    coeff_file = None
    for candidate in pp_dir.glob("**/coefficient*.dat"):
        coeff_file = candidate
        break
    if coeff_file is None:
        for candidate in pp_dir.glob("**/forceCoeffs*.dat"):
            coeff_file = candidate
            break

    if coeff_file and coeff_file.exists():
        lines = [l for l in coeff_file.read_text().splitlines()
                 if l.strip() and not l.startswith('#')]
        lines = lines[-avg_n:]
        cls, cds, cms = [], [], []
        for line in lines:
            cols = line.split()
            if len(cols) >= 4:
                try:
                    # Typical columns: time Cm Cd Cl Cl(f) Cl(r)
                    cms.append(float(cols[1]))
                    cds.append(float(cols[2]))
                    cls.append(float(cols[3]))
                except ValueError:
                    pass
        if cls:
            return {"Cl": sum(cls)/len(cls),
                    "Cd": sum(cds)/len(cds),
                    "Cm": sum(cms)/len(cms)}

    # Fallback: parse simpleFoam log for last avg_n force writes
    log = log_path
    if not log.exists():
        return {"Cl": None, "Cd": None, "Cm": None}

    cl_vals, cd_vals, cm_vals = [], [], []
    txt = log.read_text()
    for block in re.finditer(
        r'Cm\s+=\s+([\d.eE+\-]+).*?Cd\s+=\s+([\d.eE+\-]+).*?Cl\s+=\s+([\d.eE+\-]+)',
        txt, re.DOTALL
    ):
        cm_vals.append(float(block.group(1)))
        cd_vals.append(float(block.group(2)))
        cl_vals.append(float(block.group(3)))

    if not cl_vals:
        return {"Cl": None, "Cd": None, "Cm": None}

    cl_vals  = cl_vals[-avg_n:]
    cd_vals  = cd_vals[-avg_n:]
    cm_vals  = cm_vals[-avg_n:]
    return {
        "Cl": sum(cl_vals) / len(cl_vals),
        "Cd": sum(cd_vals) / len(cd_vals),
        "Cm": sum(cm_vals) / len(cm_vals),
    }


def parse_yplus(case_dir: Path) -> float:
    """Parse maximum y+ from yPlus postProcessing output or simpleFoam log."""
    # Try postProcessing/yPlus first
    for yplus_file in (case_dir / "postProcessing").glob("**/yPlus*.dat"):
        lines = [l for l in yplus_file.read_text().splitlines()
                 if l.strip() and not l.startswith('#')]
        if lines:
            # Find last timestep and take max across all patches
            # columns: time  patch  min  max  average
            times = []
            for line in lines:
                cols = line.split()
                if len(cols) >= 4:
                    try:
                        times.append(float(cols[0]))
                    except ValueError:
                        pass
            if not times:
                continue
            last_time = max(times)
            max_val = 0.0
            for line in lines:
                cols = line.split()
                if len(cols) >= 4:
                    try:
                        if float(cols[0]) == last_time:
                            max_val = max(max_val, float(cols[3]))  # col3=max
                    except ValueError:
                        pass
            if max_val > 0:
                return max_val

    # Fallback: parse simpleFoam log
    log = case_dir / "log.simpleFoam"
    if not log.exists():
        return -1.0
    txt = log.read_text()
    vals = re.findall(r'y\+\s*:\s*max\s*=\s*([\d.eE+\-]+)', txt)
    if vals:
        return float(vals[-1])
    return -1.0


def gci_analysis(results: dict) -> dict:
    """
    Compute Grid Convergence Index via Richardson extrapolation.
    Uses CL and CD on M2, M3, M4 (three finest meshes).
    Returns dict with GCI values and whether criterion is met.
    """
    keys = ["M1", "M2", "M3", "M4"]
    n_cells = [results[k]["n_cells"] for k in keys]

    # Grid refinement ratios (use cell count ratio^(1/2) for 2D)
    def r(n1, n2):
        return (n2 / n1) ** 0.5

    gci_results = {}
    for qty in ["Cl", "Cd"]:
        vals = [results[k][qty] for k in keys]
        if any(v is None for v in vals):
            gci_results[qty] = {"gci_fine": None, "gci_medium": None, "met": False}
            continue

        # Use M2, M3, M4 for GCI
        f1, f2, f3 = vals[3], vals[2], vals[1]   # M4=fine, M3=medium, M2=coarse
        n1, n2, n3 = n_cells[3], n_cells[2], n_cells[1]

        r21 = r(n2, n1)   # M4/M3
        r32 = r(n3, n2)   # M3/M2

        # Apparent order of convergence
        try:
            eps21 = f2 - f1
            eps32 = f3 - f2
            if eps21 == 0 or eps32 == 0:
                raise ValueError("Zero difference")
            q = math.log(abs(eps32 / eps21)) / math.log(r21)
            p_ord = abs(q)
        except (ValueError, ZeroDivisionError):
            p_ord = 2.0   # assume second order

        Fs = 1.25   # safety factor

        # GCI for fine grid (M4)
        gci_fine   = Fs * abs(eps21) / (r21**p_ord - 1) / abs(f1) * 100
        # GCI for medium grid (M3)
        gci_medium = Fs * abs(eps32) / (r32**p_ord - 1) / abs(f2) * 100

        # Criterion: variation M3→M4 < 1%
        var_pct = abs(f2 - f1) / abs(f1) * 100 if f1 != 0 else 999

        gci_results[qty] = {
            "order":       round(p_ord, 2),
            "gci_fine":    round(gci_fine, 4),
            "gci_medium":  round(gci_medium, 4),
            "var_M3_M4":   round(var_pct, 4),
            "met":         var_pct < 1.0,
        }

    return gci_results


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_mesh_level(name: str, params: dict, submit: bool) -> dict:
    """Set up, mesh, and solve one refinement level. Returns result dict."""
    case_dir = STUDY_DIR / name
    print(f"\n{'='*60}")
    print(f"  Processing {name}")
    print(f"{'='*60}")

    # ---- 1. Copy snappy case ----
    snappy_dir = case_dir / "snappy"
    snappy_done = (snappy_dir / "log.snappyHexMesh").exists() and \
                  any(snappy_dir.iterdir()) \
                  if snappy_dir.exists() else False
    if not snappy_done:
        if snappy_dir.exists():
            shutil.rmtree(snappy_dir)
        shutil.copytree(SNAPPY_CASE, snappy_dir)
    else:
        print(f"  Snappy already done, skipping mesh generation...")


    # Patch snappyHexMeshDict (always, in case params changed)
    src_snappy = snappy_dir / "system" / "snappyHexMeshDict"
    if not snappy_done:
        patch_snappy(src_snappy, src_snappy, params)
        print(f"  Patched snappyHexMeshDict: {params}")

    # ---- 2. Run blockMesh + snappyHexMesh ----
    if not snappy_done:
        print(f"  Running blockMesh...")
        rc = run_in_container("blockMesh", snappy_dir, snappy_dir / "log.blockMesh")
        if rc != 0:
            print(f"  [ERROR] blockMesh failed (rc={rc})")
            return {"error": "blockMesh failed"}

        print(f"  Running surfaceFeatureExtract...")
        run_in_container("surfaceFeatureExtract", snappy_dir, snappy_dir / "log.surfaceFeatureExtract")

        print(f"  Running snappyHexMesh...")
        rc = run_in_container("snappyHexMesh -overwrite", snappy_dir, snappy_dir / "log.snappyHexMesh")
        if rc != 0:
            print(f"  [ERROR] snappyHexMesh failed (rc={rc})")
            return {"error": "snappyHexMesh failed"}

        run_in_container("checkMesh -latestTime", snappy_dir, snappy_dir / "log.checkMesh")
    n_cells = count_cells(snappy_dir)
    print(f"  Cells: {n_cells:,}")

    # ---- 3. Copy simple case and mesh ----
    simple_dir = case_dir / "simpleFoam"
    if simple_dir.exists():
        shutil.rmtree(simple_dir)
    shutil.copytree(SIMPLE_CASE, simple_dir)
    # Remove postProcessing copied from template
    pp_dir = simple_dir / "postProcessing"
    if pp_dir.exists():
        shutil.rmtree(pp_dir)


    # Copy mesh
    poly_dst = simple_dir / "constant" / "polyMesh"
    poly_dst.mkdir(parents=True, exist_ok=True)

    # Find latest time in snappy
    snappy_times = sorted(
        [d for d in snappy_dir.iterdir() if d.is_dir() and d.name.replace('.','').isdigit()],
        key=lambda d: float(d.name)
    )
    if snappy_times:
        poly_src = snappy_times[-1] / "polyMesh"
    else:
        poly_src = snappy_dir / "constant" / "polyMesh"

    if poly_src.exists():
        shutil.copytree(poly_src, poly_dst, dirs_exist_ok=True)
    else:
        print(f"  [ERROR] polyMesh not found at {poly_src}")
        return {"error": "polyMesh not found"}

    # Clean time dirs and reset 0
    for d in simple_dir.iterdir():
        if d.is_dir() and d.name.replace('.','').isdigit() and d.name != '0':
            shutil.rmtree(d)
    if (simple_dir / "0").exists():
        shutil.rmtree(simple_dir / "0")
    shutil.copytree(simple_dir / "0.orig", simple_dir / "0")

    # ---- 4. extrudeMesh + createPatch (3D snappy → 2D) ----
    # Fix extrudeMeshDict to use absolute path to snappy case
    extrude_dict = simple_dir / "system" / "extrudeMeshDict"
    if extrude_dict.exists():
        txt = extrude_dict.read_text()
        txt = re.sub(
            r'sourceCase\s+"[^"]*"',
            f'sourceCase "{snappy_dir}"',
            txt
        )
        extrude_dict.write_text(txt)

    print(f"  Running extrudeMesh...")
    rc = run_in_container("extrudeMesh", simple_dir, simple_dir / "log.extrudeMesh")
    if rc != 0:
        print(f"  [WARN] extrudeMesh rc={rc}")

    print(f"  Running createPatch...")
    rc = run_in_container("createPatch -overwrite", simple_dir, simple_dir / "log.createPatch")
    if rc != 0:
        print(f"  [WARN] createPatch rc={rc}")

    # Reset 0 after mesh operations
    if (simple_dir / "0").exists():
        shutil.rmtree(simple_dir / "0")
    shutil.copytree(simple_dir / "0.orig", simple_dir / "0")

    # ---- 5. Run simpleFoam in parallel ----
    N_PROCS = 4
    decompose_dict = simple_dir / "system" / "decomposeParDict"
    decompose_dict.write_text("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n    object      decomposeParDict;\n}\nnumberOfSubdomains 16;\nmethod          simple;\nsimpleCoeffs { n (4 4 1); delta 0.001; }\n")
    print(f"  Running decomposePar...")
    run_in_container("decomposePar", simple_dir, simple_dir / "log.decomposePar")
    print(f"  Running simpleFoam -parallel...")
    rc = run_in_container(
        "mpirun --oversubscribe --mca btl_base_warn_component_unused 0 -np 16 simpleFoam -parallel",
        simple_dir, simple_dir / "log.simpleFoam"
    )
    if rc != 0:
        print(f"  [WARN] simpleFoam returned rc={rc} — parsing results anyway")
    print(f"  Running reconstructPar...")
    run_in_container("reconstructPar", simple_dir, simple_dir / "log.reconstructPar")


    # ---- 6. Extract results ----
    coeffs = parse_forces_log(simple_dir / "log.simpleFoam", AVG_LAST_N)
    yplus  = parse_yplus(simple_dir)

    result = {
        "n_cells": n_cells,
        "n_layers": params["layers"],
        "yplus_max": yplus,
        "Cl": coeffs["Cl"],
        "Cd": coeffs["Cd"],
        "Cm": coeffs["Cm"],
        "params": params,
    }
    print(f"  Results: cells={n_cells:,}  y+_max={yplus:.1f}  "
          f"CL={coeffs['Cl']}  CD={coeffs['Cd']}  CM={coeffs['Cm']}")
    return result


def write_report(results: dict, gci: dict, out_path: Path):
    """Write the final text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  GRID CONVERGENCE STUDY — NACA 2312")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Mesh':<6} {'N_cells':>10} {'Layers':>7} {'y+_max':>8} "
                 f"{'CL':>10} {'CD':>10} {'CM':>10}")
    lines.append("-" * 70)
    for name in ["M1", "M2", "M3", "M4"]:
        r = results[name]
        cl = f"{r['Cl']:.6f}" if r['Cl'] is not None else "N/A"
        cd = f"{r['Cd']:.6f}" if r['Cd'] is not None else "N/A"
        cm = f"{r['Cm']:.6f}" if r['Cm'] is not None else "N/A"
        lines.append(f"{name:<6} {r['n_cells']:>10,} {r['n_layers']:>7} "
                     f"{r['yplus_max']:>8.1f} {cl:>10} {cd:>10} {cm:>10}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  GCI ANALYSIS (Richardson Extrapolation, Fs=1.25)")
    lines.append("=" * 70)
    for qty in ["Cl", "Cd"]:
        g = gci.get(qty, {})
        lines.append(f"\n  Quantity: {qty}")
        if g.get("gci_fine") is None:
            lines.append("    Insufficient data for GCI computation.")
            continue
        lines.append(f"    Apparent order p          : {g['order']}")
        lines.append(f"    GCI (fine,   M4)          : {g['gci_fine']:.4f} %")
        lines.append(f"    GCI (medium, M3)          : {g['gci_medium']:.4f} %")
        lines.append(f"    Variation M3→M4           : {g['var_M3_M4']:.4f} %")
        met = "YES ✓" if g["met"] else "NO ✗"
        lines.append(f"    1% criterion met          : {met}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("  SELECTED MESH FOR PRODUCTION")
    lines.append("=" * 70)
    # Find coarsest mesh meeting <1% variation vs M4
    selected = "M4"
    for name in ["M1", "M2", "M3"]:
        r_name = results[name]
        r_fine = results["M4"]
        if (r_name["Cl"] is not None and r_fine["Cl"] is not None and
                abs(r_name["Cl"] - r_fine["Cl"]) / abs(r_fine["Cl"]) * 100 < 1.0 and
                abs(r_name["Cd"] - r_fine["Cd"]) / abs(r_fine["Cd"]) * 100 < 1.0):
            selected = name
            break
    lines.append(f"  Coarsest mesh with <1% variation in CL and CD vs M4: {selected}")
    lines.append(f"  Cells: {results[selected]['n_cells']:,}")
    lines.append(f"  y+_max: {results[selected]['yplus_max']:.1f}")
    lines.append("")

    report = "\n".join(lines)
    out_path.write_text(report)
    print("\n" + report)
    print(f"\nReport saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Grid convergence study for NACA 2312")
    parser.add_argument("--workdir", default="/work/u10677113/NACA2312",
                        help="Base working directory")
    parser.add_argument("--submit", action="store_true",
                        help="Submit via PBS instead of running directly")
    parser.add_argument("--mesh-only", action="store_true",
                        help="Only generate meshes, skip simpleFoam")
    parser.add_argument("--results-json", default=None,
                        help="Load existing results JSON and skip to reporting")
    args = parser.parse_args()

    WORKDIR   = Path(args.workdir)
    STUDY_DIR = WORKDIR / "grid_convergence"
    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    results_file = STUDY_DIR / "results.json"

    # Load existing results if requested
    if args.results_json:
        with open(args.results_json) as f:
            results = json.load(f)
    else:
        results = {}
        for name, params in MESH_LEVELS.items():
            case_dir = STUDY_DIR / name
            # Skip if already completed
            done_flag = case_dir / "simpleFoam" / "log.simpleFoam"
            if done_flag.exists():
                log_txt = done_flag.read_text()
                if "End" in log_txt:
                    print(f"  {name}: already completed, loading results...")
                    simple_dir = case_dir / "simpleFoam"
                    snappy_dir = case_dir / "snappy"
                    coeffs = parse_forces_log(done_flag, AVG_LAST_N)
                    yplus  = parse_yplus(simple_dir)
                    n_cells = count_cells(snappy_dir)
                    results[name] = {
                        "n_cells":  n_cells,
                        "n_layers": params["layers"],
                        "yplus_max": yplus,
                        "Cl": coeffs["Cl"],
                        "Cd": coeffs["Cd"],
                        "Cm": coeffs["Cm"],
                        "params": params,
                    }
                    continue

            result = run_mesh_level(name, params, args.submit)
            results[name] = result

            # Save intermediate results
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

    # Save final results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # GCI analysis
    gci = gci_analysis(results)

    # Write report
    report_path = STUDY_DIR / "grid_convergence_report.txt"
    write_report(results, gci, report_path)


if __name__ == "__main__":
    main()
