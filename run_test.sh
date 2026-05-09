#!/bin/bash
# run_test.sh — Generate inputs and run the imposed-motion test case
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="$SCRIPT_DIR/testCase_imposed"
NCORES=4

# ── Source OpenFOAM ────────────────────────────────────────────────────────────
source /opt/openfoam7/etc/bashrc

echo "=== Generating input time series ==="
python3 "$SCRIPT_DIR/generate_inputs.py"

echo ""
echo "=== Checking mesh ==="
cd "$CASE_DIR"
checkMesh -allGeometry -allTopology > checkMesh.log 2>&1
grep -E "Mesh OK|FAILED|cells:" checkMesh.log || true
echo "  Full log: testCase_imposed/checkMesh.log"

echo ""
echo "=== Decomposing for $NCORES cores ==="
decomposePar -force > log.decomposePar 2>&1
echo "  Done."

echo ""
echo "=== Copying .dat files to processor directories ==="
for d in "$CASE_DIR"/processor*/constant; do
    cp "$CASE_DIR"/constant/*.dat "$d/"
done
echo "  Done."

echo ""
echo "=== Running pimpleFoam (parallel, $NCORES cores) ==="
mpirun -np $NCORES pimpleFoam -parallel > log.pimpleFoam 2>&1 &
SOLVER_PID=$!
echo "  Solver PID: $SOLVER_PID"
echo "  Monitoring residuals (Ctrl+C to stop monitoring, solver keeps running)..."
echo ""

# Monitor until solver finishes
tail -f log.pimpleFoam | grep --line-buffered -E "Time|Solving|PIMPLE:|Courant" &
TAIL_PID=$!

wait $SOLVER_PID
SOLVER_EXIT=$?
kill $TAIL_PID 2>/dev/null || true

echo ""
if [ $SOLVER_EXIT -eq 0 ]; then
    echo "=== Solver finished successfully ==="
else
    echo "=== Solver exited with code $SOLVER_EXIT — check log.pimpleFoam ==="
    echo "  Last 20 lines of log:"
    tail -20 log.pimpleFoam
    exit 1
fi

echo ""
echo "=== Reconstructing parallel case ==="
reconstructPar > log.reconstructPar 2>&1
echo "  Done."

echo ""
echo "=== Checking convergence ==="
python3 "$SCRIPT_DIR/check_results.py"

echo ""
echo "=== All done. ==="
echo "  Results:  $CASE_DIR/postProcessing/"
echo "  Log:      $CASE_DIR/log.pimpleFoam"
