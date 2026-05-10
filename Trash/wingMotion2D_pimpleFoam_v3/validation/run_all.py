"""
run_all.py — Master validation script.

Runs all validation checks and prints a PASS/FAIL summary.
Exit code: 0 if all pass, 1 if any fail.

Usage:
    cd wingMotion2D_pimpleFoam/validation
    python run_all.py
"""

import sys
import time
from pathlib import Path

# Ensure figures dir exists
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODULES = [
    ("forced_validation", "Forced harmonic response vs analytical"),
    ("free_vibration",    "Free vibration: energy conservation + log-decrement"),
    ("convergence",       "Timestep convergence study (RK45 order)"),
    ("flutter_check",     "Linear flutter analysis (quasi-steady, informational)"),
]

WIDTH = 52


def _separator():
    print("─" * (WIDTH + 30))


def main():
    print()
    print("=" * (WIDTH + 30))
    print("  Structural integrator validation — wingMotion2D_pimpleFoam")
    print("=" * (WIDTH + 30))

    results = []
    for mod_name, description in MODULES:
        print(f"\n[{mod_name}]  {description}")
        _separator()
        t0 = time.time()
        try:
            mod = __import__(mod_name)
            passed, msg = mod.run()
        except Exception as exc:
            passed = False
            msg = f"EXCEPTION: {exc}"
            import traceback
            traceback.print_exc()
        elapsed = time.time() - t0
        results.append((mod_name, passed, msg, elapsed))

    _separator()
    print("\nSUMMARY")
    _separator()
    all_pass = True
    for mod_name, passed, msg, elapsed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {status:4s}  {mod_name:<30s}  ({elapsed:.1f}s)")
        print(f"        {msg}")
    _separator()
    print(f"\n  Figures saved to: {FIG_DIR}")
    overall = "ALL PASS" if all_pass else "SOME FAILED"
    print(f"  Overall result:   {overall}")
    print()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
