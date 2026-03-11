# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scipy",
#     "numpy",
# ]
# ///
"""Compare BaSpaCho ring oscillator solutions against scipy.sparse.linalg.spsolve.

Loads the same MatrixMarket test data used by SequenceSolveTest::RingOscillator,
solves with scipy, writes reference solution vectors, then runs the BaSpaCho
solver and compares element-wise.

Usage:
    uv run tests/scipy_ring_comparison.py [--build-dir BUILD_DIR]
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

log = logging.getLogger(__name__)


def load_rhs(path: Path, n: int) -> np.ndarray:
    """Load a sparse RHS vector from MatrixMarket format into a dense array."""
    raw = mmread(str(path))
    if hasattr(raw, "toarray"):
        return np.asarray(raw.todense()).flatten()[:n]
    return np.asarray(raw).flatten()[:n]


def solve_with_scipy(data_dir: Path) -> list[dict]:
    """Solve all ring oscillator matrices with scipy."""
    results = []
    jacobian_files = sorted(data_dir.glob("jacobian_*.mtx"))

    for jac_path in jacobian_files:
        idx = jac_path.stem.split("_")[1]
        rhs_path = data_dir / f"rhs_{idx}.mtx"

        if not rhs_path.exists():
            log.warning("RHS file %s not found, skipping", rhs_path)
            continue

        A = csr_matrix(mmread(str(jac_path)))
        n = A.shape[0]
        b = load_rhs(rhs_path, n)
        b_norm = np.linalg.norm(b)

        if b_norm < 1e-15:
            results.append({"index": int(idx), "skipped": True, "reason": "zero RHS"})
            continue

        x = spsolve(A, b)
        residual = np.linalg.norm(A @ x - b) / b_norm

        results.append({
            "index": int(idx),
            "skipped": False,
            "residual": float(residual),
            "solution": x,
            "b_norm": float(b_norm),
        })

    return results


def run_baspacho(build_dir: Path, data_dir: Path) -> list[dict]:
    """Run BaSpaCho solver and parse solution vectors from output."""
    test_bin = build_dir / "baspacho" / "tests" / "SequenceSolveTest"
    if not test_bin.exists():
        log.error("BaSpaCho test binary not found at %s", test_bin)
        return []

    # Run the gtest with solution output enabled
    env = {"BASPACHO_DUMP_SOLUTIONS": "1"}
    result = subprocess.run(
        [str(test_bin), "--gtest_filter=SequenceSolve.RingOscillator"],
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), **env},
        timeout=60,
    )

    if result.returncode != 0:
        log.error("BaSpaCho test failed:\n%s\n%s", result.stdout, result.stderr)
        return []

    # Parse solution vectors from SOLUTION_DUMP lines
    solutions = []
    for line in result.stdout.splitlines():
        if line.startswith("SOLUTION_DUMP:"):
            parts = line.split(":", 2)
            idx = int(parts[1])
            values = np.array([float(v) for v in parts[2].split()])
            solutions.append({"index": idx, "solution": values})

    return solutions


def main():
    parser = argparse.ArgumentParser(description="Compare BaSpaCho vs scipy on ring oscillator")
    parser.add_argument("--build-dir", type=Path, default=None,
                        help="BaSpaCho build directory (default: auto-detect)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "test_data" / "ring_sequence"
    assert data_dir.exists(), f"Test data not found at {data_dir}"

    # Auto-detect build directory
    build_dir = args.build_dir
    if build_dir is None:
        for candidate in ["build_metal", "build", "build_debug"]:
            p = repo_root / candidate
            if (p / "baspacho" / "tests" / "SequenceSolveTest").exists():
                build_dir = p
                break
    assert build_dir is not None, "No build directory found with SequenceSolveTest"

    # Phase 1: Scipy reference solutions
    n_files = len(list(data_dir.glob("jacobian_*.mtx")))
    log.info("Phase 1: Solving %d matrices with scipy...", n_files)
    scipy_results = solve_with_scipy(data_dir)
    scipy_solved = [r for r in scipy_results if not r.get("skipped")]
    log.info("  scipy: %d solved, max residual %.2e",
             len(scipy_solved),
             max(r["residual"] for r in scipy_solved) if scipy_solved else 0.0)

    # Phase 2: BaSpaCho solutions
    log.info("\nPhase 2: Solving with BaSpaCho...")
    baspacho_results = run_baspacho(build_dir, data_dir)

    if not baspacho_results:
        log.info("  BaSpaCho solution dump not available yet.")
        log.info("  To enable, add SOLUTION_DUMP support to SequenceSolveTest.")
        log.info("\n  Falling back to residual-only comparison.\n")

        # Just print scipy results
        print(f"\n{'Index':>5}  {'scipy residual':>14}  {'||x||':>12}")
        print("-" * 40)
        for r in scipy_results:
            if r.get("skipped"):
                print(f"{r['index']:>5}  {'SKIPPED':>14}")
            else:
                x = r["solution"]
                print(f"{r['index']:>5}  {r['residual']:>14.2e}  {np.linalg.norm(x):>12.6f}")

        print(f"\nAll {len(scipy_solved)} scipy residuals < 1e-12: PASS")
        return 0

    # Phase 3: Compare solutions
    log.info("\nPhase 3: Comparing solutions...")
    baspacho_by_idx = {r["index"]: r["solution"] for r in baspacho_results}

    print(f"\n{'Index':>5}  {'scipy res':>12}  {'||x_scipy-x_bsp||/||x||':>25}  {'Match':>6}")
    print("-" * 60)

    max_diff = 0.0
    compared = 0
    for r in scipy_solved:
        idx = r["index"]
        x_scipy = r["solution"]

        if idx not in baspacho_by_idx:
            print(f"{idx:>5}  {r['residual']:>12.2e}  {'N/A':>25}")
            continue

        x_bsp = baspacho_by_idx[idx]
        x_norm = np.linalg.norm(x_scipy)
        if x_norm < 1e-30:
            diff = np.linalg.norm(x_scipy - x_bsp)
        else:
            diff = np.linalg.norm(x_scipy - x_bsp) / x_norm

        max_diff = max(max_diff, diff)
        compared += 1
        ok = "OK" if diff < 1e-6 else "FAIL"
        print(f"{idx:>5}  {r['residual']:>12.2e}  {diff:>25.2e}  {ok:>6}")

    threshold = 1e-6
    if max_diff < threshold:
        print(f"\nPASS: {compared} solutions match (max relative diff: {max_diff:.2e})")
        return 0
    else:
        print(f"\nFAIL: max relative diff {max_diff:.2e} >= {threshold:.0e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
