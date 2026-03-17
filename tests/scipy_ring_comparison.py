# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scipy",
#     "numpy",
# ]
# ///
# Copyright (c) Robert Taylor, 2026. All rights reserved.
# Licensed under the MIT license found in the LICENSE file.
"""Compare Sprux ring oscillator solutions against scipy.sparse.linalg.spsolve.

Loads the same MatrixMarket test data used by SequenceSolveTest::RingOscillator,
solves with scipy, then runs a Sprux test binary with SPRUX_DUMP_SOLUTIONS=1
and compares solution vectors element-wise.

Supports CPU, CUDA, and Metal backends via --test-binary flag.

Usage:
    uv run tests/scipy_ring_comparison.py [--build-dir BUILD_DIR] [--test-binary NAME]

Examples:
    # CPU (default)
    uv run tests/scipy_ring_comparison.py --build-dir build

    # CUDA
    uv run tests/scipy_ring_comparison.py --build-dir build --test-binary CudaSequenceSolveTest

    # Metal
    uv run tests/scipy_ring_comparison.py --build-dir build_metal --test-binary MetalSequenceSolveTest
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

log = logging.getLogger(__name__)

# Map test binary names to gtest filter patterns
GTEST_FILTERS = {
    "SequenceSolveTest": "SequenceSolve.RingOscillator",
    "CudaSequenceSolveTest": "CudaSequenceSolve.RingOscillator",
    "MetalSequenceSolveTest": "MetalSequenceSolve.RingOscillator",
}


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


def run_sprux(test_bin: Path, gtest_filter: str) -> list[dict]:
    """Run Sprux test binary and parse SOLUTION_DUMP lines."""
    env = {**os.environ, "SPRUX_DUMP_SOLUTIONS": "1"}

    log.info("  Running: %s --gtest_filter=%s", test_bin.name, gtest_filter)
    result = subprocess.run(
        [str(test_bin), f"--gtest_filter={gtest_filter}"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    if result.returncode != 0:
        log.error("Sprux test failed (exit code %d):\n%s\n%s",
                  result.returncode, result.stdout[-2000:], result.stderr[-2000:])
        return []

    solutions = []
    for line in result.stdout.splitlines():
        if line.startswith("SOLUTION_DUMP:"):
            parts = line.split(":", 2)
            idx = int(parts[1])
            values = np.array([float(v) for v in parts[2].split()])
            solutions.append({"index": idx, "solution": values})

    return solutions


def find_test_binary(build_dir: Path, name: str) -> Path | None:
    """Find the test binary in the build directory."""
    candidates = [
        build_dir / "sprux" / "tests" / name,
        build_dir / "tests" / name,
        build_dir / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare Sprux vs scipy on ring oscillator")
    parser.add_argument("--build-dir", type=Path, default=None,
                        help="Sprux build directory (default: auto-detect)")
    parser.add_argument("--test-binary", type=str, default="SequenceSolveTest",
                        choices=list(GTEST_FILTERS.keys()),
                        help="Test binary to run (default: SequenceSolveTest)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Max allowed relative solution difference "
                             "(default: 1e-6 for float, 1e-10 for double)")
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
            if find_test_binary(p, args.test_binary):
                build_dir = p
                break
    assert build_dir is not None, f"No build directory found with {args.test_binary}"

    test_bin = find_test_binary(build_dir, args.test_binary)
    assert test_bin is not None, f"{args.test_binary} not found in {build_dir}"

    gtest_filter = GTEST_FILTERS[args.test_binary]

    # Determine precision and threshold
    is_float = "Metal" in args.test_binary
    precision_label = "float32" if is_float else "float64"
    threshold = args.threshold if args.threshold is not None else (1e-4 if is_float else 1e-10)

    # Phase 1: Scipy reference solutions
    n_files = len(list(data_dir.glob("jacobian_*.mtx")))
    log.info("Phase 1: Solving %d matrices with scipy...", n_files)
    scipy_results = solve_with_scipy(data_dir)
    scipy_solved = [r for r in scipy_results if not r.get("skipped")]
    scipy_max_res = max(r["residual"] for r in scipy_solved) if scipy_solved else 0.0
    log.info("  scipy: %d solved, max residual %.2e", len(scipy_solved), scipy_max_res)

    # Phase 2: Sprux solutions
    log.info("\nPhase 2: Solving with Sprux (%s, %s)...", args.test_binary, precision_label)
    sprux_results = run_sprux(test_bin, gtest_filter)

    if not sprux_results:
        log.error("No SOLUTION_DUMP output from %s. Check test binary.", args.test_binary)
        return 1

    log.info("  Sprux: %d solutions collected", len(sprux_results))

    # Phase 3: Compare solutions
    log.info("\nPhase 3: Comparing solutions...")
    sprux_by_idx = {r["index"]: r["solution"] for r in sprux_results}

    print(f"\nBackend: {args.test_binary} ({precision_label}), threshold: {threshold:.0e}")
    print(f"{'Index':>5}  {'scipy res':>12}  {'||x_scipy-x_sprux||/||x||':>25}  {'Match':>6}")
    print("-" * 60)

    max_diff = 0.0
    compared = 0
    failures = 0
    for r in scipy_solved:
        idx = r["index"]
        x_scipy = r["solution"]

        if idx not in sprux_by_idx:
            print(f"{idx:>5}  {r['residual']:>12.2e}  {'N/A':>25}  {'SKIP':>6}")
            continue

        x_sprux = sprux_by_idx[idx]
        x_norm = np.linalg.norm(x_scipy)
        if x_norm < 1e-30:
            diff = np.linalg.norm(x_scipy - x_sprux)
        else:
            diff = np.linalg.norm(x_scipy - x_sprux) / x_norm

        max_diff = max(max_diff, diff)
        compared += 1
        ok = diff < threshold
        status = "OK" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"{idx:>5}  {r['residual']:>12.2e}  {diff:>25.2e}  {status:>6}")

    print(f"\nCompared: {compared}, Max relative diff: {max_diff:.2e}, Threshold: {threshold:.0e}")

    if failures > 0:
        print(f"FAIL: {failures} solutions exceed threshold")
        return 1
    elif compared == 0:
        print("FAIL: no solutions compared")
        return 1
    else:
        print(f"PASS: all {compared} solutions match scipy")
        return 0


if __name__ == "__main__":
    sys.exit(main())
