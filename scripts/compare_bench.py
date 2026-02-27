"""Compare two bench JSON outputs and detect performance regressions.

Usage:
    python scripts/compare_bench.py baseline.json current.json [--threshold 0.20]

Exit codes:
    0 - No regressions exceeding threshold
    1 - One or more regressions exceeding threshold
    2 - Usage / file error
"""

import json
import sys


def load_results(path):
    """Load bench JSON file and return dict keyed by (problem, solver, operation)."""
    with open(path) as f:
        data = json.load(f)
    results = {}
    for entry in data.get("results", []):
        key = (entry["problem"], entry["solver"], entry["operation"])
        results[key] = entry
    return results


def main():
    threshold = 0.20
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--threshold" and i < len(sys.argv) - 1:
            threshold = float(sys.argv[i + 1])

    if len(args) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    baseline_path, current_path = args[0], args[1]

    try:
        baseline = load_results(baseline_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading baseline {baseline_path}: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        current = load_results(current_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading current {current_path}: {e}", file=sys.stderr)
        sys.exit(2)

    # Find matching keys
    common_keys = sorted(set(baseline.keys()) & set(current.keys()))
    if not common_keys:
        print("WARNING: No matching (problem, solver, operation) tuples found.")
        print(f"  Baseline has {len(baseline)} entries, current has {len(current)} entries.")
        sys.exit(0)

    # Compare
    regressions = 0
    rows = []
    for key in common_keys:
        b_median = baseline[key]["median_sec"]
        c_median = current[key]["median_sec"]
        if b_median <= 0:
            delta_pct = 0.0
        else:
            delta_pct = (c_median - b_median) / b_median
        is_regression = delta_pct > threshold
        if is_regression:
            regressions += 1
        rows.append((*key, b_median, c_median, delta_pct, is_regression))

    # Print summary table
    print(f"\nPerformance Comparison (threshold: {threshold:.0%})")
    hdr = f"{'Problem':<45} | {'Solver':<25} | {'Op':<10} | {'Baseline':>10} | {'Current':>10} | {'Delta':>10}"
    print(hdr)
    print("-" * len(hdr))
    for problem, solver, op, b, c, delta, is_reg in rows:
        # Truncate long problem names
        prob_short = problem[:44] if len(problem) > 44 else problem
        flag = " REGRESSION" if is_reg else ""
        print(
            f"{prob_short:<45} | {solver:<25} | {op:<10} | "
            f"{b:>9.4f}s | {c:>9.4f}s | {delta:>+8.1%}{flag}"
        )

    print()
    if regressions > 0:
        print(f"RESULT: FAIL -- {regressions} regression(s) exceeded {threshold:.0%} threshold")
    else:
        print(f"RESULT: PASS -- no regressions exceeded {threshold:.0%} threshold")

    # Write JSON summary for artifact upload
    summary = {
        "threshold": threshold,
        "regressions": regressions,
        "comparisons": [
            {
                "problem": p,
                "solver": s,
                "operation": o,
                "baseline_sec": b,
                "current_sec": c,
                "delta_pct": round(d * 100, 2),
                "regression": r,
            }
            for p, s, o, b, c, d, r in rows
        ],
    }
    with open("comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    sys.exit(1 if regressions > 0 else 0)


if __name__ == "__main__":
    main()
