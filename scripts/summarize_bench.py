"""Generate GitHub Job Summary markdown from bench JSON results.

Usage:
    python scripts/summarize_bench.py [--results FILE ...] [--comparison FILE ...]

Reads bench results JSON files (from bench -J) and optional comparison_summary
JSON files (from compare_bench.py) and writes a markdown summary to stdout.
Designed to be appended to $GITHUB_STEP_SUMMARY.

Examples:
    # Single platform results
    python scripts/summarize_bench.py --results results-metal.json

    # Results with baseline comparison
    python scripts/summarize_bench.py \
        --results results-metal.json \
        --comparison comparison-metal.json

    # Multiple platforms
    python scripts/summarize_bench.py \
        --results results-metal.json results-cuda.json \
        --comparison comparison-metal.json comparison-cuda.json
"""

import argparse
import json
import sys
from pathlib import Path


def format_time(sec):
    """Format seconds into a human-readable string."""
    if sec >= 1.0:
        return f"{sec:.3f}s"
    elif sec >= 0.001:
        return f"{sec * 1000:.2f}ms"
    else:
        return f"{sec * 1e6:.1f}us"


def infer_platform(filename):
    """Infer platform name from filename."""
    name = Path(filename).stem.lower()
    if "metal" in name:
        return "macOS Metal"
    elif "cuda" in name:
        return "CUDA"
    # Fall back to filename
    return Path(filename).stem


def emit_results_table(results, platform, out):
    """Emit a markdown table of benchmark results."""
    if not results:
        return

    # Group by problem
    problems = {}
    solvers_seen = []
    for r in results:
        prob = r["problem"]
        solver = r["solver"]
        if solver not in solvers_seen:
            solvers_seen.append(solver)
        problems.setdefault(prob, {})[solver] = r

    # Header
    out.append(f"### {platform} Results\n")
    out.append(f"| Problem | Operation | " + " | ".join(f"`{s}`" for s in solvers_seen) + " |")
    out.append("| --- | --- |" + " ---: |" * len(solvers_seen))

    for prob in sorted(problems.keys()):
        solver_data = problems[prob]
        # All entries for this problem should have the same operation
        ops = set()
        for s_data in solver_data.values():
            ops.add(s_data["operation"])

        for op in sorted(ops):
            cells = []
            for solver in solvers_seen:
                entry = solver_data.get(solver)
                if entry and entry["operation"] == op:
                    cells.append(format_time(entry["median_sec"]))
                else:
                    cells.append("-")
            # Shorten problem name: strip common prefix numbers
            prob_short = prob
            out.append(f"| {prob_short} | {op} | " + " | ".join(cells) + " |")

    out.append("")


def emit_comparison_table(comparisons, platform, out):
    """Emit a markdown table showing baseline vs current with delta."""
    if not comparisons:
        return

    threshold = comparisons.get("threshold", 0.20)
    entries = comparisons.get("comparisons", [])
    regressions = comparisons.get("regressions", 0)

    if not entries:
        return

    status = ":x: FAIL" if regressions > 0 else ":white_check_mark: PASS"
    out.append(f"### {platform} vs Baseline ({status})\n")
    out.append("| Problem | Solver | Op | Baseline | Current | Delta |")
    out.append("| --- | --- | --- | ---: | ---: | ---: |")

    for e in entries:
        baseline = format_time(e["baseline_sec"])
        current = format_time(e["current_sec"])
        delta_pct = e["delta_pct"]
        sign = "+" if delta_pct > 0 else ""

        if e.get("regression"):
            delta_str = f"**{sign}{delta_pct:.1f}%** :warning:"
        elif delta_pct < -5:
            delta_str = f"{sign}{delta_pct:.1f}% :rocket:"
        else:
            delta_str = f"{sign}{delta_pct:.1f}%"

        out.append(
            f"| {e['problem']} | `{e['solver']}` | {e['operation']} "
            f"| {baseline} | {current} | {delta_str} |"
        )

    out.append("")
    if regressions > 0:
        out.append(
            f"> :warning: **{regressions} regression(s)** exceeded "
            f"{threshold:.0%} threshold\n"
        )
    else:
        out.append(
            f"> All benchmarks within {threshold:.0%} regression threshold\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", nargs="*", default=[], help="Bench results JSON files"
    )
    parser.add_argument(
        "--comparison", nargs="*", default=[], help="Comparison summary JSON files"
    )
    parser.add_argument(
        "--title",
        default="Performance Benchmark Results",
        help="Summary title",
    )
    args = parser.parse_args()

    out = []
    out.append(f"## {args.title}\n")

    # Emit results tables
    for results_file in args.results:
        try:
            with open(results_file) as f:
                data = json.load(f)
            platform = infer_platform(results_file)
            emit_results_table(data.get("results", []), platform, out)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            out.append(f"> :warning: Could not load `{results_file}`: {e}\n")

    # Emit comparison tables
    for comp_file in args.comparison:
        try:
            with open(comp_file) as f:
                data = json.load(f)
            platform = infer_platform(comp_file)
            emit_comparison_table(data, platform, out)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            out.append(f"> :warning: Could not load `{comp_file}`: {e}\n")

    print("\n".join(out))


if __name__ == "__main__":
    main()
