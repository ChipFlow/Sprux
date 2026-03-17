# Sprux Benchmarks

Sprux provides three benchmark tools for evaluating solver performance across different
problem types, backends, and hardware.

## Tools Overview

| Tool | Binary | Purpose |
|------|--------|---------|
| `bench` | `build/sprux/benchmarking/bench` | Cholesky on synthetic problems (FLAT, GRID, MERI) |
| `BAL_bench` | `build/sprux/benchmarking/BAL_bench` | Bundle Adjustment in the Large |
| `lu_bench` | `build/sprux/benchmarking/lu_bench` | LU on real circuit Jacobians (sequences) |

## bench — Cholesky Benchmarks

Benchmarks Cholesky factorization on synthetic SPD problems with varying structure.

### Problem Types

| Type | Structure | Use Case |
|------|-----------|----------|
| FLAT | Flat elimination tree | Worst case for supernodal |
| GRID | 2D grid connectivity | Typical FEM/FVM problems |
| MERI | 3D meridian connectivity | Dense supernodes |

### Basic Usage

```bash
# Run all problems, compare with CHOLMOD baseline
build/sprux/benchmarking/bench -B 1_CHOLMOD

# Run specific problem types
build/sprux/benchmarking/bench -P GRID

# Factor operation only (default)
build/sprux/benchmarking/bench -B 1_CHOLMOD -O factor
```

### Collecting Timing Statistics

For fitting a computation model to your hardware:

```bash
# Collect per-operation timings
build/sprux/benchmarking/bench -B 1_CHOLMOD -Z

# This generates CSV files:
#   stats_cpu_f64_potrf.csv
#   stats_cpu_f64_trsm.csv
#   stats_cpu_f64_syge.csv
#   stats_cpu_f64_asmbl.csv

# Fit a computation model
build/sprux/examples/opt_comp_model \
  -p stats_cpu_f64_potrf.csv \
  -a stats_cpu_f64_asmbl.csv \
  -t stats_cpu_f64_trsm.csv \
  -g stats_cpu_f64_syge.csv
```

### Command-Line Options

Run `build/sprux/benchmarking/bench -h` for all options. Key flags:
- `-B <baseline>`: Baseline solver (e.g., `1_CHOLMOD`)
- `-P <pattern>`: Problem filter (regex)
- `-O <operation>`: Operation to benchmark (`factor`, `analysis`, `solve-X`)
- `-Z`: Collect detailed timing statistics

## BAL_bench — Bundle Adjustment

Benchmarks on problems from [Bundle Adjustment in the Large](https://grail.cs.washington.edu/projects/bal/).

```bash
# Download a BAL problem
wget https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-49-7776-pre.txt.bz2
bunzip2 problem-49-7776-pre.txt.bz2

# Run benchmark
build/sprux/benchmarking/BAL_bench -i problem-49-7776-pre.txt
```

Tests both:
- Point elimination + reduced camera-camera solve (Sprux)
- Direct solve on reduced problem (CHOLMOD baseline, if installed)

## lu_bench — LU Benchmarks

Benchmarks LU factorization with partial pivoting on real circuit Jacobian sequences.
Supports multiple backends and outputs JSON for CI regression tracking.

### Backend Selection

```bash
# CPU (double precision)
build/sprux/benchmarking/lu_bench -d test_data/c6288_sequence -b CPU

# Metal (float + iterative refinement)
build/sprux/benchmarking/lu_bench -d test_data/c6288_sequence -b Metal_Sparse

# Metal dense baseline (Accelerate sgetrf/sgetrs)
build/sprux/benchmarking/lu_bench -d test_data/c6288_sequence -b Metal_Dense

# CUDA (double precision)
build/sprux/benchmarking/lu_bench -d test_data/c6288_sequence -b CUDA

# cuDSS (NVIDIA's sparse direct solver)
build/sprux/benchmarking/lu_bench -d test_data/c6288_sequence -b cuDSS
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-d <dir>` | Directory with jacobian/rhs .mtx files | required |
| `-b <backend>` | Backend: `CPU`, `Metal_Sparse`, `Metal_Dense`, `CUDA`, `cuDSS` | `CPU` |
| `-n <count>` | Number of matrices to process | all |
| `-M <iters>` | Iterative refinement iterations (Metal) | 7 |
| `--json` | Output JSON (for CI) | off |

### JSON Output

```bash
build/sprux/benchmarking/lu_bench -d test_data/c6288_sequence -b Metal_Sparse --json
```

Output format:
```json
{
  "backend": "Metal_Sparse",
  "problem": "c6288_sequence",
  "matrices": 20,
  "symbolic_ms": 45.2,
  "avg_factor_ms": 7.3,
  "avg_solve_ms": 5.0,
  "avg_total_ms": 12.3,
  "avg_residual": 1.2e-7
}
```

## Test Data

### c6288_sequence

20 Jacobian matrices from transient simulation of c6288 multiplier circuit.
- Size: 25,380 × 25,380, ~95,814 nnz each
- Source: vajax circuit simulator (t_stop=2ns, dt=2ps)
- Properties: general (non-symmetric), same sparsity pattern, different values
- Location: `test_data/c6288_sequence/`

### ring_sequence (ring oscillator)

50 tiny Jacobian matrices from ring oscillator simulation.
- Size: 47 × 47, ~189 nnz each
- Location: `test_data/LU_ring_jacobian/`
- Good for: quick functional testing

### mul64

50 large Jacobian matrices from 64-bit multiplier circuit.
- Size: 666,118 × 666,118
- Location: `test_data/mul64/` (xz-compressed, auto-decompressed by lu_bench)
- Good for: large-scale performance evaluation

### tb_dp

50 Jacobian matrices from differential pair testbench.
- Location: `test_data/tb_dp/`

## CI Performance Regression

The project includes GitHub Actions workflows for performance tracking:

- **`perf-regression.yml`**: Runs `lu_bench` with `--json` on each PR, compares against
  cached baselines per platform (macOS Metal, Linux CUDA).
- Baselines stored in GitHub Actions cache, keyed by platform and backend.
- Regression threshold: >20% slowdown triggers a warning.

### Running Locally

```bash
# Build with Metal
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSPRUX_USE_CUBLAS=0 -DSPRUX_USE_METAL=1 -DBLA_VENDOR=Apple
cmake --build build -j16

# Run benchmark and save baseline
build/sprux/benchmarking/lu_bench -d test_data/c6288_sequence -b Metal_Sparse --json \
  > baseline.json

# After changes, compare
build/sprux/benchmarking/lu_bench -d test_data/c6288_sequence -b Metal_Sparse --json \
  > current.json
# Compare avg_factor_ms and avg_solve_ms
```
