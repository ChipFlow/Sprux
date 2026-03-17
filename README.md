<p align="center">
    <a href="https://github.com/chipflow/sprux/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" height="20">
    </a>
    <a href="https://isocpp.org/">
        <img src="https://img.shields.io/badge/C++-17-blue.svg" alt="C++" height="20">
    </a>
    <a href="https://github.com/pre-commit/pre-commit">
        <img src="https://img.shields.io/badge/pre--commit-enabled-green?logo=pre-commit&logoColor=white" alt="pre-commit" height="20">
    </a>
</p>

# Sprux

Sprux is a high-performance sparse direct solver with GPU acceleration.

*Formerly BaSpaCho (Batched Sparse Cholesky). The C++ namespace remains `BaSpaCho` for backward compatibility.*

## Features

- **Cholesky** (SPD), **LU with partial pivoting** (general), **LDL^T** (symmetric indefinite)
- **GPU backends**: CUDA (NVIDIA), Metal (Apple Silicon), OpenCL (experimental)
- **CPU backends**: OpenBLAS, Intel MKL, Apple Accelerate
- Supernodal sparse elimination with **level-set parallelism**
- **Preprocessing pipeline**: BTF max transversal, equilibration, static pivoting
- **External encoder API** for GPU pipeline embedding (IREE, XLA custom-calls)
- **Mixed-precision iterative refinement** (float GPU factor + double CPU accumulation)
- **Block-structured matrices** with partial factor/solve for marginal computation
- **Python bindings** via pybind11

## Quick Start

```bash
# Configure (CPU with OpenBLAS, no GPU)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPRUX_USE_CUBLAS=0

# Build
cmake --build build -j16

# Test
ctest --test-dir build
```

For Metal (Apple Silicon):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSPRUX_USE_CUBLAS=0 -DSPRUX_USE_METAL=1 -DBLA_VENDOR=Apple
cmake --build build -j16
```

For CUDA (NVIDIA):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j16
```

## Backend Selection

| Backend | Flag | Precision | GPU | Best For |
|---------|------|-----------|-----|----------|
| CPU BLAS | `-DSPRUX_USE_BLAS=1` (default) | float/double | No | General use, double precision |
| CUDA | `-DSPRUX_USE_CUBLAS=1` (default) | float/double | NVIDIA | Large problems, double precision on GPU |
| Metal | `-DSPRUX_USE_METAL=1` | float only | Apple Silicon | macOS, mixed-precision with refinement |
| OpenCL | `-DSPRUX_USE_OPENCL=1` | float/double | Any | Experimental portable GPU |

Runtime selection:
```cpp
Settings settings;
settings.backend = BackendAuto;  // Auto-detect: CUDA > Metal > OpenCL > CPU
auto solver = createSolver(settings, paramSize, structure);
```

## Build Options

| CMake Option | Default | Description |
|-------------|---------|-------------|
| `SPRUX_USE_CUBLAS` | ON | Enable CUDA support |
| `SPRUX_USE_METAL` | OFF | Enable Metal support (macOS only) |
| `SPRUX_USE_OPENCL` | OFF | Enable OpenCL + CLBlast |
| `SPRUX_USE_BLAS` | ON | Enable CPU BLAS |
| `SPRUX_CUDA_ARCHS` | "detect" | CUDA architectures ("detect", "torch", or "60;70;75") |
| `SPRUX_USE_SUITESPARSE_AMD` | OFF | Use SuiteSparse AMD instead of Eigen |
| `SPRUX_BUILD_TESTS` | ON | Build unit tests |
| `SPRUX_BUILD_EXAMPLES` | ON | Build examples and benchmarks |
| `BLA_VENDOR` | (auto) | BLAS: ATLAS, OpenBLAS, Intel10_64lp_seq, Apple |

## Usage

### Cholesky (SPD)

```cpp
#include "sprux/sprux/Solver.h"
using namespace BaSpaCho;

Settings settings;
settings.backend = BackendFast;
auto solver = createSolver(settings, paramSize, sparseStructure);

solver->factor(data.data());
solver->solve(data.data(), rhs.data(), n, 1);
```

### LU (General Matrices)

```cpp
Settings settings;
settings.backend = BackendMetal;
settings.matrixType = MTYPE_GENERAL;
settings.staticPivotThreshold = 0.0;  // auto
auto solver = createSolver(settings, paramSize, sparseStructure);

std::vector<int64_t> pivots(solver->numSpans());
solver->factorLU(data.data(), pivots.data());
solver->solveLU(data.data(), pivots.data(), rhs.data(), n, 1);
```

### LDL^T (Symmetric Indefinite)

```cpp
solver->factorLDLT(data.data());
solver->solveLDLT(data.data(), rhs.data(), n, 1);
```

See [docs/api-guide.md](docs/api-guide.md) for detailed usage including Metal embedding,
persistent contexts, preprocessing pipeline, and Python bindings.

## Benchmarks

```bash
# Cholesky — compare with CHOLMOD baseline
build/sprux/benchmarking/bench -B 1_CHOLMOD

# Bundle Adjustment in the Large
build/sprux/benchmarking/BAL_bench -i ~/BAL/problem-871-527480-pre.txt

# LU — circuit Jacobians with Metal/CUDA/CPU backends
build/sprux/benchmarking/lu_bench -d test_data/c6288_sequence -b Metal_Sparse
```

See [docs/benchmarks.md](docs/benchmarks.md) for benchmark tools, test data, and CI setup.

## Architecture

The solver pipeline: **symbolic analysis** (AMD ordering, supernode merging, level-set scheduling) → **numeric factorization** (GPU sparse elimination + CPU/GPU dense BLAS) → **solve** (forward/backward substitution).

Key design decisions:
- **Hybrid GPU/CPU execution**: sparse elimination runs on GPU; dense operations use CPU BLAS
  for small blocks (exploiting Apple Silicon unified memory or cheap D↔H copies on CUDA)
- **Level-set parallelism**: independent eliminations are batched into single GPU dispatches
- **External encoder API**: factor and solve operations encode into a caller-provided
  Metal command encoder for zero-overhead pipeline fusion

See [docs/architecture.md](docs/architecture.md) for full details on data structures,
backend design, and memory management.

## Python Bindings

```python
import sprux
solver = sprux.create_solver(param_sizes, row_ptrs, col_inds,
                                matrix_type="general", backend="metal")
solver.factor_lu(data, pivots)
solver.solve_lu(data, pivots, rhs)
```

Build with: `cmake -DSPRUX_BUILD_PYTHON=ON` (requires pybind11).

## Examples

- **`Optimizer.h`** — Levenberg-Marquardt optimizer with direct and mixed direct/iterative solvers
  - `OptimizeSimple.cpp` — spring-connected points
  - `OptimizeBaAtLarge.cpp` — bundle adjustment from BAL
  - `OptimizeCompModel.cpp` — fit BLAS computation model to hardware timings
- **`PCG_Sample.cpp`** — partial elimination + preconditioned conjugate gradient

## Dependencies

Fetched automatically by CMake:
- Eigen 3.4.0, GoogleTest, dispenso (multithreading), SuiteSparse BTF
- Sophus (BA examples only)

Optional:
- CUDA Toolkit 10.2+ (arch ≥ 60 for double atomics)
- CHOLMOD (SuiteSparse) — benchmarking baseline
- OpenCL 1.2+ & CLBlast — OpenCL backend
- pybind11 — Python bindings

## Caveats

- **Block structure**: the library works with block-structured matrices. Purely scalar
  matrices (all 1×1 blocks) work but won't benefit from supernodal BLAS. Best performance
  with parameter blocks of size 1–12.
- **CUDA determinism**: sparse elimination uses `atomicAdd` on GPU, making CUDA results
  non-deterministic by default. Use the two-phase deterministic elimination option if needed.
  CUDA architecture ≥ 6.0 required for double-precision `atomicAdd`.
- **Metal precision**: float only. Use `BackendFast` or `BackendCuda` for double precision.
  Mixed-precision iterative refinement recovers double-precision accuracy on Metal.
- **Ordering**: only AMD (Approximate Minimum Degree) reordering is supported.

## License

MIT — see [LICENSE](LICENSE).

Original BaSpaCho: Copyright (c) Meta Platforms, Inc. and affiliates.
Sprux extensions: Copyright (c) Robert Taylor, 2026.
