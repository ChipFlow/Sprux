# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sprux (formerly BaSpaCho — Batched Sparse Cholesky) is a high-performance sparse direct solver with GPU acceleration. It supports:

- **Cholesky** (SPD), **LU with partial pivoting** (general), **LDL^T** (symmetric indefinite)
- GPU backends: **CUDA** (NVIDIA), **Metal** (Apple Silicon), **OpenCL** (experimental)
- Supernodal sparse elimination with level-set parallelism
- Preprocessing: BTF max transversal, equilibration, static pivoting
- External encoder API for GPU pipeline embedding (IREE, XLA custom-calls)
- Mixed-precision iterative refinement (float factor + double accumulation)
- Block-structured matrices with partial factor/solve for marginals

The C++ namespace is `Sprux`.

## Build Commands

**Configure (CPU-only, using OpenBLAS):**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPRUX_USE_CUBLAS=0
```

**Configure (with CUDA):**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

**Configure (with Intel MKL):**
```bash
. /opt/intel/oneapi/setvars.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=Intel10_64lp
```

**Configure (with Apple Metal, macOS only):**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPRUX_USE_CUBLAS=0 -DSPRUX_USE_METAL=1 -DBLA_VENDOR=Apple
```

**Build:**
```bash
cmake --build build -- -j16
```

**Run all tests:**
```bash
ctest --test-dir build
```

**List available tests:**
```bash
ctest --test-dir build --show-only
```

**Run a single test:**
```bash
ctest --test-dir build -R <test_name>
```

**Using pixi (alternative):**
```bash
pixi run prepare       # Configure without CUDA
pixi run build         # Build
pixi run test          # Run tests
pixi run build_and_test # Full workflow
```

## Code Style

- C++17 standard
- Google style base with modifications (see `.clang-format`)
- Column limit: 100 characters
- Pointer alignment: left (`int* ptr` not `int *ptr`)
- Format with: `clang-format -i <file>`
- Pre-commit hook runs clang-format automatically

## Architecture

See [docs/architecture.md](docs/architecture.md) for full details.

### Solver Pipeline

```
Input (CSR + param sizes) → Symbolic Analysis → Numeric Factorization → Solve
```

1. **Symbolic analysis** (`createSolver()`): AMD ordering, supernode detection, level-set scheduling, factor storage allocation
2. **Numeric factorization** (`factor()` / `factorLU()` / `factorLDLT()`): sparse elimination on GPU (level-set parallel kernels), then dense loop (BLAS)
3. **Solve** (`solve()` / `solveLU()` / `solveLDLT()`): forward/backward substitution with optional pivot application

### Core Data Structures

**SparseStructure** (`sprux/sprux/SparseStructure.h`): CSR-format sparse structure storing `ptrs` and `inds` vectors representing block indices (not individual elements).

**CoalescedBlockMatrixSkel** (`sprux/sprux/CoalescedBlockMatrix.h`): Block matrix skeleton with coalesced columns. Key terminology:
- **span**: basic parameter block grouping
- **lump**: aggregation of consecutive spans (supernode)
- **chain**: span rows × lump cols
- **board**: all spans in a lump of rows × lump cols

**Solver** (`sprux/sprux/Solver.h`): Main interface created via `createSolver()`. Provides:
- `factor()` / `factorLU()` / `factorLDLT()`: factorization
- `solve()` / `solveLU()` / `solveLDLT()`: triangular solves
- `factorUpTo()` / `solveLUpTo()`: partial factorization for marginals
- `beginFactorLU()` / `finishFactorLU()`: split factorization for GPU overlap
- Backends: `BackendFast`, `BackendCuda`, `BackendMetal`, `BackendOpenCL`, `BackendAuto`

### Backend Context Hierarchy

Each backend implements three context types:
- **SymbolicCtx**: created once during `createSolver()`, holds GPU structure buffers
- **NumericCtx<T>**: created per factorization (or reused via persistent API), holds work buffers
- **SolveCtx<T>**: created per solve (or reused via persistent API), holds solve work buffers

### Preprocessing Pipeline (LU)

For general matrices, preprocessing improves numerical stability:
1. **BTF max transversal** (`Preprocessing.h`): structural row permutation
2. **Row/column equilibration**: scales entries to O(1)
3. **Static pivoting**: perturbs small/non-finite pivots (`Settings.staticPivotThreshold`)
4. **Iterative refinement**: float factor + double residual for mixed-precision accuracy

### External Encoder API (Metal)

The Metal backend supports embedding into external GPU pipelines:
- `setExternalEncoder()` / `clearExternalEncoder()`: encode Sprux ops into caller's encoder
- **Encoder cycling**: transparent CPU↔GPU interleaving when CPU BLAS fallback needed
- Enables fusion with IREE custom-calls, XLA operations, etc.

### Directory Structure

```
sprux/
  sprux/          # Core library sources
  testing/        # Test utilities (TestingMatGen, TestingUtils, MatrixMarketReader)
  tests/          # Unit tests (gtest)
  benchmarking/   # Performance benchmarks (bench, BAL_bench, lu_bench)
  examples/       # Example applications (Optimizer, PCG)
python/           # Python bindings (pybind11)
docs/             # Architecture, API guide, benchmarks documentation
test_data/        # Test matrices (c6288_sequence, mul64, tb_dp)
```

### Key CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `SPRUX_USE_CUBLAS` | ON | Enable CUDA support |
| `SPRUX_USE_METAL` | OFF | Enable Apple Metal support (macOS only, float only) |
| `SPRUX_USE_OPENCL` | OFF | Enable OpenCL support with CLBlast (experimental) |
| `SPRUX_USE_BLAS` | ON | Enable BLAS support |
| `SPRUX_CUDA_ARCHS` | "detect" | CUDA architectures ("detect", "torch", or "60;70;75") |
| `SPRUX_USE_SUITESPARSE_AMD` | OFF | Use SuiteSparse AMD instead of Eigen's |
| `SPRUX_BUILD_TESTS` | ON | Build tests |
| `SPRUX_BUILD_EXAMPLES` | ON | Build examples/benchmarks |
| `BLA_VENDOR` | (auto) | BLAS implementation (ATLAS, OpenBLAS, Intel10_64lp_seq, Apple) |

## GPU Backend Notes

### Metal Backend (Apple Silicon)

The Metal backend provides GPU acceleration on Apple Silicon Macs (M1, M2, M3, M4).

**Float-only precision.** Apple Silicon GPUs lack native double-precision FP64 support. The Metal backend only supports `float` operations. For double precision, use `BackendFast` (CPU) or `BackendCuda`.

**Hybrid execution strategy:**
- Sparse elimination: GPU compute kernels (thousands of parallel 1×1 lumps)
- Dense factorization (n ≤ 256): CPU Accelerate BLAS on unified memory (zero-copy)
- Dense factorization (n > 256): MPS (Metal Performance Shaders)
- Dense solve: CPU Eigen on unified memory

**Key pattern**: Apple Silicon unified memory allows CPU BLAS to operate directly on Metal shared buffers with no data transfer overhead.

```cpp
// Metal backend usage (float only)
Settings settings;
settings.backend = BackendMetal;
settings.matrixType = MTYPE_GENERAL;
auto solver = createSolver(settings, paramSize, structure);

MetalMirror<float> dataGpu(hostData);
solver->factorLU(dataGpu.ptr(), pivots.data());
dataGpu.get(hostData);
```

### CUDA Backend (NVIDIA)

The CUDA backend supports both float and double precision on NVIDIA GPUs with compute capability >= 6.0.

**Hybrid execution**: sparse elimination on GPU, small dense lumps via CPU BLAS (D→H + BLAS + H→D), large dense lumps via cuSolver/cuBLAS.

### OpenCL Backend (Experimental)

Portable GPU acceleration using CLBlast for BLAS operations. Infrastructure in place but most operations use CPU fallbacks. For production use, prefer CUDA or Metal.

## Dependencies

Fetched automatically by CMake:
- Eigen 3.4.0
- GoogleTest
- dispenso (multithreading)
- SuiteSparse (BTF for LU preprocessing, optionally AMD for reordering)
- Sophus (for BA examples only)

Optional external:
- CUDA Toolkit (10.2+, architecture >=60 for double atomics)
- CHOLMOD (SuiteSparse) - for benchmarking comparisons
- OpenCL 1.2+ + CLBlast - for OpenCL backend

## Running Benchmarks

See [docs/benchmarks.md](docs/benchmarks.md) for full details.

```bash
# Cholesky benchmarks with CHOLMOD baseline
build/sprux/benchmarking/bench -B 1_CHOLMOD

# Bundle Adjustment problem
build/sprux/benchmarking/BAL_bench -i ~/BAL/problem-871-527480-pre.txt

# LU benchmarks on circuit Jacobians
build/sprux/benchmarking/lu_bench -d test_data/c6288_sequence -b Metal_Sparse

# Collect timing statistics for computation model fitting
build/sprux/benchmarking/bench -B 1_CHOLMOD -Z
```
