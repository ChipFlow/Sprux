# Sprux API Guide

Sprux (formerly BaSpaCho) provides a C++ API for sparse direct solving. This guide
covers common usage patterns for each decomposition type and backend.

## Input Format

Sprux takes block-structured sparse matrices in CSR format:

```cpp
#include "sprux/sprux/Solver.h"
#include "sprux/sprux/SparseStructure.h"

using namespace Sprux;

// Define parameter block sizes
std::vector<int64_t> paramSize = {3, 3, 6, 6};  // 4 blocks of varying size

// Define block-level sparsity in CSR format
// ptrs[i] = start of row i's column indices, ptrs[n] = total nnz blocks
// inds[k] = column index of k-th nonzero block
SparseStructure ss;
ss.ptrs = {0, 2, 4, 7, 9};
ss.inds = {0, 2, 1, 3, 0, 2, 3, 1, 3};  // block connectivity
```

## Cholesky (SPD Matrices)

For symmetric positive-definite matrices:

```cpp
Settings settings;
settings.backend = BackendFast;  // CPU BLAS
settings.matrixType = MTYPE_SPD; // default

auto solver = createSolver(settings, paramSize, ss);

// Allocate and fill factor data (lower triangle in internal ordering)
std::vector<double> data(solver->dataSize());
auto acc = solver->accessor();
// Fill data via accessor...

// Factor and solve
solver->factor(data.data());
std::vector<double> rhs(solver->order());
// Fill rhs...
solver->solve(data.data(), rhs.data(), solver->order(), 1);
// rhs now contains the solution
```

### Partial Factorization (for Marginals)

```cpp
// Factor only up to span index `k`
solver->factorUpTo(data.data(), k);

// Solve with partial factor
solver->solveLUpTo(data.data(), k, rhs.data(), solver->order(), 1);
```

## LU Factorization (General Matrices)

For non-symmetric matrices (e.g., circuit Jacobians):

```cpp
Settings settings;
settings.backend = BackendMetal;     // or BackendCuda, BackendFast
settings.matrixType = MTYPE_GENERAL;
settings.staticPivotThreshold = 0.0; // auto threshold

auto solver = createSolver(settings, paramSize, ss);

// Allocate data (lower + upper triangle)
std::vector<float> data(solver->totalDataSize());
auto acc = solver->accessor();
// Fill both triangles via accessor...

// Allocate pivots
std::vector<int64_t> pivots(solver->numSpans());

// Factor and solve
solver->factorLU(data.data(), pivots.data());
std::vector<float> rhs(solver->order());
solver->solveLU(data.data(), pivots.data(), rhs.data(), solver->order(), 1);
```

### LU with Preprocessing

For ill-conditioned matrices, use BTF + equilibration:

```cpp
#include "sprux/sprux/Preprocessing.h"

// Load your matrix (e.g., from Matrix Market format)
int n = ...;
std::vector<int64_t> rowPtr = ...;
std::vector<int64_t> colInd = ...;
std::vector<double> values = ...;

// Step 1: BTF max transversal (structural row permutation)
auto rowPerm = btfMaxTransversal(n, rowPtr, colInd);
// Apply row permutation to matrix...

// Step 2: Row/column equilibration
auto [rowScale, colScale] = equilibrate(n, rowPtr, colInd, values);
// Apply scaling...

// Step 3: Create solver and factor
Settings settings;
settings.matrixType = MTYPE_GENERAL;
settings.staticPivotThreshold = 0.0;
auto solver = createSolver(settings, paramSize, ss);

// Step 4: Iterative refinement (for float GPU + double accuracy)
for (int iter = 0; iter < 7; iter++) {
    // Compute residual in double: r = b - A*x
    // Solve correction in float: A*dx = r
    // Update: x += dx
}
```

### Split Factorization (GPU Overlap)

For maximum throughput when processing multiple matrices:

```cpp
// Matrix 1: start factorization (GPU sparse elim runs async)
solver->beginFactorLU(data1, pivots1);

// While GPU is busy with matrix 1, solve matrix 0 on CPU
solver->solveLU(data0, pivots0, rhs0, stride, 1);

// Finish matrix 1 factorization (waits for GPU, runs dense loop)
solver->finishFactorLU(data1, pivots1);
```

## LDL^T Factorization (Symmetric Indefinite)

For symmetric matrices that may not be positive-definite:

```cpp
Settings settings;
settings.matrixType = MTYPE_SPD;  // uses lower triangle storage
auto solver = createSolver(settings, paramSize, ss);

std::vector<double> data(solver->dataSize());
// Fill lower triangle...

solver->factorLDLT(data.data());
solver->solveLDLT(data.data(), rhs.data(), solver->order(), 1);
```

## Metal Backend

### Basic Usage (Float Only)

```cpp
#include "sprux/sprux/MetalDefs.h"

Settings settings;
settings.backend = BackendMetal;
settings.matrixType = MTYPE_GENERAL;
auto solver = createSolver(settings, paramSize, ss);

// Use MetalMirror for GPU memory management
std::vector<float> hostData(solver->totalDataSize());
// Fill hostData...

MetalMirror<float> dataGpu(hostData);
std::vector<int64_t> pivots(solver->numSpans());

solver->factorLU(dataGpu.ptr(), pivots.data());
dataGpu.get(hostData);  // Copy results back to CPU
```

### External Encoder (Pipeline Embedding)

For embedding Sprux into a larger GPU pipeline:

```cpp
auto* metalCtx = solver->getSymbolicCtx()->getMetalContext();

// Create persistent contexts (reuse across calls)
auto numCtx = solver->createNumericCtx<float>();
auto solveCtx = solver->createSolveCtx<float>();

// Get command queue for your pipeline
id<MTLCommandQueue> queue = metalCtx->getCommandQueue();
id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

// Encode upstream work...

// Set external encoder — Sprux encodes into YOUR encoder
metalCtx->setExternalEncoder(encoder);

solver->factorLU(dataGpu.ptr(), devPivots, *numCtx, PivotLocation::Device);
solver->solveLU(dataGpu.ptr(), devPivots, vecGpu.ptr(), n, 1,
                *solveCtx, PivotLocation::Device);

metalCtx->clearExternalEncoder();

// Encode downstream work...
[encoder endEncoding];
[cmdBuf commit];
```

### Persistent Contexts

Avoid per-call allocation overhead by reusing contexts:

```cpp
auto numCtx = solver->createNumericCtx<float>();
auto solveCtx = solver->createSolveCtx<float>();

for (int i = 0; i < numMatrices; i++) {
    // Contexts are reset internally, no re-allocation
    solver->factorLU(data[i], pivots.data(), *numCtx);
    solver->solveLU(data[i], pivots.data(), rhs[i], n, 1, *solveCtx);
}
```

## CUDA Backend

```cpp
#include "sprux/sprux/CudaDefs.h"

Settings settings;
settings.backend = BackendCuda;
settings.matrixType = MTYPE_GENERAL;
auto solver = createSolver(settings, paramSize, ss);

// Use CudaMirror for device memory
std::vector<double> hostData(solver->totalDataSize());
CudaMirror<double> dataGpu(hostData);

std::vector<int64_t> pivots(solver->numSpans());
solver->factorLU(dataGpu.ptr(), pivots.data());
dataGpu.get(hostData);
```

### CUDA Stream Support

```cpp
// Use a specific CUDA stream (e.g., from JAX/XLA FFI)
cudaStream_t stream = ...;
solver->setStream(stream);
solver->factorLU(dataGpu.ptr(), pivots.data());
```

## Settings Reference

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `backend` | `BackendType` | `BackendFast` | `BackendFast`, `BackendCuda`, `BackendMetal`, `BackendOpenCL`, `BackendAuto` |
| `matrixType` | `MatrixType` | `MTYPE_SPD` | `MTYPE_SPD` (Cholesky), `MTYPE_GENERAL` (LU) |
| `addFillPolicy` | `AddFillPolicy` | `AddFillComplete` | Fill strategy: `AddFillNone`, `AddFillForGivenElims`, `AddFillForAutoElims`, `AddFillComplete` |
| `findSparseEliminationRanges` | `bool` | `true` | Auto-detect independent sparse elimination blocks |
| `numThreads` | `int` | `16` | CPU thread count for multithreaded operations |
| `staticPivotThreshold` | `double` | `-1.0` | `< 0`: disabled, `0`: auto, `> 0`: manual threshold |
| `supernodeMergeFillTolerance` | `double` | `0.0` | Max extra-zero fraction when merging supernodes (0.0–1.0) |
| `maxSupernodeSize` | `int64_t` | `0` | Max supernode size (0 = merging disabled) |
| `computationModel` | `ComputationModel*` | `nullptr` | Custom BLAS timing model for merge heuristic |

## Python Bindings

The Python module is built via `-DSPRUX_BUILD_PYTHON=ON` and provides
a numpy-based interface to all solver operations.

```python
import sprux_py as sprux
import numpy as np

# Create solver from block-CSR sparsity pattern
# param_sizes: size of each parameter block
# row_ptrs: CSR row pointers [num_blocks + 1]
# col_inds: CSR column indices of nonzero blocks
solver = sprux.create_solver(
    param_sizes=[1, 1, 1, 1],          # scalar blocks
    row_ptrs=np.array([0, 2, 4, 7, 9], dtype=np.int64),
    col_inds=np.array([0, 2, 1, 3, 0, 2, 3, 1, 3], dtype=np.int64),
    matrix_type="general",              # "spd" or "general"
    backend="auto",                     # "cpu", "cuda", "metal", "opencl", "auto"
    static_pivot_threshold=-1.0         # <0 disabled, 0 auto, >0 manual
)

# Query properties
print(f"Order: {solver.order}")
print(f"Data size: {solver.data_size}")
print(f"Num spans: {solver.num_spans}")
print(f"Best backend: {sprux.detect_best_backend()}")

# Allocate buffers (numpy arrays, contiguous C order)
data = np.zeros(solver.total_data_size, dtype=np.float64)
pivots = np.zeros(solver.num_spans, dtype=np.int64)

# Fill data via CSR loader or direct indexing...
# solver.load_from_csr(csr_row_ptrs, csr_col_inds, block_sizes, csr_values, data)

# LU factorization (A = P * L * U) — modifies data in place
solver.factor_lu(data, pivots, verbose=True)

# Solve — re-use pivots with different RHS vectors
rhs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
solver.solve_lu(data, pivots, rhs)
# rhs now contains the solution

# Statistics
solver.enable_stats()
solver.factor_lu(data, pivots)
solver.print_stats()
solver.reset_stats()
```

### Cholesky (SPD matrices)

```python
solver = sprux.create_solver(param_sizes, row_ptrs, col_inds,
                              matrix_type="spd", backend="cpu")
data = np.zeros(solver.data_size, dtype=np.float64)
solver.factor(data)
rhs = np.array([...], dtype=np.float64)
solver.solve(data, rhs)
```

### LDL^T (symmetric indefinite)

```python
solver = sprux.create_solver(param_sizes, row_ptrs, col_inds,
                              matrix_type="spd", backend="cpu")
data = np.zeros(solver.data_size, dtype=np.float64)
solver.factor_ldlt(data, verbose=True)
rhs = np.array([...], dtype=np.float64)
solver.solve_ldlt(data, rhs)
```

## Matrix-Vector Multiplication

For iterative methods (e.g., PCG with partial elimination):

```cpp
// y += alpha * A[spanIndex:, spanIndex:] * x
solver->addMvFrom(data.data(), spanIndex,
                  x.data(), stride,
                  y.data(), stride, 1, alpha);
```

## Error Handling

All errors throw `std::runtime_error` with descriptive messages. Common errors:

- **"Metal backend only supports float"**: Attempting `double` with `BackendMetal`
- **"Zero pivot encountered"**: Singular matrix in LDL^T
- **"Static pivot threshold must be >= -1.0"**: Invalid settings

Assertion macros (`SPRUX_CHECK`, `SPRUX_CHECK_GE`, etc.) throw `std::runtime_error`
in both debug and release builds.

### Python errors

Python bindings raise `ValueError` or `RuntimeError` corresponding to the above
C++ errors. Additional errors:

- **`ValueError`**: Unknown `backend` or `matrix_type` string, or buffer too small
- **`RuntimeError`**: Backend not available at compile time (e.g., CUDA not enabled)
