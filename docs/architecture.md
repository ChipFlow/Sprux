# Sprux Architecture

Sprux (formerly BaSpaCho) is a high-performance sparse direct solver. This document
describes its internal architecture: core data structures, the solver pipeline, GPU
backend design, and the external encoder API for GPU pipeline embedding.

## Solver Pipeline

```
Input (CSR + param sizes)
        │
        ▼
┌─────────────────────────────┐
│  Symbolic Analysis          │  createSolver()
│  ─ Fill-reducing order      │  ─ AMD permutation
│  ─ Supernode detection      │  ─ Merge heuristic
│  ─ Sparse-elim ranges      │  ─ Level-set schedule
│  ─ Allocate factor storage  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Numeric Factorization      │  factor() / factorLU() / factorLDLT()
│  ─ Sparse elimination (GPU) │  ─ Level-set parallel kernels
│  ─ Dense loop (BLAS)        │  ─ potrf/getrf + trsm + gemm + assemble
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Solve                      │  solve() / solveLU() / solveLDLT()
│  ─ Forward substitution     │  ─ Apply pivots (LU)
│  ─ Backward substitution    │
└─────────────────────────────┘
```

## Core Data Structures

### SparseStructure

Defined in `sprux/sprux/SparseStructure.h`.

CSR-format sparse structure storing block-level connectivity (not individual elements).
Two vectors:
- `ptrs`: row pointers (size = num_blocks + 1)
- `inds`: column indices

Key operations:
- `transpose()`: CSR → CSC (lower ↔ upper triangle)
- `symmetricPermutation()`: reorder rows and columns
- `addIndependentEliminationFill()`: add fill from independent sparse elimination
- `fillReducingPermutation()`: compute AMD ordering

### CoalescedBlockMatrixSkel

Defined in `sprux/sprux/CoalescedBlockMatrix.h`.

The skeleton (structure without numeric data) of the block-sparse factor. Key terminology:

| Term | Meaning |
|------|---------|
| **span** | A single parameter block (original user parameter). |
| **lump** | An aggregation of consecutive spans. After supernode merging, a lump groups spans whose rows have similar or identical sparsity patterns. |
| **chain** | The entries in one span-row within a lump-column. Chains within a lump are stored contiguously for BLAS efficiency. |
| **board** | All chains (span-rows) in a lump-of-rows × lump-of-columns block. |

The coalesced layout enables dense BLAS calls on supernodal blocks:
```
                  lump j
              ┌───────────┐
  lump j      │ diagonal  │  ← potrf / getrf
              │  block    │
              ├───────────┤
  lump k      │  board    │  ← trsm
  (rows in    │ (k,j)     │
   lump k ×   │           │
   cols in j)  └───────────┘
```

### Solver

Defined in `sprux/sprux/Solver.h`.

The main user-facing class. Created via `createSolver()`. Provides:
- `factor()`: Cholesky factorization (SPD matrices)
- `factorLU()`: LU with partial pivoting (general matrices)
- `factorLDLT()`: LDL^T (symmetric indefinite)
- `solve()` / `solveLU()` / `solveLDLT()`: triangular solves
- `factorUpTo()` / `solveLUpTo()`: partial factorization for marginals
- `beginFactorLU()` / `finishFactorLU()`: split factorization for GPU overlap
- `accessor()` / `deviceAccessor()`: block-level access to factor data

## Sparse Elimination

Sparse elimination handles the "easy" part of the factorization — small, independent
parameter blocks (typically 1×1 scalar blocks from circuit simulation or point parameters
from bundle adjustment) that don't benefit from supernodal dense BLAS.

### Level-Set Parallelism

The elimination tree is partitioned into **level sets** — groups of lumps whose
eliminations are independent and can execute in parallel on the GPU:

```
Level 0:  [lump 0, lump 1, ..., lump N-1]   ← all independent, GPU parallel
Level 1:  [lump N, lump N+1, ...]            ← depend on level 0, GPU parallel
  ...
Level K:  [lump M]                           ← root of elimination tree
```

For LU factorization, each 1×1 scalar lump elimination:
1. Divides the below-diagonal column by the diagonal (L factor)
2. Applies L×U rank-1 Schur complement updates to all target blocks
3. Updates both lower and upper triangle targets (n² pairs per lump, vs n(n+1)/2 for Cholesky)

### GPU Kernel Dispatch

- **Metal**: All level-set kernels encoded into a single command buffer with memory barriers
  between levels. Single `commitAndWait()` at the end.
- **CUDA**: Similar batched dispatch with stream synchronization between levels.
- **Two-phase determinism option**: Separate accumulation and application phases to avoid
  non-deterministic atomicAdd results (at the cost of extra memory).

## LU Pipeline

For general (non-symmetric) matrices, the full pipeline includes preprocessing:

```
BTF Max Transversal          ← Structural row permutation
        │
        ▼
Row/Column Equilibration     ← Scale to O(1) entries
        │
        ▼
Symbolic Analysis            ← Same as Cholesky path
        │
        ▼
Numeric LU Factorization     ← GPU sparse elim + CPU/GPU dense loop
        │
        ▼
LU Solve (P⁻¹ L U)          ← Forward L, backward U, unpivot
        │
        ▼
Iterative Refinement         ← Mixed-precision: float factor, double residual
```

### Static Pivoting

`Settings.staticPivotThreshold` controls perturbation of small pivots:
- `< 0`: disabled (default)
- `= 0`: automatic threshold: `cbrt(eps) * maxDiag`
- `> 0`: manual threshold

Post-getrf, any diagonal with `|diag| < threshold` or `!isfinite(diag)` is replaced.

### Iterative Refinement

For Metal (float-only GPU), iterative refinement recovers double-precision accuracy:
1. Factor A in float on GPU
2. Compute residual r = b - A*x in double on CPU
3. Solve A*dx = r in float on GPU
4. Update x += dx in double on CPU
5. Repeat until convergence (typically 5-7 iterations)

## Backend Architecture

### Context Hierarchy

Each backend implements three context types:

```
SymbolicCtx              ← Created once during createSolver()
    │                       Holds GPU buffers for structure data
    │
    ├── NumericCtx<T>    ← Created per factorization (or reused via persistent API)
    │                       Holds work buffers, pivot status
    │
    └── SolveCtx<T>      ← Created per solve (or reused via persistent API)
                            Holds solve work buffers
```

### Backend Implementations

| Backend | File | GPU | Precision | Notes |
|---------|------|-----|-----------|-------|
| `BackendFast` | `MatOpsFast.cpp` | No | float/double | CPU BLAS (OpenBLAS, MKL, Accelerate) |
| `BackendCuda` | `MatOpsCuda.cu` | CUDA | float/double | cuBLAS + custom kernels |
| `BackendMetal` | `MatOpsMetal.mm` | Metal | float only | MPS + custom kernels + CPU BLAS fallback |
| `BackendOpenCL` | `MatOpsOpenCL.cpp` | OpenCL | float/double | CLBlast (experimental) |

### Metal Backend Details

The Metal backend uses a hybrid execution strategy on Apple Silicon:

- **Sparse elimination**: GPU compute kernels (thousands of parallel 1×1 lumps)
- **Dense factorization** (n ≤ 256): CPU Accelerate BLAS on unified memory (no copy needed)
- **Dense factorization** (n > 256): MPS (Metal Performance Shaders)
- **Dense solve**: CPU Eigen on unified memory (GPU dispatch overhead > compute time)

Key optimization: Apple Silicon's unified memory allows CPU BLAS to operate directly on
Metal shared buffers with zero-copy overhead.

### CUDA Backend Details

- **Sparse elimination**: Custom CUDA kernels with atomicAdd
- **Dense factorization** (small lumps): CPU BLAS with D→H/H→D copies (~1.6MB, <0.5ms)
- **Dense factorization** (large lumps): cuSolver getrf + cuBLAS trsm/gemm
- **GPU-resident pivots**: Pivots stay on device, no per-lump D→H roundtrip

## External Encoder API

The Metal backend supports embedding Sprux operations into an external GPU command encoder,
enabling fusion with upstream GPU pipelines (e.g., IREE custom-calls, XLA operations).

```cpp
// Get the Metal command queue
id<MTLCommandQueue> queue = metalContext->getCommandQueue();

// Create your own command buffer and encoder
id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

// ... encode your upstream GPU work ...

// Set external encoder — Sprux will encode into YOUR encoder
metalContext->setExternalEncoder(encoder);

// Factor + solve encode into the same encoder (no extra command buffers)
solver.factorLU(data, pivots, numCtx, PivotLocation::Device);
solver.solveLU(data, pivots, vec, stride, 1, solveCtx, PivotLocation::Device);

// Clear external encoder when done
metalContext->clearExternalEncoder();

// ... encode more GPU work ...

[encoder endEncoding];
[cmdBuf commit];
[cmdBuf waitUntilCompleted];
```

**Encoder cycling**: When Sprux needs CPU fallback during external encoder mode
(e.g., CPU BLAS for dense blocks), it temporarily ends the encoder, performs CPU
work on the unified memory buffer, then creates a new encoder and restores it as
the external encoder. This is transparent to the caller.

## Memory Management

### MetalMirror / CudaMirror

RAII wrappers for GPU memory:

```cpp
// Metal: allocates MTLBuffer with shared storage mode
MetalMirror<float> dataGpu(hostData);  // host vector → GPU buffer
solver.factor(dataGpu.ptr());          // operate on GPU
dataGpu.get(hostData);                 // GPU → host

// CUDA: allocates device memory
CudaMirror<double> dataGpu(hostData);  // host → device
solver.factor(dataGpu.ptr());
dataGpu.get(hostData);                 // device → host
```

### Apple Silicon Unified Memory

On Apple Silicon, `MetalMirror` buffers use shared storage mode. The CPU and GPU
see the same physical memory — no copies are needed for CPU BLAS fallback operations.
This is why the Metal backend can freely mix GPU kernels with CPU BLAS calls without
any data transfer overhead.

## Directory Structure

```
sprux/
  sprux/             # Core library sources
    Solver.h/.cpp      # Main Solver class and createSolver()
    MatOpsFast.cpp     # CPU BLAS backend
    MatOpsCuda.cu      # CUDA backend
    MatOpsMetal.mm     # Metal backend
    MatOpsOpenCL.cpp   # OpenCL backend (experimental)
    MetalKernels.metal # Metal compute shaders
    MetalDefs.h/.mm    # Metal context and utilities
    CoalescedBlockMatrix.h  # Block matrix skeleton
    SparseStructure.h  # CSR sparse structure
    Preprocessing.h/.cpp    # BTF, equilibration
    EliminationTree.h/.cpp  # Symbolic analysis
    SupernodeMerger.h/.cpp  # Supernode merging
    LevelSetSchedule.h/.cpp # Level-set parallelism
  testing/           # Test utilities
    TestingMatGen.h      # Synthetic test matrix generation
    TestingUtils.h       # Common test helpers
    MatrixMarketReader.h # Matrix Market file I/O
  tests/             # Unit tests (gtest)
  benchmarking/      # Performance benchmarks
    Bench.cpp            # Cholesky benchmarks (bench)
    LUBench.cpp          # LU benchmarks (lu_bench)
  examples/          # Example applications
    Optimizer.h          # Levenberg-Marquardt optimizer
    PCG_Sample.cpp       # Partial elimination + PCG
  python/            # Python bindings (pybind11)
```
