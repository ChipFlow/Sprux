# CUDA Sparse LU Implementation in Sprux

This document describes how Sprux implements sparse LU factorization on NVIDIA GPUs using CUDA, with particular attention to what runs on the CPU vs GPU.

## Overview

Sprux performs supernodal LU factorization of a block-sparse matrix. The sparse structure (which blocks exist) is analyzed on the **CPU** during a symbolic phase, while the dense numeric operations within each block run on the **GPU** using cuBLAS, cuSOLVER, and custom CUDA kernels.

The factorization computes **PA = LU** with partial pivoting, where:
- **P** is a row permutation (pivots)
- **L** is unit lower triangular
- **U** is upper triangular

## Architecture Layers

```
Solver (Solver.cpp)              -- CPU: orchestration loop, symbolic indexing
  -> NumericCtx<T> interface     -- abstract per-operation dispatch
    -> CudaNumericCtx<T>         -- GPU: cuBLAS/cuSOLVER/kernels (MatOpsCuda.cu)
```

### Key source files

| File | Role |
|------|------|
| `Solver.cpp` | CPU-side factorization loop, elimination ordering, solve orchestration |
| `MatOpsCuda.cu` | All GPU operations: kernels, cuBLAS/cuSOLVER wrappers, memory management |
| `CudaDefs.h` | `DevMirror<T>` (GPU memory wrapper), error-checking macros |
| `CudaAtomic.cuh` | `atomicAdd` workaround for compute capability < 6.0 |
| `CoalescedBlockMatrix.h` | Block matrix skeleton data structures |

## Data Structures

### Block Matrix Skeleton (CPU, mirrored to GPU)

The sparse matrix is stored as a **coalesced block matrix** (`CoalescedBlockMatrixSkel`):

- **Span**: a contiguous group of rows/columns (basic parameter block)
- **Lump**: one or more consecutive spans coalesced together (the "supernode")
- **Chain**: a column-slice of the block column below a lump's diagonal
- **Board**: grouping of chains for a specific off-diagonal block interaction

Index arrays like `lumpStart`, `chainColPtr`, `chainData`, `spanStart`, `spanToLump`, etc. describe the block structure. These are computed once on the CPU during symbolic analysis, then copied to GPU memory via `DevMirror<T>` in `CudaSymbolicCtx`.

### GPU Memory Management

- **`DevMirror<T>`**: RAII wrapper around `cudaMalloc`/`cudaFree` with `load(vector<T>)` for H->D copy. Used for all index arrays.
- **Matrix data pointer**: The caller allocates GPU memory and passes a `T*` device pointer directly. No bulk matrix H<->D copy during factorization.
- **Pivot arrays**: Small per-lump arrays (`n` ints) copied D->H after each `getrf` for format conversion, then H->D for the `applyRowPerm` kernel.

## Factorization Pipeline

### CPU: Symbolic Phase (one-time setup)

Runs entirely on **CPU**:

1. **Reordering**: AMD/METIS fill-reducing permutation computed on the sparsity pattern
2. **Supernodal analysis**: Group spans into lumps, compute the elimination tree
3. **Index construction**: Build all `chainColPtr`, `boardColPtr`, `chainData`, etc.
4. **GPU mirror**: Copy all index arrays to GPU via `CudaSymbolicCtx` constructor

### CPU + GPU: Numeric Factorization

The main factorization loop in `Solver::internalFactorRangeLU()` (Solver.cpp:657) runs on the **CPU** but dispatches GPU work through the `NumericCtx<T>` interface:

```
for each lump l (CPU loop):
    1. prepareAssemble(l)                          -- CPU: set up assembly indices
    2. for each board below diagonal of l:
         eliminateBoardLU(numCtx, data, rPtr)      -- dispatches GPU GEMM
    3. factorLumpLU(numCtx, data, pivots, l)       -- dispatches GPU getrf + trsm + permutation
```

All GPU calls are **asynchronous** -- the CPU loop submits work to the CUDA stream without waiting. A final `numCtx->flush()` synchronizes at the end.

#### Step 1: Schur Complement Update (`eliminateBoardLU`)

**CPU**: Iterates over L-column and U-row blocks of a previously-factored lump, computes which target blocks need updating, and locates them in the block structure.

**GPU** (via `saveGemm`): For each (L-block, U-block) pair:
```
C -= L * U
```
Dispatched as **cuBLAS `Dgemm`/`Sgemm`**. The row-major storage is handled by swapping operand order in the col-major cuBLAS call:
```cpp
// Row-major C -= L * U  becomes  col-major C_cm -= U_cm * L_cm
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
            U + offU, ldU, L + offL, ldL, &beta, C + offC, ldC);
```

#### Step 2: Diagonal Block Factorization (`factorLumpLU`)

For each lump, the diagonal block is factored and the result is used to solve for the off-diagonal L and U blocks.

##### 2a. Dense LU with partial pivoting (`getrf`)

This is the most complex GPU operation due to the row-major vs col-major mismatch:

| Step | Where | Operation |
|------|-------|-----------|
| Transpose row->col major | **GPU** | Custom `transposeSquareInPlaceKernel` |
| LU factorization | **GPU** | `cusolverDnDgetrf` / `cusolverDnSgetrf` |
| Transpose col->row major | **GPU** | Custom `transposeSquareInPlaceKernel` (same kernel) |
| Copy pivots D->H | **GPU->CPU** | `cudaMemcpy` (small: `n` ints) |
| Convert pivot format | **CPU** | 1-based -> 0-based, `int` -> `int64_t` |

The transpose workaround is necessary because cuSOLVER's `getrf` expects column-major input, but Sprux stores blocks in row-major order. Rather than maintaining a separate col-major copy, the diagonal block is transposed in-place on the GPU before and after the factorization.

**Pivot format**: cuSOLVER outputs 1-based `int` pivots in column-major convention. Sprux converts these to 0-based `int64_t` pivots on the CPU. This is the **only per-lump D->H transfer** during factorization.

##### 2b. Apply row permutation to off-diagonal blocks (`applyRowPerm`)

| Step | Where | Operation |
|------|-------|-----------|
| Copy pivots H->D | **CPU->GPU** | `cudaMemcpy` (small: `n * sizeof(int64_t)`) |
| Swap rows | **GPU** | Custom `applyRowPermKernel` |

The kernel applies sequential row swaps (matching LAPACK pivot convention) with parallelism across columns. **Must use exactly 1 thread block** because `__syncthreads()` only synchronizes within a block, and each swap depends on the previous one completing.

Applied to:
- L blocks below the diagonal (column below diagonal block)
- U blocks to the right of the diagonal (each upper triangle block)

##### 2c. Triangular solves for L and U blocks

| Operation | Where | Implementation |
|-----------|-------|----------------|
| Solve for U column: `X * U_diag = A_below` -> L blocks | **GPU** | cuBLAS `Dtrsm`/`Strsm` (SIDE_LEFT, LOWER, NON_UNIT) |
| Solve for L row: `L_diag * X = A_right` -> U blocks | **GPU** | cuBLAS `Dtrsm`/`Strsm` (SIDE_RIGHT, UPPER, UNIT) |

The row-major to col-major mapping swaps the SIDE and transposition flags compared to what you'd expect from the mathematical description.

## Solve Phase

After factorization, solving `Ax = b` given `PA = LU`:

```
1. y = P * b           (apply row permutation)
2. Solve L * z = y     (forward substitution, L is unit lower triangular)
3. Solve U * x = z     (backward substitution)
```

### Step 1: Apply permutation

**CPU loop** over lumps, dispatching per-lump GPU kernels:

| Step | Where | Operation |
|------|-------|-----------|
| Copy pivots H->D | **CPU->GPU** | `cudaMemcpy` per lump |
| Permute RHS vector | **GPU** | `applyRowPermVecKernel` (1 block, sequential swaps) |

### Step 2: Forward substitution (L solve)

**CPU loop** over lumps (forward order):

| Operation | Where | Implementation |
|-----------|-------|----------------|
| Solve diagonal block: `L_diag * x_l = z_l` | **GPU** | cuBLAS `Dtrsm`/`Strsm` |
| Update below-diagonal: `y -= L_below * x_l` | **GPU** | cuBLAS `Dgemv`/`Sgemv` |
| Scatter results to target spans | **GPU** | Custom assembly kernel |

### Step 3: Backward substitution (U solve)

**CPU loop** over lumps (reverse order):

| Operation | Where | Implementation |
|-----------|-------|----------------|
| Gather from target spans | **GPU** | Custom assembly kernel |
| Update: `z -= U_right * x_col` | **GPU** | cuBLAS `Dgemv`/`Sgemv` |
| Solve diagonal block: `U_diag * x_l = z_l` | **GPU** | cuBLAS `Dtrsm`/`Strsm` |

## CPU vs GPU Summary

### Always on CPU

- Sparse structure analysis (reordering, elimination tree, supernode detection)
- Symbolic index construction (chainColPtr, boardColPtr, etc.)
- Factorization loop control (which lumps/boards to process, in what order)
- Pivot format conversion (1-based int -> 0-based int64_t)
- Solve loop control (lump iteration order)

### Always on GPU

- Dense block LU factorization (cuSOLVER getrf)
- Triangular solves (cuBLAS trsm)
- GEMM Schur complement updates (cuBLAS gemm)
- Row permutation application (custom kernels)
- Matrix transpose for row/col-major conversion (custom kernel)
- Schur complement assembly/scatter (custom kernel)
- Solve vector permutation, gemv, assembly (cuBLAS + custom kernels)

### CPU <-> GPU Transfers During Factorization

| Transfer | Direction | Size | Frequency |
|----------|-----------|------|-----------|
| Index arrays (symbolic structure) | H->D | O(nnz_blocks) | Once at setup |
| Pivots after getrf | D->H | `n` ints per lump | Once per lump |
| Pivots for applyRowPerm | H->D | `n` int64_t per lump | Once per lump per target block |

The matrix data itself stays on the GPU throughout. The only per-lump transfers are the small pivot arrays.

## Custom CUDA Kernels

| Kernel | Purpose | Grid size |
|--------|---------|-----------|
| `transposeSquareInPlaceKernel` | Row-major <-> col-major for cuSOLVER | `ceil(n*(n-1)/2 / 256)` blocks x 256 threads |
| `applyRowPermKernel` | Sequential row swaps, parallel across columns | **1 block** x min(256, numCols) threads |
| `applyRowPermVecKernel` | Forward permutation of RHS vector | **1 block** x min(256, nRHS) threads |
| `applyRowPermVecInvKernel` | Inverse permutation of RHS vector | **1 block** x min(256, nRHS) threads |
| `assemble_kernel` | Scatter Schur complement results to target blocks | standard grid |

## Batched LU Support

`CudaNumericCtx<vector<T*>>` handles multiple independent matrices with the same sparsity pattern. The same CPU loop drives factorization, but each GPU call processes all batch members:

- `cublasDgemmBatched` / `cublasSgemmBatched` for GEMM
- Separate getrf calls per batch element (cuSOLVER doesn't have batched getrf for general sizes)
- 2D kernel grids: `(numBlocks, batchSize)` for custom kernels

## Key Design Decisions

1. **Row-major storage with col-major libraries**: Sprux stores blocks row-major for cache-friendly access patterns in the supernodal structure. cuBLAS/cuSOLVER expect col-major. Rather than maintaining dual layouts, the code uses transpose tricks (swap operands for GEMM, explicit transpose for getrf).

2. **Asynchronous dispatch**: All GPU operations are submitted to a CUDA stream without synchronization. The CPU loop runs ahead, preparing the next lump's work while the GPU processes the current one. Only pivot D->H copies force implicit synchronization (via `cudaMemcpy`).

3. **Single-block permutation kernels**: The `applyRowPermKernel` must use exactly 1 thread block because sequential pivot swaps have data dependencies that require `__syncthreads()`, which only works within a block.

4. **Pivots are the bottleneck transfer**: The only CPU<->GPU data movement per lump is the pivot array (~`n` elements where `n` is lump size). This is inherently sequential due to cuSOLVER's API returning pivots on the device in a format that needs CPU conversion.
