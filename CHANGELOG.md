# Changelog

All notable changes to Sprux (formerly BaSpaCho) are documented here.

## [Unreleased] — Rename to Sprux

### Changed
- Project renamed from BaSpaCho to Sprux
- Documentation overhauled: new architecture guide, API guide, benchmarks guide
- Copyright headers updated for ChipFlow contributions

## Metal Performance Optimization — March 2026

### Added
- Custom parallel Metal kernels for LU factorization, TRSM, and solve
- Fused dense solve kernel (forward L + backward U in single dispatch)
- Batched all-lumps dense solve kernels (16 dispatches → 1)
- Batched cross-matrix SpMV for iterative refinement
- CPU SpMV refinement with native double precision (7 orders of magnitude better residuals)
- CPU BLAS fallback for dense LU getrf on Metal (n ≥ 64)
- Two-phase deterministic sparse elimination (Metal and CUDA)
- External encoder API for GPU pipeline embedding (IREE, XLA custom-calls)
- Persistent NumericCtx/SolveCtx for reuse across factorizations
- GPU-resident pivots (no per-call D→H roundtrip)
- `lu_bench` Metal_Dense backend using Accelerate sgetrf/sgetrs baseline

### Changed
- Replaced MPS LU with custom Metal kernel to eliminate encoder transitions
- Single atomic kernel for LU sparse elimination (replaced two-phase)
- Removed per-kernel Metal profiling infrastructure
- Consolidated Metal LU benchmarks into Metal_Sparse/Metal_Dense

## CUDA LU Dense Loop Optimization — March 2026

### Added
- Custom batched GEMM kernel (eliminates per-call cuBLAS overhead)
- CPU BLAS fallback for small dense lumps (D→H copy + BLAS + H→D)
- Lazy bulk-copy cache for readValue (eliminates 25K × 10μs cudaMemcpy calls)
- BASPACHO_PROFILE_LU env var for per-phase timing
- Async H→D transfer via pinned memory

### Performance
- CUDA LU factor: 278ms → 3.0ms (93×), matches cuDSS baseline
- Total: 7.3ms vs cuDSS 3.7ms (2× parity)

## CUDA LU Sparse Elimination — March 2026

### Added
- CUDA LU sparse elimination kernels (factor + solve)
- `solveUpperRowMajor()` for row-major upper triangular solve
- Upload upper triangle DevMirrors for general matrices

### Performance
- C6288: Factor 7848ms → 261ms (30×), Solve ~100s → 5.1ms (~20,000×)

## Metal LU Solve Optimization — February–March 2026

### Added
- CPU BLAS fallback for Metal dense solve (unified memory, zero-copy)
- Batch Metal solve dispatches (11 sync calls → deferred encode)
- Batch pivot upload (single memcpy instead of per-lump)
- CPU BLAS fallback for Metal dense factorization
- CPU BLAS fallback for Cholesky factor (GRID/MERI problems)

### Performance
- Metal solve: 403ms → 4.98ms (81×)
- Metal factor: 64.7ms → 7.3ms (8.8×)
- Metal now 4.1× faster than CPU BLAS for LU

## Metal Cholesky Factor Optimization — February 2026

### Added
- Three-tier potrf routing: Eigen (n<4), MPS (4–128), BLAS spotrf (n>128)
- MPS threshold tuning for trsm/GEMM
- BLAS trsm fallback for large matrices

### Performance
- Metal Cholesky factor: ~500ms → ~32ms (15.6×)

## Metal LU Sparse Elimination — February 2026

### Added
- GPU sparse elimination kernels for LU: `lu_factor_lumps_kernel` + `lu_sparse_elim_kernel`
- GPU sparse elimination solve kernels (forward L + backward U)
- Level-set batched dispatch (all levels in single command buffer)
- Pre-computed work list for LU sparse elim (CPU pre-computed offsets)

### Performance
- C6288 factor: 3.97s → 0.08s (50×)
- C6288 solve: ~100s → 0.42s (240×)

## MC64 Preprocessing — February 2026

### Added
- BTF max transversal for structural row permutation
- Row/column equilibration for O(1) matrix entries
- Integration with SuiteSparse BTF library

### Notes
- C6288: 0 perturbations, residual ~5e-11 after refinement (was NaN without preprocessing)

## Static Pivoting — February 2026

### Added
- `Settings.staticPivotThreshold`: configurable pivot perturbation
- `perturbSmallDiagonals` in CPU/CUDA/Metal backends
- Post-getrf perturbation for non-finite and small pivots

## LU Benchmark Tool — February 2026

### Added
- `lu_bench` CLI with CPU, Metal, CUDA, cuDSS backends
- JSON output for CI regression tracking
- Auto-decompress .mtx.xz sequence files

## Sequence Tests — February 2026

### Added
- `MatrixMarketReader.h` for Matrix Market file I/O
- Ring oscillator sequence test (50 × 47×47 matrices)
- C6288 sequence test
- Metal variant with CPU float cross-validation

## Metal Sync Optimization — February 2026

### Added
- Deferred GPU sync: `commitPending()`/`waitForGpu()` pattern
- Batch potrf+trsm and GEMM+assemble into single command buffers

### Performance
- 25% improvement for Metal single factorization

## Supernode Merging + Level-Sets — February 2026

### Added
- Supernode merging with fill tolerance heuristic
- Level-set scheduling for parallel sparse elimination
- Integration into Solver with proper index alignment

## Fast Symbolic Analysis — February 2026

### Added
- CHOLMOD-based symbolic analysis via SuiteSparse FetchContent
- Proper handling of CHOLMOD postorder flag
- CSR lower → CSC upper triangle conversion

## LDL^T Factorization — February 2026

### Added
- `factorLDLT()` / `solveLDLT()` for symmetric indefinite matrices
- Unit lower triangular L with diagonal D storage

## LU Factorization — January 2026

### Added
- `factorLU()` / `solveLU()` with partial pivoting for general matrices
- Upper triangle storage in CoalescedBlockMatrixSkel
- `MTYPE_GENERAL` matrix type

## Metal Backend — December 2025

### Added
- Metal compute shaders for sparse operations
- MPS integration for dense matrix operations
- `MetalMirror` RAII wrapper for GPU memory
- `BackendMetal` and `BackendAuto` detection

## OpenCL Backend (Experimental) — January 2026

### Added
- OpenCL 1.2+ backend with CLBlast BLAS
- `BackendOpenCL` option

## 1.0 — August 2022

### BaSpaCho initial release (Meta/Facebook Research)
- Supernodal Cholesky decomposition
- CUDA GPU acceleration with batching
- Block-structured sparse matrices
- AMD fill-reducing ordering
- Partial factor/solve for marginals
- Preconditioned conjugate gradient example
