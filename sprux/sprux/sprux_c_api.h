/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) 2024-2026 ChipFlow
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sprux_solver* sprux_solver_t;

/**
 * Backend selection for solver creation.
 * Maps to Sprux::BackendType (Sprux C++ namespace) enum values.
 */
enum sprux_backend {
  SPRUX_BACKEND_CPU = 0,    /* CPU with BLAS */
  SPRUX_BACKEND_CUDA = 1,   /* NVIDIA GPU (float and double) */
  SPRUX_BACKEND_METAL = 2,  /* Apple Metal GPU (float only) */
  SPRUX_BACKEND_OPENCL = 3, /* OpenCL with CLBlast */
  SPRUX_BACKEND_AUTO = 4,   /* Auto-detect best available */
};

/**
 * Create a solver for general (LU) matrices from CSR block structure.
 *
 * @param paramSizes  Size of each parameter block [numBlocks]
 * @param numBlocks   Number of parameter blocks
 * @param ptrs        CSR row pointers [numBlocks+1]
 * @param inds        CSR column indices [nnzBlocks]
 * @param nnzBlocks   Number of nonzero blocks (ptrs[numBlocks])
 * @param backend     Backend selection (see sprux_backend enum)
 * @param staticPivotThreshold  <0 disabled, 0 auto, >0 manual
 * @return Solver handle, or NULL on error
 */
sprux_solver_t sprux_create_lu_solver(const int64_t* paramSizes, int64_t numBlocks,
                                      const int64_t* ptrs, const int64_t* inds, int64_t nnzBlocks,
                                      int backend, double staticPivotThreshold);

/**
 * Destroy a solver and free all associated resources.
 */
void sprux_destroy(sprux_solver_t h);

/**
 * Query total factor data size (lower + upper triangle).
 * Caller allocates a float buffer of this size for factor/solve operations.
 */
int64_t sprux_data_size(sprux_solver_t h);

/**
 * Query the number of spans (parameter blocks after reordering).
 * Caller allocates a pivot array of this size for LU operations.
 */
int64_t sprux_num_spans(sprux_solver_t h);

/**
 * Load CSR values into internal coalesced format.
 *
 * @param h           Solver handle
 * @param ptrs        CSR row pointers [numBlocks+1]
 * @param inds        CSR column indices
 * @param blockSizes  Size of each block [numBlocks]
 * @param values      Block values in CSR order (row-major within blocks)
 * @param data        Output buffer (must be sized to sprux_data_size())
 * @return 0 on success, -1 on error
 */
int sprux_load_from_csr_f32(sprux_solver_t h, const int64_t* ptrs, const int64_t* inds,
                            const int64_t* blockSizes, const float* values, float* data);

/**
 * Load double-precision CSR values, converting to float during load.
 * Useful when the simulator works in f64 but the GPU factors in f32.
 * Same parameters as sprux_load_from_csr_f32 but values are double*.
 *
 * @return 0 on success, -1 on error
 */
int sprux_load_from_csr_f64_to_f32(sprux_solver_t h, const int64_t* ptrs, const int64_t* inds,
                                   const int64_t* blockSizes, const double* values, float* data);

/**
 * Full LU factorization (blocking).
 *
 * @param h       Solver handle
 * @param data    Factor data buffer (modified in place)
 * @param pivots  Pivot array [sprux_num_spans()]
 * @return 0 on success, -1 on error
 */
int sprux_factor_lu_f32(sprux_solver_t h, float* data, int64_t* pivots);

/**
 * Split-phase LU factorization, phase 1: submit GPU sparse elimination.
 * Returns immediately while GPU processes asynchronously.
 * CPU is free for other work (e.g., solving the previous matrix).
 * Must be followed by sprux_finish_factor_lu_f32().
 *
 * @param h       Solver handle
 * @param data    Factor data buffer
 * @param pivots  Pivot array [sprux_num_spans()]
 * @return 0 on success, -1 on error
 */
int sprux_begin_factor_lu_f32(sprux_solver_t h, float* data, int64_t* pivots);

/**
 * Split-phase LU factorization, phase 2: wait for GPU, run dense loop.
 * Must be called after sprux_begin_factor_lu_f32().
 *
 * @param h       Solver handle
 * @param data    Same buffer passed to begin
 * @param pivots  Same pivots passed to begin
 * @return 0 on success, -1 on error
 */
int sprux_finish_factor_lu_f32(sprux_solver_t h, float* data, int64_t* pivots);

/**
 * LU triangular solve (blocking).
 * Applies pivot permutation, then solves L and U in place.
 *
 * @param h       Solver handle
 * @param data    Factored data (from factor_lu or finish_factor_lu)
 * @param pivots  Pivot array (from factorization)
 * @param rhs     Right-hand side vector(s), overwritten with solution
 * @param stride  Leading dimension of rhs (must be >= order)
 * @param nrhs    Number of right-hand side columns
 * @return 0 on success, -1 on error
 */
int sprux_solve_lu_f32(sprux_solver_t h, const float* data, const int64_t* pivots, float* rhs,
                       int64_t stride, int nrhs);

/* =========================================================================
 * FFI Solver: encapsulated Metal LU with preprocessing + iterative refinement
 *
 * High-level API for circuit simulation (SPICE-like NR loops).
 * Caller provides f64 CSR matrix + f64 RHS, gets f64 solution.
 * Internally: BTF preprocessing, equilibration, f32 GPU factor/solve,
 * CPU f64 iterative refinement. Sparsity pattern is fixed at creation.
 * ========================================================================= */

typedef struct sprux_ffi_solver* sprux_ffi_solver_t;

/**
 * Create an FFI solver for a given sparsity pattern.
 *
 * Performs one-time setup: BTF max transversal, symmetric structure,
 * solver creation with auto-computed pivot threshold,
 * and MPS shader warmup (Metal) or CPU BLAS setup.
 *
 * @param n               Matrix dimension (n x n)
 * @param nnz             Number of non-zeros
 * @param csr_indptr      CSR row pointers [n+1], int32
 * @param csr_indices     CSR column indices [nnz], int32
 * @param csr_data_init   First matrix values [nnz], f64 (for pivot threshold)
 * @param max_refine_steps Number of iterative refinement steps (typically 1)
 * @return Solver handle, or NULL on error
 */
sprux_ffi_solver_t sprux_ffi_create(int32_t n, int32_t nnz, const int32_t* csr_indptr,
                                    const int32_t* csr_indices, const double* csr_data_init,
                                    int max_refine_steps);

/**
 * Destroy an FFI solver and free all associated resources.
 */
void sprux_ffi_destroy(sprux_ffi_solver_t h);

/**
 * Solve Ax = b with Metal-accelerated LU and iterative refinement.
 *
 * Equilibrates, factors in f32 on GPU, refines in f64 on CPU.
 * Returns near-f64 accuracy despite f32 factorization.
 *
 * @param h         Solver handle
 * @param csr_data  CSR non-zero values [nnz], f64
 * @param rhs       Right-hand side vector [n], f64
 * @param x_out     Solution vector [n], f64 (output)
 * @return 0 on success, -1 on error
 */
int sprux_ffi_solve(sprux_ffi_solver_t h, const double* csr_data, const double* rhs,
                    double* x_out);

/**
 * Sparse matrix-vector multiply: b_out = A @ x (CPU, f64).
 *
 * Uses the original CSR structure (no permutation).
 *
 * @param h         Solver handle
 * @param csr_data  CSR non-zero values [nnz], f64
 * @param x         Input vector [n], f64
 * @param b_out     Output vector [n], f64
 * @return 0 on success, -1 on error
 */
int sprux_ffi_dot(sprux_ffi_solver_t h, const double* csr_data, const double* x, double* b_out);

/**
 * Begin a GPU trace capture. Writes a .gputrace file for analysis in
 * Xcode or apple-profiler tools.
 *
 * @param output_path  Path for .gputrace output (e.g. "/tmp/sprux.gputrace")
 * @return 1 if capture started, 0 if not (Metal not available or already capturing)
 */
int sprux_begin_capture(const char* output_path);

/**
 * End a GPU trace capture started with sprux_begin_capture.
 */
void sprux_end_capture(void);

#ifdef __cplusplus
}
#endif
