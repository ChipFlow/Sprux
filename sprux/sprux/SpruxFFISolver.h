/*
 * Copyright (c) 2024-2026 ChipFlow
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>

namespace Sprux {

/**
 * High-level FFI solver for circuit simulation (SPICE-like NR loops).
 *
 * Encapsulates the full Metal-accelerated LU pipeline:
 *   - One-time setup: BTF max transversal, symmetric structure, solver creation,
 *     recording pass, MPS shader warmup, static pivot threshold
 *   - Per-solve: equilibration, f64→f32 CSR load, GPU factorLU + solveLU
 *     (Metal float32 only), CPU f64 iterative refinement (SpMV residual
 *     with original f64 matrix values), unpermute — recovers near-f64 accuracy
 *
 * The sparsity pattern (indptr, indices) is fixed at construction time.
 * Only CSR values and RHS change between solves.
 *
 * Reference implementation: benchmarking/LUBench.cpp:benchmarkLUMetalFFI
 */
class SpruxFFISolver {
 public:
  /**
   * Create solver for a given sparsity pattern.
   *
   * Performs all one-time setup:
   *  1. Convert int32 CSR to int64
   *  2. Compute BTF max transversal (row permutation for diagonal quality)
   *  3. Build symmetric SparseStructure for AMD ordering
   *  4. Compute static pivot threshold from equilibrated first matrix
   *  5. Create Solver with Metal backend, persistent contexts
   *  6. Recording pass (captures GemmWorkItem schedule)
   *  7. MPS warmup (forces shader JIT compilation)
   *
   * @param n            Matrix dimension (n x n)
   * @param nnz          Number of non-zeros
   * @param csr_indptr   CSR row pointers [n+1], int32
   * @param csr_indices  CSR column indices [nnz], int32
   * @param csr_data_init First matrix values [nnz], f64 — used for pivot threshold
   * @param max_refine_steps Max iterative refinement steps (default 1)
   * @param refine_tol Early termination tolerance for relative residual (default 1e-12).
   *                   Set to 0 to disable early termination and always run max steps.
   */
  SpruxFFISolver(int32_t n, int32_t nnz, const int32_t* csr_indptr, const int32_t* csr_indices,
                 const double* csr_data_init, int max_refine_steps = 1,
                 double refine_tol = 1e-12);

  ~SpruxFFISolver();

  // Not copyable or movable (owns GPU resources)
  SpruxFFISolver(const SpruxFFISolver&) = delete;
  SpruxFFISolver& operator=(const SpruxFFISolver&) = delete;

  /**
   * Solve Ax = b with Metal-accelerated LU and iterative refinement.
   *
   * Per-call workflow:
   *  1. Equilibrate CSR values (row/column scaling to O(1))
   *  2. Load equilibrated f32 values into solver's coalesced format
   *  3. Permute RHS by BTF row perm + AMD ordering + equilibration
   *  4. GPU: factorLU + solveLU (Metal command buffer)
   *  5. CPU: f64 SpMV residual with original matrix values
   *  6. GPU: solveLU on correction (refinement)
   *  7. Accumulate, unpermute, write f64 result
   *
   * @param csr_data  CSR non-zero values [nnz], f64
   * @param rhs       Right-hand side vector [n], f64
   * @param x_out     Solution vector [n], f64 (output)
   */
  /**
   * @return Number of refinement iterations actually performed (may be less than
   *         max_refine_steps if early termination triggered).
   */
  int solve(const double* csr_data, const double* rhs, double* x_out);

  /**
   * Sparse matrix-vector multiply: b_out = A @ x (CPU, f64).
   *
   * Uses the original CSR structure directly (no permutation).
   * Useful for residual computation in the caller.
   *
   * @param csr_data  CSR non-zero values [nnz], f64
   * @param x         Input vector [n], f64
   * @param b_out     Output vector [n], f64
   */
  void dot(const double* csr_data, const double* x, double* b_out);

  /** Matrix dimension. */
  int64_t n() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace Sprux
