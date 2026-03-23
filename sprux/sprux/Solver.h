/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright (c) Robert Taylor, 2026. All rights reserved.
// Licensed under the MIT license found in the LICENSE file.

#pragma once

#include <memory>
#include <unordered_set>
#include "sprux/sprux/CoalescedBlockMatrix.h"
#include "sprux/sprux/CsrTypes.h"
#include "sprux/sprux/MatOps.h"
#include "sprux/sprux/SparseStructure.h"
#include "sprux/sprux/SupernodeMerger.h"

namespace Sprux {

/**
 * @brief Where pivot data resides (host or device memory).
 *
 * Used by persistent-context factorLU/solveLU overloads to control whether
 * pivot data is copied between host and device memory.
 *
 * - **Host**: pivots are in host (CPU) memory. Standard behavior — the GPU
 *   backend copies pivots to host after factorization and uploads them before solve.
 * - **Device**: pivots remain in device (GPU) memory throughout. Eliminates
 *   D2H→H2D roundtrips when the caller doesn't need to inspect pivots on CPU.
 *   `devPivots` must be device-allocated with at least `numSpans()` int64_t elements.
 */
enum class PivotLocation { Host, Device };

/**
 * @brief Class solver represents a symbolic decomposition with the operations required to
 * operate on (externally allocated) numeric matrix/vector data.
 *
 * You will never have to create a Solver class yourself, use the createSolver function below,
 * which performs the symbolic analysis in order to create a proper factor, computing a param
 * reordering, and adding required fill to the reordered sparse structure.
 *
 * Note that this is a low-level interface, and does not provide reordering of numeric data,
 * and all solve functions assume internal ordering. This might not be required when solving
 * eg. see optimizer example.
 *
 * The provided parameters are called `spans` in the factor, so for convenience and clarity
 * we say that the reordering maps the user parameter index to the internal span index:
 *   spanIndex = reord[spanIndex].
 */
class Solver {
 public:
  // constructor, from RAW factor skeleton (do not call directly, use createSolver)
  Solver(CoalescedBlockMatrixSkel&& factorSkel, std::vector<int64_t>&& sparseElimRanges,
         std::vector<int64_t>&& permutation, OpsPtr&& ops, int64_t canFactorUpTo = -1,
         LevelSetSchedule&& levelSetSchedule = {}, double staticPivotThreshold = -1.0);

  // return a (permuted) accessor to access factor's block (re-ordering is auto-applied)
  PermutedCoalescedAccessor accessor() const {
    PermutedCoalescedAccessor retv;
    retv.init(factorSkel.accessor(), permutation.data());
    return retv;
  }

  // return an accessor to be used by an on-device kernel (if supported by backend)
  PermutedCoalescedAccessor deviceAccessor() const { return symCtx->deviceAccessor(); }

  // Set the CUDA stream for all GPU operations (cuBLAS, cuSOLVER, kernels).
  // Must be called before factorLU/solveLU when using a non-default stream
  // (e.g., JAX's XLA stream from the FFI plugin). No-op for CPU backends.
  void setStream(void* stream) { symCtx->setStream(stream); }

  // enable stat collection (default: disabled)
  void enableStats(bool enabled = true);

  // print some statistics about timings
  void printStats() const;

  // reset statistics
  void resetStats();

  /**
   * @brief Cholesky factorization for SPD matrices (A = L * L^T).
   *
   * Computes the Cholesky factor L in place. The solver must have been created
   * with `settings.matrixType = MTYPE_SPD` (the default).
   *
   * @param matData  Factor data buffer (modified in place). Must be sized to dataSize().
   *                 Must contain the lower triangle in internal ordering.
   * @param verbose  If true, prints timing information to stdout.
   */
  template <typename T>
  void factor(T* matData, bool verbose = false) const;

  /**
   * @brief LU factorization with partial pivoting for general (non-symmetric) matrices.
   *
   * Computes A = P * L * U in place. The solver must have been created
   * with `settings.matrixType = MTYPE_GENERAL`.
   *
   * @param data    Factor data buffer (modified in place). Must be sized to totalDataSize()
   *                and contain both lower and upper triangle data.
   * @param pivots  Pivot permutation output array. Must be sized to numSpans().
   * @param verbose If true, prints timing information to stdout.
   */
  template <typename T>
  void factorLU(T* data, int64_t* pivots, bool verbose = false) const;

  /**
   * @brief LU factorization with persistent numeric context (avoids per-call allocation).
   *
   * Same as factorLU() but reuses a caller-provided NumericCtx across calls,
   * eliminating per-call cudaMalloc/cudaFreeHost overhead on GPU backends.
   * Create the context via `solver->internalSymbolicContext().createNumericCtx<T>(...)`.
   *
   * @param data    Factor data buffer (modified in place). Must be sized to totalDataSize().
   * @param pivots  Pivot permutation output array. Must be sized to numSpans().
   * @param ctx     Numeric context to reuse. Reset internally on each call.
   * @param verbose If true, prints timing information to stdout.
   */
  template <typename T>
  void factorLU(T* data, int64_t* pivots, NumericCtx<T>& ctx, bool verbose = false) const;

  /**
   * @brief LU factorization with device-resident pivots (avoids D2H pivot copy).
   *
   * Same as the persistent-context overload but keeps pivots in device (GPU) memory,
   * avoiding the device-to-host copy at the end of factorization.
   *
   * @param data      Factor data buffer (modified in place). Must be sized to totalDataSize().
   * @param devPivots Device-allocated pivot array. Must have at least numSpans() int64_t elements.
   * @param ctx       Numeric context to reuse.
   * @param pivLoc    Must be PivotLocation::Device.
   * @param verbose   If true, prints timing information to stdout.
   */
  template <typename T>
  void factorLU(T* data, int64_t* devPivots, NumericCtx<T>& ctx, PivotLocation pivLoc,
                bool verbose = false) const;

  /**
   * @brief Begin LU factorization (phase 1): submit sparse elimination to GPU.
   *
   * Creates the numeric context, computes the static pivot threshold, and
   * dispatches sparse elimination kernels to the GPU. The GPU work is submitted
   * but NOT waited on — it runs asynchronously.
   *
   * Call finishFactorLU() to complete the factorization (waits for GPU, runs
   * dense loop). Between beginFactorLU and finishFactorLU, the GPU is busy
   * processing sparse elimination while the CPU is free for other work (e.g.,
   * solving the previous matrix).
   *
   * @param data Matrix data buffer
   * @param pivots Pivot array (sized to numSpans())
   * @param verbose If true, prints timing information
   */
  template <typename T>
  void beginFactorLU(T* data, int64_t* pivots, bool verbose = false) const;

  /**
   * @brief Finish LU factorization (phase 2): wait for GPU, run dense loop.
   *
   * Must be called after beginFactorLU(). Waits for the GPU sparse elimination
   * to complete, then runs the dense factorization loop on CPU.
   *
   * @param data Same data buffer passed to beginFactorLU
   * @param pivots Same pivots array passed to beginFactorLU
   * @param verbose If true, prints timing information
   */
  template <typename T>
  void finishFactorLU(T* data, int64_t* pivots, bool verbose = false) const;

  /**
   * @brief Solve A*x = b using Cholesky factorization (L * L^T).
   *
   * Solves in place: forward substitution with L, then backward substitution with L^T.
   * The vector must already be permuted to internal ordering.
   *
   * @param matData Factored matrix data (from factor())
   * @param vecData Right-hand side vector(s), overwritten with solution. Must be in permuted ordering.
   * @param stride  Leading dimension of vecData (must be >= order())
   * @param nRHS    Number of right-hand side vectors (columns)
   */
  template <typename T>
  void solve(const T* matData, T* vecData, int64_t stride, int nRHS) const;

  /**
   * @brief Solve L*x = b (forward substitution only).
   *
   * Applies only the lower-triangular solve. The vector must be in permuted ordering.
   * Useful for partial solves in domain decomposition or preconditioning.
   *
   * @param matData Factored matrix data (from factor())
   * @param vecData Right-hand side vector(s), overwritten with solution
   * @param stride  Leading dimension of vecData (must be >= order())
   * @param nRHS    Number of right-hand side vectors
   */
  template <typename T>
  void solveL(const T* matData, T* vecData, int64_t stride, int nRHS) const;

  /**
   * @brief Solve L^T * x = b (backward substitution only).
   *
   * Applies only the upper-triangular (transpose of L) solve. The vector must be
   * in permuted ordering. Useful for partial solves in domain decomposition.
   *
   * @param matData Factored matrix data (from factor())
   * @param vecData Right-hand side vector(s), overwritten with solution
   * @param stride  Leading dimension of vecData (must be >= order())
   * @param nRHS    Number of right-hand side vectors
   */
  template <typename T>
  void solveLt(const T* matData, T* vecData, int64_t stride, int nRHS) const;

  /**
   * @brief Solve A*x = b using LU factorization with partial pivoting.
   *
   * Applies the pivot permutation P, then forward substitution with L (unit lower
   * triangular), then backward substitution with U. The pivots array must be the
   * same one produced by factorLU().
   *
   * @param matData  Factored matrix data (from factorLU())
   * @param pivots   Pivot permutation array (from factorLU()). Must have numSpans() elements.
   * @param vecData  Right-hand side vector(s), overwritten with solution
   * @param stride   Leading dimension of vecData (must be >= order())
   * @param nRHS     Number of right-hand side vectors
   */
  template <typename T>
  void solveLU(const T* matData, const int64_t* pivots, T* vecData, int64_t stride, int nRHS) const;

  /**
   * @brief Solve with LU factorization using a persistent solve context.
   *
   * Same as solveLU() but reuses a caller-provided SolveCtx across calls,
   * eliminating per-call temporary allocation overhead on GPU backends.
   * Create the context via `solver->internalSymbolicContext().createSolveCtx<T>(...)`.
   *
   * @param matData  Factored matrix data (from factorLU())
   * @param pivots   Pivot permutation array (from factorLU())
   * @param vecData  Right-hand side vector(s), overwritten with solution
   * @param stride   Leading dimension of vecData (must be >= order())
   * @param nRHS     Number of right-hand side vectors
   * @param ctx      Solve context to reuse across calls
   */
  template <typename T>
  void solveLU(const T* matData, const int64_t* pivots, T* vecData, int64_t stride, int nRHS,
               SolveCtx<T>& ctx) const;

  /**
   * @brief Solve with LU factorization using device-resident pivots.
   *
   * Same as the persistent-context overload but reads pivots from device (GPU) memory,
   * avoiding the host-to-device pivot upload at the start of each solve.
   *
   * @param matData    Factored matrix data (from factorLU())
   * @param devPivots  Device-allocated pivot array (from factorLU with PivotLocation::Device)
   * @param vecData    Right-hand side vector(s), overwritten with solution
   * @param stride     Leading dimension of vecData (must be >= order())
   * @param nRHS       Number of right-hand side vectors
   * @param ctx        Solve context to reuse across calls
   * @param pivLoc     Must be PivotLocation::Device
   */
  template <typename T>
  void solveLU(const T* matData, const int64_t* devPivots, T* vecData, int64_t stride, int nRHS,
               SolveCtx<T>& ctx, PivotLocation pivLoc) const;

  /**
   * @brief Factor using LDL^T decomposition for symmetric indefinite matrices.
   *
   * Computes A = L * D * L^T where:
   * - L is unit lower triangular (stored below diagonal, diagonal implicitly 1)
   * - D is diagonal (stored on diagonal, can have negative entries)
   *
   * Uses the same lower-triangle storage as Cholesky, making it a drop-in
   * replacement for applications that only store the lower triangle.
   *
   * Use this instead of Cholesky (factor()) when:
   * - The matrix may have negative eigenvalues (e.g., Hessians at saddle points)
   * - You need to handle general symmetric matrices, not just SPD
   * - The matrix definiteness is unknown or variable
   *
   * Throws std::runtime_error if a zero pivot is encountered (matrix is singular).
   *
   * Note: Currently uses dense elimination for each lump. The sparse elimination
   * path is not yet implemented for LDL^T.
   *
   * @param data Matrix data buffer (modified in place with L and D factors)
   * @param verbose If true, prints timing information
   */
  template <typename T>
  void factorLDLT(T* data, bool verbose = false) const;

  /**
   * @brief Solve in place with LDL^T factorization.
   *
   * Solves A*x = b where A = L*D*L^T by computing:
   * 1. Forward substitution: L*y = b (unit lower triangular)
   * 2. Diagonal solve: D*z = y
   * 3. Backward substitution: L^T*x = z (unit upper triangular)
   *
   * The solution overwrites the input vector.
   *
   * @param matData Factored matrix data (from factorLDLT)
   * @param vecData Right-hand side vector(s), overwritten with solution
   * @param stride Leading dimension of vecData (must be >= order())
   * @param nRHS Number of right-hand side vectors
   */
  template <typename T>
  void solveLDLT(const T* matData, T* vecData, int64_t stride, int nRHS) const;

  /**
   * @brief Partial Cholesky factorization up to a given span index.
   *
   * Factors only spans [0, spanIndex). Useful for incremental/partial factorization
   * in domain decomposition or Schur complement methods.
   *
   * @param data      Factor data buffer (modified in place)
   * @param spanIndex Upper bound span index (exclusive)
   * @param verbose   If true, prints timing information
   */
  template <typename T>
  void factorUpTo(T* data, int64_t spanIndex, bool verbose = false) const;

  /**
   * @brief Partial forward substitution (L solve) up to a given span index.
   *
   * Solves L*x = b for spans [0, spanIndex) only.
   *
   * @param data      Factored matrix data
   * @param spanIndex Upper bound span index (exclusive)
   * @param vecData   Right-hand side vector(s), overwritten with partial solution
   * @param stride    Leading dimension of vecData
   * @param nRHS      Number of right-hand side vectors
   */
  template <typename T>
  void solveLUpTo(const T* data, int64_t spanIndex, T* vecData, int64_t stride, int nRHS) const;

  /**
   * @brief Partial backward substitution (L^T solve) up to a given span index.
   *
   * Solves L^T * x = b for spans [0, spanIndex) only.
   *
   * @param data      Factored matrix data
   * @param spanIndex Upper bound span index (exclusive)
   * @param vecData   Right-hand side vector(s), overwritten with partial solution
   * @param stride    Leading dimension of vecData
   * @param nRHS      Number of right-hand side vectors
   */
  template <typename T>
  void solveLtUpTo(const T* data, int64_t spanIndex, T* vecData, int64_t stride, int nRHS) const;

  /**
   * @brief Sparse matrix-vector multiply: outVec += alpha * M * inVec, from a given span.
   *
   * Applies only the bottom-right corner of the matrix starting from `spanIndex`.
   * Used in domain decomposition for computing Schur complement contributions.
   *
   * @param matData    Factor data
   * @param spanIndex  Starting span index
   * @param inVecData  Input vector
   * @param inStride   Leading dimension of input vector
   * @param outVecData Output vector (accumulated into)
   * @param outStride  Leading dimension of output vector
   * @param nRHS       Number of right-hand side vectors
   * @param alpha      Scalar multiplier (default 1.0)
   */
  template <typename T>
  void addMvFrom(const T* matData, int64_t spanIndex, const T* inVecData, int64_t inStride,
                 T* outVecData, int64_t outStride, int nRHS, BaseType<T> alpha = 1.0) const;

  /**
   * @brief Pseudo-factorization from a given span index.
   *
   * Divides off-diagonal blocks by the L^T factors of the corresponding diagonal blocks.
   * Used as a preprocessing step in domain decomposition.
   *
   * @param data      Factor data (modified in place)
   * @param spanIndex Starting span index
   * @param verbose   If true, prints timing information
   */
  template <typename T>
  void pseudoFactorFrom(T* data, int64_t spanIndex, bool verbose = false) const;

  /**
   * @brief Factor from a given span index onwards.
   *
   * Only processes spans [spanIndex, end). Uses factor data starting from
   * spanMatrixOffset(spanIndex).
   *
   * @param data      Factor data (modified in place)
   * @param spanIndex Starting span index
   * @param verbose   If true, prints timing information
   */
  template <typename T>
  void factorFrom(T* data, int64_t spanIndex, bool verbose = false) const;

  /**
   * @brief Forward substitution (L solve) from a given span index onwards.
   *
   * Only processes spans [spanIndex, end). Uses factor data from
   * spanMatrixOffset(spanIndex) and vector data from spanVectorOffset(spanIndex).
   *
   * @param data      Factored matrix data
   * @param spanIndex Starting span index
   * @param vecData   Right-hand side vector(s), overwritten with partial solution
   * @param stride    Leading dimension of vecData
   * @param nRHS      Number of right-hand side vectors
   */
  template <typename T>
  void solveLFrom(const T* data, int64_t spanIndex, T* vecData, int64_t stride, int nRHS) const;

  /**
   * @brief Backward substitution (L^T solve) from a given span index onwards.
   *
   * Only processes spans [spanIndex, end). Uses factor data from
   * spanMatrixOffset(spanIndex) and vector data from spanVectorOffset(spanIndex).
   *
   * @param data      Factored matrix data
   * @param spanIndex Starting span index
   * @param vecData   Right-hand side vector(s), overwritten with partial solution
   * @param stride    Leading dimension of vecData
   * @param nRHS      Number of right-hand side vectors
   */
  template <typename T>
  void solveLtFrom(const T* data, int64_t spanIndex, T* vecData, int64_t stride, int nRHS) const;

  /// Number of spans (parameter blocks after reordering). Pivot arrays must be this size.
  int64_t numSpans() const { return factorSkel.numSpans(); }

  // order of the factor
  int64_t order() const { return factorSkel.order(); }

  // storage data size (lower triangle / L factor)
  int64_t dataSize() const { return factorSkel.dataSize(); }

  // storage data size for upper triangle / U factor (0 for symmetric matrices)
  int64_t upperDataSize() const { return factorSkel.upperDataSize(); }

  // total storage size (lower + upper for general, just lower for symmetric)
  int64_t totalDataSize() const { return factorSkel.totalDataSize(); }

  // return the matrix type (SPD, SYMMETRIC, or GENERAL)
  MatrixType matrixType() const { return factorSkel.matrixType; }

  // returns the upper span index limit for proper factorization (if the factor doesn't have fill
  // for full factorization this might not include all parameters)
  int64_t canFactorUpToSpan() const { return canFactorUpTo; }

  // offset of span vector data
  int64_t spanVectorOffset(int64_t spanIndex) const {
    return factorSkel.spanVectorOffset(spanIndex);
  }

  // offset of span matrix data
  int64_t spanMatrixOffset(int64_t spanIndex) const {
    return factorSkel.spanMatrixOffset(spanIndex);
  }

  // return sparse structure of the factor
  const CoalescedBlockMatrixSkel& skel() const { return factorSkel; }

  // return the (span/lump) ranges set to undergo sparse elimination
  const std::vector<int64_t>& sparseEliminationRanges() const { return sparseElimRanges; }

  // return the reordering applied to parameters (i's position is perm[i] in the factor)
  const std::vector<int64_t>& paramToSpan() const { return permutation; }

  // return the level-set schedule for parallel factorization
  const LevelSetSchedule& levelSetSchedule() const { return levelSetSchedule_; }

  // return the count of diagonal elements perturbed during the last factorLU call
  int64_t staticPivotPerturbCount() const { return staticPivotPerturbCount_; }

  // TESTING: return the internal symbolic context for advanced use cases
  SymbolicCtx& internalSymbolicContext() { return *symCtx; }

  SymElimCtx& internalGetElimCtx(size_t i) {
    SPRUX_CHECK_LT(i, elimCtxs.size());
    return *elimCtxs[i];
  }

  /**
   * Load values from CSR format into internal data buffer.
   *
   * Maps block values from CSR order (row-major within blocks, blocks in
   * CSR traversal order) to Sprux's internal coalesced format.
   *
   * @param csrRowStart  CSR row pointers [numBlocks+1]
   * @param csrColInds   CSR column indices [numBlockNonzeros]
   * @param blockSizes   Size of each block [numBlocks]
   * @param csrValues    Values in CSR order (row-major within blocks)
   * @param data         Output data buffer (must be sized to dataSize())
   */
  template <typename T>
  void loadFromCsr(const int64_t* csrRowStart, const int64_t* csrColInds,
                   const int64_t* blockSizes, const T* csrValues, T* data) const;

  /**
   * Extract values to CSR format from internal data buffer.
   *
   * Inverse of loadFromCsr - extracts block values from internal format
   * to CSR order.
   *
   * @param csrRowStart  CSR row pointers [numBlocks+1]
   * @param csrColInds   CSR column indices [numBlockNonzeros]
   * @param blockSizes   Size of each block [numBlocks]
   * @param data         Input data buffer
   * @param csrValues    Output CSR values (must be pre-sized)
   */
  template <typename T>
  void extractToCsr(const int64_t* csrRowStart, const int64_t* csrColInds,
                    const int64_t* blockSizes, const T* data, T* csrValues) const;

 private:
  void initElimination();

  int64_t boardElimTempSize(int64_t lump, int64_t boardIndexInSN) const;

  template <typename T>
  void factorLump(NumericCtx<T>& numCtx, T* data, int64_t lump) const;

  template <typename T>
  void eliminateBoard(NumericCtx<T>& numCtx, T* data, int64_t ptr) const;

  template <typename T>
  void internalFactorRange(T* data, int64_t startSpanIndex, int64_t endSpanIndex,
                           bool verbose = false) const;

  template <typename T>
  void internalSolveLRange(SolveCtx<T>& slvCtx, const T* data, int64_t startSpanIndex,
                           int64_t endSpanIndex, T* vecData, int64_t stride, int nRHS) const;

  template <typename T>
  void internalSolveLtRange(SolveCtx<T>& slvCtx, const T* data, int64_t startSpanIndex,
                            int64_t endSpanIndex, T* vecData, int64_t stride, int nRHS) const;

  // For LU solve: uses unit lower triangular L (diagonal = 1)
  template <typename T>
  void internalSolveLRangeUnit(SolveCtx<T>& slvCtx, const T* data, int64_t startSpanIndex,
                               int64_t endSpanIndex, T* vecData, int64_t stride, int nRHS) const;

  // LU factorization internal methods
  template <typename T>
  void factorLumpLU(NumericCtx<T>& numCtx, T* data, int64_t* pivots, int64_t lump) const;

  template <typename T>
  void eliminateBoardLU(NumericCtx<T>& numCtx, T* data, int64_t ptr) const;

  template <typename T>
  void internalFactorRangeLU(T* data, int64_t* pivots, int64_t startSpanIndex, int64_t endSpanIndex,
                             bool verbose = false) const;

  // Split-phase internal methods for pipelined LU factorization
  template <typename T>
  void beginInternalFactorRangeLU(T* data, int64_t* pivots, int64_t startSpanIndex,
                                  int64_t endSpanIndex, bool verbose = false) const;

  template <typename T>
  void finishInternalFactorRangeLU(T* data, int64_t* pivots, int64_t startSpanIndex,
                                   int64_t endSpanIndex, bool verbose = false) const;

  template <typename T>
  void internalSolveURange(SolveCtx<T>& slvCtx, const T* data, int64_t startSpanIndex,
                           int64_t endSpanIndex, T* vecData, int64_t stride, int nRHS) const;

  // LDL^T factorization internal methods
  template <typename T>
  void factorLumpLDLT(NumericCtx<T>& numCtx, T* data, int64_t lump) const;

  template <typename T>
  void eliminateBoardLDLT(NumericCtx<T>& numCtx, T* data, int64_t ptr) const;

  template <typename T>
  void internalFactorRangeLDLT(T* data, int64_t startSpanIndex, int64_t endSpanIndex,
                               bool verbose = false) const;

  template <typename T>
  void internalSolveLRangeLDLT(SolveCtx<T>& slvCtx, const T* data, int64_t startSpanIndex,
                               int64_t endSpanIndex, T* vecData, int64_t stride, int nRHS) const;

  template <typename T>
  void internalSolveDRange(SolveCtx<T>& slvCtx, const T* data, int64_t startSpanIndex,
                           int64_t endSpanIndex, T* vecData, int64_t stride, int nRHS) const;

  template <typename T>
  void internalSolveLtRangeLDLT(SolveCtx<T>& slvCtx, const T* data, int64_t startSpanIndex,
                                int64_t endSpanIndex, T* vecData, int64_t stride, int nRHS) const;

  CoalescedBlockMatrixSkel factorSkel;
  std::vector<int64_t> sparseElimRanges;
  std::vector<int64_t> permutation;  // *on indices*: v'[p[i]] = v[i];
  int64_t canFactorUpTo;
  LevelSetSchedule levelSetSchedule_;
  double staticPivotThreshold_;
  mutable int64_t staticPivotPerturbCount_ = 0;
  mutable double effectiveStaticPivotThreshold_ = 0.0;

  // Pending numeric context for split-phase factorization (beginFactorLU/finishFactorLU).
  // Type-erased to support both float and double template instantiations.
  mutable std::unique_ptr<NumericCtxBase> pendingNumCtx_;

  OpsPtr ops;
  SymbolicCtxPtr symCtx;
  std::vector<SymElimCtxPtr> elimCtxs;
  std::vector<SymElimCtxPtr> luElimCtxs;  // LU sparse elimination contexts (for MTYPE_GENERAL)
  std::vector<int64_t> startElimRowPtr;
  int64_t maxElimTempSize;
};

using SolverPtr = std::unique_ptr<Solver>;

/**
 * The backend type selectes the engine that will be used for numerical operations. Note that
 * device (Cuda/Metal) engines will expect memory allocated on the device, and will crash when
 * provided data on the CPU. The Blas engine on the other hand will only work with CPU data.
 **/
enum BackendType {
  BackendFast,    // CPU with BLAS (recommended for CPU)
  BackendCuda,    // NVIDIA GPU with cuBLAS (float and double)
  BackendMetal,   // Apple Metal GPU backend (macOS/iOS, float only)
  BackendOpenCL,  // OpenCL GPU backend with CLBlast (portable, float and double)
  BackendAuto,    // Automatically select best available backend
};

/**
 * @brief Detect the best available backend for the current system.
 *
 * Priority order:
 * 1. CUDA (if compiled with SPRUX_USE_CUBLAS and GPU available)
 * 2. Metal (if compiled with SPRUX_USE_METAL on macOS with Apple Silicon)
 * 3. OpenCL (if compiled with SPRUX_USE_OPENCL and GPU available)
 * 4. Fast (CPU with BLAS, always available)
 *
 * @return BackendType The detected best backend (never returns BackendAuto)
 */
BackendType detectBestBackend();

/**
 * @brief Policy controlling how fill is added to the sparse structure during symbolic analysis.
 *
 * Fill entries are additional nonzeros that arise during factorization. This policy
 * controls whether createSolver() computes and adds them, which determines whether
 * the solver can perform complete or only partial factorization.
 */
enum AddFillPolicy {
  /// Compute fill-reducing ordering and add all fill needed for complete factorization.
  /// Required if using elimLastIds in createSolver().
  AddFillComplete,

  /// Add fill for both user-specified and auto-detected sparse elimination ranges.
  /// Includes fill-reducing reordering. Supports partial factorization up to the
  /// end of the elimination ranges.
  AddFillForAutoElims,

  /// Add fill only for user-specified elimination ranges. No reordering.
  /// Supports partial factorization up to the end of the given ranges.
  AddFillForGivenElims,

  /// No fill added, no reordering. Use when the sparsity pattern already
  /// includes all necessary fill (e.g., from an external symbolic analysis).
  AddFillNone,
};

// forward def, represent the computation model to tune for
struct ComputationModel;

/**
 * Settings for `createSolver` function below (symbolic analysis of a sparse block matrix).
 */
struct Settings {
  bool findSparseEliminationRanges = true;
  int numThreads = 16;
  BackendType backend = BackendFast;
  AddFillPolicy addFillPolicy = AddFillComplete;
  const ComputationModel* computationModel = nullptr;
  double supernodeMergeFillTolerance = 0.0;  // max extra-zero fraction: 0.0 = exact only, 0.25 = 25%
  int64_t maxSupernodeSize = 0;              // 0 = merging disabled, 256 = typical max size
  MatrixType matrixType = MTYPE_SPD;         // MTYPE_SPD for Cholesky, MTYPE_GENERAL for LU
  double staticPivotThreshold = -1.0;  // <0 disabled, 0 = auto (sqrt(eps)), >0 = manual
};

/**
 * @brief Primary entry point for creating a solver. Performs symbolic analysis
 * (fill-reducing reordering, elimination tree, supernode merging) and creates
 * the solver with proper fill structure.
 *
 * Supports both SPD (Cholesky) and general (LU) matrices via settings.matrixType.
 * For general matrices, automatically initializes upper triangle storage.
 *
 * @param settings settings as explained above
 * @param paramSizes vector with size of the n-th parameter block
 * @param ss a csr structure (ptrs/inds) representing the *blocks*
 * @param sparseElimRanges [a_0, a_1, ..., a_n] where [a_i,a_{i+1}] will be treated as a sparse
 * elimination range
 * @param elimLastIds ids to be kept at the very end in reordering, allowing partial factor
 * that will eliminate all other parameters. Must all be after sparse elimination ranges. If
 * non-empty then the settings.addFillPolicy MUST be AddFillComplete (there is no point in
 * having this set if it's not possible to eliminate up to there).
 * @return SolverPtr
 */
SolverPtr createSolver(const Settings& settings, const std::vector<int64_t>& paramSizes,
                       const SparseStructure& ss, const std::vector<int64_t>& sparseElimRanges = {},
                       const std::unordered_set<int64_t>& elimLastIds = {});

}  // end namespace Sprux
