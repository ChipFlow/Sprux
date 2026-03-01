/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <unordered_set>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/CsrTypes.h"
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/SupernodeMerger.h"

namespace BaSpaCho {

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

  // enable stat collection (default: disabled)
  void enableStats(bool enabled = true);

  // print some statistics about timings
  void printStats() const;

  // reset statistics
  void resetStats();

  // factor the data stored in the factor (Cholesky for SPD, LU for general)
  template <typename T>
  void factor(T* data, bool verbose = false) const;

  // factor using LU decomposition with partial pivoting (for MTYPE_GENERAL)
  // pivots array must be sized to numSpans()
  template <typename T>
  void factorLU(T* data, int64_t* pivots, bool verbose = false) const;

  // solve in place with LLt (vector must be permuted)
  template <typename T>
  void solve(const T* matData, T* vecData, int64_t stride, int nRHS) const;

  // solve in place with L (vector must be permuted)
  template <typename T>
  void solveL(const T* matData, T* vecData, int64_t stride, int nRHS) const;

  // solve in place with Lt (vector must be permuted)
  template <typename T>
  void solveLt(const T* matData, T* vecData, int64_t stride, int nRHS) const;

  // solve in place with LU factorization (applies P, then solves L, then U)
  // pivots array must match the one used in factorLU
  template <typename T>
  void solveLU(const T* matData, const int64_t* pivots, T* vecData, int64_t stride, int nRHS) const;

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

  // apply partial factor, up to a given span
  template <typename T>
  void factorUpTo(T* data, int64_t spanIndex, bool verbose = false) const;

  // apply partial solve (lower triangular), up to a given span
  template <typename T>
  void solveLUpTo(const T* data, int64_t spanIndex, T* vecData, int64_t stride, int nRHS) const;

  // apply partial solve (lower triangular), up to a given span
  template <typename T>
  void solveLtUpTo(const T* data, int64_t spanIndex, T* vecData, int64_t stride, int nRHS) const;

  // outVec += M * inVec, applying M's bottom right corner from `spanIndex`
  template <typename T>
  void addMvFrom(const T* matData, int64_t spanIndex, const T* inVecData, int64_t inStride,
                 T* outVecData, int64_t outStride, int nRHS, BaseType<T> alpha = 1.0) const;

  // apply pseudo-factor ( /= diagBlockLt where diagBlockLt has Lt factors of diagonal blocks)
  template <typename T>
  void pseudoFactorFrom(T* data, int64_t spanIndex, bool verbose = false) const;

  // factor from given spanIndex, only uses factor data from spanMatrixOffset(spanInedex)
  template <typename T>
  void factorFrom(T* data, int64_t spanIndex, bool verbose = false) const;

  // factor from given spanIndex, only uses factor data from spanMatrixOffset(spanInedex), and
  // vector data from spanVectorOffset(spanIndex)
  template <typename T>
  void solveLFrom(const T* data, int64_t spanIndex, T* vecData, int64_t stride, int nRHS) const;

  // factor from given spanIndex, only uses factor data from spanMatrixOffset(spanInedex), and
  // vector data from spanVectorOffset(spanIndex)
  template <typename T>
  void solveLtFrom(const T* data, int64_t spanIndex, T* vecData, int64_t stride, int nRHS) const;

  // order of the factor
  int64_t order() const { return factorSkel.order(); }

  // storge data size (lower triangle / L factor)
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

  // TESTING: return
  SymbolicCtx& internalSymbolicContext() { return *symCtx; }

  SymElimCtx& internalGetElimCtx(size_t i) {
    BASPACHO_CHECK_LT(i, elimCtxs.size());
    return *elimCtxs[i];
  }

  /**
   * Load values from CSR format into internal data buffer.
   *
   * Maps block values from CSR order (row-major within blocks, blocks in
   * CSR traversal order) to BaSpaCho's internal coalesced format.
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
 * provided data on the CPU. Ref/Blas engines in the other hand will only work with CPU data.
 **/
enum BackendType {
  BackendRef,     // reference implementation, not recommended
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
 * 1. CUDA (if compiled with BASPACHO_USE_CUBLAS and GPU available)
 * 2. Metal (if compiled with BASPACHO_USE_METAL on macOS with Apple Silicon)
 * 3. OpenCL (if compiled with BASPACHO_USE_OPENCL and GPU available)
 * 4. Fast (CPU with BLAS, always available)
 *
 * @return BackendType The detected best backend (never returns BackendAuto or BackendRef)
 */
BackendType detectBestBackend();

/**
 * Policy on fill adding to sparse matrix structure. Note that this controls the factor's sparse
 * structure, and therefore if the solver will support total/partial factor
 **/
enum AddFillPolicy {
  AddFillComplete,       // add fill for complete factoring, reorder
  AddFillForAutoElims,   // add fill for give+auto elim-ranges, reorder
  AddFillForGivenElims,  // fill for elimination of elim ranges, no reorder
  AddFillNone,           // no fill added, no reorder
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

}  // end namespace BaSpaCho
