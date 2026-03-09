/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <cxxabi.h>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <vector>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"

namespace BaSpaCho {

struct Ops;
struct SymbolicCtx;
struct SymElimCtx;
template <typename T>
struct NumericCtx;
template <typename T>
struct SolveCtx;
using OpsPtr = std::unique_ptr<Ops>;
using SymbolicCtxPtr = std::unique_ptr<SymbolicCtx>;
using SymElimCtxPtr = std::unique_ptr<SymElimCtx>;
template <typename T>
using NumericCtxPtr = std::unique_ptr<NumericCtx<T>>;
template <typename T>
using SolveCtxPtr = std::unique_ptr<SolveCtx<T>>;

template <typename T>
struct Batch {
  using BaseType = T;
  static int getSize(const T*) { return 1; }
};

template <typename T>
struct Batch<std::vector<T*>> {
  using BaseType = T;
  static int getSize(const std::vector<T*>* data) { return data->size(); }
};

template <typename T>
using BaseType = typename Batch<T>::BaseType;

// generator class for operation contexts
struct Ops {
  virtual ~Ops() {}

  // creates a symbolic context from a matrix structure
  virtual SymbolicCtxPtr createSymbolicCtx(const CoalescedBlockMatrixSkel& skel,
                                           const std::vector<int64_t>& permutation) = 0;
};

struct NumericCtxBase {
  virtual ~NumericCtxBase() {}
  // Flush pending GPU operations (no-op for CPU backends)
  virtual void flush() {}
  // Get count of deferred perturbSmallDiagonals perturbations (GPU backends
  // defer the count to avoid per-lump CPU sync). Returns 0 for CPU backends.
  virtual int64_t deferredPerturbCount() { return 0; }
  // Pre-allocate all GPU buffers to max needed sizes so no cudaMalloc occurs
  // during the hot factorization path. Required for CUDA graph capture.
  // No-op for CPU backends.
  virtual void preAllocateForLU(int64_t maxDenseBlockSize, int64_t totalDensePivots) {
    (void)maxDenseBlockSize;
    (void)totalDensePivots;
  }
};

struct SolveCtxBase {
  virtual ~SolveCtxBase() {}
  // Flush pending GPU operations (no-op for CPU backends)
  virtual void flush() {}
};

// (symbolic) context for factorization, constant indices (and GPU copies)
struct SymbolicCtx {
  virtual ~SymbolicCtx() {}

  // Set the stream for all GPU operations (cuBLAS, cuSOLVER, kernel launches).
  // Must be called before factorLU/solveLU when using a non-default stream.
  // No-op for CPU backends.
  virtual void setStream(void* stream) { (void)stream; }

  // prepares data for a parallel elimination op (Cholesky)
  virtual SymElimCtxPtr prepareElimination(int64_t lumpsBegin, int64_t lumpsEnd) = 0;

  // prepares data for LU sparse elimination (non-symmetric matrices)
  // Default: returns nullptr (backend doesn't support LU sparse elimination)
  virtual SymElimCtxPtr prepareLUElimination(int64_t lumpsBegin, int64_t lumpsEnd) {
    (void)lumpsBegin;
    (void)lumpsEnd;
    return nullptr;
  }

  virtual NumericCtxBase* createNumericCtxForType(std::type_index tIdx, int64_t tempBufSize,
                                                  int batchSize) = 0;

  virtual SolveCtxBase* createSolveCtxForType(std::type_index tIdx, int nRHS, int batchSize) = 0;

  virtual PermutedCoalescedAccessor deviceAccessor() = 0;

  template <typename T>
  NumericCtxPtr<T> createNumericCtx(int64_t tempBufSize, const T* data);

  template <typename T>
  SolveCtxPtr<T> createSolveCtx(int nRHS, const T* data);

  mutable OpStat<int, int> potrfStat;
  mutable int64_t potrfBiggestN = 0;
  mutable OpStat<int, int, int> trsmStat;
  mutable OpStat<int, int, int, int> sygeStat;
  mutable int64_t gemmCalls = 0;
  mutable int64_t syrkCalls = 0;
  mutable OpStat<int, int, int> asmblStat;

  // LU factorization stats
  mutable OpStat<int, int> getrfStat;
  mutable int64_t getrfBiggestN = 0;
  mutable int64_t luGemmCalls = 0;

  mutable OpStat<> solveSparseLStat;
  mutable OpStat<> solveSparseLtStat;
  mutable OpStat<> pseudoFactorStat;
  mutable OpStat<> symmStat;
  mutable OpStat<> solveLStat;
  mutable OpStat<> solveLtStat;
  mutable OpStat<> solveGemvStat;
  mutable OpStat<> solveGemvTStat;
  mutable OpStat<> solveAssVStat;
  mutable OpStat<> solveAssVTStat;
};

// (symbolic) context for sparse elimination of a range of parameters
struct SymElimCtx {
  virtual ~SymElimCtx() {}

  mutable OpStat<> elimStat;
};

// ops and contexts depending on the float/double type
template <typename T>
struct NumericCtx : NumericCtxBase {
  virtual ~NumericCtx() {}

  // does 1. diag factor 2. solve on colunm
  virtual void pseudoFactorSpans(T* data, int64_t spanBegin, int64_t spanEnd) = 0;

  // does (possibly parallel) elimination on a lump of aggregs
  virtual void doElimination(const SymElimCtx& elimData, T* data, int64_t lumpsBegin,
                             int64_t lumpsEnd) = 0;

  // Batch all Cholesky sparse elimination levels into a single GPU submission.
  // Default: calls doElimination per level (CPU/non-batching backends).
  virtual void doAllEliminations(const std::vector<SymElimCtxPtr>& elimCtxs,
                                 const std::vector<int64_t>& ranges, T* data) {
    for (size_t l = 0; l + 1 < ranges.size(); l++) {
      if (elimCtxs[l]) {
        doElimination(*elimCtxs[l], data, ranges[l], ranges[l + 1]);
      }
    }
  }

  // LU sparse elimination for non-symmetric matrices (1x1 scalar lumps)
  // Uses L from lower chain and U from upper chain for Schur complement
  virtual void doEliminationLU(const SymElimCtx& elimData, T* data, int64_t lumpsBegin,
                               int64_t lumpsEnd, T staticPivotThreshold,
                               int64_t& perturbCount) {
    (void)elimData;
    (void)data;
    (void)lumpsBegin;
    (void)lumpsEnd;
    (void)staticPivotThreshold;
    (void)perturbCount;
    throw std::runtime_error("doEliminationLU: LU sparse elimination not supported by this backend");
  }

  // Batch all LU sparse elimination levels into a single GPU submission.
  // Default: calls doEliminationLU per level (CPU/non-batching backends).
  virtual void doAllEliminationsLU(const std::vector<SymElimCtxPtr>& elimCtxs,
                                   const std::vector<int64_t>& ranges, T* data,
                                   T staticPivotThreshold, int64_t& totalPerturbCount) {
    for (size_t l = 0; l + 1 < ranges.size(); l++) {
      if (elimCtxs[l]) {
        int64_t perturbCount = 0;
        doEliminationLU(*elimCtxs[l], data, ranges[l], ranges[l + 1], staticPivotThreshold,
                        perturbCount);
        totalPerturbCount += perturbCount;
      }
    }
  }

  // Read a single value from data buffer. Default: direct CPU read.
  // GPU backends override to copy from device memory.
  virtual T readValue(const T* data, int64_t offset) { return data[offset]; }

  // Compute max|diag| across all lumps in [startLump, upToLump).
  // Default: uses readValue loop (works for CPU and GPU with lazy cache).
  // GPU backends can override with reduction kernel for pure-GPU operation.
  virtual double maxAbsDiag(const T* data, const int64_t* lumpStart, const int64_t* chainColPtr,
                            const int64_t* chainData, int64_t startLump, int64_t upToLump) {
    if constexpr (std::is_arithmetic_v<T>) {
      double maxVal = 0;
      for (int64_t l = startLump; l < upToLump; l++) {
        int64_t lumpSize = lumpStart[l + 1] - lumpStart[l];
        int64_t diagOff = chainData[chainColPtr[l]];
        for (int64_t i = 0; i < lumpSize; i++) {
          double absVal = std::abs(static_cast<double>(readValue(data, diagOff + i * lumpSize + i)));
          if (absVal > maxVal) maxVal = absVal;
        }
      }
      return maxVal;
    } else {
      (void)data; (void)lumpStart; (void)chainColPtr; (void)chainData;
      (void)startLump; (void)upToLump;
      throw std::runtime_error("maxAbsDiag: not supported for batched types");
    }
  }

  // dense Cholesky on dense row-major matrix A (in place)
  virtual void potrf(int64_t n, T* data, int64_t offA) = 0;

  // solve: X * A.lowerHalf().transpose() = B (in place, B becomes X)
  virtual void trsm(int64_t n, int64_t k, T* data, int64_t offA, int64_t offB) = 0;

  // computes (A|B) * A', upper diag part of A*A' doesn't matter
  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const T* data, int64_t offset) = 0;

  virtual void prepareAssemble(int64_t targetLump) = 0;

  virtual void assemble(T* data, int64_t rectRowBegin, int64_t dstStride, int64_t srcColDataOffset,
                        int64_t srcRectWidth, int64_t numBlockRows, int64_t numBlockCols) = 0;

  // ============ LU factorization methods ============
  // These have default implementations that throw for backends not yet supporting LU.

  // LU factorization with partial pivoting on dense row-major matrix A (in place)
  // pivots must be sized to min(m,n), returns info (0 = success)
  virtual int getrf(int64_t m, int64_t n, T* data, int64_t offA, int64_t* pivots) {
    (void)m;
    (void)n;
    (void)data;
    (void)offA;
    (void)pivots;
    throw std::runtime_error("getrf: LU factorization not supported by this backend");
  }

  // solve: L * X = B where L is lower triangular with unit diagonal (in place, B becomes X)
  // Used for: solving for U row (right of diagonal block)
  virtual void trsmLowerUnit(int64_t m, int64_t n, const T* L, int64_t offL, T* B, int64_t offB,
                             int64_t ldb) {
    (void)m;
    (void)n;
    (void)L;
    (void)offL;
    (void)B;
    (void)offB;
    (void)ldb;
    throw std::runtime_error("trsmLowerUnit: LU not supported by this backend");
  }

  // solve: X * U = B where U is upper triangular (in place, B becomes X)
  // Used for: solving for L column (below diagonal block)
  virtual void trsmUpperRight(int64_t m, int64_t n, const T* U, int64_t offU, T* B, int64_t offB,
                              int64_t ldb) {
    (void)m;
    (void)n;
    (void)U;
    (void)offU;
    (void)B;
    (void)offB;
    (void)ldb;
    throw std::runtime_error("trsmUpperRight: LU not supported by this backend");
  }

  // C -= L * U, general matrix multiply for LU elimination
  // L is m x k, U is k x n, C is m x n
  virtual void saveGemm(int64_t m, int64_t n, int64_t k, const T* L, int64_t offL, int64_t ldL,
                        const T* U, int64_t offU, int64_t ldU, T* C, int64_t offC, int64_t ldC) {
    (void)m;
    (void)n;
    (void)k;
    (void)L;
    (void)offL;
    (void)ldL;
    (void)U;
    (void)offU;
    (void)ldU;
    (void)C;
    (void)offC;
    (void)ldC;
    throw std::runtime_error("saveGemm: LU not supported by this backend");
  }

  // Apply row permutation from pivots array to a portion of the matrix
  virtual void applyRowPerm(int64_t* pivots, int64_t n, T* data, int64_t offData, int64_t ld,
                            int64_t numCols) {
    (void)pivots;
    (void)n;
    (void)data;
    (void)offData;
    (void)ld;
    (void)numCols;
    throw std::runtime_error("applyRowPerm: LU not supported by this backend");
  }

  // Signal start of dense LU operations — GPU backends can sync and copy data
  // to host for CPU BLAS fallback (avoids per-op GPU dispatch overhead).
  // Called before the dense loop in internalFactorRangeLU.
  virtual void beginDenseOps(T* data, int64_t totalDataSize) {
    (void)data;
    (void)totalDataSize;
  }

  // Scan diagonal of row-major n×n matrix at data+offset with given stride,
  // perturb elements with |value| < threshold. Returns count of perturbed elements.
  virtual int64_t perturbSmallDiagonals(int64_t n, T* data, int64_t offset, int64_t stride,
                                        T threshold) {
    (void)n;
    (void)data;
    (void)offset;
    (void)stride;
    (void)threshold;
    throw std::runtime_error("perturbSmallDiagonals not supported by this backend");
  }

  // ============ LDL^T factorization methods ============
  // For symmetric indefinite matrices: A = L * D * L^T
  // L is unit lower triangular (stored below diagonal), D is diagonal (stored on diagonal)

  // LDL^T factorization on dense row-major matrix A (in place)
  // After: diagonal contains D, lower triangle contains L (unit diagonal implicit)
  // Returns 0 on success, >0 if matrix is singular (zero pivot at position info)
  virtual int ldlt(int64_t n, T* data, int64_t offA) {
    (void)n;
    (void)data;
    (void)offA;
    throw std::runtime_error("ldlt: LDL^T factorization not supported by this backend");
  }

  // Scale rows of matrix by diagonal: B[i,:] *= D[i]
  // Used for computing L*D from L
  virtual void scaleRowsByDiag(int64_t m, int64_t n, const T* D, int64_t offD, T* B, int64_t offB,
                               int64_t ldb) {
    (void)m;
    (void)n;
    (void)D;
    (void)offD;
    (void)B;
    (void)offB;
    (void)ldb;
    throw std::runtime_error("scaleRowsByDiag: LDL^T not supported by this backend");
  }

  // C -= L * D * L^T, symmetric rank-k update with diagonal scaling
  // L is m x k, D is k (diagonal), result is m x m symmetric (lower triangle updated)
  virtual void saveSyrkScaled(int64_t m, int64_t k, const T* L, int64_t offL, int64_t ldL,
                              const T* D, int64_t offD, T* C, int64_t offC, int64_t ldC) {
    (void)m;
    (void)k;
    (void)L;
    (void)offL;
    (void)ldL;
    (void)D;
    (void)offD;
    (void)C;
    (void)offC;
    (void)ldC;
    throw std::runtime_error("saveSyrkScaled: LDL^T not supported by this backend");
  }

  // LDL^T off-diagonal column solve: solve X * L^T = B for unit L, then scale by D^{-1}
  // L is n x n with unit diagonal (stored with D on diagonal, which is used for scaling)
  // B is k x n, result is stored back in B
  // This computes: B <- B * L^{-T} * D^{-1}
  virtual void trsmUnitScaleInv(int64_t n, int64_t k, T* data, int64_t offA, int64_t offB) {
    (void)n;
    (void)k;
    (void)data;
    (void)offA;
    (void)offB;
    throw std::runtime_error("trsmUnitScaleInv: LDL^T not supported by this backend");
  }

  // For LDL^T Schur complement: computes temp = (L * D) * L^T and stores in tempData
  // m1 = number of columns in symmetric part (same rows for left and right L)
  // m2 = total rows in L (for rectangular gemm part)
  // k = number of columns in L (diagonal block size)
  // L is m2 x k starting at offL
  // D is k diagonal elements starting at offD with stride ldD
  // Result stored in tempData (same layout as saveSyrkGemm)
  virtual void saveSyrkGemmScaled(int64_t m1, int64_t m2, int64_t k, T* data, int64_t offL,
                                  int64_t offD, int64_t ldD) {
    (void)m1;
    (void)m2;
    (void)k;
    (void)data;
    (void)offL;
    (void)offD;
    (void)ldD;
    throw std::runtime_error("saveSyrkGemmScaled: LDL^T not supported by this backend");
  }
};

// methods (and possibly context) for solve operations
template <typename T>
struct SolveCtx : SolveCtxBase {
  virtual ~SolveCtx() {}

  virtual void sparseElimSolveL(const SymElimCtx& elimData, const T* data, int64_t lumpsBegin,
                                int64_t lumpsEnd, T* C, int64_t ldc) = 0;

  virtual void sparseElimSolveLt(const SymElimCtx& elimData, const T* data, int64_t lumpsBegin,
                                 int64_t lumpsEnd, T* C, int64_t ldc) = 0;

  // LU sparse elimination forward solve: L has unit diagonal (skip diag solve)
  // Only does below-diagonal scatter: v[rowSpan] -= L[rowSpan, lump] * v[lump]
  virtual void sparseElimSolveLUnit(const SymElimCtx& elimData, const T* data, int64_t lumpsBegin,
                                    int64_t lumpsEnd, T* C, int64_t ldc) {
    (void)elimData;
    (void)data;
    (void)lumpsBegin;
    (void)lumpsEnd;
    (void)C;
    (void)ldc;
    throw std::runtime_error("sparseElimSolveLUnit: LU sparse solve not supported");
  }

  // LU sparse elimination backward solve: gather from upper triangle then divide by U diagonal
  virtual void sparseElimSolveU(const SymElimCtx& elimData, const T* data, int64_t lumpsBegin,
                                int64_t lumpsEnd, T* C, int64_t ldc) {
    (void)elimData;
    (void)data;
    (void)lumpsBegin;
    (void)lumpsEnd;
    (void)C;
    (void)ldc;
    throw std::runtime_error("sparseElimSolveU: LU sparse solve not supported");
  }

  virtual void symm(const T* data, int64_t offset, int64_t n, const T* C, int64_t offC, int64_t ldc,
                    T* D, int64_t ldd, BaseType<T> alpha) = 0;

  virtual void solveL(const T* data, int64_t offset, int64_t n, T* C, int64_t offC,
                      int64_t ldc) = 0;

  // Solve L * x = b where L is unit lower triangular (diagonal = 1) (in place)
  // This is for LU solve where L from getrf has implicit unit diagonal
  virtual void solveLUnit(const T* data, int64_t offset, int64_t n, T* C, int64_t offC,
                          int64_t ldc) {
    (void)data;
    (void)offset;
    (void)n;
    (void)C;
    (void)offC;
    (void)ldc;
    throw std::runtime_error("solveLUnit: LU not supported by this backend");
  }

  virtual void gemv(const T* data, int64_t offset, int64_t nRows, int64_t nCols, const T* A,
                    int64_t offA, int64_t lda, BaseType<T> alpha) = 0;

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, T* C, int64_t ldc) = 0;

  virtual void solveLt(const T* data, int64_t offset, int64_t n, T* C, int64_t offC,
                       int64_t ldc) = 0;

  virtual void gemvT(const T* data, int64_t offset, int64_t nRows, int64_t nCols, T* A,
                     int64_t offA, int64_t lda, BaseType<T> alpha) = 0;

  virtual void assembleVecT(const T* C, int64_t ldc, int64_t chainColPtr, int64_t numColItems) = 0;

  virtual bool hasFragmentedOps() { return false; }

  virtual void fragmentedMV(const T* /*data*/, const T* /*x*/, int64_t /*spanBegin*/,
                            int64_t /*spanEnd*/, T* /*y*/, BaseType<T> /*alpha*/) {
    throw std::runtime_error("fragmentedMV: not supported");
  }

  virtual void fragmentedSolveL(const T* /*data*/, int64_t /*spanBegin*/, int64_t /*spanEnd*/,
                                T* /*y*/) {
    throw std::runtime_error("fragmentedSolveL: not supported");
  }

  virtual void fragmentedSolveLt(const T* /*data*/, int64_t /*spanBegin*/, int64_t /*spanEnd*/,
                                 T* /*y*/) {
    throw std::runtime_error("fragmentedSolveLt: not supported");
  }

  // ============ LU solve methods ============
  // These have default implementations that throw for backends not yet supporting LU.

  // Solve U * x = b where U is upper triangular (in place)
  virtual void solveU(const T* data, int64_t offset, int64_t n, T* C, int64_t offC, int64_t ldc) {
    (void)data;
    (void)offset;
    (void)n;
    (void)C;
    (void)offC;
    (void)ldc;
    throw std::runtime_error("solveU: LU not supported by this backend");
  }

  // Pre-upload all pivots to GPU for batched permutation dispatch.
  // When called, subsequent applyRowPermVec/Inv calls can skip per-call upload.
  // Default: no-op (CPU backends don't need this).
  virtual void uploadPivots(const int64_t* pivots, int64_t totalSize) {
    (void)pivots;
    (void)totalSize;
  }

  // Apply row permutation P to vector: y = P * x (for LU solve, applies pivots)
  virtual void applyRowPermVec(const int64_t* pivots, int64_t n, T* vec, int64_t ldVec) {
    (void)pivots;
    (void)n;
    (void)vec;
    (void)ldVec;
    throw std::runtime_error("applyRowPermVec: LU not supported by this backend");
  }

  // Apply inverse row permutation P^T to vector: y = P^T * x (for LU solve, reverse pivots)
  virtual void applyRowPermVecInv(const int64_t* pivots, int64_t n, T* vec, int64_t ldVec) {
    (void)pivots;
    (void)n;
    (void)vec;
    (void)ldVec;
    throw std::runtime_error("applyRowPermVecInv: LU not supported by this backend");
  }

  // Direct gemv for U backward solve: result += alpha * M * x
  // M is row-major matrix of shape (nRows x nCols) at data+offset
  // x is at vec+srcOff with length nCols
  // result is updated at vec+dstOff with length nRows
  virtual void gemvDirect(const T* data, int64_t offset, int64_t nRows, int64_t nCols, T* vec,
                          int64_t srcOff, int64_t dstOff, int64_t ldVec, BaseType<T> alpha) {
    (void)data;
    (void)offset;
    (void)nRows;
    (void)nCols;
    (void)vec;
    (void)srcOff;
    (void)dstOff;
    (void)ldVec;
    (void)alpha;
    throw std::runtime_error("gemvDirect: LU not supported by this backend");
  }

  // ============ LDL^T solve methods ============
  // For symmetric indefinite factorization A = L * D * L^T

  // Solve L * x = b where L is unit lower triangular (in place)
  // Same as solveLUnit but named for clarity in LDL^T context
  virtual void solveLUnitLDLT(const T* data, int64_t offset, int64_t n, T* C, int64_t offC,
                              int64_t ldc) {
    // Default: use solveLUnit if available
    solveLUnit(data, offset, n, C, offC, ldc);
  }

  // Solve D * x = b where D is diagonal (stored on diagonal of factored matrix)
  virtual void solveDiag(const T* data, int64_t offset, int64_t n, T* C, int64_t offC,
                         int64_t ldc) {
    (void)data;
    (void)offset;
    (void)n;
    (void)C;
    (void)offC;
    (void)ldc;
    throw std::runtime_error("solveDiag: LDL^T not supported by this backend");
  }

  // Solve L^T * x = b where L is unit lower triangular (in place)
  // (backward substitution with unit diagonal)
  virtual void solveLtUnit(const T* data, int64_t offset, int64_t n, T* C, int64_t offC,
                           int64_t ldc) {
    (void)data;
    (void)offset;
    (void)n;
    (void)C;
    (void)offC;
    (void)ldc;
    throw std::runtime_error("solveLtUnit: LDL^T not supported by this backend");
  }
};

// introspection shortcuts
template <typename T>
std::string prettyTypeName(const T& t) {
  char* c_str = abi::__cxa_demangle(typeid(t).name(), nullptr, nullptr, nullptr);
  std::string retv(c_str);
  free(c_str);
  return retv;
}

template <typename T>
NumericCtxPtr<T> SymbolicCtx::createNumericCtx(int64_t tempBufSize, const T* data) {
  static const std::type_index T_tIdx(typeid(T));
  int batchSize = Batch<T>::getSize(data);
  NumericCtxBase* ctx = createNumericCtxForType(T_tIdx, tempBufSize, batchSize);
  NumericCtx<T>* typedCtx = dynamic_cast<NumericCtx<T>*>(ctx);
  BASPACHO_CHECK_NOTNULL(typedCtx);
  return NumericCtxPtr<T>(typedCtx);
}

template <typename T>
SolveCtxPtr<T> SymbolicCtx::createSolveCtx(int nRHS, const T* data) {
  static const std::type_index T_tIdx(typeid(T));
  int batchSize = Batch<T>::getSize(data);
  SolveCtxBase* ctx = createSolveCtxForType(T_tIdx, nRHS, batchSize);
  SolveCtx<T>* typedCtx = dynamic_cast<SolveCtx<T>*>(ctx);
  BASPACHO_CHECK_NOTNULL(typedCtx);
  return SolveCtxPtr<T>(typedCtx);
}

OpsPtr simpleOps();

OpsPtr fastOps(int numThreads = 16);

#ifdef BASPACHO_USE_CUBLAS
OpsPtr cudaOps();
#endif

#ifdef BASPACHO_USE_METAL
OpsPtr metalOps();
#endif

#ifdef BASPACHO_USE_OPENCL
OpsPtr openclOps();
#endif

}  // end namespace BaSpaCho
