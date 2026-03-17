/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <alloca.h>
#include <dispenso/parallel_for.h>
#include <chrono>
#include "sprux/sprux/DebugMacros.h"
#include "sprux/sprux/MatOpsCpuBase.h"
#include "sprux/sprux/Utils.h"

#ifdef SPRUX_USE_BLAS
#ifdef SPRUX_USE_MKL
#include "mkl.h"
#define BLAS_INT MKL_INT
#define SPRUX_USE_TRSM_WORAROUND 0
#else
#include "sprux/sprux/BlasDefs.h"
#define SPRUX_USE_TRSM_WORAROUND 1
#endif
#endif  // SPRUX_USE_BLAS

namespace BaSpaCho {

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

struct BlasSymbolicCtx : CpuBaseSymbolicCtx {
  BlasSymbolicCtx(const CoalescedBlockMatrixSkel& skel, int numThreads)
      : CpuBaseSymbolicCtx(skel),
        useThreads(numThreads > 1),
        threadPool(useThreads ? numThreads : 0) {}

  virtual NumericCtxBase* createNumericCtxForType(std::type_index tIdx, int64_t tempBufSize,
                                                  int batchSize) override;

  virtual SolveCtxBase* createSolveCtxForType(std::type_index tIdx, int nRHS,
                                              int batchSize) override;

  bool useThreads;
  mutable dispenso::ThreadPool threadPool;
};

// Blas ops aiming at high performance using BLAS/LAPACK
struct BlasOps : Ops {
  explicit BlasOps(int numThreads) : numThreads(numThreads) {}
  virtual SymbolicCtxPtr createSymbolicCtx(const CoalescedBlockMatrixSkel& skel,
                                           const std::vector<int64_t>& /* permutation */) override {
    // todo: use settings
    return SymbolicCtxPtr(new BlasSymbolicCtx(skel, numThreads));
  }
  int numThreads;
};

template <typename T>
struct BlasNumericCtx : CpuBaseNumericCtx<T> {
  BlasNumericCtx(const BlasSymbolicCtx& sym, int64_t bufSize, int64_t numSpans)
      : CpuBaseNumericCtx<T>(sym, bufSize, numSpans), sym(sym) {}

  virtual void pseudoFactorSpans(T* data, int64_t spanBegin, int64_t spanEnd) override {
    auto timer = sym.pseudoFactorStat.instance();
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    if (!sym.useThreads) {
      for (int64_t s = spanBegin; s < spanEnd; s++) {
        factorSpan(skel, data, s);
      }
    } else {
      dispenso::TaskSet taskSet(sym.threadPool);
      dispenso::parallel_for(taskSet, dispenso::makeChunkedRange(spanBegin, spanEnd, 1L),
                             [&](int64_t sBegin, int64_t sEnd) {
                               for (int64_t s = sBegin; s < sEnd; s++) {
                                 factorSpan(skel, data, s);
                               }
                             });
    }
  }

  virtual void doElimination(const SymElimCtx& elimData, T* data, int64_t lumpsBegin,
                             int64_t lumpsEnd) override {
    const CpuBaseSymElimCtx* pElim = dynamic_cast<const CpuBaseSymElimCtx*>(&elimData);
    SPRUX_CHECK_NOTNULL(pElim);
    const CpuBaseSymElimCtx& elim = *pElim;
    const CoalescedBlockMatrixSkel& skel = sym.skel;
    auto timer = elim.elimStat.instance();

    if (!sym.useThreads) {
      for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
        factorLump(skel, data, l);
      }
    } else {
      dispenso::TaskSet taskSet(sym.threadPool);
      dispenso::parallel_for(taskSet, dispenso::makeChunkedRange(lumpsBegin, lumpsEnd, 5L),
                             [&](int64_t lBegin, int64_t lEnd) {
                               for (int64_t l = lBegin; l < lEnd; l++) {
                                 factorLump(skel, data, l);
                               }
                             });
    }

    int64_t numElimRows = elim.rowPtr.size() - 1;
    if (elim.colLump.size() > 3 * (elim.rowPtr.size() - 1)) {
      int64_t numSpans = skel.spanStart.size() - 1;
      if (!sym.useThreads) {
        std::vector<int64_t> spanToChainOffset(numSpans);
        for (int64_t sRel = 0L; sRel < numElimRows; sRel++) {
          eliminateRowChain(elim, skel, data, sRel, spanToChainOffset);
        }
      } else {
        struct ElimContext {
          std::vector<int64_t> spanToChainOffset;
          explicit ElimContext(int64_t numSpans) : spanToChainOffset(numSpans) {}
        };
        vector<ElimContext> contexts;
        dispenso::TaskSet taskSet(sym.threadPool);
        dispenso::parallel_for(
            taskSet, contexts, [=]() -> ElimContext { return ElimContext(numSpans); },
            dispenso::makeChunkedRange(0L, numElimRows, 5L),
            [&](ElimContext& ctx, int64_t sBegin, int64_t sEnd) {
              for (int64_t sRel = sBegin; sRel < sEnd; sRel++) {
                eliminateRowChain(elim, skel, data, sRel, ctx.spanToChainOffset);
              }
            });
      }
    } else {
      if (!sym.useThreads) {
        vector<T> reusableTmpBuf;
        for (int64_t sRel = 0L; sRel < numElimRows; sRel++) {
          eliminateVerySparseRowChain(elim, skel, data, sRel, reusableTmpBuf);
        }
      } else {
        dispenso::TaskSet taskSet(sym.threadPool);
        vector<vector<T>> contexts;
        dispenso::parallel_for(
            taskSet, contexts, [=]() -> vector<T> { return vector<T>(); },
            dispenso::makeChunkedRange(0L, numElimRows, 5L),
            [&](vector<T>& reusableTmpBuf, int64_t sBegin, int64_t sEnd) {
              for (int64_t sRel = sBegin; sRel < sEnd; sRel++) {
                eliminateVerySparseRowChain(elim, skel, data, sRel, reusableTmpBuf);
              }
            });
      }
    }
  }

#ifdef SPRUX_USE_BLAS
  virtual void potrf(int64_t n, T* data, int64_t offA) override;

  virtual void trsm(int64_t n, int64_t k, T* data, int64_t offA, int64_t offB) override;

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const T* data,
                            int64_t offset) override;

  // LU factorization methods
  virtual int getrf(int64_t m, int64_t n, T* data, int64_t offA, int64_t* pivots) override;

  virtual void trsmLowerUnit(int64_t m, int64_t n, const T* L, int64_t offL, T* B, int64_t offB,
                             int64_t ldb) override;

  virtual void trsmUpperRight(int64_t m, int64_t n, const T* U, int64_t offU, T* B, int64_t offB,
                              int64_t ldb) override;

  virtual void applyRowPerm(int64_t* pivots, int64_t n, T* data, int64_t offData, int64_t ld,
                            int64_t numCols) override;

  virtual void saveGemm(int64_t m, int64_t n, int64_t k, const T* L, int64_t offL, int64_t ldL,
                        const T* U, int64_t offU, int64_t ldU, T* C, int64_t offC,
                        int64_t ldC) override;
#endif  // SPRUX_USE_BLAS

  virtual int64_t perturbSmallDiagonals(int64_t n, T* data, int64_t offset, int64_t stride,
                                        T threshold) override {
    int64_t count = 0;
    for (int64_t i = 0; i < n; i++) {
      T& diag = data[offset + i * stride + i];
      if (!std::isfinite(diag) || std::abs(diag) < threshold) {
        diag = (diag >= T(0)) ? threshold : -threshold;
        count++;
      }
    }
    return count;
  }

  virtual void prepareAssemble(int64_t targetLump) override {
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    for (int64_t i = skel.chainColPtr[targetLump], iEnd = skel.chainColPtr[targetLump + 1];
         i < iEnd; i++) {
      spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
    }
  }

  virtual void assemble(T* data, int64_t rectRowBegin,
                        int64_t dstStride,  //
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    auto timer = sym.asmblStat.instance(sizeof(T), numBlockRows, numBlockCols);
    const CoalescedBlockMatrixSkel& skel = sym.skel;
    const int64_t* chainRowsTillEnd = skel.chainRowsTillEnd.data() + srcColDataOffset;
    const int64_t* pToSpan = skel.chainRowSpan.data() + srcColDataOffset;
    const int64_t* pSpanToChainOffset = spanToChainOffset.data();
    const int64_t* pSpanOffsetInLump = skel.spanOffsetInLump.data();
    const T* matRectPtr = tempBuffer.data();

    if (!sym.useThreads) {
      // non-threaded reference implementation:
      for (int64_t r = 0; r < numBlockRows; r++) {
        int64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
        int64_t rSize = chainRowsTillEnd[r] - rBegin - rectRowBegin;
        int64_t rParam = pToSpan[r];
        int64_t rOffset = pSpanToChainOffset[rParam];
        const T* matRowPtr = matRectPtr + rBegin * srcRectWidth;

        int64_t cEnd = std::min(numBlockCols, r + 1);
        for (int64_t c = 0; c < cEnd; c++) {
          int64_t cStart = chainRowsTillEnd[c - 1] - rectRowBegin;
          int64_t cSize = chainRowsTillEnd[c] - cStart - rectRowBegin;
          int64_t offset = rOffset + pSpanOffsetInLump[pToSpan[c]];

          T* dst = data + offset;
          const T* src = matRowPtr + cStart;
          stridedMatSub(dst, dstStride, src, srcRectWidth, rSize, cSize);
        }
      }
    } else {
      dispenso::TaskSet taskSet(sym.threadPool);
      dispenso::parallel_for(taskSet, dispenso::makeChunkedRange(0L, numBlockRows, 3L),
                             [&](int64_t rFrom, int64_t rTo) {
                               for (int64_t r = rFrom; r < rTo; r++) {
                                 int64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
                                 int64_t rSize = chainRowsTillEnd[r] - rBegin - rectRowBegin;
                                 int64_t rParam = pToSpan[r];
                                 int64_t rOffset = pSpanToChainOffset[rParam];
                                 const T* matRowPtr = matRectPtr + rBegin * srcRectWidth;

                                 int64_t cEnd = std::min(numBlockCols, r + 1);
                                 int64_t nextCStart = chainRowsTillEnd[-1] - rectRowBegin;
                                 for (int64_t c = 0; c < cEnd; c++) {
                                   int64_t cStart = nextCStart;
                                   nextCStart = chainRowsTillEnd[c] - rectRowBegin;
                                   int64_t cSize = nextCStart - cStart;
                                   int64_t offset = rOffset + pSpanOffsetInLump[pToSpan[c]];

                                   T* dst = data + offset;
                                   const T* src = matRowPtr + cStart;
                                   stridedMatSub(dst, dstStride, src, srcRectWidth, rSize, cSize);
                                 }
                               }
                             });
    }
  }

  using CpuBaseNumericCtx<T>::factorLump;
  using CpuBaseNumericCtx<T>::factorSpan;
  using CpuBaseNumericCtx<T>::eliminateRowChain;
  using CpuBaseNumericCtx<T>::eliminateVerySparseRowChain;
  using CpuBaseNumericCtx<T>::stridedMatSub;

  using CpuBaseNumericCtx<T>::tempBuffer;
  using CpuBaseNumericCtx<T>::spanToChainOffset;

  const BlasSymbolicCtx& sym;
};

#ifdef SPRUX_USE_BLAS
template <>
void BlasNumericCtx<double>::potrf(int64_t n, double* data, int64_t offA) {
  auto timer = sym.potrfStat.instance(sizeof(double), n);
  sym.potrfBiggestN = std::max(sym.potrfBiggestN, n);

  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n, data + offA, n);
}

template <>
void BlasNumericCtx<float>::potrf(int64_t n, float* data, int64_t offA) {
  auto timer = sym.potrfStat.instance(sizeof(float), n);
  sym.potrfBiggestN = std::max(sym.potrfBiggestN, n);

  LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', n, data + offA, n);
}

template <>
void BlasNumericCtx<double>::trsm(int64_t n, int64_t k, double* data, int64_t offA, int64_t offB) {
  auto timer = sym.trsmStat.instance(sizeof(double), n, k);

  // TSRM should be fast but appears very slow in OpenBLAS
  static constexpr bool slowTrsmWorkaround = SPRUX_USE_TRSM_WORAROUND;
  if (slowTrsmWorkaround) {
    using MatCMajD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    // col-major's upper = (row-major's lower).transpose()
    Eigen::Map<const MatCMajD> matA(data + offA, n, n);
    dispenso::TaskSet taskSet(sym.threadPool);
    dispenso::parallel_for(
        taskSet, dispenso::makeChunkedRange(0L, k, 16L), [&](int64_t k1, int64_t k2) {
          Eigen::Map<MatRMaj<double>> matB(data + offB + n * k1, k2 - k1, n);
          matA.template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(
              matB);
        });
    return;
  }

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, n, k, 1.0,
              data + offA, n, data + offB, n);
}

template <>
void BlasNumericCtx<float>::trsm(int64_t n, int64_t k, float* data, int64_t offA, int64_t offB) {
  auto timer = sym.trsmStat.instance(sizeof(float), n, k);

  // TSRM should be fast but appears very slow in OpenBLAS
  static constexpr bool slowTrsmWorkaround = SPRUX_USE_TRSM_WORAROUND;
  if (slowTrsmWorkaround) {
    using MatCMajD = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    // col-major's upper = (row-major's lower).transpose()
    Eigen::Map<const MatCMajD> matA(data + offA, n, n);
    dispenso::TaskSet taskSet(sym.threadPool);
    dispenso::parallel_for(
        taskSet, dispenso::makeChunkedRange(0L, k, 16L), [&](int64_t k1, int64_t k2) {
          Eigen::Map<MatRMaj<float>> matB(data + offB + n * k1, k2 - k1, n);
          matA.template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(
              matB);
        });
    return;
  }

  cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, n, k, 1.0,
              data + offA, n, data + offB, n);
}

template <>
void BlasNumericCtx<double>::saveSyrkGemm(int64_t m, int64_t n, int64_t k, const double* data,
                                          int64_t offset) {
  auto timer = sym.sygeStat.instance(sizeof(double), m, n, k);
  SPRUX_CHECK_LE(m * n, (int64_t)tempBuffer.size());

  // in some cases it could be faster with syrk+gemm
  // as it saves some computation, not the case in practice
  bool doSyrk = (m == n) || (m + n + k > 150);
  bool doGemm = !(doSyrk && m == n);

  if (doSyrk) {
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasConjTrans, m, k, 1.0, data + offset, k, 0.0,
                tempBuffer.data(), m);
    sym.syrkCalls++;
  }

  if (doGemm) {
    int64_t gemmStart = doSyrk ? m : 0;
    int64_t gemmInOffset = doSyrk ? m * k : 0;
    int64_t gemmOutOffset = doSyrk ? m * m : 0;
    cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, m, n - gemmStart, k, 1.0,
                data + offset, k, data + offset + gemmInOffset, k, 0.0,
                tempBuffer.data() + gemmOutOffset, m);
    sym.gemmCalls++;
  }
}

template <>
void BlasNumericCtx<float>::saveSyrkGemm(int64_t m, int64_t n, int64_t k, const float* data,
                                         int64_t offset) {
  auto timer = sym.sygeStat.instance(sizeof(float), m, n, k);
  SPRUX_CHECK_LE(m * n, (int64_t)tempBuffer.size());

  // in some cases it could be faster with syrk+gemm
  // as it saves some computation, not the case in practice
  bool doSyrk = (m == n) || (m + n + k > 150);
  bool doGemm = !(doSyrk && m == n);

  if (doSyrk) {
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasConjTrans, m, k, 1.0, data + offset, k, 0.0,
                tempBuffer.data(), m);
    sym.syrkCalls++;
  }

  if (doGemm) {
    int64_t gemmStart = doSyrk ? m : 0;
    int64_t gemmInOffset = doSyrk ? m * k : 0;
    int64_t gemmOutOffset = doSyrk ? m * m : 0;
    cblas_sgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, m, n - gemmStart, k, 1.0,
                data + offset, k, data + offset + gemmInOffset, k, 0.0,
                tempBuffer.data() + gemmOutOffset, m);
    sym.gemmCalls++;
  }
}

// ============ LU Factorization BLAS implementations ============

// Helper to transpose a square matrix in-place
template <typename T>
static void transposeSquareInPlace(T* data, int64_t n) {
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = i + 1; j < n; j++) {
      std::swap(data[i * n + j], data[j * n + i]);
    }
  }
}

// LAPACK getrf with workaround for row-major storage:
// BaSpaCho stores blocks row-major, but LAPACKE only supports COL_MAJOR for getrf.
// When LAPACK interprets row-major data as col-major, it sees A^T instead of A.
// To get correct P * A = L * U with L in lower and U in upper (row-major view),
// we transpose A before getrf (so LAPACK sees A), then transpose back after.
// This gives us L*U stored in standard row-major positions.

template <>
int BlasNumericCtx<double>::getrf(int64_t m, int64_t n, double* data, int64_t offA,
                                  int64_t* pivots) {
  auto timer = sym.getrfStat.instance(sizeof(double), m);
  sym.getrfBiggestN = std::max(sym.getrfBiggestN, m);

  // Transpose before: row-major A → col-major view sees A (not A^T)
  if (m == n) {
    transposeSquareInPlace(data + offA, n);
  }

  std::vector<BLAS_INT> ipiv(std::min(m, n));
  int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, data + offA, m, ipiv.data());

  // Transpose after: col-major L+U → row-major L+U
  if (m == n) {
    transposeSquareInPlace(data + offA, n);
  }

  // Copy pivots back (convert from 1-based to 0-based)
  for (int64_t i = 0; i < std::min(m, n); i++) {
    pivots[i] = ipiv[i] - 1;  // LAPACK is 1-based
  }
  return info;
}

template <>
int BlasNumericCtx<float>::getrf(int64_t m, int64_t n, float* data, int64_t offA,
                                 int64_t* pivots) {
  auto timer = sym.getrfStat.instance(sizeof(float), m);
  sym.getrfBiggestN = std::max(sym.getrfBiggestN, m);

  // Transpose before: row-major A → col-major view sees A (not A^T)
  if (m == n) {
    transposeSquareInPlace(data + offA, n);
  }

  std::vector<BLAS_INT> ipiv(std::min(m, n));
  int info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, m, n, data + offA, m, ipiv.data());

  // Transpose after: col-major L+U → row-major L+U
  if (m == n) {
    transposeSquareInPlace(data + offA, n);
  }

  // Copy pivots back (convert from 1-based to 0-based)
  for (int64_t i = 0; i < std::min(m, n); i++) {
    pivots[i] = ipiv[i] - 1;  // LAPACK is 1-based
  }
  return info;
}

template <>
void BlasNumericCtx<double>::trsmLowerUnit(int64_t m, int64_t n, const double* L, int64_t offL,
                                           double* B, int64_t offB, int64_t ldb) {
  // Solve L * X = B where L is m x m unit lower triangular, X and B are m x n
  // Both L and B are stored ROW-MAJOR.
  // Row-major L_row(m x m) lower = Col-major L_col^T(m x m) upper (same data)
  // Row-major B_row(m x n) = Col-major B_col^T(n x m) (same data, transposed dims)
  // The equation L_row * X_row = B_row becomes X_col * L_col = B_col after transpose
  // So: solve X * U = B with CblasRight, where U is upper unit (L^T)
  // BLAS params: M=n (rows of B_col), N=m (cols of B_col), lda=m, ldb=n
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasUnit, n, m, 1.0, L + offL, m,
              B + offB, ldb);
}

template <>
void BlasNumericCtx<float>::trsmLowerUnit(int64_t m, int64_t n, const float* L, int64_t offL,
                                          float* B, int64_t offB, int64_t ldb) {
  // Same as above for float
  cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasUnit, n, m, 1.0, L + offL, m,
              B + offB, ldb);
}

template <>
void BlasNumericCtx<double>::trsmUpperRight(int64_t m, int64_t n, const double* U, int64_t offU,
                                            double* B, int64_t offB, int64_t ldb) {
  // Solve X * U = B where U is n x n upper triangular (ROW-MAJOR storage)
  // X and B are m x n, also stored ROW-MAJOR with row stride ldb
  // For row-major: X * U = B becomes (in col-major view) U^T * X^T = B^T
  // U^T is lower triangular, X^T and B^T have swapped dimensions
  // So solve with CblasLeft, CblasLower: L(n,n) * X_col(n,m) = B_col(n,m)
  (void)ldb;  // Row stride in row-major = n for tightly packed storage
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, n, m, 1.0, U + offU,
              n, B + offB, n);
}

template <>
void BlasNumericCtx<float>::trsmUpperRight(int64_t m, int64_t n, const float* U, int64_t offU,
                                           float* B, int64_t offB, int64_t ldb) {
  (void)ldb;
  cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, n, m, 1.0, U + offU,
              n, B + offB, n);
}

template <typename T>
void BlasNumericCtx<T>::applyRowPerm(int64_t* pivots, int64_t n, T* data, int64_t offData,
                                     int64_t ld, int64_t numCols) {
  // Apply row permutation to a portion of the matrix
  // pivots[i] indicates row i should be swapped with row pivots[i]
  for (int64_t i = 0; i < n; i++) {
    int64_t swapRow = pivots[i];
    if (swapRow != i) {
      // Swap rows i and swapRow for all columns
      for (int64_t c = 0; c < numCols; c++) {
        std::swap(data[offData + i + c * ld], data[offData + swapRow + c * ld]);
      }
    }
  }
}

// Explicit template instantiation
template void BlasNumericCtx<double>::applyRowPerm(int64_t*, int64_t, double*, int64_t, int64_t,
                                                   int64_t);
template void BlasNumericCtx<float>::applyRowPerm(int64_t*, int64_t, float*, int64_t, int64_t,
                                                  int64_t);

template <>
void BlasNumericCtx<double>::saveGemm(int64_t m, int64_t n, int64_t k, const double* L, int64_t offL,
                                      int64_t ldL, const double* U, int64_t offU, int64_t ldU,
                                      double* C, int64_t offC, int64_t ldC) {
  // C -= L * U
  // L is m x k row-major with ld=ldL
  // U is k x n row-major with ld=ldU
  // C is m x n row-major with ld=ldC
  // For row-major C -= L * U, in column-major: C^T -= U^T * L^T
  // So: call gemm with A=U^T (n,k), B=L^T (k,m), C=C^T (n,m)
  // But our row-major data viewed as col-major is already transposed:
  // L_row(m,k) = L_col^T, U_row(k,n) = U_col^T, C_row(m,n) = C_col^T
  // So L_col is (k,m), U_col is (n,k), C_col is (n,m)
  // C_col -= U_col * L_col = (n,k) * (k,m) = (n,m)
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, k, -1.0, U + offU, ldU, L + offL, ldL,
              1.0, C + offC, ldC);
  sym.luGemmCalls++;
}

template <>
void BlasNumericCtx<float>::saveGemm(int64_t m, int64_t n, int64_t k, const float* L, int64_t offL,
                                     int64_t ldL, const float* U, int64_t offU, int64_t ldU,
                                     float* C, int64_t offC, int64_t ldC) {
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, k, -1.0, U + offU, ldU, L + offL, ldL,
              1.0, C + offC, ldC);
  sym.luGemmCalls++;
}

#endif  // SPRUX_USE_BLAS

using OuterStride = Eigen::OuterStride<>;
template <typename T>
using OuterStridedMatM =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, OuterStride>;
template <typename T>
using OuterStridedCMajMatM =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, OuterStride>;
template <typename T>
using OuterStridedCMajMatK =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
               OuterStride>;

template <typename T>
struct BlasSolveCtx : CpuBaseSolveCtx<T> {
  BlasSolveCtx(const BlasSymbolicCtx& sym, int nRHS) : CpuBaseSolveCtx<T>(sym, nRHS), sym(sym) {}
  virtual ~BlasSolveCtx() override {}

  virtual void sparseElimSolveL(const SymElimCtx& elimData, const T* data, int64_t lumpsBegin,
                                int64_t lumpsEnd, T* C, int64_t ldc) override {
    auto timer = sym.solveSparseLStat.instance();
    const CpuBaseSymElimCtx* pElim = dynamic_cast<const CpuBaseSymElimCtx*>(&elimData);
    SPRUX_CHECK_NOTNULL(pElim);
    const CpuBaseSymElimCtx& elim = *pElim;
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    if (!sym.useThreads) {
      for (int64_t lump = lumpsBegin; lump < lumpsEnd; lump++) {
        int64_t lumpStart = skel.lumpStart[lump];
        int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
        int64_t colStart = skel.chainColPtr[lump];
        int64_t diagDataPtr = skel.chainData[colStart];

        // in-place lower diag cholesky dec on diagonal block
        Eigen::Map<const MatRMaj<T>> diagBlock(data + diagDataPtr, lumpSize, lumpSize);
        OuterStridedCMajMatM<T> matC(C + lumpStart, lumpSize, nRHS, OuterStride(ldc));
        diagBlock.template triangularView<Eigen::Lower>().solveInPlace(matC);
      }
    } else {
      dispenso::TaskSet taskSet(sym.threadPool);
      dispenso::parallel_for(
          taskSet, dispenso::makeChunkedRange(lumpsBegin, lumpsEnd, 5L),
          [&](int64_t lBegin, int64_t lEnd) {
            for (int64_t lump = lBegin; lump < lEnd; lump++) {
              int64_t lumpStart = skel.lumpStart[lump];
              int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
              int64_t colStart = skel.chainColPtr[lump];
              int64_t diagDataPtr = skel.chainData[colStart];

              // in-place lower diag cholesky dec on diagonal block
              Eigen::Map<const MatRMaj<T>> diagBlock(data + diagDataPtr, lumpSize, lumpSize);
              OuterStridedCMajMatM<T> matC(C + lumpStart, lumpSize, nRHS, OuterStride(ldc));
              diagBlock.template triangularView<Eigen::Lower>().solveInPlace(matC);
            }
          });
    }

    if (!sym.useThreads) {
      int64_t numElimRows = elim.rowPtr.size() - 1;
      for (int64_t sRel = 0L; sRel < numElimRows; sRel++) {
        int64_t rowSpan = sRel + elim.spanRowBegin;
        int64_t rowSpanStart = skel.spanStart[rowSpan];
        int64_t rowSpanSize = skel.spanStart[rowSpan + 1] - rowSpanStart;
        OuterStridedCMajMatM<T> matQ(C + rowSpanStart, rowSpanSize, nRHS, OuterStride(ldc));

        for (int64_t i = elim.rowPtr[sRel], iEnd = elim.rowPtr[sRel + 1]; i < iEnd; i++) {
          int64_t lump = elim.colLump[i];
          int64_t lumpStart = skel.lumpStart[lump];
          int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
          int64_t chainColOrd = elim.chainColOrd[i];
          SPRUX_CHECK_GE(chainColOrd,
                            1);  // there must be a diagonal block

          int64_t ptr = skel.chainColPtr[lump] + chainColOrd;
          SPRUX_CHECK_EQ(skel.chainRowSpan[ptr], rowSpan);
          int64_t blockPtr = skel.chainData[ptr];

          Eigen::Map<const MatRMaj<T>> block(data + blockPtr, rowSpanSize, lumpSize);
          OuterStridedCMajMatM<T> matC(C + lumpStart, lumpSize, nRHS, OuterStride(ldc));
          matQ -= block * matC;
        }
      }
    } else {
      int64_t numElimRows = elim.rowPtr.size() - 1;
      dispenso::TaskSet taskSet(sym.threadPool);
      dispenso::parallel_for(
          taskSet, dispenso::makeChunkedRange(0L, numElimRows, 5L),
          [&, this](int64_t sBegin, int64_t sEnd) {
            for (int64_t sRel = sBegin; sRel < sEnd; sRel++) {
              int64_t rowSpan = sRel + elim.spanRowBegin;
              int64_t rowSpanStart = skel.spanStart[rowSpan];
              int64_t rowSpanSize = skel.spanStart[rowSpan + 1] - rowSpanStart;
              OuterStridedCMajMatM<T> matQ(C + rowSpanStart, rowSpanSize, nRHS, OuterStride(ldc));

              for (int64_t i = elim.rowPtr[sRel], iEnd = elim.rowPtr[sRel + 1]; i < iEnd; i++) {
                int64_t lump = elim.colLump[i];
                int64_t lumpStart = skel.lumpStart[lump];
                int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
                int64_t chainColOrd = elim.chainColOrd[i];
                SPRUX_CHECK_GE(chainColOrd,
                                  1);  // there must be a diagonal block

                int64_t ptr = skel.chainColPtr[lump] + chainColOrd;
                SPRUX_CHECK_EQ(skel.chainRowSpan[ptr], rowSpan);
                int64_t blockPtr = skel.chainData[ptr];

                Eigen::Map<const MatRMaj<T>> block(data + blockPtr, rowSpanSize, lumpSize);
                OuterStridedCMajMatM<T> matC(C + lumpStart, lumpSize, nRHS, OuterStride(ldc));
                matQ -= block * matC;
              }
            }
          });
    }
  }

  virtual void sparseElimSolveLt(const SymElimCtx& /* elimData */, const T* data,
                                 int64_t lumpsBegin, int64_t lumpsEnd, T* C, int64_t ldc) override {
    auto timer = sym.solveSparseLtStat.instance();
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    if (!sym.useThreads) {
      for (int64_t lump = lumpsBegin; lump < lumpsEnd; lump++) {
        int64_t lumpStart = skel.lumpStart[lump];
        int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
        int64_t colStart = skel.chainColPtr[lump];
        int64_t colEnd = skel.chainColPtr[lump + 1];
        OuterStridedCMajMatM<T> matC(C + lumpStart, lumpSize, nRHS, OuterStride(ldc));

        for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
          int64_t rowSpan = skel.chainRowSpan[colPtr];
          int64_t rowSpanStart = skel.spanStart[rowSpan];
          int64_t rowSpanSize = skel.spanStart[rowSpan + 1] - rowSpanStart;
          int64_t blockPtr = skel.chainData[colPtr];
          Eigen::Map<const MatRMaj<T>> block(data + blockPtr, rowSpanSize, lumpSize);
          OuterStridedCMajMatM<T> matQ(C + rowSpanStart, rowSpanSize, nRHS, OuterStride(ldc));
          matC -= block.transpose() * matQ;
        }

        int64_t diagDataPtr = skel.chainData[colStart];
        Eigen::Map<const MatRMaj<T>> diagBlock(data + diagDataPtr, lumpSize, lumpSize);
        diagBlock.template triangularView<Eigen::Lower>().adjoint().solveInPlace(matC);
      }
    } else {
      dispenso::TaskSet taskSet(sym.threadPool);
      dispenso::parallel_for(
          taskSet, dispenso::makeChunkedRange(lumpsBegin, lumpsEnd, 5L),
          [&](int64_t lBegin, int64_t lEnd) {
            for (int64_t lump = lBegin; lump < lEnd; lump++) {
              int64_t lumpStart = skel.lumpStart[lump];
              int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
              int64_t colStart = skel.chainColPtr[lump];
              int64_t colEnd = skel.chainColPtr[lump + 1];
              OuterStridedCMajMatM<T> matC(C + lumpStart, lumpSize, nRHS, OuterStride(ldc));

              for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
                int64_t rowSpan = skel.chainRowSpan[colPtr];
                int64_t rowSpanStart = skel.spanStart[rowSpan];
                int64_t rowSpanSize = skel.spanStart[rowSpan + 1] - rowSpanStart;
                int64_t blockPtr = skel.chainData[colPtr];
                Eigen::Map<const MatRMaj<T>> block(data + blockPtr, rowSpanSize, lumpSize);
                OuterStridedCMajMatM<T> matQ(C + rowSpanStart, rowSpanSize, nRHS, OuterStride(ldc));
                matC -= block.transpose() * matQ;
              }

              int64_t diagDataPtr = skel.chainData[colStart];
              Eigen::Map<const MatRMaj<T>> diagBlock(data + diagDataPtr, lumpSize, lumpSize);
              diagBlock.template triangularView<Eigen::Lower>().adjoint().solveInPlace(matC);
            }
          });
    }
  }

#ifdef SPRUX_USE_BLAS
  virtual void symm(const T* data, int64_t offset, int64_t n, const T* C, int64_t offC, int64_t ldc,
                    T* D, int64_t ldd, T alpha) override;

  virtual void solveL(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                      int64_t ldc) override;

  virtual void solveLUnit(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                          int64_t ldc) override;

  virtual void gemv(const T* data, int64_t offM, int64_t nRows, int64_t nCols, const T* A,
                    int64_t offA, int64_t lda, T alpha) override;
#endif  // SPRUX_USE_BLAS

  static inline void stridedTransAdd(T* dst, int64_t dstStride, const T* src, int64_t srcStride,
                                     int64_t rSize, int64_t cSize) {
    for (uint j = 0; j < rSize; j++) {
      T* pDst = dst + j;
      for (uint i = 0; i < cSize; i++) {
        *pDst += src[i];
        pDst += dstStride;
      }
      src += srcStride;
    }
  }

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, T* C, int64_t ldc) override {
    auto timer = sym.solveAssVStat.instance();
    const T* A = tmpBuf.data();
    const CoalescedBlockMatrixSkel& skel = sym.skel;
    const int64_t* chainRowsTillEnd = skel.chainRowsTillEnd.data() + chainColPtr;
    const int64_t* toSpan = skel.chainRowSpan.data() + chainColPtr;
    int64_t startRow = chainRowsTillEnd[-1];
    for (int64_t i = 0; i < numColItems; i++) {
      int64_t rowOffset = chainRowsTillEnd[i - 1] - startRow;
      int64_t span = toSpan[i];
      int64_t spanStart = skel.spanStart[span];
      int64_t spanSize = skel.spanStart[span + 1] - spanStart;

      stridedTransAdd(C + spanStart, ldc, A + rowOffset * nRHS, nRHS, spanSize, nRHS);
    }
  }

#ifdef SPRUX_USE_BLAS
  virtual void solveLt(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                       int64_t ldc) override;

  virtual void gemvT(const T* data, int64_t offM, int64_t nRows, int64_t nCols, T* A, int64_t offA,
                     int64_t lda, T alpha) override;

  // LU solve methods
  virtual void solveU(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                      int64_t ldc) override;

  virtual void applyRowPermVec(const int64_t* pivots, int64_t n, T* vec, int64_t ldVec) override;

  virtual void applyRowPermVecInv(const int64_t* pivots, int64_t n, T* vec, int64_t ldVec) override;

  virtual void gemvDirect(const T* data, int64_t offset, int64_t nRows, int64_t nCols, T* vec,
                          int64_t srcOff, int64_t dstOff, int64_t ldVec, T alpha) override;
#endif

  static inline void stridedTransSet(T* dst, int64_t dstStride, const T* src, int64_t srcStride,
                                     int64_t rSize, int64_t cSize) {
    for (uint j = 0; j < rSize; j++) {
      T* pDst = dst + j;
      for (uint i = 0; i < cSize; i++) {
        *pDst = src[i];
        pDst += dstStride;
      }
      src += srcStride;
    }
  }

  virtual void assembleVecT(const T* C, int64_t ldc, /* int64_t nRHS, T* A, */
                            int64_t chainColPtr, int64_t numColItems) override {
    auto timer = sym.solveAssVTStat.instance();
    T* A = tmpBuf.data();
    const CoalescedBlockMatrixSkel& skel = sym.skel;
    const int64_t* chainRowsTillEnd = skel.chainRowsTillEnd.data() + chainColPtr;
    const int64_t* toSpan = skel.chainRowSpan.data() + chainColPtr;
    int64_t startRow = chainRowsTillEnd[-1];
    for (int64_t i = 0; i < numColItems; i++) {
      int64_t rowOffset = chainRowsTillEnd[i - 1] - startRow;
      int64_t span = toSpan[i];
      int64_t spanStart = skel.spanStart[span];
      int64_t spanSize = skel.spanStart[span + 1] - spanStart;

      stridedTransSet(A + rowOffset * nRHS, nRHS, C + spanStart, ldc, nRHS, spanSize);
    }
  }

  virtual bool hasFragmentedOps() override { return true; }

  virtual void fragmentedMV(const T* data, const T* x, int64_t spanBegin, int64_t spanEnd, T* y,
                            T alpha) override {
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    if (!sym.useThreads) {
      for (int64_t s = spanBegin; s < spanEnd; s++) {
        int64_t sBegin = skel.spanStart[s];
        int64_t sSize = skel.spanStart[s + 1] - sBegin;
        Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> outVecS(y + sBegin, sSize);
        Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> inVecS(x + sBegin, sSize);

        // diagonal
        int64_t cPtr = skel.chainColPtr[s];
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dBlock(
            data + skel.chainData[cPtr], sSize, sSize);
        outVecS.noalias() += dBlock.template triangularView<Eigen::Lower>() * inVecS * alpha;
        outVecS.noalias() +=
            dBlock.template triangularView<Eigen::StrictlyLower>().adjoint() * inVecS * alpha;

        // blocks below diag
        for (int64_t p = cPtr + 1, pEnd = skel.chainColPtr[s + 1]; p < pEnd; p++) {
          int64_t r = skel.chainRowSpan[p];
          int64_t rBegin = skel.spanStart[r];
          int64_t rSize = skel.spanStart[r + 1] - rBegin;
          int64_t offset = skel.chainData[p];
          Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> inVecR(x + rBegin, rSize);
          Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> block(
              data + offset, sSize, rSize);  // swapped
          outVecS.noalias() += block * inVecR * alpha;

          Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> outVecR(y + rBegin, rSize);
          outVecR.noalias() += block.transpose() * inVecS * alpha;
        }
      }
      return;
    }

    dispenso::TaskSet taskSet(sym.threadPool);
    dispenso::parallel_for(
        taskSet, dispenso::makeChunkedRange(spanBegin, spanEnd, 8L),
        [&](int64_t rangeBegin, int64_t rangeEnd) {
          SPRUX_CHECK_LE(rangeEnd, rangeBegin + 8);
          int64_t dataStart = skel.spanStart[rangeBegin];
          int64_t dataSize = skel.spanStart[rangeEnd] - dataStart;
          T* outData = (T*)alloca(sizeof(T) * dataSize);
          Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(outData, dataSize).setZero();

#if 1
          int64_t rangeSize = rangeEnd - rangeBegin;
          int64_t* boardRowPtr = (int64_t*)alloca(sizeof(int64_t) * rangeSize);
          int64_t* boardRowPtrEnd = (int64_t*)alloca(sizeof(int64_t) * rangeSize);
          std::pair<int64_t, int64_t>* colIndex = (std::pair<int64_t, int64_t>*)alloca(
              sizeof(std::pair<int64_t, int64_t>) * (rangeSize + 1));
          int64_t added = 0;
          for (int64_t i = 0; i < rangeSize; i++) {
            int64_t s = i + rangeBegin;
            int64_t b = boardRowPtr[i] = skel.boardRowPtr[s];
            int64_t bEnd = boardRowPtrEnd[i] = skel.boardRowPtr[s + 1] - 1;
            if (b < bEnd) {
              colIndex[added++] = {skel.boardColLump[b], i};
            }
          }
          rangeSize = added;
          std::sort(colIndex, colIndex + rangeSize);
          colIndex[rangeSize] = {std::numeric_limits<int64_t>::max(), 0};

          while (rangeSize) {
            auto [c, i] = colIndex[0];
            int64_t s = i + rangeBegin;
            int64_t b = boardRowPtr[i];

            if (c >= spanBegin) {
              int64_t sBegin = skel.spanStart[s];
              int64_t sSize = skel.spanStart[s + 1] - sBegin;
              Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> outVec(outData + (sBegin - dataStart),
                                                                  sSize);

              int64_t cBegin = skel.spanStart[c];
              int64_t cSize = skel.spanStart[c + 1] - cBegin;
              int64_t offset = skel.chainData[skel.chainColPtr[c] + skel.boardColOrd[b]];

              Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> inVec(x + cBegin, cSize);
              Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                  block(data + offset, sSize, cSize);
              outVec.noalias() += block * inVec;
            }

            b++;
            bool beforeEnd = b < boardRowPtrEnd[i];
            boardRowPtr[i] = b;
            std::pair<int64_t, int64_t> newColIndex(
                beforeEnd ? skel.boardColLump[b] : (std::numeric_limits<int64_t>::max() - 1), i);

            // re-sort column indices
            int64_t j = 1;
            while (colIndex[j] < newColIndex) {
              colIndex[j - 1] = colIndex[j];
              j++;
            }
            colIndex[j - 1] = newColIndex;

            if (!beforeEnd) {
              rangeSize--;
            }
          }
#endif
          for (int64_t s = rangeBegin; s < rangeEnd; s++) {
            int64_t sBegin = skel.spanStart[s];
            int64_t sSize = skel.spanStart[s + 1] - sBegin;
            Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> outVec(outData + (sBegin - dataStart),
                                                                sSize);

#if 0
            for (int64_t b = skel.boardRowPtr[s + 1] - 2, bEnd = skel.boardRowPtr[s]; b >= bEnd; b--) {
              int64_t c = skel.boardColLump[b];
              if (c < spanBegin) {
                break;
              }

              int64_t cBegin = skel.spanStart[c];
              int64_t cSize = skel.spanStart[c + 1] - cBegin;
              int64_t offset = skel.chainData[skel.chainColPtr[c] + skel.boardColOrd[b]];

              Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> inVec(x + cBegin, cSize);
              Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> block(
                  data + offset, sSize, cSize);
              outVec.noalias() += block * inVec;
            }
#endif

            // diagonal
            int64_t cPtr = skel.chainColPtr[s];
            Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> inVec0(x + sBegin, sSize);
            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                dBlock(data + skel.chainData[cPtr], sSize, sSize);
            outVec.noalias() += dBlock.template triangularView<Eigen::Lower>() * inVec0;
            outVec.noalias() +=
                dBlock.template triangularView<Eigen::StrictlyLower>().adjoint() * inVec0;

            // blocks below diag
            for (int64_t p = cPtr + 1, pEnd = skel.chainColPtr[s + 1]; p < pEnd; p++) {
              int64_t r = skel.chainRowSpan[p];
              int64_t rBegin = skel.spanStart[r];
              int64_t rSize = skel.spanStart[r + 1] - rBegin;
              int64_t offset = skel.chainData[p];
              Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> inVec(x + rBegin, rSize);
              Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
                  block(data + offset, sSize, rSize);  // swapped
              outVec.noalias() += block * inVec;
            }
          }

          Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(y + dataStart, dataSize) +=
              Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(outData, dataSize) * alpha;
        });
  }

  virtual void fragmentedSolveL(const T* data, int64_t spanBegin, int64_t spanEnd, T* y) override {
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    if (!sym.useThreads) {
      for (int64_t s = spanBegin; s < spanEnd; s++) {
        int64_t sBegin = skel.spanStart[s];
        int64_t sSize = skel.spanStart[s + 1] - sBegin;
        int64_t cPtr = skel.chainColPtr[s];
        Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> dVec(y + sBegin, sSize);

        // diagonal
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dBlock(
            data + skel.chainData[cPtr], sSize, sSize);
        dBlock.template triangularView<Eigen::Lower>().solveInPlace(dVec);

        // blocks below diag
        for (int64_t p = cPtr + 1, pEnd = skel.chainColPtr[s + 1]; p < pEnd; p++) {
          int64_t r = skel.chainRowSpan[p];
          int64_t rBegin = skel.spanStart[r];
          int64_t rSize = skel.spanStart[r + 1] - rBegin;
          int64_t offset = skel.chainData[p];
          Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> outVec(y + rBegin, rSize);
          Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> block(
              data + offset, rSize, sSize);  // swapped
          outVec.noalias() -= block * dVec;
        }
      }

      return;
    }

    // iterate [i, i+k] up to spanEnd, then [spanEnd, numSpans]
    static constexpr int kSpan = 128;
    for (int64_t i = spanBegin; i < spanEnd + kSpan; i += kSpan) {
      int64_t subBegin, subEnd;
      if (i < spanEnd) {
        subBegin = i;
        subEnd = std::min(subBegin + kSpan, spanEnd);
      } else {
        subBegin = spanEnd;
        subEnd = skel.numSpans();
      }

      if (subBegin > spanBegin && subBegin < subEnd) {
        dispenso::TaskSet taskSet(sym.threadPool);
        dispenso::parallel_for(
            taskSet, dispenso::makeChunkedRange(subBegin, subEnd, 4L),
            [&](int64_t rangeBegin, int64_t rangeEnd) {
              SPRUX_CHECK_LE(rangeBegin, rangeEnd + 4);
              int64_t dataStart = skel.spanStart[rangeBegin];
              int64_t dataSize = skel.spanStart[rangeEnd] - dataStart;
              T* outData = (T*)alloca(sizeof(T) * dataSize);
              Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(outData, dataSize).setZero();

              int64_t rangeSize = rangeEnd - rangeBegin;
              int64_t* boardRowPtr = (int64_t*)alloca(sizeof(int64_t) * rangeSize);
              int64_t* boardRowPtrEnd = (int64_t*)alloca(sizeof(int64_t) * rangeSize);
              std::pair<int64_t, int64_t>* colIndex = (std::pair<int64_t, int64_t>*)alloca(
                  sizeof(std::pair<int64_t, int64_t>) * (rangeSize + 1));
              int64_t added = 0;
              for (int64_t r = 0; r < rangeSize; r++) {
                int64_t s = r + rangeBegin;
                int64_t b = boardRowPtr[r] = skel.boardRowPtr[s];
                int64_t bEnd = boardRowPtrEnd[r] = skel.boardRowPtr[s + 1] - 1;
                if (b < bEnd && skel.boardColLump[b] < subBegin) {
                  colIndex[added++] = {skel.boardColLump[b], r};
                }
              }
              rangeSize = added;
              std::sort(colIndex, colIndex + rangeSize);
              colIndex[rangeSize] = {std::numeric_limits<int64_t>::max(), 0};

              while (rangeSize) {
                auto [c, k] = colIndex[0];
                int64_t s = k + rangeBegin;
                int64_t b = boardRowPtr[k];

                if (c >= spanBegin) {
                  int64_t sBegin = skel.spanStart[s];
                  int64_t sSize = skel.spanStart[s + 1] - sBegin;
                  Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> outVec(
                      outData + (sBegin - dataStart), sSize);

                  int64_t cBegin = skel.spanStart[c];
                  int64_t cSize = skel.spanStart[c + 1] - cBegin;
                  int64_t offset = skel.chainData[skel.chainColPtr[c] + skel.boardColOrd[b]];

                  Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> inVec(y + cBegin, cSize);
                  Eigen::Map<
                      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                      block(data + offset, sSize, cSize);
                  outVec.noalias() -= block * inVec;
                }

                b++;
                bool beforeEnd = (b < boardRowPtrEnd[k]) && (skel.boardColLump[b] < subBegin);
                boardRowPtr[k] = b;
                std::pair<int64_t, int64_t> newColIndex(
                    beforeEnd ? skel.boardColLump[b] : std::numeric_limits<int64_t>::max() - 1, k);

                // re-sort column indices
                int64_t j = 1;
                while (colIndex[j] < newColIndex) {
                  colIndex[j - 1] = colIndex[j];
                  j++;
                }
                colIndex[j - 1] = newColIndex;

                if (!beforeEnd) {
                  rangeSize--;
                }
              }

              Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(y + dataStart, dataSize) +=
                  Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(outData, dataSize);
            });
      }  // if

      if (i >= spanEnd) {
        break;
      }

      for (int64_t s = subBegin; s < subEnd; s++) {
        int64_t sBegin = skel.spanStart[s];
        int64_t sSize = skel.spanStart[s + 1] - sBegin;
        int64_t cPtr = skel.chainColPtr[s];
        Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> dVec(y + sBegin, sSize);

        // diagonal
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dBlock(
            data + skel.chainData[cPtr], sSize, sSize);
        dBlock.template triangularView<Eigen::Lower>().solveInPlace(dVec);

        // blocks below diag
        for (int64_t p = cPtr + 1, pEnd = skel.chainColPtr[s + 1]; p < pEnd; p++) {
          int64_t r = skel.chainRowSpan[p];
          if (r >= subEnd) {
            break;
          }
          int64_t rBegin = skel.spanStart[r];
          int64_t rSize = skel.spanStart[r + 1] - rBegin;
          int64_t offset = skel.chainData[p];
          Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> outVec(y + rBegin, rSize);
          Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> block(
              data + offset, rSize, sSize);  // swapped
          outVec.noalias() -= block * dVec;
        }
      }
    }
  }

  virtual void fragmentedSolveLt(const T* data, int64_t spanBegin, int64_t spanEnd, T* y) override {
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    if (!sym.useThreads) {
      for (int64_t s = spanEnd - 1; s >= spanBegin; s--) {
        int64_t sBegin = skel.spanStart[s];
        int64_t sSize = skel.spanStart[s + 1] - sBegin;
        int64_t cPtr = skel.chainColPtr[s];
        Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> dVec(y + sBegin, sSize);

        // blocks below diag
        for (int64_t p = cPtr + 1, pEnd = skel.chainColPtr[s + 1]; p < pEnd; p++) {
          int64_t r = skel.chainRowSpan[p];
          int64_t rBegin = skel.spanStart[r];
          int64_t rSize = skel.spanStart[r + 1] - rBegin;
          int64_t offset = skel.chainData[p];
          Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> inVec(y + rBegin, rSize);
          Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> block(
              data + offset, sSize, rSize);  // swapped
          dVec.noalias() -= block * inVec;
        }

        // diagonal
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> dBlock(
            data + skel.chainData[cPtr], sSize, sSize);
        dBlock.template triangularView<Eigen::Upper>().solveInPlace(dVec);
      }
      return;
    }

    for (int64_t subEnd = spanEnd; subEnd > spanBegin; subEnd -= 64) {
      int64_t subBegin = std::max(subEnd - 64, spanBegin);

      if (subEnd < skel.numSpans()) {
        dispenso::TaskSet taskSet(sym.threadPool);
        dispenso::parallel_for(
            taskSet, dispenso::makeChunkedRange(subBegin, subEnd, 2L),
            [&](int64_t thBegin, int64_t thEnd) {
              SPRUX_CHECK_LE(thEnd, thBegin + 2);

              for (int64_t s = thEnd - 1; s >= thBegin; s--) {
                int64_t sBegin = skel.spanStart[s];
                int64_t sSize = skel.spanStart[s + 1] - sBegin;
                T* dVecData = (T*)alloca(sizeof(T) * sSize);
                Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> dVec(dVecData, sSize);
                dVec.setZero();

                // blocks below diag
                for (int64_t p = skel.chainColPtr[s + 1] - 1; /* */; p--) {
                  int64_t r = skel.chainRowSpan[p];
                  if (r < subEnd) {
                    break;
                  }
                  int64_t rBegin = skel.spanStart[r];
                  int64_t rSize = skel.spanStart[r + 1] - rBegin;
                  int64_t offset = skel.chainData[p];
                  Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> inVec(y + rBegin, rSize);
                  Eigen::Map<
                      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
                      block(data + offset, sSize, rSize);  // swapped
                  dVec.noalias() -= block * inVec;
                }

                Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(y + sBegin, sSize) += dVec;
              }
            });
      }

      for (int64_t s = subEnd - 1; s >= subBegin; s--) {
        int64_t sBegin = skel.spanStart[s];
        int64_t sSize = skel.spanStart[s + 1] - sBegin;
        int64_t cPtr = skel.chainColPtr[s];
        Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> dVec(y + sBegin, sSize);

        // blocks below diag
        for (int64_t p = cPtr + 1, pEnd = skel.chainColPtr[s + 1]; p < pEnd; p++) {
          int64_t r = skel.chainRowSpan[p];
          if (r >= subEnd) {
            break;
          }
          int64_t rBegin = skel.spanStart[r];
          int64_t rSize = skel.spanStart[r + 1] - rBegin;
          int64_t offset = skel.chainData[p];
          Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> inVec(y + rBegin, rSize);
          Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> block(
              data + offset, sSize, rSize);  // swapped
          dVec.noalias() -= block * inVec;
        }

        // diagonal
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> dBlock(
            data + skel.chainData[cPtr], sSize, sSize);
        dBlock.template triangularView<Eigen::Upper>().solveInPlace(dVec);
      }
    }
  }

  using CpuBaseSolveCtx<T>::nRHS;
  using CpuBaseSolveCtx<T>::tmpBuf;

  const BlasSymbolicCtx& sym;
};

#ifdef SPRUX_USE_BLAS
template <>
void BlasSolveCtx<double>::symm(const double* data, int64_t offM, int64_t n, const double* C,
                                int64_t offC, int64_t ldc, double* D, int64_t ldd, double alpha) {
  auto timer = sym.symmStat.instance();
  cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, n, nRHS, alpha, data + offM, n, C + offC, ldc,
              1.0, D + offC, ldd);
}

template <>
void BlasSolveCtx<float>::symm(const float* data, int64_t offM, int64_t n, const float* C,
                               int64_t offC, int64_t ldc, float* D, int64_t ldd, float alpha) {
  auto timer = sym.symmStat.instance();
  cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, n, nRHS, alpha, data + offM, n, C + offC, ldc,
              1.0, D + offC, ldd);
}

template <>
void BlasSolveCtx<double>::solveL(const double* data, int64_t offM, int64_t n, double* C,
                                  int64_t offC, int64_t ldc) {
  auto timer = sym.solveLStat.instance();
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, n, nRHS, 1.0,
              data + offM, n, C + offC, ldc);
}

template <>
void BlasSolveCtx<float>::solveL(const float* data, int64_t offM, int64_t n, float* C, int64_t offC,
                                 int64_t ldc) {
  auto timer = sym.solveLStat.instance();
  cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, n, nRHS, 1.0,
              data + offM, n, C + offC, ldc);
}

// solveLUnit: solve L * X = B where L is unit lower triangular (for LU factorization)
// After getrf with transpose workaround, L is stored in row-major lower.
// For row-major L, BLAS with CblasColMajor sees it in upper triangle.
template <>
void BlasSolveCtx<double>::solveLUnit(const double* data, int64_t offM, int64_t n, double* C,
                                      int64_t offC, int64_t ldc) {
  auto timer = sym.solveLStat.instance();
  // Row-major lower L is seen as col-major upper L^T, so use CblasUpper with CblasConjTrans
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasUnit, n, nRHS, 1.0,
              data + offM, n, C + offC, ldc);
}

template <>
void BlasSolveCtx<float>::solveLUnit(const float* data, int64_t offM, int64_t n, float* C,
                                     int64_t offC, int64_t ldc) {
  auto timer = sym.solveLStat.instance();
  cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasUnit, n, nRHS, 1.0,
              data + offM, n, C + offC, ldc);
}

template <>
void BlasSolveCtx<double>::gemv(const double* data, int64_t offM, int64_t nRows, int64_t nCols,
                                const double* A, int64_t offA, int64_t lda, double alpha) {
  auto timer = sym.solveGemvStat.instance();
  cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nRHS, nRows, nCols, alpha, A + offA, lda,
              data + offM, nCols, 0.0, tmpBuf.data(), nRHS);
}

template <>
void BlasSolveCtx<float>::gemv(const float* data, int64_t offM, int64_t nRows, int64_t nCols,
                               const float* A, int64_t offA, int64_t lda, float alpha) {
  auto timer = sym.solveGemvStat.instance();
  cblas_sgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nRHS, nRows, nCols, alpha, A + offA, lda,
              data + offM, nCols, 0.0, tmpBuf.data(), nRHS);
}

template <>
void BlasSolveCtx<double>::solveLt(const double* data, int64_t offM, int64_t n, double* C,
                                   int64_t offC, int64_t ldc) {
  auto timer = sym.solveLtStat.instance();
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, nRHS, 1.0,
              data + offM, n, C + offC, ldc);
}

template <>
void BlasSolveCtx<float>::solveLt(const float* data, int64_t offM, int64_t n, float* C,
                                  int64_t offC, int64_t ldc) {
  auto timer = sym.solveLtStat.instance();
  cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, nRHS, 1.0,
              data + offM, n, C + offC, ldc);
}

template <>
void BlasSolveCtx<double>::gemvT(const double* data, int64_t offM, int64_t nRows, int64_t nCols,
                                 double* A, int64_t offA, int64_t lda, double alpha) {
  auto timer = sym.solveGemvTStat.instance();
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, nCols, nRHS, nRows, alpha, data + offM,
              nCols, tmpBuf.data(), nRHS, 1.0, A + offA, lda);
}

template <>
void BlasSolveCtx<float>::gemvT(const float* data, int64_t offM, int64_t nRows, int64_t nCols,
                                float* A, int64_t offA, int64_t lda, float alpha) {
  auto timer = sym.solveGemvTStat.instance();
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, nCols, nRHS, nRows, alpha, data + offM,
              nCols, tmpBuf.data(), nRHS, 1.0, A + offA, lda);
}

// ============ LU Solve BLAS implementations ============

// solveU: solve U * X = B where U is upper triangular (for LU factorization)
// After getrf with transpose workaround, U is stored in row-major upper.
// For row-major U, BLAS with CblasColMajor sees it in lower triangle.
template <>
void BlasSolveCtx<double>::solveU(const double* data, int64_t offM, int64_t n, double* C,
                                  int64_t offC, int64_t ldc) {
  // Row-major upper U is seen as col-major lower U^T, so use CblasLower with CblasConjTrans
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, CblasNonUnit, n, nRHS, 1.0,
              data + offM, n, C + offC, ldc);
}

template <>
void BlasSolveCtx<float>::solveU(const float* data, int64_t offM, int64_t n, float* C,
                                 int64_t offC, int64_t ldc) {
  cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, CblasNonUnit, n, nRHS, 1.0,
              data + offM, n, C + offC, ldc);
}

template <typename T>
void BlasSolveCtx<T>::applyRowPermVec(const int64_t* pivots, int64_t n, T* vec, int64_t ldVec) {
  // Apply row permutation P to vector: y = P * x
  // For each i, swap row i with row pivots[i]
  for (int64_t i = 0; i < n; i++) {
    int64_t swapRow = pivots[i];
    if (swapRow != i) {
      // Swap elements for all RHS
      for (int rhs = 0; rhs < nRHS; rhs++) {
        std::swap(vec[i + rhs * ldVec], vec[swapRow + rhs * ldVec]);
      }
    }
  }
}

template <typename T>
void BlasSolveCtx<T>::applyRowPermVecInv(const int64_t* pivots, int64_t n, T* vec, int64_t ldVec) {
  // Apply inverse row permutation P^T to vector: y = P^T * x
  // Reverse order from applyRowPermVec
  for (int64_t i = n - 1; i >= 0; i--) {
    int64_t swapRow = pivots[i];
    if (swapRow != i) {
      // Swap elements for all RHS
      for (int rhs = 0; rhs < nRHS; rhs++) {
        std::swap(vec[i + rhs * ldVec], vec[swapRow + rhs * ldVec]);
      }
    }
  }
}

// Direct gemv for U backward solve: result += alpha * M * x
// M is row-major matrix of shape (nRows x nCols) at data+offset
// x is at vec+srcOff, result is updated at vec+dstOff
template <>
void BlasSolveCtx<double>::gemvDirect(const double* data, int64_t offset, int64_t nRows,
                                       int64_t nCols, double* vec, int64_t srcOff, int64_t dstOff,
                                       int64_t ldVec, double alpha) {
  // For row-major M(nRows x nCols), we want y += alpha * M * x
  // Row-major M(nRows x nCols) stored as Col-major M_col^T where M_col has shape (nCols x nRows)
  // y(nRows x nRHS) += alpha * M_col^T(nRows x nCols) * x(nCols x nRHS)
  // Use gemm with m=nRows, n=nRHS, k=nCols
  cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nRows, nRHS, nCols, alpha,
              data + offset, nCols, vec + srcOff, ldVec, 1.0, vec + dstOff, ldVec);
}

template <>
void BlasSolveCtx<float>::gemvDirect(const float* data, int64_t offset, int64_t nRows,
                                      int64_t nCols, float* vec, int64_t srcOff, int64_t dstOff,
                                      int64_t ldVec, float alpha) {
  cblas_sgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nRows, nRHS, nCols, alpha,
              data + offset, nCols, vec + srcOff, ldVec, 1.0, vec + dstOff, ldVec);
}

// Explicit template instantiations for LU solve methods
template void BlasSolveCtx<double>::applyRowPermVec(const int64_t*, int64_t, double*, int64_t);
template void BlasSolveCtx<float>::applyRowPermVec(const int64_t*, int64_t, float*, int64_t);
template void BlasSolveCtx<double>::applyRowPermVecInv(const int64_t*, int64_t, double*, int64_t);
template void BlasSolveCtx<float>::applyRowPermVecInv(const int64_t*, int64_t, float*, int64_t);

#endif  // SPRUX_USE_BLAS

NumericCtxBase* BlasSymbolicCtx::createNumericCtxForType(std::type_index tIdx, int64_t tempBufSize,
                                                         int batchSize) {
  SPRUX_CHECK_EQ(batchSize, 1);
  if (tIdx == std::type_index(typeid(double))) {
    return new BlasNumericCtx<double>(*this, tempBufSize, skel.spanStart.size() - 1);
  } else if (tIdx == std::type_index(typeid(float))) {
    return new BlasNumericCtx<float>(*this, tempBufSize, skel.spanStart.size() - 1);
  } else {
    return nullptr;
  }
}

SolveCtxBase* BlasSymbolicCtx::createSolveCtxForType(std::type_index tIdx, int nRHS,
                                                     int batchSize) {
  SPRUX_CHECK_EQ(batchSize, 1);
  if (tIdx == std::type_index(typeid(double))) {
    return new BlasSolveCtx<double>(*this, nRHS);
  } else if (tIdx == std::type_index(typeid(float))) {
    return new BlasSolveCtx<float>(*this, nRHS);
  } else {
    return nullptr;
  }
}

OpsPtr fastOps(int numThreads) { return OpsPtr(new BlasOps(numThreads)); }

}  // end namespace BaSpaCho
