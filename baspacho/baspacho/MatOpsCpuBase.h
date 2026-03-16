/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <limits>
#include <vector>
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/Utils.h"

namespace BaSpaCho {

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

struct CpuBaseSymElimCtx : SymElimCtx {
  CpuBaseSymElimCtx() {}
  virtual ~CpuBaseSymElimCtx() override {}

  // per-row pointers to chains in a rectagle:
  // * span-rows from lumpToSpan[lumpsEnd],
  // * board cols in interval lumpsBegin:lumpsEnd
  int64_t spanRowBegin;
  int64_t maxBufferSize;
  std::vector<int64_t> rowPtr;       // row data pointer
  std::vector<int64_t> colLump;      // col-lump
  std::vector<int64_t> chainColOrd;  // order in col chain elements
};

// common code for ref and blas implementations
struct CpuBaseSymbolicCtx : SymbolicCtx {
  explicit CpuBaseSymbolicCtx(const CoalescedBlockMatrixSkel& skel) : skel(skel) {}

  virtual PermutedCoalescedAccessor deviceAccessor() override {
    throw std::runtime_error("no device accessor can be create from cpu-only backed!");
  }

  static int64_t computeMaxBufSize(const CpuBaseSymElimCtx& elim,
                                   const CoalescedBlockMatrixSkel& skel, int64_t sRel) {
    int64_t maxBufferSize = 0;

    // iterate over chains present in this row
    for (int64_t i = elim.rowPtr[sRel], iEnd = elim.rowPtr[sRel + 1]; i < iEnd; i++) {
      int64_t lump = elim.colLump[i];
      int64_t chainColOrd = elim.chainColOrd[i];
      SPRUX_CHECK_GE(chainColOrd,
                        1);  // there must be a diagonal block

      int64_t ptrStart = skel.chainColPtr[lump] + chainColOrd;
      int64_t ptrEnd = skel.chainColPtr[lump + 1];

      int64_t nRowsAbove = skel.chainRowsTillEnd[ptrStart - 1];
      int64_t nRowsChain = skel.chainRowsTillEnd[ptrStart] - nRowsAbove;
      int64_t nRowsOnward = skel.chainRowsTillEnd[ptrEnd - 1];

      maxBufferSize = std::max(maxBufferSize, nRowsOnward * nRowsChain);
    }

    return maxBufferSize;
  }

  virtual SymElimCtxPtr prepareElimination(int64_t lumpsBegin, int64_t lumpsEnd) override {
    CpuBaseSymElimCtx* elim = new CpuBaseSymElimCtx;

    int64_t spanRowBegin = skel.lumpToSpan[lumpsEnd];
    int64_t numSpanRows = skel.spanStart.size() - 1 - spanRowBegin;
    elim->rowPtr.assign(numSpanRows + 1, 0);
    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      for (int64_t i = skel.chainColPtr[l], iEnd = skel.chainColPtr[l + 1]; i < iEnd; i++) {
        int64_t s = skel.chainRowSpan[i];
        if (s < spanRowBegin) {
          continue;
        }
        int64_t sRel = s - spanRowBegin;
        elim->rowPtr[sRel]++;
      }
    }
    int64_t totNumChains = cumSumVec(elim->rowPtr);
    elim->colLump.resize(totNumChains);
    elim->chainColOrd.resize(totNumChains);
    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      for (int64_t iBegin = skel.chainColPtr[l], iEnd = skel.chainColPtr[l + 1], i = iBegin;
           i < iEnd; i++) {
        int64_t s = skel.chainRowSpan[i];
        if (s < spanRowBegin) {
          continue;
        }
        int64_t sRel = s - spanRowBegin;
        elim->colLump[elim->rowPtr[sRel]] = l;
        elim->chainColOrd[elim->rowPtr[sRel]] = i - iBegin;
        elim->rowPtr[sRel]++;
      }
    }
    rewindVec(elim->rowPtr);
    elim->spanRowBegin = spanRowBegin;

    elim->maxBufferSize = 0;
    for (int64_t r = 0; r < (int64_t)elim->rowPtr.size() - 1; r++) {
      elim->maxBufferSize = std::max(elim->maxBufferSize, computeMaxBufSize(*elim, skel, r));
    }
    return SymElimCtxPtr(elim);
  }

  const CoalescedBlockMatrixSkel& skel;
};

template <typename T>
struct CpuBaseNumericCtx : NumericCtx<T> {
  CpuBaseNumericCtx(const CpuBaseSymbolicCtx& sym, int64_t bufSize, int64_t numSpans)
      : sym(sym), tempBuffer(bufSize), spanToChainOffset(numSpans) {}

  virtual void potrf(int64_t n, T* data, int64_t offA) override {
    auto timer = sym.potrfStat.instance(sizeof(T), n);
    sym.potrfBiggestN = std::max(sym.potrfBiggestN, n);

    Eigen::Map<MatRMaj<T>> matA(data + offA, n, n);
    Eigen::LLT<Eigen::Ref<MatRMaj<T>>> llt(matA);
  }

  // LDL^T factorization: A = L * D * L^T
  // L is stored below diagonal (unit diagonal implicit), D on diagonal
  virtual int ldlt(int64_t n, T* data, int64_t offA) override {
    // Manual LDL^T implementation to ensure correct in-place storage format:
    // - L factors stored below diagonal (unit diagonal implicit)
    // - D factors stored on diagonal
    Eigen::Map<MatRMaj<T>> matA(data + offA, n, n);
    for (int64_t j = 0; j < n; j++) {
      // Compute D[j] = A[j,j] - sum_{k<j} L[j,k]^2 * D[k]
      T Djj = matA(j, j);
      for (int64_t k = 0; k < j; k++) {
        T Ljk = matA(j, k);
        T Dk = matA(k, k);
        Djj -= Ljk * Ljk * Dk;
      }
      // Use relative tolerance for numerical stability
      // Pivot is considered singular if |Djj| < eps * max_seen_pivot
      constexpr T eps = std::numeric_limits<T>::epsilon() * T(100);
      if (std::abs(Djj) < eps) {
        return j + 1;  // Singular or near-singular matrix
      }
      matA(j, j) = Djj;

      // Compute L[i,j] for i > j: L[i,j] = (A[i,j] - sum_{k<j} L[i,k]*L[j,k]*D[k]) / D[j]
      for (int64_t i = j + 1; i < n; i++) {
        T Lij = matA(i, j);
        for (int64_t k = 0; k < j; k++) {
          T Lik = matA(i, k);
          T Ljk = matA(j, k);
          T Dk = matA(k, k);
          Lij -= Lik * Ljk * Dk;
        }
        matA(i, j) = Lij / Djj;
      }
    }
    return 0;
  }

  // Scale rows of matrix by diagonal: B[i,:] *= D[i]
  virtual void scaleRowsByDiag(int64_t m, int64_t n, const T* D, int64_t offD, T* B, int64_t offB,
                               int64_t ldb) override {
    for (int64_t i = 0; i < m; i++) {
      T di = D[offD + i * (m + 1)];  // D stored on diagonal, stride = n+1 for row-major
      for (int64_t j = 0; j < n; j++) {
        B[offB + i * ldb + j] *= di;
      }
    }
  }

  // C -= L * D * L^T where L is m x k, D is k diagonal elements
  virtual void saveSyrkScaled(int64_t m, int64_t k, const T* L, int64_t offL, int64_t ldL,
                              const T* D, int64_t offD, T* C, int64_t offC, int64_t ldC) override {
    // Compute C -= L * D * L^T
    // This updates the lower triangle of C
    for (int64_t i = 0; i < m; i++) {
      for (int64_t j = 0; j <= i; j++) {
        T sum = T(0);
        for (int64_t q = 0; q < k; q++) {
          T Liq = L[offL + i * ldL + q];
          T Ljq = L[offL + j * ldL + q];
          T Dq = D[offD + q * (k + 1)];  // D on diagonal
          sum += Liq * Dq * Ljq;
        }
        C[offC + i * ldC + j] -= sum;
      }
    }
  }

  // LDL^T off-diagonal column solve: solve X * L^T = B for unit L, then scale by D^{-1}
  // L is n x n with unit diagonal (stored with D on diagonal, which is used for scaling)
  // B is k x n (k rows, n cols), result is stored back in B
  // This computes: B <- B * L^{-T} * D^{-1}
  virtual void trsmUnitScaleInv(int64_t n, int64_t k, T* data, int64_t offA, int64_t offB) override {
    // The stored row-major matrix M at offA has:
    // - D on diagonal (D_0, D_1, ..., D_{n-1})
    // - L (unit lower triangular, L_{ij} for i > j) below diagonal
    //
    // B is k x n stored row-major at offB
    //
    // We want to compute: B <- B * L^{-T} * D^{-1}
    // where L is unit lower triangular (1's on diagonal, values below diagonal from M)
    //
    // Step 1: Solve X * L^T = B for unit upper triangular L^T
    // Step 2: Scale each column j by 1/D_j

    Eigen::Map<const MatRMaj<T>> matA(data + offA, n, n);
    Eigen::Map<MatRMaj<T>> matB(data + offB, k, n);

    // Step 1: Solve X * L^T = B where L is unit lower triangular
    // L^T is unit upper triangular
    // Use Eigen's UnitLower for L, then its adjoint (transpose) for L^T solve
    matA.template triangularView<Eigen::UnitLower>()
        .adjoint()
        .template solveInPlace<Eigen::OnTheRight>(matB);

    // Step 2: Scale each column of B by 1/D_j
    for (int64_t j = 0; j < n; j++) {
      T Dj = matA(j, j);  // D_j is on diagonal
      for (int64_t i = 0; i < k; i++) {
        matB(i, j) /= Dj;
      }
    }
  }

  virtual void trsm(int64_t n, int64_t k, T* data, int64_t offA, int64_t offB) override {
    auto timer = sym.trsmStat.instance(sizeof(T), n, k);

    using MatCMajD = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    // col-major's upper = (row-major's lower).transpose()
    Eigen::Map<const MatCMajD> matA(data + offA, n, n);
    Eigen::Map<MatRMaj<T>> matB(data + offB, k, n);
    matA.template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(matB);
  }

  // For LDL^T Schur complement: computes temp = (L * D) * L^T
  // This is like saveSyrkGemm but each column of L is scaled by D[j]
  virtual void saveSyrkGemmScaled(int64_t m, int64_t n, int64_t k, T* data, int64_t offL,
                                  int64_t offD, int64_t ldD) override {
    SPRUX_CHECK_LE(m * n, (int64_t)tempBuffer.size());

    const T* L = data + offL;
    const T* D = data + offD;  // Diagonal block, D[j] at position D[j * ldD + j]
    T* C = tempBuffer.data();

    Eigen::Map<const MatRMaj<T>> matL_sub(L, m, k);       // First m rows of L
    Eigen::Map<const MatRMaj<T>> matL_full(L, n, k);      // All n rows of L
    Eigen::Map<MatRMaj<T>> matC(C, n, m);

    // Compute W = L_full * diag(D) (scale each column of L by D[j])
    // Then C = W * L_sub^T
    // We do this efficiently by noting: C[i,j] = sum_q L_full[i,q] * D[q] * L_sub[j,q]

    for (int64_t i = 0; i < n; i++) {
      for (int64_t j = 0; j < m; j++) {
        T sum = T(0);
        for (int64_t q = 0; q < k; q++) {
          T Dq = D[q * ldD + q];  // D[q] is at diagonal position (q, q)
          sum += matL_full(i, q) * Dq * matL_sub(j, q);
        }
        matC(i, j) = sum;
      }
    }
  }

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const T* data,
                            int64_t offset) override {
    auto timer = sym.sygeStat.instance(sizeof(T), m, n, k);
    SPRUX_CHECK_LE(m * n, (int64_t)tempBuffer.size());

    const T* AB = data + offset;
    T* C = tempBuffer.data();
    Eigen::Map<const MatRMaj<T>> matA(AB, m, k);
    Eigen::Map<const MatRMaj<T>> matB(AB, n, k);
    Eigen::Map<MatRMaj<T>> matC(C, n, m);
    matC.noalias() = matB * matA.transpose();

    sym.gemmCalls++;
  }

  // helper for elimination, does
  // 1. potrf on diagonal block corresponding to a supernode (lump), and
  // 2. trsm on all rows below
  static inline void factorLump(const CoalescedBlockMatrixSkel& skel, T* data, int64_t lump) {
    int64_t lumpStart = skel.lumpStart[lump];
    int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
    int64_t colStart = skel.chainColPtr[lump];
    int64_t dataPtr = skel.chainData[colStart];

    // in-place lower diag cholesky dec on diagonal block
    Eigen::Map<MatRMaj<T>> diagBlock(data + dataPtr, lumpSize, lumpSize);
    { Eigen::LLT<Eigen::Ref<MatRMaj<T>>> llt(diagBlock); }

    int64_t gatheredStart = skel.boardColPtr[lump];
    int64_t gatheredEnd = skel.boardColPtr[lump + 1];
    int64_t rowDataStart = skel.boardChainColOrd[gatheredStart + 1];
    int64_t rowDataEnd = skel.boardChainColOrd[gatheredEnd - 1];
    int64_t belowDiagStart = skel.chainData[colStart + rowDataStart];
    int64_t numRows = skel.chainRowsTillEnd[colStart + rowDataEnd - 1] -
                      skel.chainRowsTillEnd[colStart + rowDataStart - 1];

    Eigen::Map<MatRMaj<T>> belowDiagBlock(data + belowDiagStart, numRows, lumpSize);
    diagBlock.template triangularView<Eigen::Lower>()
        .transpose()
        .template solveInPlace<Eigen::OnTheRight>(belowDiagBlock);
  }

  static inline void factorSpan(const CoalescedBlockMatrixSkel& skel, T* data, int64_t span) {
    int64_t lump = skel.spanToLump[span];
    int64_t spanOffsetInLump = skel.spanOffsetInLump[span];
    int64_t spanIndexInLump = span - skel.lumpToSpan[lump];
    int64_t spanSize = skel.spanStart[span + 1] - skel.spanStart[span];
    int64_t lumpStart = skel.lumpStart[lump];
    int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
    int64_t colStart = skel.chainColPtr[lump];
    int64_t diagBlockPtr = skel.chainData[colStart + spanIndexInLump] + spanOffsetInLump;

    // in-place lower diag cholesky dec on diagonal block
    OuterStridedMatM<T> diagBlock(data + diagBlockPtr, spanSize, spanSize, OuterStride(lumpSize));
    { Eigen::LLT<Eigen::Ref<MatRMaj<T>>> llt(diagBlock); }

    int64_t gatheredEnd = skel.boardColPtr[lump + 1];
    int64_t rowDataEnd = skel.boardChainColOrd[gatheredEnd - 1];
    int64_t belowDiagPtr = skel.chainData[colStart + spanIndexInLump + 1] + spanOffsetInLump;
    int64_t numRows = skel.chainRowsTillEnd[colStart + rowDataEnd - 1] -
                      skel.chainRowsTillEnd[colStart + spanIndexInLump];

    OuterStridedMatM<T> belowDiagBlock(data + belowDiagPtr, numRows, spanSize,
                                       OuterStride(lumpSize));
    diagBlock.template triangularView<Eigen::Lower>()
        .transpose()
        .template solveInPlace<Eigen::OnTheRight>(belowDiagBlock);
  }

  static inline void stridedMatSub(T* dst, int64_t dstStride, const T* src, int64_t srcStride,
                                   int64_t rSize, int64_t cSize) {
    for (uint j = 0; j < rSize; j++) {
      for (uint i = 0; i < cSize; i++) {
        dst[i] -= src[i];
      }
      dst += dstStride;
      src += srcStride;
    }
  }

  static void prepareContextForTargetLump(const CoalescedBlockMatrixSkel& skel, int64_t targetLump,
                                          std::vector<int64_t>& spanToChainOffset) {
    for (int64_t i = skel.chainColPtr[targetLump], iEnd = skel.chainColPtr[targetLump + 1];
         i < iEnd; i++) {
      spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
    }
  }

  static inline T* elimDiagBlock(int m, int k, T* A, int lda, T* B) {
    T* A_row = A;
    T* B_row = B;
    for (int i = 0; i < m; i++) {
      T* C_row = B;
      for (int j = 0; j <= i; j++) {
        T& v = A_row[j];
        for (int q = 0; q < k; q++) {
          v -= B_row[q] * C_row[q];
        }
        C_row += k;
      }
      A_row += lda;
      B_row += k;
    }
    return B_row;
  }

  static inline T* elimBlock(int m, int n, int k, T* A, int lda, T* B, T* C) {
    T* A_row = A;
    T* B_row = B;
    for (int i = 0; i < m; i++) {
      T* C_row = C;
      for (int j = 0; j < n; j++) {
        T& v = A_row[j];
        for (int q = 0; q < k; q++) {
          v -= B_row[q] * C_row[q];
        }
        C_row += k;
      }
      A_row += lda;
      B_row += k;
    }
    return B_row;
  }

  static void eliminateRowChain(const CpuBaseSymElimCtx& elim, const CoalescedBlockMatrixSkel& skel,
                                T* data, int64_t sRel, std::vector<int64_t>& spanToChainOffset) {
    int64_t s = sRel + elim.spanRowBegin;
    if (elim.rowPtr[sRel] == elim.rowPtr[sRel + 1]) {
      return;
    }
    int64_t targetLump = skel.spanToLump[s];
    int64_t targetLumpSize = skel.lumpStart[targetLump + 1] - skel.lumpStart[targetLump];
    int64_t spanOffsetInLump = skel.spanStart[s] - skel.lumpStart[targetLump];
    prepareContextForTargetLump(skel, targetLump, spanToChainOffset);

    const int64_t* pChainRowsTillEnd = skel.chainRowsTillEnd.data();
    const int64_t* pChainRowSpan = skel.chainRowSpan.data();
    const int64_t* pSpanToChainOffset = spanToChainOffset.data();

    // iterate over chains present in this row
    for (int64_t i = elim.rowPtr[sRel], iEnd = elim.rowPtr[sRel + 1]; i < iEnd; i++) {
      int64_t lump = elim.colLump[i];
      int64_t chainColOrd = elim.chainColOrd[i];
      SPRUX_CHECK_GE(chainColOrd,
                        1);  // there must be a diagonal block

      int64_t ptrStart = skel.chainColPtr[lump] + chainColOrd;
      int64_t ptrEnd = skel.chainColPtr[lump + 1];
      SPRUX_CHECK_EQ(pChainRowSpan[ptrStart], s);

      int64_t nRowsAbove = pChainRowsTillEnd[ptrStart - 1];
      int64_t nRowsChain = pChainRowsTillEnd[ptrStart] - nRowsAbove;
      T* origDataStart = data + skel.chainData[ptrStart];
      SPRUX_CHECK_EQ(nRowsChain, skel.spanStart[s + 1] - skel.spanStart[s]);
      int64_t lumpSize = skel.lumpStart[lump + 1] - skel.lumpStart[lump];

      T* headTargetData = data + spanOffsetInLump + pSpanToChainOffset[pChainRowSpan[ptrStart]];
      T* nextDataStart =
          elimDiagBlock(nRowsChain, lumpSize, headTargetData, targetLumpSize, origDataStart);

      // assemble blocks, iterating on chain and below chains
      int64_t nextRelRow = pChainRowsTillEnd[ptrStart] - nRowsAbove;
      for (int64_t ptr = ptrStart + 1; ptr < ptrEnd; ptr++) {
        int64_t s2 = pChainRowSpan[ptr];
        int64_t relRow = nextRelRow;
        nextRelRow = pChainRowsTillEnd[ptr] - nRowsAbove;
        int64_t s2_size = nextRelRow - relRow;

        // incomment below if check is needed
        // SPRUX_CHECK(spanToChainOffset[s2] != kInvalid);
        T* targetData = data + spanOffsetInLump + pSpanToChainOffset[s2];

        nextDataStart = elimBlock(s2_size, nRowsChain, lumpSize, targetData, targetLumpSize,
                                  nextDataStart, origDataStart);
      }
    }
  }

  static void eliminateVerySparseRowChain(const CpuBaseSymElimCtx& elim,
                                          const CoalescedBlockMatrixSkel& skel, T* data,
                                          int64_t sRel, std::vector<T>& reusableTmpBuf) {
    int64_t s = sRel + elim.spanRowBegin;
    if (elim.rowPtr[sRel] == elim.rowPtr[sRel + 1]) {
      return;
    }
    int64_t targetLump = skel.spanToLump[s];
    int64_t targetLumpSize = skel.lumpStart[targetLump + 1] - skel.lumpStart[targetLump];
    int64_t spanOffsetInLump = skel.spanStart[s] - skel.lumpStart[targetLump];
    int64_t bisectStart = skel.chainColPtr[targetLump];
    int64_t bisectEnd = skel.chainColPtr[targetLump + 1];

    // iterate over chains present in this row
    for (int64_t i = elim.rowPtr[sRel], iEnd = elim.rowPtr[sRel + 1]; i < iEnd; i++) {
      int64_t lump = elim.colLump[i];
      int64_t chainColOrd = elim.chainColOrd[i];
      SPRUX_CHECK_GE(chainColOrd,
                        1);  // there must be a diagonal block

      int64_t ptrStart = skel.chainColPtr[lump] + chainColOrd;
      int64_t ptrEnd = skel.chainColPtr[lump + 1];
      SPRUX_CHECK_EQ(skel.chainRowSpan[ptrStart], s);

      int64_t nRowsAbove = skel.chainRowsTillEnd[ptrStart - 1];
      int64_t nRowsChain = skel.chainRowsTillEnd[ptrStart] - nRowsAbove;
      int64_t nRowsOnward = skel.chainRowsTillEnd[ptrEnd - 1];
      int64_t dataOffset = skel.chainData[ptrStart];
      SPRUX_CHECK_EQ(nRowsChain, skel.spanStart[s + 1] - skel.spanStart[s]);
      int64_t lumpSize = skel.lumpStart[lump + 1] - skel.lumpStart[lump];

      Eigen::Map<MatRMaj<T>> chainSubMat(data + dataOffset, nRowsChain, lumpSize);
      Eigen::Map<MatRMaj<T>> chainOnwardSubMat(data + dataOffset, nRowsOnward, lumpSize);

      reusableTmpBuf.resize(nRowsOnward * nRowsChain);
      Eigen::Map<MatRMaj<T>> prod(reusableTmpBuf.data(), nRowsOnward, nRowsChain);
      prod.noalias() = chainOnwardSubMat * chainSubMat.transpose();

      // assemble blocks, iterating on chain and below chains
      for (int64_t ptr = ptrStart; ptr < ptrEnd; ptr++) {
        int64_t s2 = skel.chainRowSpan[ptr];
        int64_t relRow = skel.chainRowsTillEnd[ptr - 1] - nRowsAbove;
        int64_t s2_size = skel.chainRowsTillEnd[ptr] - nRowsAbove - relRow;

        int64_t pos = bisect(skel.chainRowSpan.data() + bisectStart, bisectEnd - bisectStart, s2);
        int64_t chainOffset = skel.chainData[bisectStart + pos];
        T* targetData = data + spanOffsetInLump + chainOffset;

        stridedMatSub(targetData, targetLumpSize, prod.data() + nRowsChain * relRow, nRowsChain,
                      s2_size, nRowsChain);
      }
    }
  }

  // temporary data
  const CpuBaseSymbolicCtx& sym;
  std::vector<T> tempBuffer;
  std::vector<int64_t> spanToChainOffset;
};

template <typename T>
struct CpuBaseSolveCtx : SolveCtx<T> {
  CpuBaseSolveCtx(const CpuBaseSymbolicCtx& sym, int nRHS)
      : sym(sym), nRHS(nRHS), tmpBuf(sym.skel.order() * nRHS) {}
  virtual ~CpuBaseSolveCtx() override {}

  virtual void symm(const T* data, int64_t offM, int64_t n, const T* C, int64_t offC, int64_t ldc,
                    T* D, int64_t ldd, T alpha) override {
    auto timer = sym.symmStat.instance();
    Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
    OuterStridedCMajMatK<T> matC(C + offC, n, nRHS, OuterStride(ldc));
    OuterStridedCMajMatM<T> matD(D + offC, n, nRHS, OuterStride(ldd));

    matD.noalias() += alpha * (MatRMaj<T>(matA.template selfadjointView<Eigen::Lower>()) * matC);
  }

  virtual void solveL(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                      int64_t ldc) override {
    auto timer = sym.solveLStat.instance();
    Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
    OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
    matA.template triangularView<Eigen::Lower>().solveInPlace(matC);
  }

  // Solve L * x = b where L is unit lower triangular (forward substitution)
  virtual void solveLUnit(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                          int64_t ldc) override {
    Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
    OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
    matA.template triangularView<Eigen::UnitLower>().solveInPlace(matC);
  }

  virtual void gemv(const T* data, int64_t offM, int64_t nRows, int64_t nCols, const T* A,
                    int64_t offA, int64_t lda, T alpha) override {
    auto timer = sym.solveGemvStat.instance();
    Eigen::Map<const MatRMaj<T>> matM(data + offM, nRows, nCols);
    OuterStridedCMajMatK<T> matA(A + offA, nCols, nRHS, OuterStride(lda));
    Eigen::Map<MatRMaj<T>> matC(tmpBuf.data(), nRows, nRHS);
    matC.noalias() = alpha * (matM * matA);
  }

  virtual void solveLt(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                       int64_t ldc) override {
    auto timer = sym.solveLtStat.instance();
    Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
    OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
    matA.template triangularView<Eigen::Lower>().adjoint().solveInPlace(matC);
  }

  virtual void gemvT(const T* data, int64_t offM, int64_t nRows, int64_t nCols, T* A, int64_t offA,
                     int64_t lda, T alpha) override {
    auto timer = sym.solveGemvTStat.instance();
    Eigen::Map<const MatRMaj<T>> matM(data + offM, nRows, nCols);
    OuterStridedCMajMatM<T> matA(A + offA, nCols, nRHS, OuterStride(lda));
    Eigen::Map<const MatRMaj<T>> matC(tmpBuf.data(), nRows, nRHS);
    matA.noalias() += alpha * (matM.transpose() * matC);
  }

  // LDL^T solve methods

  // Solve D * x = b where D is diagonal (stored on diagonal of matrix)
  virtual void solveDiag(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                         int64_t ldc) override {
    OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
    for (int64_t i = 0; i < n; i++) {
      T di = data[offM + i * n + i];  // Diagonal element at (i,i)
      for (int64_t j = 0; j < nRHS; j++) {
        matC(i, j) /= di;
      }
    }
  }

  // Solve L^T * x = b where L is unit lower triangular (backward substitution)
  virtual void solveLtUnit(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                           int64_t ldc) override {
    Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
    OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
    // L^T is unit upper triangular, solve from bottom up
    // Using Eigen's unit lower triangular transpose solve
    matA.template triangularView<Eigen::UnitLower>().adjoint().solveInPlace(matC);
  }

  const CpuBaseSymbolicCtx& sym;
  int64_t nRHS;
  std::vector<T> tmpBuf;
};

}  // end namespace BaSpaCho
