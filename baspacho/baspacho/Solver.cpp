/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/Solver.h"
#include <chrono>
#include <dispenso/parallel_for.h>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include "baspacho/baspacho/ComputationModel.h"
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/Utils.h"

#ifdef __APPLE__
#include <os/signpost.h>
static os_log_t baspachoSignpostLog() {
  // OS_LOG_CATEGORY_POINTS_OF_INTEREST makes signposts appear in the
  // "Points of Interest" track in Instruments without a custom template.
  static os_log_t log =
      os_log_create("com.baspacho.solver", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
  return log;
}
#define BASPACHO_SIGNPOST_BEGIN(name) \
  os_signpost_interval_begin(baspachoSignpostLog(), OS_SIGNPOST_ID_EXCLUSIVE, name)
#define BASPACHO_SIGNPOST_END(name) \
  os_signpost_interval_end(baspachoSignpostLog(), OS_SIGNPOST_ID_EXCLUSIVE, name)
#else
#define BASPACHO_SIGNPOST_BEGIN(name) ((void)0)
#define BASPACHO_SIGNPOST_END(name) ((void)0)
#endif

namespace BaSpaCho {

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

Solver::Solver(CoalescedBlockMatrixSkel&& factorSkel_, std::vector<int64_t>&& sparseElimRanges_,
               std::vector<int64_t>&& permutation_, OpsPtr&& ops_, int64_t canFactorUpTo_,
               LevelSetSchedule&& levelSetSchedule, double staticPivotThreshold)
    : factorSkel(std::move(factorSkel_)),
      sparseElimRanges(std::move(sparseElimRanges_)),
      permutation(std::move(permutation_)),
      canFactorUpTo(canFactorUpTo_),
      levelSetSchedule_(std::move(levelSetSchedule)),
      staticPivotThreshold_(staticPivotThreshold),
      ops(std::move(ops_)) {
  if (canFactorUpTo < 0) {
    canFactorUpTo = factorSkel.numSpans();
  }
  symCtx = ops->createSymbolicCtx(factorSkel, permutation);
  for (int64_t l = 0; l + 1 < (int64_t)sparseElimRanges.size(); l++) {
    elimCtxs.push_back(symCtx->prepareElimination(sparseElimRanges[l], sparseElimRanges[l + 1]));
  }

  // Create LU sparse elimination contexts for non-symmetric matrices.
  // Only populate luElimCtxs if the backend supports LU sparse elimination
  // (returns non-null contexts). Otherwise leave empty to use the dense LU path.
  if (factorSkel.isGeneral() && !sparseElimRanges.empty()) {
    std::vector<SymElimCtxPtr> tmpCtxs;
    bool anyValid = false;
    for (int64_t l = 0; l + 1 < (int64_t)sparseElimRanges.size(); l++) {
      auto ctx = symCtx->prepareLUElimination(sparseElimRanges[l], sparseElimRanges[l + 1]);
      if (ctx) anyValid = true;
      tmpCtxs.push_back(std::move(ctx));
    }
    if (anyValid) luElimCtxs = std::move(tmpCtxs);
  }

  initElimination();
}

template <typename T>
void Solver::factorLump(NumericCtx<T>& numCtx, T* data, int64_t lump) const {
  int64_t lumpStart = factorSkel.lumpStart[lump];
  int64_t lumpSize = factorSkel.lumpStart[lump + 1] - lumpStart;
  int64_t chainColBegin = factorSkel.chainColPtr[lump];
  int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

  // compute lower diag cholesky dec on diagonal block
  numCtx.potrf(lumpSize, data, diagBlockOffset);

  int64_t boardColBegin = factorSkel.boardColPtr[lump];
  int64_t boardColEnd = factorSkel.boardColPtr[lump + 1];
  int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + 1];
  int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
  int64_t belowDiagOffset = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
  int64_t numRowsBelowDiag = factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
                             factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
  if (numRowsBelowDiag == 0) {
    return;
  }

  numCtx.trsm(lumpSize, numRowsBelowDiag, data, diagBlockOffset, belowDiagOffset);
}

template <typename T>
void Solver::eliminateBoard(NumericCtx<T>& numCtx, T* data, int64_t ptr) const {
  int64_t origLump = factorSkel.boardColLump[ptr];
  int64_t boardIndexInCol = factorSkel.boardColOrd[ptr];

  int64_t origLumpSize = factorSkel.lumpStart[origLump + 1] - factorSkel.lumpStart[origLump];
  int64_t chainColBegin = factorSkel.chainColPtr[origLump];

  int64_t boardColBegin = factorSkel.boardColPtr[origLump];
  int64_t boardColEnd = factorSkel.boardColPtr[origLump + 1];

  int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol];
  int64_t rowDataEnd0 = factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol + 1];
  int64_t rowDataEnd1 = factorSkel.boardChainColOrd[boardColEnd - 1];

  int64_t belowDiagStart = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
  int64_t rectRowBegin = factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
  int64_t numRowsSub = factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] - rectRowBegin;
  int64_t numRowsFull = factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] - rectRowBegin;

  numCtx.saveSyrkGemm(numRowsSub, numRowsFull, origLumpSize, data, belowDiagStart);

  int64_t targetLump = factorSkel.boardRowLump[boardColBegin + boardIndexInCol];
  int64_t targetLumpSize = factorSkel.lumpStart[targetLump + 1] - factorSkel.lumpStart[targetLump];
  int64_t srcColDataOffset = chainColBegin + belowDiagChainColOrd;
  int64_t numBlockRows = rowDataEnd1 - belowDiagChainColOrd;
  int64_t numBlockCols = rowDataEnd0 - belowDiagChainColOrd;

  numCtx.assemble(data, rectRowBegin,
                  targetLumpSize,    //
                  srcColDataOffset,  //
                  numRowsSub, numBlockRows, numBlockCols);
}

int64_t Solver::boardElimTempSize(int64_t lump, int64_t boardIndexInCol) const {
  int64_t chainColBegin = factorSkel.chainColPtr[lump];

  int64_t boardColBegin = factorSkel.boardColPtr[lump];
  int64_t boardColEnd = factorSkel.boardColPtr[lump + 1];

  int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol];
  int64_t rowDataEnd0 = factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol + 1];
  int64_t rowDataEnd1 = factorSkel.boardChainColOrd[boardColEnd - 1];

  int64_t rectRowBegin = factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
  int64_t numRowsSub = factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] - rectRowBegin;
  int64_t numRowsFull = factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] - rectRowBegin;

  return numRowsSub * numRowsFull;
}

void Solver::initElimination() {
  int64_t denseOpsFromLump = sparseElimRanges.size() ? sparseElimRanges.back() : 0;

  startElimRowPtr.resize(factorSkel.chainColPtr.size() - 1 - denseOpsFromLump);
  maxElimTempSize = 0;
  for (int64_t l = denseOpsFromLump; l < (int64_t)factorSkel.chainColPtr.size() - 1; l++) {
    //  iterate over columns having a non-trivial a-block
    int64_t rPtr0 = factorSkel.boardRowPtr[l];
    int64_t rEnd0 = factorSkel.boardRowPtr[l + 1];
    BASPACHO_CHECK_EQ(factorSkel.boardColLump[rEnd0 - 1], l);
    while (factorSkel.boardColLump[rPtr0] < denseOpsFromLump) {
      rPtr0++;
    }
    BASPACHO_CHECK_LT(rPtr0,
                      rEnd0);  // will stop before end as l > denseOpsFromLump
    startElimRowPtr[l - denseOpsFromLump] = rPtr0;

    for (int64_t rPtr = startElimRowPtr[l - denseOpsFromLump],
                 rEnd = factorSkel.boardRowPtr[l + 1];      //
         rPtr < rEnd && factorSkel.boardColLump[rPtr] < l;  //
         rPtr++) {
      int64_t origLump = factorSkel.boardColLump[rPtr];
      int64_t boardIndexInCol = factorSkel.boardColOrd[rPtr];
      int64_t boardSNDataStart = factorSkel.boardColPtr[origLump];
      int64_t boardSNDataEnd = factorSkel.boardColPtr[origLump + 1];
      BASPACHO_CHECK_LT(boardIndexInCol, boardSNDataEnd - boardSNDataStart);
      BASPACHO_CHECK_EQ(l, factorSkel.boardRowLump[boardSNDataStart + boardIndexInCol]);
      maxElimTempSize = max(maxElimTempSize, boardElimTempSize(origLump, boardIndexInCol));
    }
  }
}

template <typename T>
void Solver::factor(T* data, bool verbose) const {
  factorUpTo(data, factorSkel.numSpans(), verbose);
}

template <typename T>
void Solver::factorUpTo(T* data, int64_t spanIndex, bool verbose) const {
  internalFactorRange(data, 0, spanIndex, verbose);
}

template <typename T>
void Solver::factorFrom(T* data, int64_t spanIndex, bool verbose) const {
  internalFactorRange(data, spanIndex, factorSkel.numSpans(), verbose);
}

template <typename T>
void Solver::internalFactorRange(T* data, int64_t startSpanIndex, int64_t endSpanIndex,
                                 bool verbose) const {
  BASPACHO_CHECK_GE(startSpanIndex, 0);
  BASPACHO_CHECK_LE(startSpanIndex, endSpanIndex);
  BASPACHO_CHECK_LT(endSpanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[startSpanIndex], 0);
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[endSpanIndex], 0);
  BASPACHO_CHECK_LE(endSpanIndex, canFactorUpTo);
  int64_t startLump = factorSkel.spanToLump[startSpanIndex];
  int64_t upToLump = factorSkel.spanToLump[endSpanIndex];

  NumericCtxPtr<T> numCtx = symCtx->createNumericCtx<T>(maxElimTempSize, data);

  if (!sparseElimRanges.empty() && startLump == 0 && upToLump >= sparseElimRanges.back()) {
    // Common case: full range — batch all levels in one GPU submission
    if (verbose) {
      for (int64_t l = 0; l + 1 < (int64_t)sparseElimRanges.size(); l++) {
        std::cout << "Elim set: " << l << " (" << sparseElimRanges[l] << ".."
                  << sparseElimRanges[l + 1] << ")" << std::endl;
      }
    }
    numCtx->doAllEliminations(elimCtxs, sparseElimRanges, data);
  } else {
    for (int64_t l = 0; l + 1 < (int64_t)sparseElimRanges.size(); l++) {
      if (sparseElimRanges[l + 1] > upToLump) {
        BASPACHO_CHECK_EQ(sparseElimRanges[l], upToLump);
        return;
      } else if (startLump > sparseElimRanges[l]) {
        BASPACHO_CHECK_GE(startLump, sparseElimRanges[l + 1]);
        continue;
      }
      if (verbose) {
        std::cout << "Elim set: " << l << " (" << sparseElimRanges[l] << ".."
                  << sparseElimRanges[l + 1] << ")" << std::endl;
      }
      numCtx->doElimination(*elimCtxs[l], data, sparseElimRanges[l], sparseElimRanges[l + 1]);
    }
  }

  int64_t denseOpsFromLump = sparseElimRanges.empty() ? 0 : sparseElimRanges.back();
  if (verbose) {
    std::cout << "Block-Fact from: " << denseOpsFromLump << std::endl;
  }

  for (int64_t l = std::max(startLump, denseOpsFromLump);
       l < (int64_t)factorSkel.chainColPtr.size() - 1; l++) {
    numCtx->prepareAssemble(l);

    //  iterate over columns having a non-trivial a-block
    for (int64_t rPtr = startElimRowPtr[l - denseOpsFromLump],
                 rEnd = factorSkel.boardRowPtr[l + 1] - 1;  // skip last (diag block)
         rPtr < rEnd; rPtr++) {
      int64_t origLump = factorSkel.boardColLump[rPtr];
      if (origLump >= upToLump) {
        break;
      } else if (origLump < startLump) {
        continue;
      }
      eliminateBoard(*numCtx, data, rPtr);
    }

    if (l < upToLump) {
      factorLump(*numCtx, data, l);
    }
  }
}

template <typename T>
void Solver::solve(const T* matData, T* vecData, int64_t stride, int nRHS) const {
  SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);
  internalSolveLRange(*slvCtx, matData, 0, factorSkel.numSpans(), vecData, stride, nRHS);
  internalSolveLtRange(*slvCtx, matData, 0, factorSkel.numSpans(), vecData, stride, nRHS);
  slvCtx->flush();  // Final flush: ensure all GPU solve work is complete
}

template <typename T>
void Solver::solveL(const T* matData, T* vecData, int64_t stride, int nRHS) const {
  solveLUpTo(matData, factorSkel.numSpans(), vecData, stride, nRHS);
}

template <typename T>
void Solver::solveLt(const T* matData, T* vecData, int64_t stride, int nRHS) const {
  solveLtUpTo(matData, factorSkel.numSpans(), vecData, stride, nRHS);
}

template <typename T>
void Solver::solveLUpTo(const T* matData, int64_t spanIndex, T* vecData, int64_t stride,
                        int nRHS) const {
  SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);
  internalSolveLRange(*slvCtx, matData, 0, spanIndex, vecData, stride, nRHS);
}

template <typename T>
void Solver::solveLtUpTo(const T* matData, int64_t spanIndex, T* vecData, int64_t stride,
                         int nRHS) const {
  SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);
  internalSolveLtRange(*slvCtx, matData, 0, spanIndex, vecData, stride, nRHS);
}

template <typename T>
void Solver::solveLFrom(const T* matData, int64_t spanIndex, T* vecData, int64_t stride,
                        int nRHS) const {
  SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);
  internalSolveLRange(*slvCtx, matData, spanIndex, factorSkel.numSpans(), vecData, stride, nRHS);
}

template <typename T>
void Solver::solveLtFrom(const T* matData, int64_t spanIndex, T* vecData, int64_t stride,
                         int nRHS) const {
  SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);
  internalSolveLtRange(*slvCtx, matData, spanIndex, factorSkel.numSpans(), vecData, stride, nRHS);
}

static constexpr bool SparseElimSolve = true;

template <typename T>
void Solver::internalSolveLRange(SolveCtx<T>& slvCtx, const T* matData, int64_t startSpanIndex,
                                 int64_t endSpanIndex, T* vecData, int64_t stride, int nRHS) const {
  BASPACHO_CHECK_GE(startSpanIndex, 0);
  BASPACHO_CHECK_LE(startSpanIndex, endSpanIndex);
  BASPACHO_CHECK_LT(endSpanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[startSpanIndex], 0);
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[endSpanIndex], 0);
  int64_t startLump = factorSkel.spanToLump[startSpanIndex];
  int64_t upToLump = factorSkel.spanToLump[endSpanIndex];

  int64_t denseOpsFromLump;
  if (SparseElimSolve) {
    for (int64_t l = 0; l + 1 < (int64_t)sparseElimRanges.size(); l++) {
      if (sparseElimRanges[l + 1] > upToLump) {
        BASPACHO_CHECK_EQ(sparseElimRanges[l], upToLump);
        return;
      } else if (startLump > sparseElimRanges[l]) {
        BASPACHO_CHECK_GE(startLump, sparseElimRanges[l + 1]);
        continue;
      }
      slvCtx.sparseElimSolveL(*elimCtxs[l], matData, sparseElimRanges[l], sparseElimRanges[l + 1],
                              vecData, stride);
    }

    denseOpsFromLump =
        std::max(startLump, (int64_t)(sparseElimRanges.empty() ? 0 : sparseElimRanges.back()));
  } else {
    denseOpsFromLump = startLump;
  }

  if (factorSkel.numSpans() == factorSkel.numLumps() && slvCtx.hasFragmentedOps() && nRHS == 1) {
    BASPACHO_CHECK_EQ(factorSkel.lumpToSpan[denseOpsFromLump], denseOpsFromLump);
    slvCtx.fragmentedSolveL(matData, denseOpsFromLump, upToLump, vecData);
  } else {
    for (int64_t l = denseOpsFromLump; l < upToLump; l++) {
      int64_t lumpStart = factorSkel.lumpStart[l];
      int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
      int64_t chainColBegin = factorSkel.chainColPtr[l];
      int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

      slvCtx.solveL(matData, diagBlockOffset, lumpSize, vecData, lumpStart, stride);

      int64_t boardColBegin = factorSkel.boardColPtr[l];
      int64_t boardColEnd = factorSkel.boardColPtr[l + 1];
      int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + 1];
      int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
      int64_t belowDiagOffset = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
      int64_t numRowsBelowDiag =
          factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
          factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
      if (numRowsBelowDiag == 0) {
        continue;
      }

      slvCtx.gemv(matData, belowDiagOffset, numRowsBelowDiag, lumpSize, vecData, lumpStart, stride,
                  -1.0);

      int64_t chainColPtr = chainColBegin + belowDiagChainColOrd;
      slvCtx.assembleVec(chainColPtr, numColChains - belowDiagChainColOrd, vecData, stride);
    }
  }
}

template <typename T>
void Solver::internalSolveLtRange(SolveCtx<T>& slvCtx, const T* matData, int64_t startSpanIndex,
                                  int64_t endSpanIndex, T* vecData, int64_t stride,
                                  int nRHS) const {
  BASPACHO_CHECK_GE(startSpanIndex, 0);
  BASPACHO_CHECK_LE(startSpanIndex, endSpanIndex);
  BASPACHO_CHECK_LT(endSpanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[startSpanIndex], 0);
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[endSpanIndex], 0);
  int64_t startLump = factorSkel.spanToLump[startSpanIndex];
  int64_t upToLump = factorSkel.spanToLump[endSpanIndex];

  int64_t denseOpsFromLump;
  if (SparseElimSolve) {
    denseOpsFromLump =
        std::max(startLump, (int64_t)(sparseElimRanges.empty() ? 0 : sparseElimRanges.back()));
  } else {
    denseOpsFromLump = 0;
  }

  int64_t numSpans = factorSkel.lumpToSpan[upToLump] - factorSkel.lumpToSpan[denseOpsFromLump];
  if (numSpans == upToLump - denseOpsFromLump && slvCtx.hasFragmentedOps() && nRHS == 1) {
    BASPACHO_CHECK_EQ(factorSkel.lumpToSpan[denseOpsFromLump], denseOpsFromLump);
    slvCtx.fragmentedSolveLt(matData, denseOpsFromLump, upToLump, vecData);
  } else {
    for (int64_t l = upToLump - 1; l >= denseOpsFromLump; l--) {
      int64_t lumpStart = factorSkel.lumpStart[l];
      int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
      int64_t chainColBegin = factorSkel.chainColPtr[l];

      int64_t boardColBegin = factorSkel.boardColPtr[l];
      int64_t boardColEnd = factorSkel.boardColPtr[l + 1];
      int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + 1];
      int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
      int64_t belowDiagOffset = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
      int64_t numRowsBelowDiag =
          factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
          factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];

      if (numRowsBelowDiag > 0) {
        int64_t chainColPtr = chainColBegin + belowDiagChainColOrd;
        slvCtx.assembleVecT(vecData, stride, chainColPtr, numColChains - belowDiagChainColOrd);

        slvCtx.gemvT(matData, belowDiagOffset, numRowsBelowDiag, lumpSize, vecData, lumpStart,
                     stride, -1.0);
      }

      int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];
      slvCtx.solveLt(matData, diagBlockOffset, lumpSize, vecData, lumpStart, stride);
    }
  }

  if (SparseElimSolve) {
    for (int64_t l = (int64_t)sparseElimRanges.size() - 2; l >= 0; l--) {
      if (sparseElimRanges[l + 1] > upToLump) {
        BASPACHO_CHECK_LE(sparseElimRanges[l], upToLump);
        continue;
      } else if (sparseElimRanges[l] < startLump) {
        BASPACHO_CHECK_GE(startLump, sparseElimRanges[l + 1]);
        return;
      }
      slvCtx.sparseElimSolveLt(*elimCtxs[l], matData, sparseElimRanges[l], sparseElimRanges[l + 1],
                               vecData, stride);
    }
  }
}

template <typename T>
void Solver::addMvFrom(const T* matData, int64_t spanIndex, const T* inVecData, int64_t inStride,
                       T* outVecData, int64_t outStride, int nRHS, BaseType<T> alpha) const {
  SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);

  BASPACHO_CHECK_GE(spanIndex, 0);
  BASPACHO_CHECK_LT(spanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[spanIndex], 0);
  int64_t startFromLump = factorSkel.spanToLump[spanIndex];
  int64_t denseOpsFromLump = startFromLump;  // sparse ops not supported yet

  int64_t upToLump = factorSkel.lumpStart.size() - 1;

  int64_t numSpans = factorSkel.lumpToSpan[upToLump] - factorSkel.lumpToSpan[denseOpsFromLump];
  if (numSpans == upToLump - denseOpsFromLump && slvCtx->hasFragmentedOps() && nRHS == 1) {
    BASPACHO_CHECK_EQ(factorSkel.lumpToSpan[denseOpsFromLump], denseOpsFromLump);
    slvCtx->fragmentedMV(matData, inVecData, denseOpsFromLump, upToLump, outVecData, alpha);
    return;
  }

  for (int64_t l = denseOpsFromLump; l < upToLump; l++) {
    int64_t lumpStart = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
    int64_t chainColBegin = factorSkel.chainColPtr[l];

    int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];
    slvCtx->symm(matData, diagBlockOffset, lumpSize, inVecData, lumpStart, inStride, outVecData,
                 outStride, alpha);

    int64_t boardColBegin = factorSkel.boardColPtr[l];
    int64_t boardColEnd = factorSkel.boardColPtr[l + 1];
    int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + 1];
    int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
    int64_t belowDiagOffset = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
    int64_t numRowsBelowDiag =
        factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
        factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
    if (numRowsBelowDiag == 0) {
      continue;
    }

    int64_t chainColPtr = chainColBegin + belowDiagChainColOrd;
    slvCtx->gemv(matData, belowDiagOffset, numRowsBelowDiag, lumpSize, inVecData, lumpStart,
                 inStride, alpha);
    slvCtx->assembleVec(chainColPtr, numColChains - belowDiagChainColOrd, outVecData, outStride);

    slvCtx->assembleVecT(inVecData, inStride, chainColPtr, numColChains - belowDiagChainColOrd);
    slvCtx->gemvT(matData, belowDiagOffset, numRowsBelowDiag, lumpSize, outVecData, lumpStart,
                  outStride, alpha);
  }
}

template <typename T>
void Solver::pseudoFactorFrom(T* data, int64_t spanIndex, bool /* verbose */) const {
  NumericCtxPtr<T> numCtx = symCtx->createNumericCtx<T>(maxElimTempSize, data);
  numCtx->pseudoFactorSpans(data, spanIndex, factorSkel.numSpans());
}

// ============ LU Factorization Implementation ============

template <typename T>
void Solver::factorLumpLU(NumericCtx<T>& numCtx, T* data, int64_t* pivots, int64_t lump) const {
  int64_t lumpStart = factorSkel.lumpStart[lump];
  int64_t lumpSize = factorSkel.lumpStart[lump + 1] - lumpStart;
  int64_t chainColBegin = factorSkel.chainColPtr[lump];
  int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

  // LU factorization with partial pivoting on diagonal block
  // pivots array stores the row permutation for this lump
  int64_t pivotOffset = factorSkel.lumpStart[lump];  // Pivot index (row-based, not span-based)

  int info = numCtx.getrf(lumpSize, lumpSize, data, diagBlockOffset, pivots + pivotOffset);
  if (info != 0 && staticPivotThreshold_ < 0) {
    throw std::runtime_error("getrf failed with info = " + std::to_string(info));
  }

  // Static pivoting: scan all diagonals when enabled (near-zero values that aren't
  // exactly zero can still cause catastrophic growth in the L factor)
  if (staticPivotThreshold_ >= 0) {
    using ValT = typename std::remove_pointer<decltype(data)>::type;
    ValT threshold = static_cast<ValT>(effectiveStaticPivotThreshold_);
    staticPivotPerturbCount_ +=
        numCtx.perturbSmallDiagonals(lumpSize, data, diagBlockOffset, lumpSize, threshold);
  }

  int64_t boardColBegin = factorSkel.boardColPtr[lump];
  int64_t boardColEnd = factorSkel.boardColPtr[lump + 1];
  int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + 1];
  int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
  int64_t belowDiagOffset = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
  int64_t numRowsBelowDiag = factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
                             factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];

  // Process L column below diagonal (if any rows below)
  if (numRowsBelowDiag > 0) {
    // Apply row permutation to column below diagonal
    numCtx.applyRowPerm(pivots + pivotOffset, lumpSize, data, belowDiagOffset, lumpSize,
                        numRowsBelowDiag);

    // Solve for L column below diagonal: L_below * U_diag = A_below
    // => solve: X * U = B where U is upper triangular part of diagonal block
    numCtx.trsmUpperRight(numRowsBelowDiag, lumpSize, data, diagBlockOffset, data, belowDiagOffset,
                          lumpSize);
  }

  // Process U row to the right of diagonal (if upper triangle storage exists)
  if (factorSkel.isGeneral()) {
    int64_t upperRowStart = factorSkel.upperChainRowPtr[lump];
    int64_t upperRowEnd = factorSkel.upperChainRowPtr[lump + 1];
    int64_t upperDataBase = factorSkel.dataSize();  // Upper data starts after lower data

    for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
      int64_t colSpan = factorSkel.upperChainColSpan[i];
      int64_t colSize = factorSkel.spanStart[colSpan + 1] - factorSkel.spanStart[colSpan];
      int64_t upperBlockOffset = upperDataBase + factorSkel.upperChainData[i];

      // Apply row permutation to upper block (pivots are within lumpSize rows)
      numCtx.applyRowPerm(pivots + pivotOffset, lumpSize, data, upperBlockOffset, colSize, 1);

      // Solve L * U_block = A_block for U_block
      // L is unit lower triangular from diagonal block
      numCtx.trsmLowerUnit(lumpSize, colSize, data, diagBlockOffset, data, upperBlockOffset,
                           colSize);
    }
  }
}

template <typename T>
void Solver::eliminateBoardLU(NumericCtx<T>& numCtx, T* data, int64_t ptr) const {
  int64_t origLump = factorSkel.boardColLump[ptr];
  int64_t boardIndexInCol = factorSkel.boardColOrd[ptr];

  int64_t origLumpSize = factorSkel.lumpStart[origLump + 1] - factorSkel.lumpStart[origLump];
  int64_t chainColBegin = factorSkel.chainColPtr[origLump];

  int64_t boardColBegin = factorSkel.boardColPtr[origLump];
  int64_t boardColEnd = factorSkel.boardColPtr[origLump + 1];

  int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol];
  int64_t rowDataEnd0 = factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol + 1];
  int64_t rowDataEnd1 = factorSkel.boardChainColOrd[boardColEnd - 1];

  int64_t belowDiagStart = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
  int64_t rectRowBegin = factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
  int64_t numRowsSub = factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] - rectRowBegin;
  int64_t numRowsFull = factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] - rectRowBegin;

  int64_t targetLump = factorSkel.boardRowLump[boardColBegin + boardIndexInCol];
  int64_t targetLumpSize = factorSkel.lumpStart[targetLump + 1] - factorSkel.lumpStart[targetLump];
  int64_t srcColDataOffset = chainColBegin + belowDiagChainColOrd;
  int64_t numBlockRows = rowDataEnd1 - belowDiagChainColOrd;
  int64_t numBlockCols = rowDataEnd0 - belowDiagChainColOrd;

  // For LU factorization with upper triangle storage, use L*U directly
  if (factorSkel.isGeneral()) {
    // Get upper triangle data base offset
    int64_t upperDataBase = factorSkel.dataSize();

    // For each pair of (L row span, U col span), compute C -= L * U
    // L blocks are in the lower triangle at chainData offsets
    // U blocks are in the upper triangle at upperChainData offsets
    //
    // IMPORTANT: Only update blocks where the target column lump is >= targetLump.
    // This prevents updating the same block multiple times from different calls.

    // Iterate over L row spans (chains in lower triangle)
    for (int64_t lChainOrd = belowDiagChainColOrd; lChainOrd < rowDataEnd1; lChainOrd++) {
      int64_t lRowSpan = factorSkel.chainRowSpan[chainColBegin + lChainOrd];
      int64_t lRowStart = factorSkel.spanStart[lRowSpan];
      int64_t lRowSize = factorSkel.spanStart[lRowSpan + 1] - lRowStart;
      int64_t lDataOffset = factorSkel.chainData[chainColBegin + lChainOrd];

      // Iterate over U col spans (blocks in upper triangle row for origLump)
      int64_t upperRowStart = factorSkel.upperChainRowPtr[origLump];
      int64_t upperRowEnd = factorSkel.upperChainRowPtr[origLump + 1];

      for (int64_t uIdx = upperRowStart; uIdx < upperRowEnd; uIdx++) {
        int64_t uColSpan = factorSkel.upperChainColSpan[uIdx];
        int64_t uColLump = factorSkel.spanToLump[uColSpan];
        int64_t lRowLump = factorSkel.spanToLump[lRowSpan];

        // Determine when to apply this update:
        // - Lower triangle (lRowSpan >= uColSpan): block is in column uColLump, update when targetLump == uColLump
        // - Upper triangle (lRowSpan < uColSpan): block is in row lRowLump, update when targetLump == lRowLump
        // - Diagonal (lRowSpan == uColSpan): update when targetLump == lRowLump (== uColLump)
        int64_t updateAtLump = (lRowSpan >= uColSpan) ? uColLump : lRowLump;
        if (updateAtLump != targetLump) {
          continue;
        }

        int64_t uColStart = factorSkel.spanStart[uColSpan];
        int64_t uColSize = factorSkel.spanStart[uColSpan + 1] - uColStart;
        int64_t uDataOffset = upperDataBase + factorSkel.upperChainData[uIdx];

        // Update target block: C -= L * U
        // Target can be in lower triangle (lRowSpan >= uColSpan) or upper triangle (lRowSpan < uColSpan)
        if (lRowSpan >= uColSpan) {
          // Target is in lower triangle at (lRowSpan row, uColSpan col)
          // This is in the chain column of lump containing uColSpan
          int64_t targetColLump = factorSkel.spanToLump[uColSpan];
          int64_t targetChainColBegin = factorSkel.chainColPtr[targetColLump];
          int64_t targetChainColEnd = factorSkel.chainColPtr[targetColLump + 1];

          // Find the chain with lRowSpan
          int64_t targetDataOffset = -1;
          int64_t targetLumpSize2 =
              factorSkel.lumpStart[targetColLump + 1] - factorSkel.lumpStart[targetColLump];
          for (int64_t tc = targetChainColBegin; tc < targetChainColEnd; tc++) {
            if (factorSkel.chainRowSpan[tc] == lRowSpan) {
              targetDataOffset = factorSkel.chainData[tc];
              // Add column offset within the lump
              int64_t colOffsetInLump = factorSkel.spanOffsetInLump[uColSpan];
              targetDataOffset += colOffsetInLump;
              break;
            }
          }

          if (targetDataOffset >= 0) {
            // C -= L * U
            // L is lRowSize x origLumpSize at lDataOffset (row-major, ld=origLumpSize)
            // U is origLumpSize x uColSize at uDataOffset (row-major, ld=uColSize)
            // C is lRowSize x uColSize at targetDataOffset (row-major, ld=targetLumpSize2)
            numCtx.saveGemm(lRowSize, uColSize, origLumpSize, data, lDataOffset, origLumpSize, data,
                            uDataOffset, uColSize, data, targetDataOffset, targetLumpSize2);
          }
        } else {
          // Target is in upper triangle at (lRowSpan row, uColSpan col)
          // lRowSpan < uColSpan, so we need to find this block in the upper triangle
          // of the lump containing lRowSpan
          int64_t targetRowLump = factorSkel.spanToLump[lRowSpan];
          int64_t targetUpperRowStart = factorSkel.upperChainRowPtr[targetRowLump];
          int64_t targetUpperRowEnd = factorSkel.upperChainRowPtr[targetRowLump + 1];

          // Find the upper triangle entry pointing to uColSpan
          int64_t targetDataOffset = -1;
          for (int64_t tu = targetUpperRowStart; tu < targetUpperRowEnd; tu++) {
            if (factorSkel.upperChainColSpan[tu] == uColSpan) {
              targetDataOffset = upperDataBase + factorSkel.upperChainData[tu];
              // Add row offset within the lump
              int64_t rowOffsetInLump = factorSkel.spanOffsetInLump[lRowSpan];
              targetDataOffset += rowOffsetInLump * uColSize;
              break;
            }
          }

          if (targetDataOffset >= 0) {
            // C -= L * U
            // L is lRowSize x origLumpSize at lDataOffset (row-major, ld=origLumpSize)
            // U is origLumpSize x uColSize at uDataOffset (row-major, ld=uColSize)
            // C is lRowSize x uColSize at targetDataOffset (row-major, ld=uColSize)
            numCtx.saveGemm(lRowSize, uColSize, origLumpSize, data, lDataOffset, origLumpSize, data,
                            uDataOffset, uColSize, data, targetDataOffset, uColSize);
          }
        }
      }
    }
  } else {
    // Fall back to symmetric (Cholesky-style) elimination for non-general matrices
    numCtx.saveSyrkGemm(numRowsSub, numRowsFull, origLumpSize, data, belowDiagStart);

    numCtx.assemble(data, rectRowBegin,
                    targetLumpSize,    //
                    srcColDataOffset,  //
                    numRowsSub, numBlockRows, numBlockCols);
  }
}

template <typename T>
void Solver::internalFactorRangeLU(T* data, int64_t* pivots, int64_t startSpanIndex,
                                   int64_t endSpanIndex, bool verbose) const {
  beginInternalFactorRangeLU(data, pivots, startSpanIndex, endSpanIndex, verbose);
  finishInternalFactorRangeLU(data, pivots, startSpanIndex, endSpanIndex, verbose);
}

template <typename T>
void Solver::beginInternalFactorRangeLU(T* data, int64_t* pivots, int64_t startSpanIndex,
                                        int64_t endSpanIndex, bool verbose) const {
  BASPACHO_CHECK_GE(startSpanIndex, 0);
  BASPACHO_CHECK_LE(startSpanIndex, endSpanIndex);
  BASPACHO_CHECK_LT(endSpanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[startSpanIndex], 0);
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[endSpanIndex], 0);
  BASPACHO_CHECK_LE(endSpanIndex, canFactorUpTo);
  int64_t startLump = factorSkel.spanToLump[startSpanIndex];
  int64_t upToLump = factorSkel.spanToLump[endSpanIndex];

  BASPACHO_SIGNPOST_BEGIN("createNumericCtx");
  NumericCtxPtr<T> numCtx = symCtx->createNumericCtx<T>(maxElimTempSize, data);
  BASPACHO_SIGNPOST_END("createNumericCtx");

  // Compute effective static pivot threshold scaled by matrix diagonal magnitude.
  BASPACHO_SIGNPOST_BEGIN("maxDiag");
  if (staticPivotThreshold_ >= 0) {
    using ValT = typename std::remove_pointer<decltype(data)>::type;
    ValT epsScale = std::cbrt(std::numeric_limits<ValT>::epsilon());
    if (staticPivotThreshold_ == 0) {
      double maxDiag = numCtx->maxAbsDiag(data, factorSkel.lumpStart.data(),
                                          factorSkel.chainColPtr.data(),
                                          factorSkel.chainData.data(), startLump, upToLump);
      effectiveStaticPivotThreshold_ = static_cast<double>(epsScale) * std::max(maxDiag, static_cast<double>(epsScale));
    } else {
      effectiveStaticPivotThreshold_ = staticPivotThreshold_;
    }
  }
  BASPACHO_SIGNPOST_END("maxDiag");

  // LU sparse elimination: submit to GPU (deferred commit, not waited).
  using ValT = typename std::remove_pointer<decltype(data)>::type;
  ValT effectiveThreshold =
      (staticPivotThreshold_ >= 0) ? static_cast<ValT>(effectiveStaticPivotThreshold_) : ValT(-1);

  BASPACHO_SIGNPOST_BEGIN("sparseElim");
  if (!luElimCtxs.empty()) {
    if (startLump == 0 && upToLump >= sparseElimRanges.back()) {
      if (verbose) {
        for (int64_t l = 0; l + 1 < (int64_t)sparseElimRanges.size(); l++) {
          if (luElimCtxs[l]) {
            std::cout << "LU Elim set: " << l << " (" << sparseElimRanges[l] << ".."
                      << sparseElimRanges[l + 1] << ")" << std::endl;
          }
        }
      }
      numCtx->doAllEliminationsLU(luElimCtxs, sparseElimRanges, data, effectiveThreshold,
                                  staticPivotPerturbCount_);
    } else {
      for (int64_t l = 0; l + 1 < (int64_t)sparseElimRanges.size(); l++) {
        if (sparseElimRanges[l + 1] > upToLump) {
          BASPACHO_CHECK_EQ(sparseElimRanges[l], upToLump);
          break;
        } else if (startLump > sparseElimRanges[l]) {
          BASPACHO_CHECK_GE(startLump, sparseElimRanges[l + 1]);
          continue;
        }
        if (luElimCtxs[l]) {
          if (verbose) {
            std::cout << "LU Elim set: " << l << " (" << sparseElimRanges[l] << ".."
                      << sparseElimRanges[l + 1] << ")" << std::endl;
          }
          int64_t perturbCount = 0;
          numCtx->doEliminationLU(*luElimCtxs[l], data, sparseElimRanges[l],
                                  sparseElimRanges[l + 1], effectiveThreshold, perturbCount);
          staticPivotPerturbCount_ += perturbCount;
        }
      }
    }
  }
  BASPACHO_SIGNPOST_END("sparseElim");

  // Store the numeric context for finishInternalFactorRangeLU to pick up.
  // Transfer ownership: NumericCtxPtr<T> (unique_ptr<NumericCtx<T>>) → unique_ptr<NumericCtxBase>.
  pendingNumCtx_.reset(numCtx.release());
}

template <typename T>
void Solver::finishInternalFactorRangeLU(T* data, int64_t* pivots, int64_t startSpanIndex,
                                         int64_t endSpanIndex, bool verbose) const {
  BASPACHO_CHECK(pendingNumCtx_ != nullptr);
  // Recover the typed NumericCtx from the type-erased base pointer.
  NumericCtx<T>* numCtxRaw = dynamic_cast<NumericCtx<T>*>(pendingNumCtx_.get());
  BASPACHO_CHECK(numCtxRaw != nullptr);

  int64_t startLump = factorSkel.spanToLump[startSpanIndex];
  int64_t upToLump = factorSkel.spanToLump[endSpanIndex];

  int64_t denseOpsFromLump =
      (!luElimCtxs.empty() && !sparseElimRanges.empty()) ? sparseElimRanges.back() : 0;
  if (verbose) {
    std::cout << "LU Block-Fact from: " << denseOpsFromLump << std::endl;
  }

  // Wait for GPU sparse elimination to complete, enter CPU BLAS mode.
  BASPACHO_SIGNPOST_BEGIN("beginDenseOps");
  numCtxRaw->beginDenseOps(data, factorSkel.totalDataSize());
  BASPACHO_SIGNPOST_END("beginDenseOps");

  BASPACHO_SIGNPOST_BEGIN("denseLoop");
  using ClockT = std::chrono::high_resolution_clock;
  double totalBoardMs = 0, totalFactorMs = 0;
  static const bool profileLU = std::getenv("BASPACHO_PROFILE_LU") != nullptr;
  if (profileLU) verbose = true;

  for (int64_t l = std::max(startLump, denseOpsFromLump);
       l < (int64_t)factorSkel.chainColPtr.size() - 1; l++) {
    auto tLumpStart = verbose ? ClockT::now() : ClockT::time_point{};

    numCtxRaw->prepareAssemble(l);

    int64_t rPtrStart = (denseOpsFromLump > 0) ? startElimRowPtr[l - denseOpsFromLump]
                                               : factorSkel.boardRowPtr[l];
    int64_t boardCount = 0;
    for (int64_t rPtr = rPtrStart,
                 rEnd = factorSkel.boardRowPtr[l + 1] - 1;
         rPtr < rEnd; rPtr++) {
      int64_t origLump = factorSkel.boardColLump[rPtr];
      if (origLump >= upToLump) {
        break;
      } else if (origLump < startLump) {
        continue;
      }
      eliminateBoardLU(*numCtxRaw, data, rPtr);
      boardCount++;
    }

    auto tBoardEnd = verbose ? ClockT::now() : ClockT::time_point{};

    if (l < upToLump) {
      factorLumpLU(*numCtxRaw, data, pivots, l);
    }

    if (verbose) {
      numCtxRaw->flush();
      auto tEnd = ClockT::now();
      int64_t lumpSize = factorSkel.lumpStart[l + 1] - factorSkel.lumpStart[l];
      double boardMs = std::chrono::duration<double, std::milli>(tBoardEnd - tLumpStart).count();
      double factorMs = std::chrono::duration<double, std::milli>(tEnd - tBoardEnd).count();
      totalBoardMs += boardMs;
      totalFactorMs += factorMs;
      std::cout << "  Dense lump " << l << ": n=" << lumpSize
                << " boards=" << boardCount
                << " boardMs=" << boardMs
                << " factorMs=" << factorMs << std::endl;
    }
  }

  if (verbose) {
    std::cout << "  Dense loop totals: boardMs=" << totalBoardMs
              << " factorMs=" << totalFactorMs << std::endl;
  }

  BASPACHO_SIGNPOST_END("denseLoop");

  BASPACHO_SIGNPOST_BEGIN("flush");
  numCtxRaw->flush();
  staticPivotPerturbCount_ += numCtxRaw->deferredPerturbCount();
  BASPACHO_SIGNPOST_END("flush");

  // Release the pending context.
  pendingNumCtx_.reset();
}

template <typename T>
void Solver::factorLU(T* data, int64_t* pivots, bool verbose) const {
  staticPivotPerturbCount_ = 0;
  beginInternalFactorRangeLU(data, pivots, 0, factorSkel.numSpans(), verbose);
  finishInternalFactorRangeLU(data, pivots, 0, factorSkel.numSpans(), verbose);
}

template <typename T>
void Solver::beginFactorLU(T* data, int64_t* pivots, bool verbose) const {
  staticPivotPerturbCount_ = 0;
  beginInternalFactorRangeLU(data, pivots, 0, factorSkel.numSpans(), verbose);
}

template <typename T>
void Solver::finishFactorLU(T* data, int64_t* pivots, bool verbose) const {
  finishInternalFactorRangeLU(data, pivots, 0, factorSkel.numSpans(), verbose);
}

template <typename T>
void Solver::solveLU(const T* matData, const int64_t* pivots, T* vecData, int64_t stride,
                     int nRHS) const {
  BASPACHO_SIGNPOST_BEGIN("solveSetup");
  SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);

  // With transpose workaround in getrf, we have P * A = L * U (standard form).
  // The solve for A*x = b is:
  //   1. Apply P: y = P * b
  //   2. Solve L * z = y (forward substitution)
  //   3. Solve U * x = z (backward substitution)

  // Pre-upload all pivots to GPU (avoids per-lump sync on GPU backends)
  slvCtx->uploadPivots(pivots, factorSkel.lumpStart[factorSkel.numLumps()]);
  BASPACHO_SIGNPOST_END("solveSetup");

  // Step 1: Apply row permutation P: y = P * b
  // Skip sparse-elim lumps when LU sparse elimination is active —
  // their pivots are identity (1x1 scalar blocks, no pivoting needed)
  BASPACHO_SIGNPOST_BEGIN("solvePerm");
  int64_t pivotStartLump =
      (!luElimCtxs.empty() && !sparseElimRanges.empty()) ? sparseElimRanges.back() : 0;
  for (int64_t l = pivotStartLump; l < factorSkel.numLumps(); l++) {
    int64_t lumpStart = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
    int64_t pivotOffset = factorSkel.lumpStart[l];  // Row-based pivot index
    slvCtx->applyRowPermVec(pivots + pivotOffset, lumpSize, vecData + lumpStart, stride);
  }
  BASPACHO_SIGNPOST_END("solvePerm");

  // Step 2: Solve L * z = y (forward substitution with unit lower triangular L)
  BASPACHO_SIGNPOST_BEGIN("solveL");
  internalSolveLRangeUnit(*slvCtx, matData, 0, factorSkel.numSpans(), vecData, stride, nRHS);
  BASPACHO_SIGNPOST_END("solveL");

  // Step 3: Solve U * x = z (backward substitution with U factor)
  BASPACHO_SIGNPOST_BEGIN("solveU");
  internalSolveURange(*slvCtx, matData, 0, factorSkel.numSpans(), vecData, stride, nRHS);
  slvCtx->flush();  // Final flush: ensure all GPU solve work is complete
  BASPACHO_SIGNPOST_END("solveU");
}

// Forward substitution for LU with unit lower triangular L
// Similar to internalSolveLRange but uses solveLUnit instead of solveL
template <typename T>
void Solver::internalSolveLRangeUnit(SolveCtx<T>& slvCtx, const T* matData, int64_t startSpanIndex,
                                     int64_t endSpanIndex, T* vecData, int64_t stride,
                                     int nRHS) const {
  (void)nRHS;
  BASPACHO_CHECK_GE(startSpanIndex, 0);
  BASPACHO_CHECK_LE(startSpanIndex, endSpanIndex);
  BASPACHO_CHECK_LT(endSpanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[startSpanIndex], 0);
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[endSpanIndex], 0);
  int64_t startLump = factorSkel.spanToLump[startSpanIndex];
  int64_t upToLump = factorSkel.spanToLump[endSpanIndex];

  int64_t denseOpsFromLump;
  if (SparseElimSolve && !luElimCtxs.empty()) {
    // Use LU sparse elimination solve for the sparse-elim ranges (Metal backend)
    for (int64_t l = 0; l + 1 < (int64_t)sparseElimRanges.size(); l++) {
      if (sparseElimRanges[l + 1] > upToLump) {
        BASPACHO_CHECK_EQ(sparseElimRanges[l], upToLump);
        return;
      } else if (startLump > sparseElimRanges[l]) {
        BASPACHO_CHECK_GE(startLump, sparseElimRanges[l + 1]);
        continue;
      }
      slvCtx.sparseElimSolveLUnit(*elimCtxs[l], matData, sparseElimRanges[l],
                                  sparseElimRanges[l + 1], vecData, stride);
    }
    denseOpsFromLump =
        std::max(startLump, (int64_t)(sparseElimRanges.empty() ? 0 : sparseElimRanges.back()));
  } else {
    denseOpsFromLump = startLump;
  }

  for (int64_t l = denseOpsFromLump; l < upToLump; l++) {
    int64_t lumpStart = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
    int64_t chainColBegin = factorSkel.chainColPtr[l];
    int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

    // Solve L * x_l = z_l with unit diagonal
    slvCtx.solveLUnit(matData, diagBlockOffset, lumpSize, vecData, lumpStart, stride);

    int64_t boardColBegin = factorSkel.boardColPtr[l];
    int64_t boardColEnd = factorSkel.boardColPtr[l + 1];
    int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + 1];
    int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
    int64_t belowDiagOffset = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
    int64_t numRowsBelowDiag = factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
                               factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
    if (numRowsBelowDiag == 0) {
      continue;
    }

    slvCtx.gemv(matData, belowDiagOffset, numRowsBelowDiag, lumpSize, vecData, lumpStart, stride,
                -1.0);

    int64_t chainColPtr = chainColBegin + belowDiagChainColOrd;
    slvCtx.assembleVec(chainColPtr, numColChains - belowDiagChainColOrd, vecData, stride);
  }
}

template <typename T>
void Solver::internalSolveURange(SolveCtx<T>& slvCtx, const T* matData, int64_t startSpanIndex,
                                 int64_t endSpanIndex, T* vecData, int64_t stride, int nRHS) const {
  (void)nRHS;  // Used implicitly in slvCtx
  BASPACHO_CHECK_GE(startSpanIndex, 0);
  BASPACHO_CHECK_LE(startSpanIndex, endSpanIndex);
  BASPACHO_CHECK_LT(endSpanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[startSpanIndex], 0);
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[endSpanIndex], 0);
  int64_t startLump = factorSkel.spanToLump[startSpanIndex];
  int64_t upToLump = factorSkel.spanToLump[endSpanIndex];

  // Upper triangle data base offset (after lower triangle data)
  int64_t upperDataBase = factorSkel.dataSize();

  int64_t denseOpsFromLump;
  if (SparseElimSolve && !luElimCtxs.empty()) {
    denseOpsFromLump =
        std::max(startLump, (int64_t)(sparseElimRanges.empty() ? 0 : sparseElimRanges.back()));
  } else {
    denseOpsFromLump = startLump;
  }

  // Dense backward substitution with U (from last lump down to denseOpsFromLump)
  for (int64_t l = upToLump - 1; l >= denseOpsFromLump; l--) {
    int64_t lumpStart = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
    int64_t chainColBegin = factorSkel.chainColPtr[l];
    int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

    // For off-diagonal U entries, subtract contributions: y(l) -= U_{l,k} * x(k) for k > l
    if (factorSkel.isGeneral()) {
      int64_t upperRowStart = factorSkel.upperChainRowPtr[l];
      int64_t upperRowEnd = factorSkel.upperChainRowPtr[l + 1];

      for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
        int64_t colSpan = factorSkel.upperChainColSpan[i];
        int64_t colStart = factorSkel.spanStart[colSpan];
        int64_t colSize = factorSkel.spanStart[colSpan + 1] - colStart;
        int64_t upperDataOffset = upperDataBase + factorSkel.upperChainData[i];

        // y(l) -= U_{l,k} * x(k)
        // U block has shape (lumpSize x colSize), stored row-major
        slvCtx.gemvDirect(matData, upperDataOffset, lumpSize, colSize, vecData, colStart, lumpStart,
                          stride, -1.0);
      }
    }

    // Solve U * x_l = y_l for diagonal block
    slvCtx.solveU(matData, diagBlockOffset, lumpSize, vecData, lumpStart, stride);
  }

  // Sparse elimination backward U solve (reverse order, like Cholesky Lt)
  if (SparseElimSolve && !luElimCtxs.empty()) {
    for (int64_t l = (int64_t)sparseElimRanges.size() - 2; l >= 0; l--) {
      if (sparseElimRanges[l + 1] > upToLump) {
        BASPACHO_CHECK_LE(sparseElimRanges[l], upToLump);
        continue;
      } else if (sparseElimRanges[l] < startLump) {
        BASPACHO_CHECK_GE(startLump, sparseElimRanges[l + 1]);
        return;
      }
      slvCtx.sparseElimSolveU(*elimCtxs[l], matData, sparseElimRanges[l],
                              sparseElimRanges[l + 1], vecData, stride);
    }
  }
}

template void Solver::factorLU<double>(double* data, int64_t* pivots, bool verbose) const;
template void Solver::factorLU<float>(float* data, int64_t* pivots, bool verbose) const;
template void Solver::beginFactorLU<double>(double* data, int64_t* pivots, bool verbose) const;
template void Solver::beginFactorLU<float>(float* data, int64_t* pivots, bool verbose) const;
template void Solver::finishFactorLU<double>(double* data, int64_t* pivots, bool verbose) const;
template void Solver::finishFactorLU<float>(float* data, int64_t* pivots, bool verbose) const;
template void Solver::solveLU<double>(const double* matData, const int64_t* pivots, double* vecData,
                                      int64_t stride, int nRHS) const;
template void Solver::solveLU<float>(const float* matData, const int64_t* pivots, float* vecData,
                                     int64_t stride, int nRHS) const;

// ============ LDL^T Factorization Implementation ============
// For symmetric indefinite matrices: A = L * D * L^T
// L is unit lower triangular (below diagonal), D is diagonal (on diagonal)
// Uses same lower-triangle storage as Cholesky.

template <typename T>
void Solver::factorLumpLDLT(NumericCtx<T>& numCtx, T* data, int64_t lump) const {
  int64_t lumpStart = factorSkel.lumpStart[lump];
  int64_t lumpSize = factorSkel.lumpStart[lump + 1] - lumpStart;
  int64_t chainColBegin = factorSkel.chainColPtr[lump];
  int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

  // LDL^T factorization on diagonal block
  int info = numCtx.ldlt(lumpSize, data, diagBlockOffset);
  if (info != 0) {
    throw std::runtime_error("ldlt failed with info = " + std::to_string(info));
  }

  int64_t boardColBegin = factorSkel.boardColPtr[lump];
  int64_t boardColEnd = factorSkel.boardColPtr[lump + 1];
  int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + 1];
  int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
  int64_t belowDiagOffset = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
  int64_t numRowsBelowDiag = factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
                             factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];

  if (numRowsBelowDiag == 0) {
    return;
  }

  // For LDL^T, we need to solve for L column below diagonal.
  // After ldlt of diagonal block, we have L_diag (unit lower) and D_diag (diagonal) stored.
  // The relationship for off-diagonal blocks is:
  //   A'_{below} = L_{below} * D_diag * L_diag^T
  // Rearranging:
  //   L_{below} = A'_{below} * L_diag^{-T} * D_diag^{-1}
  //
  // Use trsmUnitScaleInv which computes: B <- B * L^{-T} * D^{-1}
  // where L is unit lower triangular (stored with D on diagonal for the scaling step)
  numCtx.trsmUnitScaleInv(lumpSize, numRowsBelowDiag, data, diagBlockOffset, belowDiagOffset);
}

template <typename T>
void Solver::eliminateBoardLDLT(NumericCtx<T>& numCtx, T* data, int64_t ptr) const {
  int64_t origLump = factorSkel.boardColLump[ptr];
  int64_t boardIndexInCol = factorSkel.boardColOrd[ptr];

  int64_t origLumpSize = factorSkel.lumpStart[origLump + 1] - factorSkel.lumpStart[origLump];
  int64_t chainColBegin = factorSkel.chainColPtr[origLump];
  int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

  int64_t boardColBegin = factorSkel.boardColPtr[origLump];
  int64_t boardColEnd = factorSkel.boardColPtr[origLump + 1];

  int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol];
  int64_t rowDataEnd0 = factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol + 1];
  int64_t rowDataEnd1 = factorSkel.boardChainColOrd[boardColEnd - 1];

  int64_t belowDiagStart = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
  int64_t rectRowBegin = factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
  int64_t numRowsSub = factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] - rectRowBegin;
  int64_t numRowsFull = factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] - rectRowBegin;

  // For LDL^T Schur complement: C -= L * D * L^T
  // where L is the column below diagonal, D is diagonal on the diagonal block.
  //
  // After trsmUnitScaleInv in factorLumpLDLT:
  //   L_below = A'_below * L_diag^{-T} * D_diag^{-1}
  // This is the correct L factor for LDL^T.
  //
  // The Schur complement is: C -= L_below * D_diag * L_below^T
  // We use saveSyrkGemmScaled which computes (L * D) * L^T = L * D * L^T
  numCtx.saveSyrkGemmScaled(numRowsSub, numRowsFull, origLumpSize, data, belowDiagStart,
                            diagBlockOffset, origLumpSize);

  int64_t targetLump = factorSkel.boardRowLump[boardColBegin + boardIndexInCol];
  int64_t targetLumpSize = factorSkel.lumpStart[targetLump + 1] - factorSkel.lumpStart[targetLump];
  int64_t srcColDataOffset = chainColBegin + belowDiagChainColOrd;
  int64_t numBlockRows = rowDataEnd1 - belowDiagChainColOrd;
  int64_t numBlockCols = rowDataEnd0 - belowDiagChainColOrd;

  numCtx.assemble(data, rectRowBegin,
                  targetLumpSize,    //
                  srcColDataOffset,  //
                  numRowsSub, numBlockRows, numBlockCols);
}

template <typename T>
void Solver::internalFactorRangeLDLT(T* data, int64_t startSpanIndex, int64_t endSpanIndex,
                                     bool verbose) const {
  BASPACHO_CHECK_GE(startSpanIndex, 0);
  BASPACHO_CHECK_LE(startSpanIndex, endSpanIndex);
  BASPACHO_CHECK_LT(endSpanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[startSpanIndex], 0);
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[endSpanIndex], 0);
  BASPACHO_CHECK_LE(endSpanIndex, canFactorUpTo);
  int64_t startLump = factorSkel.spanToLump[startSpanIndex];
  int64_t upToLump = factorSkel.spanToLump[endSpanIndex];

  NumericCtxPtr<T> numCtx = symCtx->createNumericCtx<T>(maxElimTempSize, data);

  // Note: LDL^T does not currently support sparse elimination optimization
  // All factorization is done in the dense phase
  int64_t denseOpsFromLump = 0;
  if (verbose) {
    std::cout << "LDL^T Block-Fact from: " << denseOpsFromLump << std::endl;
  }

  for (int64_t l = std::max(startLump, denseOpsFromLump);
       l < (int64_t)factorSkel.chainColPtr.size() - 1; l++) {
    numCtx->prepareAssemble(l);

    //  iterate over columns having a non-trivial a-block
    for (int64_t rPtr = startElimRowPtr[l - denseOpsFromLump],
                 rEnd = factorSkel.boardRowPtr[l + 1] - 1;  // skip last (diag block)
         rPtr < rEnd; rPtr++) {
      int64_t origLump = factorSkel.boardColLump[rPtr];
      if (origLump >= upToLump) {
        break;
      } else if (origLump < startLump) {
        continue;
      }
      eliminateBoardLDLT(*numCtx, data, rPtr);
    }

    if (l < upToLump) {
      factorLumpLDLT(*numCtx, data, l);
    }
  }
}

template <typename T>
void Solver::factorLDLT(T* data, bool verbose) const {
  internalFactorRangeLDLT(data, 0, factorSkel.numSpans(), verbose);
}

template <typename T>
void Solver::solveLDLT(const T* matData, T* vecData, int64_t stride, int nRHS) const {
  SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);

  // LDL^T solve: A*x = b where A = L * D * L^T
  // 1. Solve L * y = b (forward substitution with unit L)
  // 2. Solve D * z = y (diagonal solve)
  // 3. Solve L^T * x = z (backward substitution with unit L^T)

  // Step 1: Forward substitution with unit lower triangular L
  internalSolveLRangeLDLT(*slvCtx, matData, 0, factorSkel.numSpans(), vecData, stride, nRHS);

  // Step 2: Diagonal solve
  internalSolveDRange(*slvCtx, matData, 0, factorSkel.numSpans(), vecData, stride, nRHS);

  // Step 3: Backward substitution with unit L^T
  internalSolveLtRangeLDLT(*slvCtx, matData, 0, factorSkel.numSpans(), vecData, stride, nRHS);
}

template <typename T>
void Solver::internalSolveLRangeLDLT(SolveCtx<T>& slvCtx, const T* matData, int64_t startSpanIndex,
                                     int64_t endSpanIndex, T* vecData, int64_t stride,
                                     int nRHS) const {
  (void)nRHS;
  BASPACHO_CHECK_GE(startSpanIndex, 0);
  BASPACHO_CHECK_LE(startSpanIndex, endSpanIndex);
  BASPACHO_CHECK_LT(endSpanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[startSpanIndex], 0);
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[endSpanIndex], 0);
  int64_t startLump = factorSkel.spanToLump[startSpanIndex];
  int64_t upToLump = factorSkel.spanToLump[endSpanIndex];

  for (int64_t l = startLump; l < upToLump; l++) {
    int64_t lumpStart = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
    int64_t chainColBegin = factorSkel.chainColPtr[l];
    int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

    // Solve L * x = b with unit diagonal L
    slvCtx.solveLUnitLDLT(matData, diagBlockOffset, lumpSize, vecData, lumpStart, stride);

    int64_t boardColBegin = factorSkel.boardColPtr[l];
    int64_t boardColEnd = factorSkel.boardColPtr[l + 1];
    int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + 1];
    int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
    int64_t belowDiagOffset = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
    int64_t numRowsBelowDiag = factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
                               factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
    if (numRowsBelowDiag == 0) {
      continue;
    }

    slvCtx.gemv(matData, belowDiagOffset, numRowsBelowDiag, lumpSize, vecData, lumpStart, stride,
                -1.0);

    int64_t chainColPtr = chainColBegin + belowDiagChainColOrd;
    slvCtx.assembleVec(chainColPtr, numColChains - belowDiagChainColOrd, vecData, stride);
  }
}

template <typename T>
void Solver::internalSolveDRange(SolveCtx<T>& slvCtx, const T* matData, int64_t startSpanIndex,
                                 int64_t endSpanIndex, T* vecData, int64_t stride, int nRHS) const {
  (void)nRHS;
  BASPACHO_CHECK_GE(startSpanIndex, 0);
  BASPACHO_CHECK_LE(startSpanIndex, endSpanIndex);
  BASPACHO_CHECK_LT(endSpanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  int64_t startLump = factorSkel.spanToLump[startSpanIndex];
  int64_t upToLump = factorSkel.spanToLump[endSpanIndex];

  for (int64_t l = startLump; l < upToLump; l++) {
    int64_t lumpStart = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
    int64_t chainColBegin = factorSkel.chainColPtr[l];
    int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

    // Solve D * x = b (divide by diagonal elements)
    slvCtx.solveDiag(matData, diagBlockOffset, lumpSize, vecData, lumpStart, stride);
  }
}

template <typename T>
void Solver::internalSolveLtRangeLDLT(SolveCtx<T>& slvCtx, const T* matData, int64_t startSpanIndex,
                                      int64_t endSpanIndex, T* vecData, int64_t stride,
                                      int nRHS) const {
  (void)nRHS;
  BASPACHO_CHECK_GE(startSpanIndex, 0);
  BASPACHO_CHECK_LE(startSpanIndex, endSpanIndex);
  BASPACHO_CHECK_LT(endSpanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[startSpanIndex], 0);
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[endSpanIndex], 0);
  int64_t startLump = factorSkel.spanToLump[startSpanIndex];
  int64_t upToLump = factorSkel.spanToLump[endSpanIndex];

  // Backward substitution (reverse order)
  for (int64_t l = upToLump - 1; l >= startLump; l--) {
    int64_t lumpStart = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
    int64_t chainColBegin = factorSkel.chainColPtr[l];

    int64_t boardColBegin = factorSkel.boardColPtr[l];
    int64_t boardColEnd = factorSkel.boardColPtr[l + 1];
    int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + 1];
    int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
    int64_t belowDiagOffset = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
    int64_t numRowsBelowDiag = factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
                               factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];

    if (numRowsBelowDiag > 0) {
      int64_t chainColPtr = chainColBegin + belowDiagChainColOrd;
      slvCtx.assembleVecT(vecData, stride, chainColPtr, numColChains - belowDiagChainColOrd);

      slvCtx.gemvT(matData, belowDiagOffset, numRowsBelowDiag, lumpSize, vecData, lumpStart, stride,
                   -1.0);
    }

    int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];
    // Solve L^T * x = b with unit diagonal L^T
    slvCtx.solveLtUnit(matData, diagBlockOffset, lumpSize, vecData, lumpStart, stride);
  }
}

template void Solver::factorLDLT<double>(double* data, bool verbose) const;
template void Solver::factorLDLT<float>(float* data, bool verbose) const;
template void Solver::solveLDLT<double>(const double* matData, double* vecData, int64_t stride,
                                        int nRHS) const;
template void Solver::solveLDLT<float>(const float* matData, float* vecData, int64_t stride,
                                       int nRHS) const;

template void Solver::factor<double>(double* data, bool verbose) const;
template void Solver::factor<float>(float* data, bool verbose) const;
template void Solver::factor<vector<double*>>(vector<double*>* data, bool verbose) const;
template void Solver::factor<vector<float*>>(vector<float*>* data, bool verbose) const;
template void Solver::solve<double>(const double* matData, double* vecData, int64_t stride,
                                    int nRHS) const;
template void Solver::solve<float>(const float* matData, float* vecData, int64_t stride,
                                   int nRHS) const;
template void Solver::solve<vector<double*>>(const vector<double*>* matData,
                                             vector<double*>* vecData, int64_t stride,
                                             int nRHS) const;
template void Solver::solve<vector<float*>>(const vector<float*>* matData, vector<float*>* vecData,
                                            int64_t stride, int nRHS) const;
template void Solver::solveL<double>(const double* matData, double* vecData, int64_t stride,
                                     int nRHS) const;
template void Solver::solveL<float>(const float* matData, float* vecData, int64_t stride,
                                    int nRHS) const;
template void Solver::solveL<vector<double*>>(const vector<double*>* matData,
                                              vector<double*>* vecData, int64_t stride,
                                              int nRHS) const;
template void Solver::solveL<vector<float*>>(const vector<float*>* matData, vector<float*>* vecData,
                                             int64_t stride, int nRHS) const;
template void Solver::solveLt<double>(const double* matData, double* vecData, int64_t stride,
                                      int nRHS) const;
template void Solver::solveLt<float>(const float* matData, float* vecData, int64_t stride,
                                     int nRHS) const;
template void Solver::solveLt<vector<double*>>(const vector<double*>* matData,
                                               vector<double*>* vecData, int64_t stride,
                                               int nRHS) const;
template void Solver::solveLt<vector<float*>>(const vector<float*>* matData,
                                              vector<float*>* vecData, int64_t stride,
                                              int nRHS) const;
template void Solver::factorUpTo<double>(double* data, int64_t spanIndex, bool verbose) const;
template void Solver::factorUpTo<float>(float* data, int64_t spanIndex, bool verbose) const;
template void Solver::factorUpTo<vector<double*>>(vector<double*>* data, int64_t spanIndex,
                                                  bool verbose) const;
template void Solver::factorUpTo<vector<float*>>(vector<float*>* data, int64_t spanIndex,
                                                 bool verbose) const;
template void Solver::factorFrom<double>(double* data, int64_t spanIndex, bool verbose) const;
template void Solver::factorFrom<float>(float* data, int64_t spanIndex, bool verbose) const;
template void Solver::factorFrom<vector<double*>>(vector<double*>* data, int64_t spanIndex,
                                                  bool verbose) const;
template void Solver::factorFrom<vector<float*>>(vector<float*>* data, int64_t spanIndex,
                                                 bool verbose) const;
template void Solver::solveLUpTo<double>(const double* matData, int64_t spanIndex, double* vecData,
                                         int64_t stride, int nRHS) const;
template void Solver::solveLUpTo<float>(const float* matData, int64_t spanIndex, float* vecData,
                                        int64_t stride, int nRHS) const;
template void Solver::solveLUpTo<vector<double*>>(const vector<double*>* matData, int64_t spanIndex,
                                                  vector<double*>* vecData, int64_t stride,
                                                  int nRHS) const;
template void Solver::solveLUpTo<vector<float*>>(const vector<float*>* matData, int64_t spanIndex,
                                                 vector<float*>* vecData, int64_t stride,
                                                 int nRHS) const;
template void Solver::solveLtUpTo<double>(const double* matData, int64_t spanIndex, double* vecData,
                                          int64_t stride, int nRHS) const;
template void Solver::solveLtUpTo<float>(const float* matData, int64_t spanIndex, float* vecData,
                                         int64_t stride, int nRHS) const;
template void Solver::solveLtUpTo<vector<double*>>(const vector<double*>* matData,
                                                   int64_t spanIndex, vector<double*>* vecData,
                                                   int64_t stride, int nRHS) const;
template void Solver::solveLtUpTo<vector<float*>>(const vector<float*>* matData, int64_t spanIndex,
                                                  vector<float*>* vecData, int64_t stride,
                                                  int nRHS) const;
template void Solver::addMvFrom<double>(const double* matData, int64_t spanIndex,
                                        const double* inVecData, int64_t inStride,
                                        double* outVecData, int64_t outStride, int nRHS,
                                        double alpha) const;
template void Solver::addMvFrom<float>(const float* matData, int64_t spanIndex,
                                       const float* inVecData, int64_t inStride, float* outVecData,
                                       int64_t outStride, int nRHS, float alpha) const;
template void Solver::pseudoFactorFrom<double>(double* data, int64_t, bool verbose) const;
template void Solver::pseudoFactorFrom<float>(float* data, int64_t, bool verbose) const;
template void Solver::solveLFrom<double>(const double* matData, int64_t spanIndex, double* vecData,
                                         int64_t stride, int nRHS) const;
template void Solver::solveLFrom<float>(const float* matData, int64_t spanIndex, float* vecData,
                                        int64_t stride, int nRHS) const;
template void Solver::solveLtFrom<double>(const double* matData, int64_t spanIndex, double* vecData,
                                          int64_t stride, int nRHS) const;
template void Solver::solveLtFrom<float>(const float* matData, int64_t spanIndex, float* vecData,
                                         int64_t stride, int nRHS) const;

void Solver::printStats() const {
  cout << "Matrix stats:" << endl;
  cout << "  data size......: " << factorSkel.dataSize() << endl;
  cout << "  solve temp data: " << maxElimTempSize << endl;
  if (sparseElimRanges.size() >= 2) {
    cout << "Sparse elimination sets:" << endl;
  }
  for (int64_t l = 0; l < (int64_t)sparseElimRanges.size() - 1; l++) {
    cout << "  elim set [" << sparseElimRanges[l] << ".." << sparseElimRanges[l + 1]
         << "]: " << elimCtxs[l]->elimStat.toString() << endl;
  }
  cout << "Factor timings and call stats:"
       << "\n  largest node size: " << symCtx->potrfBiggestN
       << "\n  potrf: " << symCtx->potrfStat.toString()
       << "\n  trsm: " << symCtx->trsmStat.toString()  //
       << "\n  syrk/gemm(" << symCtx->syrkCalls << "+" << symCtx->gemmCalls
       << "): " << symCtx->sygeStat.toString() << "\n  asmbl: " << symCtx->asmblStat.toString()
       << endl;
  // skip if no solve operation took place:
  if (symCtx->solveSparseLStat.numRuns + symCtx->solveSparseLtStat.numRuns +
          symCtx->solveLStat.numRuns + symCtx->solveLtStat.numRuns >
      0) {
    cout << "Solve timings and call stats:"
         << "\n  solveSparseLStat: " << symCtx->solveSparseLStat.toString()
         << "\n  solveSparseLtStat: " << symCtx->solveSparseLtStat.toString()
         << "\n  solveLStat: " << symCtx->solveLStat.toString()
         << "\n  solveLtStat: " << symCtx->solveLtStat.toString()
         << "\n  solveGemvStat: " << symCtx->solveGemvStat.toString()
         << "\n  solveGemvTStat: " << symCtx->solveGemvTStat.toString()
         << "\n  solveAssVStat: " << symCtx->solveAssVStat.toString()
         << "\n  solveAssVTStat: " << symCtx->solveAssVTStat.toString() << endl;
  }
}

void Solver::enableStats(bool enable) {
  for (int64_t l = 0; l < (int64_t)sparseElimRanges.size() - 1; l++) {
    elimCtxs[l]->elimStat.enabled = enable;
  }
  symCtx->potrfStat.enabled = enable;
  symCtx->trsmStat.enabled = enable;
  symCtx->sygeStat.enabled = enable;
  symCtx->asmblStat.enabled = enable;
}

void Solver::resetStats() {
  for (int64_t l = 0; l < (int64_t)sparseElimRanges.size() - 1; l++) {
    elimCtxs[l]->elimStat.reset();
  }
  symCtx->potrfBiggestN = 0;
  symCtx->potrfStat.reset();
  symCtx->trsmStat.reset();
  symCtx->syrkCalls = 0;
  symCtx->gemmCalls = 0;
  symCtx->sygeStat.reset();
  symCtx->asmblStat.reset();
}

BackendType detectBestBackend() {
  // Priority: CUDA > Metal > OpenCL > Fast (CPU)
#ifdef BASPACHO_USE_CUBLAS
  // TODO: Could add runtime CUDA device detection here
  return BackendCuda;
#elif defined(BASPACHO_USE_METAL)
  // Metal is available on macOS with Apple Silicon
  return BackendMetal;
#elif defined(BASPACHO_USE_OPENCL)
  // OpenCL is a portable fallback
  return BackendOpenCL;
#else
  return BackendFast;
#endif
}

OpsPtr getBackend(const Settings& settings) {
  BackendType backend = settings.backend;

  // Handle auto-detection
  if (backend == BackendAuto) {
    backend = detectBestBackend();
  }

  if (backend == BackendFast) {
    return fastOps(settings.numThreads);
  } else if (backend == BackendCuda) {
#ifdef BASPACHO_USE_CUBLAS
    return cudaOps();
#else
    std::cerr << "Baspacho: CUDA not enabled at compile time" << std::endl;
    abort();
#endif
  } else if (backend == BackendMetal) {
#ifdef BASPACHO_USE_METAL
    return metalOps();
#else
    std::cerr << "Baspacho: Metal not enabled at compile time" << std::endl;
    abort();
#endif
  } else if (backend == BackendOpenCL) {
#ifdef BASPACHO_USE_OPENCL
    return openclOps();
#else
    std::cerr << "Baspacho: OpenCL not enabled at compile time" << std::endl;
    abort();
#endif
  }
  BASPACHO_CHECK(backend == BackendRef);
  return simpleOps();
}

SolverPtr createSolver(const Settings& settings, const std::vector<int64_t>& paramSize,
                       const SparseStructure& ss_, const std::vector<int64_t>& sparseElimRanges,
                       const unordered_set<int64_t>& elimLastIds) {
  // no point in providing "elim last" ids if not allowing solve up to such set
  BASPACHO_CHECK(settings.addFillPolicy == AddFillComplete || elimLastIds.empty());

  // validate supernode merging settings
  BASPACHO_CHECK_GE(settings.supernodeMergeFillTolerance, 0.0);
  BASPACHO_CHECK_LE(settings.supernodeMergeFillTolerance, 1.0);
  BASPACHO_CHECK_GE(settings.maxSupernodeSize, (int64_t)0);

  // validate static pivoting threshold
  BASPACHO_CHECK_GE(settings.staticPivotThreshold, -1.0);

  BASPACHO_CHECK((int64_t)sparseElimRanges.size() != 1);
  int64_t givenSparseElimEnd = sparseElimRanges.empty() ? 0 : sparseElimRanges.back();
  if (!sparseElimRanges.empty()) {
    BASPACHO_CHECK(isStrictlyIncreasing(sparseElimRanges, 0, sparseElimRanges.size()));
    for (int64_t id : elimLastIds) {
      BASPACHO_CHECK_GE(id, givenSparseElimEnd);
    }
  }

  SparseStructure ss = ss_;
  if (settings.addFillPolicy != AddFillNone) {
    for (int64_t e = 0; e < (int64_t)sparseElimRanges.size() - 1; e++) {
      ss = ss.addIndependentEliminationFill(sparseElimRanges[e], sparseElimRanges[e + 1]);
    }
  }

  // create a factor where either no fill is added, either limited for given elims
  if (settings.addFillPolicy == AddFillNone || settings.addFillPolicy == AddFillForGivenElims) {
    vector<int64_t> spanStart;
    spanStart.reserve(paramSize.size() + 1);
    spanStart.insert(spanStart.end(), paramSize.begin(), paramSize.end());
    spanStart.push_back(0);
    cumSumVec(spanStart);

    std::vector<int64_t> lumpToSpan(paramSize.size() + 1);
    ::std::iota(lumpToSpan.begin(), lumpToSpan.end(), 0);

    std::vector<int64_t> permutation(paramSize.size());
    ::std::iota(permutation.begin(), permutation.end(), 0);

    SparseStructure ssT = ss.transpose();  // to csc
    CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, ssT.ptrs, ssT.inds);

    if (settings.matrixType == MTYPE_GENERAL) {
      factorSkel.initUpperTriangle();
    }

    std::vector<int64_t> sparseElimRangesCopy = sparseElimRanges;
    return SolverPtr(new Solver(std::move(factorSkel), std::move(sparseElimRangesCopy),
                                std::move(permutation), getBackend(settings),
                                settings.addFillPolicy == AddFillNone ? 0 : givenSparseElimEnd,
                                {}, settings.staticPivotThreshold));
  }

  SparseStructure ssBottom = ss.extractRightBottom(givenSparseElimEnd);

  // find best permutation for right-bottom corner that is left
  vector<int64_t> permutation = ssBottom.fillReducingPermutation();
  vector<int64_t> noCrossPoints;
  if (!elimLastIds.empty()) {  // force those params to go last
    vector<int64_t> parts[2];
    for (int64_t p : permutation) {
      parts[elimLastIds.count(p + givenSparseElimEnd)].push_back(p);
    }
    noCrossPoints.push_back(parts[0].size());
    permutation = parts[0];
    permutation.insert(permutation.end(), parts[1].begin(), parts[1].end());
  }
  vector<int64_t> invPerm = inversePermutation(permutation);
  SparseStructure sortedSsBottom = ssBottom.symmetricPermutation(invPerm, false);

  // apply permutation to param size of right-bottom corner
  std::vector<int64_t> sortedBottomParamSize(paramSize.size() - givenSparseElimEnd);
  for (size_t i = givenSparseElimEnd; i < paramSize.size(); i++) {
    sortedBottomParamSize[invPerm[i - givenSparseElimEnd]] = paramSize[i];
  }

  // auto select computation model (for node merge heuristic), if not provided
  const ComputationModel* compModel = settings.computationModel ? settings.computationModel
                                      : settings.backend == BackendCuda
                                          ? &ComputationModel::model_Cuda117_2080Ti
                                          : &ComputationModel::model_OpenBlas_i7_1185g7;

  // compute as ordinary elimination tree on br-corner
  EliminationTree et(sortedBottomParamSize, sortedSsBottom, compModel);
  et.buildTree();
  et.processTree(settings.findSparseEliminationRanges, noCrossPoints,
                 settings.addFillPolicy == AddFillForAutoElims,
                 settings.supernodeMergeFillTolerance, settings.maxSupernodeSize);

  // Compute level-set schedule for parallel factorization
  auto lumpParent = computeLumpParent(et);
  auto levelSetSchedule = LevelSetSchedule::build(lumpParent);

  // The ET lumps are numbered 0..N-1, but in the final solver the first
  // givenSparseElimEnd lumps are identity (one lump per sparse-elim span).
  // Shift ET lump indices and prepend the sparse-elim lumps as leaf level.
  if (givenSparseElimEnd > 0) {
    for (auto& level : levelSetSchedule.levels) {
      for (auto& l : level) {
        l += givenSparseElimEnd;
      }
    }
    std::vector<int64_t> elimLumps(givenSparseElimEnd);
    std::iota(elimLumps.begin(), elimLumps.end(), 0);
    levelSetSchedule.levels.insert(levelSetSchedule.levels.begin(), std::move(elimLumps));
  }

  et.computeAggregateStruct(settings.addFillPolicy == AddFillForAutoElims);

  // ss last rows are to be permuted according to etTotalInvPerm
  vector<int64_t> etTotalInvPerm = composePermutations(et.permInverse, invPerm);
  vector<int64_t> fullInvPerm(givenSparseElimEnd + etTotalInvPerm.size());
  ::std::iota(fullInvPerm.begin(), fullInvPerm.begin() + givenSparseElimEnd, 0);
  for (size_t i = 0; i < etTotalInvPerm.size(); i++) {
    fullInvPerm[i + givenSparseElimEnd] = givenSparseElimEnd + etTotalInvPerm[i];
  }

  // compute span start as cumSum of sorted paramSize
  vector<int64_t> fullSpanStart(paramSize.size() + 1);
  leftPermute(fullSpanStart.begin(), fullInvPerm, paramSize);
  fullSpanStart[paramSize.size()] = 0;
  cumSumVec(fullSpanStart);

  // compute lump to span, knowing up to givenSparseElimEnd it's the identity
  vector<int64_t> fullLumpToSpan;
  fullLumpToSpan.reserve(givenSparseElimEnd + et.lumpToSpan.size());
  fullLumpToSpan.resize(givenSparseElimEnd);
  ::std::iota(fullLumpToSpan.begin(), fullLumpToSpan.begin() + givenSparseElimEnd, 0);
  shiftConcat(fullLumpToSpan, givenSparseElimEnd, et.lumpToSpan.begin(), et.lumpToSpan.end());
  BASPACHO_CHECK_EQ((int64_t)fullSpanStart.size() - 1, fullLumpToSpan.back());

  // matrix with blocks not joined, we will need the first columns
  SparseStructure sortedSsT = ss.symmetricPermutation(fullInvPerm, false).transpose();

  // fullColStart joining sortedSsT.ptrs + shifted elimEndDataPtr
  vector<int64_t> fullColStart;
  fullColStart.reserve(givenSparseElimEnd + et.colStart.size());
  fullColStart.insert(fullColStart.begin(), sortedSsT.ptrs.begin(),
                      sortedSsT.ptrs.begin() + givenSparseElimEnd);
  int64_t elimEndDataPtr = sortedSsT.ptrs[givenSparseElimEnd];
  shiftConcat(fullColStart, elimEndDataPtr, et.colStart.begin(), et.colStart.end());
  BASPACHO_CHECK_EQ(fullColStart.size(), fullLumpToSpan.size());

  // fullRowParam joining sortedSsT.inds and et.rowParam (moved)
  vector<int64_t> fullRowParam;
  fullRowParam.reserve(elimEndDataPtr + et.rowParam.size());
  fullRowParam.insert(fullRowParam.begin(), sortedSsT.inds.begin(),
                      sortedSsT.inds.begin() + elimEndDataPtr);
  shiftConcat(fullRowParam, givenSparseElimEnd, et.rowParam.begin(), et.rowParam.end());
  BASPACHO_CHECK_EQ((int64_t)fullRowParam.size(), fullColStart.back());

  CoalescedBlockMatrixSkel factorSkel(fullSpanStart, fullLumpToSpan, fullColStart, fullRowParam);

  if (settings.matrixType == MTYPE_GENERAL) {
    factorSkel.initUpperTriangle();
  }

  // include (additional) progressive Schur elimination sets, shifted
  std::vector<int64_t> fullSparseElimRanges = sparseElimRanges;
  if (!et.sparseElimRanges.empty()) {
    shiftConcat(fullSparseElimRanges, givenSparseElimEnd,
                et.sparseElimRanges.begin() + (sparseElimRanges.empty() ? 0 : 1),
                et.sparseElimRanges.end());
  }
  if (fullSparseElimRanges.size() == 1) {
    fullSparseElimRanges.pop_back();
  }
  int64_t fullSparseElimEnd = fullSparseElimRanges.empty() ? 0 : fullSparseElimRanges.back();

  return SolverPtr(new Solver(
      std::move(factorSkel), std::move(fullSparseElimRanges), std::move(fullInvPerm),
      getBackend(settings),
      settings.addFillPolicy == AddFillForAutoElims ? fullSparseElimEnd : paramSize.size(),
      std::move(levelSetSchedule), settings.staticPivotThreshold));
}

template <typename T>
void Solver::loadFromCsr(const int64_t* csrRowStart, const int64_t* csrColInds,
                         const int64_t* blockSizes, const T* csrValues, T* data) const {
  // Get the accessor for mapping block positions
  // The accessor takes original (unpermuted) indices and handles permutation internally
  auto acc = accessor();
  bool isGeneral = factorSkel.matrixType == MTYPE_GENERAL;
  int64_t upperDataBase = isGeneral ? factorSkel.dataSize() : 0;

  int64_t numBlocks = permutation.size();
  int64_t valOffset = 0;  // Current offset in csrValues

  // Iterate through CSR structure (original ordering)
  for (int64_t origRow = 0; origRow < numBlocks; origRow++) {
    int64_t rowSize = blockSizes[origRow];

    for (int64_t ptr = csrRowStart[origRow]; ptr < csrRowStart[origRow + 1]; ptr++) {
      int64_t origCol = csrColInds[ptr];
      int64_t colSize = blockSizes[origCol];
      int64_t blockElements = rowSize * colSize;

      // Get internal block position - accessor handles permutation and returns flip flag
      auto [offset, stride, flipped] = acc.blockOffset(origRow, origCol);

      if (isGeneral && flipped) {
        // Upper triangle entry for general matrices (permRow < permCol after permutation).
        int64_t permRow = permutation[origRow];
        int64_t permCol = permutation[origCol];
        int64_t rowLump = acc.plainAcc.spanToLump[permRow];
        int64_t colLump = acc.plainAcc.spanToLump[permCol];

        if (rowLump == colLump) {
          // Intra-lump: both spans in same lump, store in diagonal block directly.
          // The diagonal block is a full NxN matrix (lumpSize x lumpSize).
          int64_t lumpSize = acc.plainAcc.lumpStart[rowLump + 1] - acc.plainAcc.lumpStart[rowLump];
          int64_t diagStart = acc.plainAcc.chainData[acc.plainAcc.chainColPtr[rowLump]];
          int64_t rowOff = acc.plainAcc.spanOffsetInLump[permRow];
          int64_t colOff = acc.plainAcc.spanOffsetInLump[permCol];
          int64_t baseOffset = diagStart + rowOff * lumpSize + colOff;
          for (int64_t r = 0; r < rowSize; r++) {
            for (int64_t c = 0; c < colSize; c++) {
              data[baseOffset + r * lumpSize + c] = csrValues[valOffset + r * colSize + c];
            }
          }
        } else {
          // Inter-lump: use separate upper triangle storage via upperBlockOffset.
          auto [upperOff, upperStride] = acc.plainAcc.upperBlockOffset(permRow, permCol);
          int64_t absOffset = upperDataBase + upperOff;
          for (int64_t r = 0; r < rowSize; r++) {
            for (int64_t c = 0; c < colSize; c++) {
              data[absOffset + r * upperStride + c] = csrValues[valOffset + r * colSize + c];
            }
          }
        }
      } else if (flipped) {
        // Symmetric: transpose into lower triangle (existing behavior)
        for (int64_t r = 0; r < rowSize; r++) {
          for (int64_t c = 0; c < colSize; c++) {
            data[offset + c * stride + r] = csrValues[valOffset + r * colSize + c];
          }
        }
      } else {
        // Lower/diagonal (existing behavior)
        for (int64_t r = 0; r < rowSize; r++) {
          for (int64_t c = 0; c < colSize; c++) {
            data[offset + r * stride + c] = csrValues[valOffset + r * colSize + c];
          }
        }
      }

      valOffset += blockElements;
    }
  }
}

template <typename T>
void Solver::extractToCsr(const int64_t* csrRowStart, const int64_t* csrColInds,
                          const int64_t* blockSizes, const T* data, T* csrValues) const {
  // Get the accessor for mapping block positions
  // The accessor takes original (unpermuted) indices and handles permutation internally
  auto acc = accessor();

  int64_t numBlocks = permutation.size();
  int64_t valOffset = 0;  // Current offset in csrValues

  // Iterate through CSR structure (original ordering)
  for (int64_t origRow = 0; origRow < numBlocks; origRow++) {
    int64_t rowSize = blockSizes[origRow];

    for (int64_t ptr = csrRowStart[origRow]; ptr < csrRowStart[origRow + 1]; ptr++) {
      int64_t origCol = csrColInds[ptr];
      int64_t colSize = blockSizes[origCol];
      int64_t blockElements = rowSize * colSize;

      // Get internal block position - accessor handles permutation and returns flip flag
      auto [offset, stride, flipped] = acc.blockOffset(origRow, origCol);

      // Copy values from internal format to CSR
      // When flipped, the block is stored transposed internally
      for (int64_t r = 0; r < rowSize; r++) {
        for (int64_t c = 0; c < colSize; c++) {
          if (flipped) {
            // Block is transposed in internal storage
            csrValues[valOffset + r * colSize + c] = data[offset + c * stride + r];
          } else {
            csrValues[valOffset + r * colSize + c] = data[offset + r * stride + c];
          }
        }
      }

      valOffset += blockElements;
    }
  }
}

// Explicit template instantiations
template void Solver::loadFromCsr<float>(const int64_t*, const int64_t*, const int64_t*,
                                         const float*, float*) const;
template void Solver::loadFromCsr<double>(const int64_t*, const int64_t*, const int64_t*,
                                          const double*, double*) const;
template void Solver::extractToCsr<float>(const int64_t*, const int64_t*, const int64_t*,
                                          const float*, float*) const;
template void Solver::extractToCsr<double>(const int64_t*, const int64_t*, const int64_t*,
                                           const double*, double*) const;

}  // end namespace BaSpaCho
