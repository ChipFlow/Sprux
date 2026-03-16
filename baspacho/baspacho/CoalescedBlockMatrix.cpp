/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include <iostream>
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/Utils.h"

namespace BaSpaCho {

using namespace std;

CoalescedBlockMatrixSkel::CoalescedBlockMatrixSkel(const vector<int64_t>& spanStart,
                                                   const vector<int64_t>& lumpToSpan,
                                                   const vector<int64_t>& colPtr,
                                                   const vector<int64_t>& rowInd)
    : spanStart(spanStart), lumpToSpan(lumpToSpan) {
  SPRUX_CHECK_GE(spanStart.size(), lumpToSpan.size());
  SPRUX_CHECK_GE((int64_t)lumpToSpan.size(), 1);
  SPRUX_CHECK_EQ((int64_t)spanStart.size() - 1, lumpToSpan[lumpToSpan.size() - 1]);
  SPRUX_CHECK_EQ(colPtr.size(), lumpToSpan.size());
  SPRUX_CHECK(isStrictlyIncreasing(spanStart, 0, spanStart.size()));
  SPRUX_CHECK(isStrictlyIncreasing(lumpToSpan, 0, lumpToSpan.size()));

  int64_t totSize = spanStart[spanStart.size() - 1];
  int64_t numSpans = spanStart.size() - 1;
  int64_t numLumps = lumpToSpan.size() - 1;

  spanToLump.resize(numSpans + 1);
  lumpStart.resize(numLumps + 1);
  for (int64_t l = 0; l < numLumps; l++) {
    int64_t sBegin = lumpToSpan[l];
    int64_t sEnd = lumpToSpan[l + 1];
    lumpStart[l] = spanStart[sBegin];
    for (int64_t s = sBegin; s < sEnd; s++) {
      spanToLump[s] = l;
    }
  }
  spanToLump[numSpans] = numLumps;
  lumpStart[numLumps] = totSize;
  spanOffsetInLump.resize(numSpans + 1);
  for (int64_t s = 0; s < numSpans; s++) {
    spanOffsetInLump[s] = spanStart[s] - lumpStart[spanToLump[s]];
  }
  spanOffsetInLump[numSpans] = 0;

  chainColPtr.resize(numLumps + 1);
  chainRowSpan.clear();
  chainData.clear();

  boardColPtr.resize(numLumps + 1);
  boardRowLump.clear();
  boardChainColOrd.clear();
  int64_t dataPtr = 0;
  for (int64_t l = 0; l < numLumps; l++) {
    int64_t colStart = colPtr[l];
    int64_t colEnd = colPtr[l + 1];
    SPRUX_CHECK(isStrictlyIncreasing(rowInd, colStart, colEnd));
    int64_t lSpanBegin = lumpToSpan[l];
    int64_t lSpanEnd = lumpToSpan[l + 1];
    int64_t lSpanSize = lSpanEnd - lSpanBegin;
    int64_t lDataSize = lumpStart[l + 1] - lumpStart[l];

    // check the initial section is the set of params from `a`, and
    // therefore the full diagonal block is contained in the matrix
    // Column must contain full diagonal block:
    SPRUX_CHECK_GE(colEnd - colStart, lSpanSize);
    // Column data must start at diagonal block:
    SPRUX_CHECK_EQ(rowInd[colStart], lSpanBegin);
    // Column must contain full diagonal block:
    SPRUX_CHECK_EQ(rowInd[colStart + lSpanSize - 1], lSpanEnd - 1);

    chainColPtr[l] = chainRowSpan.size();
    boardColPtr[l] = boardRowLump.size();
    int64_t currentRowAggreg = kInvalid;
    int64_t numRowsSkipped = 0;
    for (int64_t i = colStart; i < colEnd; i++) {
      int64_t p = rowInd[i];
      chainRowSpan.push_back(p);
      chainData.push_back(dataPtr);
      dataPtr += lDataSize * (spanStart[p + 1] - spanStart[p]);
      numRowsSkipped += spanStart[p + 1] - spanStart[p];
      chainRowsTillEnd.push_back(numRowsSkipped);

      int64_t rowAggreg = spanToLump[p];
      if (rowAggreg != currentRowAggreg) {
        currentRowAggreg = rowAggreg;
        boardRowLump.push_back(rowAggreg);
        boardChainColOrd.push_back(i - colStart);
      }
    }
    boardRowLump.push_back(kInvalid);
    boardChainColOrd.push_back(colEnd - colStart);
  }
  chainColPtr[numLumps] = chainRowSpan.size();
  boardColPtr[numLumps] = boardRowLump.size();
  chainData.push_back(dataPtr);

  boardRowPtr.assign(numLumps + 1, 0);
  for (int64_t l = 0; l < numLumps; l++) {
    for (int64_t i = boardColPtr[l]; i < boardColPtr[l + 1] - 1; i++) {
      int64_t rowLump = boardRowLump[i];
      boardRowPtr[rowLump]++;
    }
  }
  int64_t numBoards = cumSumVec(boardRowPtr);
  boardColLump.resize(numBoards);
  boardColOrd.resize(numBoards);
  for (int64_t l = 0; l < numLumps; l++) {
    for (int64_t i = boardColPtr[l]; i < boardColPtr[l + 1] - 1; i++) {
      int64_t rowLump = boardRowLump[i];
      boardColLump[boardRowPtr[rowLump]] = l;
      boardColOrd[boardRowPtr[rowLump]] = i - boardColPtr[l];
      boardRowPtr[rowLump]++;
    }
  }
  rewindVec(boardRowPtr);
}

template <typename T>
void CoalescedBlockMatrixSkel::densify(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& dense,
                                       const T* data, bool fillUpperHalf,
                                       int64_t startSpanIndex) const {
  SPRUX_CHECK_GE(startSpanIndex, 0);
  SPRUX_CHECK_LT(startSpanIndex, (int64_t)spanOffsetInLump.size());
  SPRUX_CHECK_EQ(spanOffsetInLump[startSpanIndex], 0);
  int64_t startLump = spanToLump[startSpanIndex];

  int64_t offset = spanStart[startSpanIndex];
  int64_t totSize = spanStart[spanStart.size() - 1] - offset;
  dense.resize(totSize, totSize);
  dense.setZero();

  for (size_t a = startLump; a < chainColPtr.size() - 1; a++) {
    int64_t lBegin = lumpStart[a];
    int64_t lSize = lumpStart[a + 1] - lBegin;
    int64_t colStart = chainColPtr[a];
    int64_t colEnd = chainColPtr[a + 1];
    for (int64_t i = colStart; i < colEnd; i++) {
      int64_t p = chainRowSpan[i];
      int64_t pStart = spanStart[p];
      int64_t pSize = spanStart[p + 1] - pStart;
      int64_t dataPtr = chainData[i];

      dense.block(pStart - offset, lBegin - offset, pSize, lSize) =
          Eigen::Map<const MatRMaj<T>>(data + dataPtr, pSize, lSize);
    }
  }

  if (fillUpperHalf) {
    dense.template triangularView<Eigen::StrictlyUpper>() =
        dense.template triangularView<Eigen::StrictlyLower>().adjoint();
  }
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> CoalescedBlockMatrixSkel::densify(
    const std::vector<T>& data, bool fillUpperHalf) const {
  int64_t totData = chainData[chainData.size() - 1];
  SPRUX_CHECK_EQ(totData, (int64_t)data.size());

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dense;
  densify(dense, data.data(), fillUpperHalf);

  return dense;
}

template <typename T>
void CoalescedBlockMatrixSkel::damp(std::vector<T>& data, T alpha, T beta) const {
  int64_t totData = chainData[chainData.size() - 1];
  SPRUX_CHECK_EQ(totData, (int64_t)data.size());

  for (size_t a = 0; a < chainColPtr.size() - 1; a++) {
    int64_t aStart = lumpStart[a];
    int64_t aSize = lumpStart[a + 1] - aStart;
    int64_t colStart = chainColPtr[a];
    int64_t dataPtr = chainData[colStart];

    Eigen::Map<MatRMaj<T>> block(data.data() + dataPtr, aSize, aSize);
    block.diagonal() *= (1 + alpha);
    block.diagonal().array() += beta;
  }
}

template void CoalescedBlockMatrixSkel::densify<double>(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>&, const double*, bool, int64_t) const;
template void CoalescedBlockMatrixSkel::densify<float>(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&, const float*, bool, int64_t) const;
template Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
CoalescedBlockMatrixSkel::densify<double>(const std::vector<double>&, bool) const;
template Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>
CoalescedBlockMatrixSkel::densify<float>(const std::vector<float>&, bool) const;
template void CoalescedBlockMatrixSkel::damp<double>(std::vector<double>& data, double alpha,
                                                     double beta) const;
template void CoalescedBlockMatrixSkel::damp<float>(std::vector<float>& data, float alpha,
                                                    float beta) const;

void CoalescedBlockMatrixSkel::initUpperTriangle() {
  // Initialize upper triangle storage for LU factorization.
  // For symmetric sparsity patterns, the upper triangle is derived from
  // the lower triangle by transposition: if block (i,j) exists in lower (i>j),
  // then block (j,i) should exist in upper.
  //
  // Upper triangle uses CSR-like (row-ordered) structure where:
  // - upperChainRowPtr[row]: start index for row's column entries
  // - upperChainColSpan[i]: column span index for entry i
  // - upperChainData[i]: data offset for entry i

  matrixType = MTYPE_GENERAL;

  int64_t numLumps = this->numLumps();

  // Count upper triangle entries per row (lump)
  // For each lower triangle entry at (row_span, col_lump), we create
  // an upper triangle entry at (col_lump, row_span) - but only for off-diagonal
  vector<int64_t> rowCounts(numLumps + 1, 0);
  for (int64_t l = 0; l < numLumps; l++) {
    int64_t colStart = chainColPtr[l];
    int64_t colEnd = chainColPtr[l + 1];
    int64_t lSpanBegin = lumpToSpan[l];
    int64_t lSpanEnd = lumpToSpan[l + 1];

    for (int64_t i = colStart; i < colEnd; i++) {
      int64_t rowSpan = chainRowSpan[i];
      // Skip diagonal blocks (rowSpan belongs to lump l)
      if (rowSpan >= lSpanBegin && rowSpan < lSpanEnd) {
        continue;
      }
      // This is a below-diagonal entry: row_span > col_lump's spans
      // In upper triangle, this becomes: col_lump row, row_span column
      rowCounts[l]++;
    }
  }

  // Compute row pointers (cumulative sum)
  upperChainRowPtr.resize(numLumps + 1);
  upperChainRowPtr[0] = 0;
  for (int64_t l = 0; l < numLumps; l++) {
    upperChainRowPtr[l + 1] = upperChainRowPtr[l] + rowCounts[l];
  }
  int64_t totalUpperEntries = upperChainRowPtr[numLumps];

  // Allocate column indices and data offsets
  upperChainColSpan.resize(totalUpperEntries);
  upperChainData.resize(totalUpperEntries + 1);

  // Reset counts for filling
  fill(rowCounts.begin(), rowCounts.end(), 0);

  // Second pass: fill column indices and compute data offsets
  int64_t dataPtr = 0;
  for (int64_t l = 0; l < numLumps; l++) {
    int64_t colStart = chainColPtr[l];
    int64_t colEnd = chainColPtr[l + 1];
    int64_t lSpanBegin = lumpToSpan[l];
    int64_t lSpanEnd = lumpToSpan[l + 1];
    int64_t lDataSize = lumpStart[l + 1] - lumpStart[l];

    for (int64_t i = colStart; i < colEnd; i++) {
      int64_t rowSpan = chainRowSpan[i];
      // Skip diagonal blocks
      if (rowSpan >= lSpanBegin && rowSpan < lSpanEnd) {
        continue;
      }

      // Lower triangle entry at (rowSpan, l) becomes upper triangle at (l, rowSpan)
      int64_t upperRow = l;
      int64_t idx = upperChainRowPtr[upperRow] + rowCounts[upperRow];
      upperChainColSpan[idx] = rowSpan;

      // Data size: lump_rows x span_cols (transposed from lower)
      int64_t spanSize = spanStart[rowSpan + 1] - spanStart[rowSpan];
      upperChainData[idx] = dataPtr;
      dataPtr += lDataSize * spanSize;

      rowCounts[upperRow]++;
    }
  }
  upperChainData[totalUpperEntries] = dataPtr;

  // Sort each row's column indices (they may not be ordered)
  for (int64_t l = 0; l < numLumps; l++) {
    int64_t rowStart = upperChainRowPtr[l];
    int64_t rowEnd = upperChainRowPtr[l + 1];
    // Simple bubble sort (rows are typically small)
    for (int64_t i = rowStart; i < rowEnd - 1; i++) {
      for (int64_t j = i + 1; j < rowEnd; j++) {
        if (upperChainColSpan[j] < upperChainColSpan[i]) {
          swap(upperChainColSpan[i], upperChainColSpan[j]);
          swap(upperChainData[i], upperChainData[j]);
        }
      }
    }
  }
}

}  // end namespace BaSpaCho
