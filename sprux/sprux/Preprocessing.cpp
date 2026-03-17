/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright (c) Robert Taylor, 2026. All rights reserved.
// Licensed under the MIT license found in the LICENSE file.

#include "sprux/sprux/Preprocessing.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

#include "sprux/sprux/DebugMacros.h"

#include "btf.h"

namespace BaSpaCho {

LUPreprocessing computeMaxTransversal(int64_t n, const int64_t* csrRowPtr,
                                      const int64_t* csrColInd) {
  SPRUX_CHECK_GT(n, 0);
  SPRUX_CHECK(csrRowPtr != nullptr);
  SPRUX_CHECK(csrColInd != nullptr);

  LUPreprocessing result;
  result.rowPerm.resize(n);

  // BTF workspace: Match (size n) + Work (size 5*n)
  std::vector<int64_t> work(5 * n);
  double workDone = 0.0;

  // Pass CSR directly to btf_l_maxtrans, which treats it as CSC of A^T.
  // This gives us a row permutation: Match[i] = j means "put original row j
  // at position i" such that A[Match,:] has a zero-free diagonal.
  int64_t matched = btf_l_maxtrans(
      n,                              // nrow (= ncol for square)
      n,                              // ncol
      const_cast<int64_t*>(csrRowPtr),  // Ap (treated as col ptrs of A^T)
      const_cast<int64_t*>(csrColInd),  // Ai (treated as row indices of A^T)
      0.0,                            // maxwork: no limit
      &workDone,                      // work performed
      result.rowPerm.data(),          // Match output
      work.data()                     // workspace
  );

  result.structuralRank = matched;

  // Handle unmatched rows (structural rank < n):
  // Assign remaining original rows to remaining positions.
  if (matched < n) {
    std::vector<bool> usedOrigRow(n, false);
    std::vector<bool> filledPos(n, false);

    for (int64_t i = 0; i < n; i++) {
      if (result.rowPerm[i] >= 0) {
        usedOrigRow[result.rowPerm[i]] = true;
        filledPos[i] = true;
      }
    }

    // Collect unmatched original rows and unfilled positions
    std::vector<int64_t> freeRows, freePositions;
    for (int64_t i = 0; i < n; i++) {
      if (!usedOrigRow[i]) freeRows.push_back(i);
      if (!filledPos[i]) freePositions.push_back(i);
    }

    SPRUX_CHECK_EQ((int64_t)freeRows.size(), (int64_t)freePositions.size());

    for (size_t k = 0; k < freeRows.size(); k++) {
      result.rowPerm[freePositions[k]] = freeRows[k];
    }
  }

  return result;
}

template <typename T>
void applyRowPermToCsr(int64_t n, const int64_t* rowPtr, const int64_t* colInd, const T* values,
                       const int64_t* rowPerm, std::vector<int64_t>& outRowPtr,
                       std::vector<int64_t>& outColInd, std::vector<T>& outValues) {
  SPRUX_CHECK_GT(n, 0);

  // Count total nnz
  int64_t totalNnz = 0;
  for (int64_t i = 0; i < n; i++) {
    int64_t origRow = rowPerm[i];
    totalNnz += rowPtr[origRow + 1] - rowPtr[origRow];
  }

  outRowPtr.resize(n + 1);
  outColInd.resize(totalNnz);
  outValues.resize(totalNnz);

  int64_t pos = 0;
  for (int64_t i = 0; i < n; i++) {
    outRowPtr[i] = pos;
    int64_t origRow = rowPerm[i];
    int64_t rowStart = rowPtr[origRow];
    int64_t rowEnd = rowPtr[origRow + 1];
    for (int64_t k = rowStart; k < rowEnd; k++) {
      outColInd[pos] = colInd[k];
      outValues[pos] = values[k];
      pos++;
    }
  }
  outRowPtr[n] = pos;
}

// Explicit template instantiations
template void applyRowPermToCsr<double>(int64_t, const int64_t*, const int64_t*, const double*,
                                        const int64_t*, std::vector<int64_t>&,
                                        std::vector<int64_t>&, std::vector<double>&);
template void applyRowPermToCsr<float>(int64_t, const int64_t*, const int64_t*, const float*,
                                       const int64_t*, std::vector<int64_t>&,
                                       std::vector<int64_t>&, std::vector<float>&);

void computeEquilibration(int64_t n, const int64_t* rowPtr, const int64_t* colInd,
                          const double* values, std::vector<double>& rowScale,
                          std::vector<double>& colScale) {
  SPRUX_CHECK_GT(n, 0);

  // Row scaling: Dr[i] = 1 / max_j |A[i,j]|
  rowScale.resize(n);
  for (int64_t i = 0; i < n; i++) {
    double maxVal = 0;
    for (int64_t k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
      maxVal = std::max(maxVal, std::abs(values[k]));
    }
    rowScale[i] = (maxVal > 0) ? 1.0 / maxVal : 1.0;
  }

  // Column scaling: Dc[j] = 1 / max_i |Dr[i] * A[i,j]|
  colScale.assign(n, 0.0);
  for (int64_t i = 0; i < n; i++) {
    for (int64_t k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
      int64_t j = colInd[k];
      double scaled = std::abs(rowScale[i] * values[k]);
      colScale[j] = std::max(colScale[j], scaled);
    }
  }
  for (int64_t j = 0; j < n; j++) {
    colScale[j] = (colScale[j] > 0) ? 1.0 / colScale[j] : 1.0;
  }
}

template <typename T>
void applyRowPermAndScaleToCsr(int64_t n, const int64_t* rowPtr, const int64_t* colInd,
                               const T* values, const int64_t* rowPerm, const T* rowScale,
                               const T* colScale, std::vector<int64_t>& outRowPtr,
                               std::vector<int64_t>& outColInd, std::vector<T>& outValues) {
  SPRUX_CHECK_GT(n, 0);

  // Count total nnz
  int64_t totalNnz = 0;
  for (int64_t i = 0; i < n; i++) {
    int64_t origRow = rowPerm[i];
    totalNnz += rowPtr[origRow + 1] - rowPtr[origRow];
  }

  outRowPtr.resize(n + 1);
  outColInd.resize(totalNnz);
  outValues.resize(totalNnz);

  int64_t pos = 0;
  for (int64_t i = 0; i < n; i++) {
    outRowPtr[i] = pos;
    int64_t origRow = rowPerm[i];
    int64_t rowStart = rowPtr[origRow];
    int64_t rowEnd = rowPtr[origRow + 1];
    T rs = rowScale ? static_cast<T>(rowScale[i]) : T(1);
    for (int64_t k = rowStart; k < rowEnd; k++) {
      outColInd[pos] = colInd[k];
      T cs = colScale ? static_cast<T>(colScale[colInd[k]]) : T(1);
      outValues[pos] = rs * values[k] * cs;
      pos++;
    }
  }
  outRowPtr[n] = pos;
}

// Explicit template instantiations
template void applyRowPermAndScaleToCsr<double>(int64_t, const int64_t*, const int64_t*,
                                                const double*, const int64_t*, const double*,
                                                const double*, std::vector<int64_t>&,
                                                std::vector<int64_t>&, std::vector<double>&);
template void applyRowPermAndScaleToCsr<float>(int64_t, const int64_t*, const int64_t*,
                                               const float*, const int64_t*, const float*,
                                               const float*, std::vector<int64_t>&,
                                               std::vector<int64_t>&, std::vector<float>&);

SparseStructure csrToSymmetricSparseStructure(int64_t n, const int64_t* rowPtr,
                                              const int64_t* colInd) {
  SPRUX_CHECK_GT(n, 0);

  // Build lower-triangle column sets (CSC lower triangle = CSR upper triangle).
  // For each (i,j) entry, add max(i,j) to column min(i,j).
  std::vector<std::set<int64_t>> colBlocks(n);
  for (int64_t i = 0; i < n; i++) {
    colBlocks[i].insert(i);  // ensure diagonal is present
    for (int64_t k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
      int64_t j = colInd[k];
      colBlocks[std::min(i, j)].insert(std::max(i, j));
    }
  }

  // Build CSC structure and transpose to get CSR lower triangle
  std::vector<int64_t> ptrs, inds;
  for (const auto& col : colBlocks) {
    ptrs.push_back(inds.size());
    inds.insert(inds.end(), col.begin(), col.end());
  }
  ptrs.push_back(inds.size());

  return SparseStructure(std::move(ptrs), std::move(inds)).transpose();
}

}  // namespace BaSpaCho
