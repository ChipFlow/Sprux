/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright (c) Robert Taylor, 2026. All rights reserved.
// Licensed under the MIT license found in the LICENSE file.

#pragma once

#include <cstdint>
#include <vector>
#include "sprux/sprux/SparseStructure.h"

namespace Sprux {

// Result of LU preprocessing: row permutation (and optionally scaling)
// to improve diagonal quality before fill-reducing ordering.
struct LUPreprocessing {
  std::vector<int64_t> rowPerm;  // rowPerm[i] = j: put original row j at position i
  std::vector<double> rowScale;  // empty if no scaling (Phase 1)
  std::vector<double> colScale;  // empty if no scaling (Phase 1)
  int64_t structuralRank = 0;
  bool hasScaling() const { return !rowScale.empty(); }
};

// Compute maximum transversal row permutation via BTF.
// Finds a row permutation that places nonzero entries on the diagonal,
// improving pivot quality for LU factorization.
//
// Input: n x n matrix in CSR format (rowPtr size n+1, colInd size nnz).
// Output: LUPreprocessing with rowPerm set. structuralRank = n if matrix
//         is structurally nonsingular.
LUPreprocessing computeMaxTransversal(int64_t n, const int64_t* csrRowPtr,
                                      const int64_t* csrColInd);

// Apply row permutation to a CSR matrix.
// Output matrix A'[i,:] = A[rowPerm[i],:] for each row i.
template <typename T>
void applyRowPermToCsr(int64_t n, const int64_t* rowPtr, const int64_t* colInd, const T* values,
                       const int64_t* rowPerm, std::vector<int64_t>& outRowPtr,
                       std::vector<int64_t>& outColInd, std::vector<T>& outValues);

// Compute row/column equilibration scaling for a CSR matrix.
// Row scaling: Dr[i] = 1 / max_j |A[i,j]|
// Column scaling: Dc[j] = 1 / max_i |Dr[i] * A[i,j]|
// After scaling, the matrix Dr*A*Dc has max absolute row/column values of ~1.
void computeEquilibration(int64_t n, const int64_t* rowPtr, const int64_t* colInd,
                          const double* values, std::vector<double>& rowScale,
                          std::vector<double>& colScale);

// Apply row permutation AND scaling to CSR matrix values.
// Output: A'[i,j] = rowScale[i] * A[rowPerm[i], j] * colScale[j]
// Pass nullptr for rowScale/colScale to skip scaling.
template <typename T>
void applyRowPermAndScaleToCsr(int64_t n, const int64_t* rowPtr, const int64_t* colInd,
                               const T* values, const int64_t* rowPerm, const T* rowScale,
                               const T* colScale, std::vector<int64_t>& outRowPtr,
                               std::vector<int64_t>& outColInd, std::vector<T>& outValues);

// Build symmetric SparseStructure from general CSR matrix.
// Creates lower-triangle CSR suitable for createSolver by symmetrizing
// the pattern: for each (i,j), adds both (max(i,j), min(i,j)) entries.
// Ensures diagonal entries are present (required by CHOLMOD symbolic analysis).
SparseStructure csrToSymmetricSparseStructure(int64_t n, const int64_t* rowPtr,
                                              const int64_t* colInd);

}  // namespace Sprux
