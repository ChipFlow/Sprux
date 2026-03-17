/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>
#include "sprux/sprux/Preprocessing.h"
#include "sprux/sprux/Solver.h"
#include "sprux/sprux/SparseStructure.h"
#include "sprux/testing/TestingUtils.h"

using namespace Sprux;
using namespace ::Sprux::testing_utils;
using namespace std;

// Helper: build a CSR matrix from COO triplets
struct Triplet {
  int64_t row, col;
  double val;
};

static void buildCsr(int64_t n, const vector<Triplet>& triplets, vector<int64_t>& rowPtr,
                     vector<int64_t>& colInd, vector<double>& values) {
  rowPtr.assign(n + 1, 0);
  for (const auto& t : triplets) {
    rowPtr[t.row + 1]++;
  }
  for (int64_t i = 0; i < n; i++) {
    rowPtr[i + 1] += rowPtr[i];
  }

  int64_t nnz = triplets.size();
  colInd.resize(nnz);
  values.resize(nnz);

  // Sort triplets by (row, col) first
  vector<Triplet> sorted = triplets;
  sort(sorted.begin(), sorted.end(), [](const Triplet& a, const Triplet& b) {
    return a.row < b.row || (a.row == b.row && a.col < b.col);
  });

  for (int64_t i = 0; i < nnz; i++) {
    colInd[i] = sorted[i].col;
    values[i] = sorted[i].val;
  }
}

// Matrix with zero-free diagonal should get identity (or equivalent) permutation
TEST(Preprocessing, IdentityPerm) {
  // 4x4 matrix with nonzero diagonal
  //  [1 0 2 0]
  //  [0 3 0 1]
  //  [1 0 4 0]
  //  [0 1 0 5]
  vector<Triplet> trips = {
      {0, 0, 1}, {0, 2, 2}, {1, 1, 3}, {1, 3, 1}, {2, 0, 1}, {2, 2, 4}, {3, 1, 1}, {3, 3, 5},
  };
  vector<int64_t> rowPtr, colInd;
  vector<double> values;
  buildCsr(4, trips, rowPtr, colInd, values);

  auto preproc = computeMaxTransversal(4, rowPtr.data(), colInd.data());

  EXPECT_EQ(preproc.structuralRank, 4);

  // Verify permuted matrix has nonzero diagonal
  for (int64_t i = 0; i < 4; i++) {
    int64_t origRow = preproc.rowPerm[i];
    bool hasDiag = false;
    for (int64_t k = rowPtr[origRow]; k < rowPtr[origRow + 1]; k++) {
      if (colInd[k] == i) {
        hasDiag = true;
        break;
      }
    }
    EXPECT_TRUE(hasDiag) << "Position (" << i << "," << i << ") is zero after permutation";
  }
}

// Matrix that requires a row swap for zero-free diagonal
TEST(Preprocessing, SimplePermutation) {
  // 4x4 matrix where rows need reordering:
  //  [0 1 0 0]   row 0 has no diagonal entry
  //  [1 0 0 0]   row 1 has no diagonal entry
  //  [0 0 0 1]   row 2 has no diagonal entry
  //  [0 0 1 0]   row 3 has no diagonal entry
  // Need: row 1 -> pos 0, row 0 -> pos 1, row 3 -> pos 2, row 2 -> pos 3
  vector<Triplet> trips = {
      {0, 1, 1.0},
      {1, 0, 1.0},
      {2, 3, 1.0},
      {3, 2, 1.0},
  };
  vector<int64_t> rowPtr, colInd;
  vector<double> values;
  buildCsr(4, trips, rowPtr, colInd, values);

  auto preproc = computeMaxTransversal(4, rowPtr.data(), colInd.data());

  EXPECT_EQ(preproc.structuralRank, 4);

  // Verify permuted matrix has nonzero diagonal
  for (int64_t i = 0; i < 4; i++) {
    int64_t origRow = preproc.rowPerm[i];
    bool hasDiag = false;
    for (int64_t k = rowPtr[origRow]; k < rowPtr[origRow + 1]; k++) {
      if (colInd[k] == i) {
        hasDiag = true;
        break;
      }
    }
    EXPECT_TRUE(hasDiag) << "Position (" << i << "," << i << ") is zero after permutation"
                         << " (rowPerm[" << i << "]=" << origRow << ")";
  }

  // Verify it's a valid permutation (bijection)
  vector<int64_t> sorted = preproc.rowPerm;
  sort(sorted.begin(), sorted.end());
  for (int64_t i = 0; i < 4; i++) {
    EXPECT_EQ(sorted[i], i);
  }
}

// Structurally singular matrix: rank < n
TEST(Preprocessing, StructuralRank) {
  // 3x3 matrix with structural rank 2:
  //  [1 0 0]
  //  [1 0 0]   <- same column pattern as row 0, cannot both match col 0
  //  [0 0 1]
  // Column 1 has no entries, so structural rank = 2.
  vector<Triplet> trips = {
      {0, 0, 1},
      {1, 0, 1},
      {2, 2, 1},
  };
  vector<int64_t> rowPtr, colInd;
  vector<double> values;
  buildCsr(3, trips, rowPtr, colInd, values);

  auto preproc = computeMaxTransversal(3, rowPtr.data(), colInd.data());

  // Structural rank should be less than n
  EXPECT_LT(preproc.structuralRank, 3);

  // rowPerm should still be a valid permutation (all positions filled)
  vector<int64_t> sorted = preproc.rowPerm;
  sort(sorted.begin(), sorted.end());
  for (int64_t i = 0; i < 3; i++) {
    EXPECT_EQ(sorted[i], i);
  }
}

// Apply row perm, factor LU, solve, check residual
TEST(Preprocessing, RoundTripSolve) {
  // 5x5 matrix where diagonal is zero without permutation:
  //  [0 2 0 0 0]    needs row swap: row that has col 0 entry -> pos 0
  //  [3 0 0 0 0]
  //  [0 0 0 4 0]
  //  [0 0 5 0 0]
  //  [0 0 0 0 6]
  int64_t n = 5;
  vector<Triplet> trips = {
      {0, 1, 2.0}, {1, 0, 3.0}, {2, 3, 4.0}, {3, 2, 5.0}, {4, 4, 6.0},
  };
  // Add some off-diagonal fill to make it more interesting
  trips.push_back({0, 3, 0.5});
  trips.push_back({1, 2, 0.7});
  trips.push_back({3, 4, 0.3});

  vector<int64_t> rowPtr, colInd;
  vector<double> values;
  buildCsr(n, trips, rowPtr, colInd, values);

  // Step 1: compute row permutation
  auto preproc = computeMaxTransversal(n, rowPtr.data(), colInd.data());
  ASSERT_EQ(preproc.structuralRank, n);

  // Step 2: apply row perm
  vector<int64_t> pRowPtr, pColInd;
  vector<double> pValues;
  applyRowPermToCsr<double>(n, rowPtr.data(), colInd.data(), values.data(), preproc.rowPerm.data(),
                            pRowPtr, pColInd, pValues);

  // Verify permuted matrix has nonzero diagonal
  for (int64_t i = 0; i < n; i++) {
    bool hasDiag = false;
    for (int64_t k = pRowPtr[i]; k < pRowPtr[i + 1]; k++) {
      if (pColInd[k] == i) {
        hasDiag = true;
        break;
      }
    }
    ASSERT_TRUE(hasDiag) << "Position (" << i << "," << i << ") zero after perm";
  }

  // Step 3: build solver from permuted pattern
  SparseStructure ss = csrToSymmetricSparseStructure(n, pRowPtr.data(), pColInd.data());
  vector<int64_t> paramSizes(n, 1);
  vector<int64_t> blockSizes(n, 1);

  Settings settings;
  settings.backend = BackendFast;
  settings.matrixType = MTYPE_GENERAL;

  auto solver = createSolver(settings, paramSizes, ss);
  vector<double> data(solver->skel().totalDataSize());
  vector<int64_t> pivots(n);

  // Step 4: load permuted matrix and factor
  fill(data.begin(), data.end(), 0.0);
  solver->loadFromCsr(pRowPtr.data(), pColInd.data(), blockSizes.data(), pValues.data(),
                      data.data());
  solver->factorLU(data.data(), pivots.data());

  // Step 5: solve Q*A*x = Q*b
  Eigen::VectorXd b = Eigen::VectorXd::Ones(n);
  b << 1.0, 2.0, 3.0, 4.0, 5.0;

  const auto& sp = solver->paramToSpan();
  Eigen::VectorXd bp(n);
  // Compose row perm with solver perm: bp[sp[j]] = b[rowPerm[j]]
  for (int64_t j = 0; j < n; j++) {
    bp(sp[j]) = b(preproc.rowPerm[j]);
  }

  solver->solveLU(data.data(), pivots.data(), bp.data(), n, 1);

  // Inverse permute solution: x[j] = bp[sp[j]]
  Eigen::VectorXd x(n);
  for (int64_t j = 0; j < n; j++) {
    x(j) = bp(sp[j]);
  }

  // Verify residual against original (unpermuted) system A*x = b
  // Build dense A for residual check
  Eigen::MatrixXd Adense = Eigen::MatrixXd::Zero(n, n);
  for (int64_t i = 0; i < n; i++) {
    for (int64_t k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
      Adense(i, colInd[k]) = values[k];
    }
  }
  double residual = (Adense * x - b).norm() / b.norm();
  EXPECT_LT(residual, 1e-10) << "Residual too large: " << residual;
}

// Verify csrToSymmetricSparseStructure produces correct pattern
TEST(Preprocessing, CsrSymmetrize) {
  // 3x3 matrix:
  //  [1 2 0]
  //  [0 3 4]
  //  [5 0 6]
  // Symmetric pattern (lower triangle CSR):
  //  row 0: {0}        (diagonal only)
  //  row 1: {0, 1}     (from A(0,1) symmetrized)
  //  row 2: {0, 1, 2}  (from A(2,0) and A(1,2) symmetrized)
  int64_t n = 3;
  vector<Triplet> trips = {
      {0, 0, 1}, {0, 1, 2}, {1, 1, 3}, {1, 2, 4}, {2, 0, 5}, {2, 2, 6},
  };
  vector<int64_t> rowPtr, colInd;
  vector<double> values;
  buildCsr(n, trips, rowPtr, colInd, values);

  SparseStructure ss = csrToSymmetricSparseStructure(n, rowPtr.data(), colInd.data());

  ASSERT_EQ(ss.order(), n);

  // Check it's CSR lower triangle format (row i contains columns <= i)
  for (int64_t i = 0; i < n; i++) {
    set<int64_t> rowEntries;
    for (int64_t k = ss.ptrs[i]; k < ss.ptrs[i + 1]; k++) {
      rowEntries.insert(ss.inds[k]);
    }
    // Diagonal must be present
    EXPECT_TRUE(rowEntries.count(i)) << "Diagonal missing at row " << i;
    // All entries must be <= i (lower triangle)
    for (int64_t col : rowEntries) {
      EXPECT_LE(col, i) << "Upper triangle entry (" << i << "," << col << ") in lower triangle";
    }
  }

  // Row 0: just {0}
  {
    set<int64_t> row0;
    for (int64_t k = ss.ptrs[0]; k < ss.ptrs[1]; k++) row0.insert(ss.inds[k]);
    EXPECT_EQ(row0, (set<int64_t>{0}));
  }

  // Row 1: {0, 1} — from A(0,1) symmetrized to (1,0) lower entry
  {
    set<int64_t> row1;
    for (int64_t k = ss.ptrs[1]; k < ss.ptrs[2]; k++) row1.insert(ss.inds[k]);
    EXPECT_EQ(row1, (set<int64_t>{0, 1}));
  }

  // Row 2: {0, 1, 2} — A(2,0) stays, A(1,2) symmetrized to (2,1)
  {
    set<int64_t> row2;
    for (int64_t k = ss.ptrs[2]; k < ss.ptrs[3]; k++) row2.insert(ss.inds[k]);
    EXPECT_EQ(row2, (set<int64_t>{0, 1, 2}));
  }
}

// Round trip with equilibration scaling: perm + scale, factor, solve, unscale, check residual
TEST(Preprocessing, RoundTripWithScaling) {
  // 5x5 matrix with wildly different scales (mimicking circuit Jacobians):
  //   row 0: [0, 2e6, 0, 5e-3, 0]
  //   row 1: [3e6, 0, 7e-1, 0, 0]
  //   row 2: [0, 0, 0, 4e3, 0]
  //   row 3: [0, 0, 5e3, 0, 3e-4]
  //   row 4: [0, 0, 0, 0, 6e-6]
  int64_t n = 5;
  vector<Triplet> trips = {
      {0, 1, 2e6},  {0, 3, 5e-3}, {1, 0, 3e6},  {1, 2, 7e-1},
      {2, 3, 4e3},  {3, 2, 5e3},  {3, 4, 3e-4}, {4, 4, 6e-6},
  };
  vector<int64_t> rowPtr, colInd;
  vector<double> values;
  buildCsr(n, trips, rowPtr, colInd, values);

  // Step 1: max transversal
  auto preproc = computeMaxTransversal(n, rowPtr.data(), colInd.data());
  ASSERT_EQ(preproc.structuralRank, n);

  // Step 2: apply row perm, then compute equilibration on Q*A
  vector<int64_t> pRowPtr, pColInd;
  vector<double> pValues;
  applyRowPermToCsr<double>(n, rowPtr.data(), colInd.data(), values.data(), preproc.rowPerm.data(),
                            pRowPtr, pColInd, pValues);

  vector<double> rowScale, colScale;
  computeEquilibration(n, pRowPtr.data(), pColInd.data(), pValues.data(), rowScale, colScale);
  preproc.rowScale = rowScale;
  preproc.colScale = colScale;

  // Step 3: build solver from permuted pattern
  SparseStructure ss = csrToSymmetricSparseStructure(n, pRowPtr.data(), pColInd.data());
  vector<int64_t> paramSizes(n, 1);
  vector<int64_t> blockSizes(n, 1);

  Settings settings;
  settings.backend = BackendFast;
  settings.matrixType = MTYPE_GENERAL;

  auto solver = createSolver(settings, paramSizes, ss);
  vector<double> data(solver->skel().totalDataSize());
  vector<int64_t> pivots(n);

  // Step 4: apply row perm + scaling, load, factor
  applyRowPermAndScaleToCsr<double>(n, rowPtr.data(), colInd.data(), values.data(),
                                   preproc.rowPerm.data(), preproc.rowScale.data(),
                                   preproc.colScale.data(), pRowPtr, pColInd, pValues);
  fill(data.begin(), data.end(), 0.0);
  solver->loadFromCsr(pRowPtr.data(), pColInd.data(), blockSizes.data(), pValues.data(),
                      data.data());
  solver->factorLU(data.data(), pivots.data());

  // Step 5: solve Dr*Q*A*Dc*y = Dr*Q*b
  Eigen::VectorXd b(n);
  b << 1.0, 2.0, 3.0, 4.0, 5.0;

  const auto& sp = solver->paramToSpan();
  Eigen::VectorXd bp(n);
  for (int64_t j = 0; j < n; j++) {
    bp(sp[j]) = preproc.rowScale[j] * b(preproc.rowPerm[j]);
  }
  solver->solveLU(data.data(), pivots.data(), bp.data(), n, 1);

  // Unscale: x = Dc * y
  Eigen::VectorXd x(n);
  for (int64_t j = 0; j < n; j++) {
    x(j) = preproc.colScale[j] * bp(sp[j]);
  }

  // Verify residual against ORIGINAL system A*x = b
  Eigen::MatrixXd Adense = Eigen::MatrixXd::Zero(n, n);
  for (int64_t i = 0; i < n; i++) {
    for (int64_t k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
      Adense(i, colInd[k]) = values[k];
    }
  }
  double residual = (Adense * x - b).norm() / b.norm();
  EXPECT_LT(residual, 1e-10) << "Residual too large: " << residual;

  // Also verify with Eigen's direct solve
  Eigen::VectorXd xRef = Adense.partialPivLu().solve(b);
  double refResidual = (Adense * xRef - b).norm() / b.norm();
  cout << "Sprux residual: " << residual << ", Eigen residual: " << refResidual << endl;
}

// ApplyRowPerm preserves matrix data correctly
TEST(Preprocessing, ApplyRowPermCorrectness) {
  // 3x3 matrix:
  //  row 0: [1 2 0]
  //  row 1: [0 3 4]
  //  row 2: [5 0 6]
  // With perm = {2, 0, 1}: put row 2 at pos 0, row 0 at pos 1, row 1 at pos 2
  // Result:
  //  row 0: [5 0 6]
  //  row 1: [1 2 0]
  //  row 2: [0 3 4]
  int64_t n = 3;
  vector<Triplet> trips = {
      {0, 0, 1}, {0, 1, 2}, {1, 1, 3}, {1, 2, 4}, {2, 0, 5}, {2, 2, 6},
  };
  vector<int64_t> rowPtr, colInd;
  vector<double> values;
  buildCsr(n, trips, rowPtr, colInd, values);

  vector<int64_t> perm = {2, 0, 1};
  vector<int64_t> outRowPtr, outColInd;
  vector<double> outValues;
  applyRowPermToCsr<double>(n, rowPtr.data(), colInd.data(), values.data(), perm.data(), outRowPtr,
                            outColInd, outValues);

  // Row 0 should be original row 2: [5 0 6] -> cols {0, 2}, vals {5, 6}
  ASSERT_EQ(outRowPtr[1] - outRowPtr[0], 2);
  EXPECT_EQ(outColInd[outRowPtr[0]], 0);
  EXPECT_EQ(outColInd[outRowPtr[0] + 1], 2);
  EXPECT_DOUBLE_EQ(outValues[outRowPtr[0]], 5.0);
  EXPECT_DOUBLE_EQ(outValues[outRowPtr[0] + 1], 6.0);

  // Row 1 should be original row 0: [1 2 0] -> cols {0, 1}, vals {1, 2}
  ASSERT_EQ(outRowPtr[2] - outRowPtr[1], 2);
  EXPECT_EQ(outColInd[outRowPtr[1]], 0);
  EXPECT_EQ(outColInd[outRowPtr[1] + 1], 1);
  EXPECT_DOUBLE_EQ(outValues[outRowPtr[1]], 1.0);
  EXPECT_DOUBLE_EQ(outValues[outRowPtr[1] + 1], 2.0);

  // Row 2 should be original row 1: [0 3 4] -> cols {1, 2}, vals {3, 4}
  ASSERT_EQ(outRowPtr[3] - outRowPtr[2], 2);
  EXPECT_EQ(outColInd[outRowPtr[2]], 1);
  EXPECT_EQ(outColInd[outRowPtr[2] + 1], 2);
  EXPECT_DOUBLE_EQ(outValues[outRowPtr[2]], 3.0);
  EXPECT_DOUBLE_EQ(outValues[outRowPtr[2] + 1], 4.0);
}
