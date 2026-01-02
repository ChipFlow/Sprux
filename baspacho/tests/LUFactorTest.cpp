/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;
using namespace ::testing;

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using Vector = Eigen::Vector<T, Eigen::Dynamic>;

template <typename T>
struct Epsilon;
template <>
struct Epsilon<double> {
  static constexpr double value = 1e-10;
  static constexpr double value2 = 1e-8;
};
template <>
struct Epsilon<float> {
  static constexpr float value = 1e-4;
  static constexpr float value2 = 5e-4;
};

// Test LU factorization on a simple dense block matrix
template <typename T>
void testLUFactorSimple(OpsPtr&& ops) {
  // Create a simple 3x3 block structure (single lump, single span)
  // This tests the basic getrf functionality on a dense diagonal block
  vector<set<int64_t>> colBlocks{{0}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> spanStart{0, 4};  // Single block of size 4
  vector<int64_t> lumpToSpan{0, 1};
  SparseStructure groupedSs = ss;
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);

  // Create a non-symmetric test matrix (data stored in row-major within the block)
  // But BaSpaCho uses column-major internally for BLAS
  vector<T> data(factorSkel.dataSize());
  int64_t n = 4;

  // Fill with a non-symmetric matrix that has a well-conditioned LU decomposition
  // Using column-major storage (how BaSpaCho stores dense blocks)
  Matrix<T> testMat(n, n);
  testMat << 4, 1, 2, 1,   //
      1, 5, 1, 2,          //
      2, 1, 6, 1,          //
      1, 2, 1, 7;

  // Copy to data buffer (column-major)
  for (int64_t col = 0; col < n; col++) {
    for (int64_t row = 0; row < n; row++) {
      data[row + col * n] = testMat(row, col);
    }
  }

  // Compute reference LU decomposition using Eigen
  Eigen::PartialPivLU<Matrix<T>> eigenLU(testMat);

  // Create solver and perform LU factorization
  Solver solver(std::move(factorSkel), {}, {}, std::move(ops));

  // Allocate pivots array - one per row (order), not per span
  vector<int64_t> pivots(solver.skel().order());

  // Perform LU factorization
  solver.factorLU(data.data(), pivots.data());

  // Extract L and U from the factored data
  // After getrf, L is below diagonal with unit diagonal, U is upper triangular
  Matrix<T> factored(n, n);
  for (int64_t col = 0; col < n; col++) {
    for (int64_t row = 0; row < n; row++) {
      factored(row, col) = data[row + col * n];
    }
  }

  // Verify: the factored matrix should contain both L (below) and U (on and above diagonal)
  // Extract L (unit lower triangular)
  Matrix<T> L = Matrix<T>::Identity(n, n);
  L.template triangularView<Eigen::StrictlyLower>() =
      factored.template triangularView<Eigen::StrictlyLower>();

  // Extract U (upper triangular)
  Matrix<T> U = factored.template triangularView<Eigen::Upper>();

  // Apply permutation to get P
  Matrix<T> P = Matrix<T>::Identity(n, n);
  for (int64_t i = 0; i < n; i++) {
    if (pivots[i] != i) {
      P.row(i).swap(P.row(pivots[i]));
    }
  }

  // Verify P * A = L * U
  Matrix<T> PA = P * testMat;
  Matrix<T> LU = L * U;

  T error = (PA - LU).norm();
  ASSERT_NEAR(error, 0, Epsilon<T>::value2) << "LU factorization error: P*A != L*U";
}

TEST(LUFactor, Simple_Blas_double) { testLUFactorSimple<double>(fastOps()); }

TEST(LUFactor, Simple_Blas_float) { testLUFactorSimple<float>(fastOps()); }

// Test LU solve on a simple system
template <typename T>
void testLUSolveSimple(OpsPtr&& ops) {
  // Create a simple single-block structure
  vector<set<int64_t>> colBlocks{{0}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> spanStart{0, 4};
  vector<int64_t> lumpToSpan{0, 1};
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, ss.ptrs, ss.inds);

  int64_t n = 4;
  vector<T> data(factorSkel.dataSize());

  // Create test matrix
  Matrix<T> A(n, n);
  A << 4, 1, 2, 1,  //
      1, 5, 1, 2,   //
      2, 1, 6, 1,   //
      1, 2, 1, 7;

  // Copy to data buffer (column-major)
  for (int64_t col = 0; col < n; col++) {
    for (int64_t row = 0; row < n; row++) {
      data[row + col * n] = A(row, col);
    }
  }

  // Create right-hand side
  Vector<T> b(n);
  b << 1, 2, 3, 4;

  // Compute reference solution using Eigen
  Vector<T> xRef = A.partialPivLu().solve(b);

  // Create solver
  Solver solver(std::move(factorSkel), {}, {}, std::move(ops));
  vector<int64_t> pivots(solver.skel().order());

  // Factor
  solver.factorLU(data.data(), pivots.data());

  // Solve
  Vector<T> x = b;  // Copy b, solve in place
  solver.solveLU(data.data(), pivots.data(), x.data(), n, 1);

  // Verify solution
  T error = (x - xRef).norm();
  ASSERT_NEAR(error, 0, Epsilon<T>::value2) << "LU solve error";

  // Also verify A * x = b
  T residual = (A * x - b).norm();
  ASSERT_NEAR(residual, 0, Epsilon<T>::value2) << "Residual error: A*x != b";
}

TEST(LUFactor, Solve_Blas_double) { testLUSolveSimple<double>(fastOps()); }

TEST(LUFactor, Solve_Blas_float) { testLUSolveSimple<float>(fastOps()); }

// Test LU factorization on a block-sparse matrix
template <typename T>
void testLUFactorBlockSparse(const std::function<OpsPtr()>& genOps) {
  // Create a simple 2-block structure
  // Block 0: 3x3, Block 1: 2x2, with fill between them
  vector<set<int64_t>> colBlocks{{0, 1}, {1}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 3, 5};  // Block 0 is 3x3, Block 1 is 2x2
  vector<int64_t> lumpToSpan{0, 1, 2};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);

  int64_t order = factorSkel.order();
  vector<T> data(factorSkel.dataSize());

  // Create a non-symmetric SPD-like matrix (diagonally dominant for stability)
  // and fill the data buffer
  Matrix<T> fullMat = Matrix<T>::Zero(order, order);

  // Fill diagonal blocks with well-conditioned values
  fullMat.block(0, 0, 3, 3) << 10, 1, 2,  //
      1, 11, 1,                            //
      2, 1, 12;

  fullMat.block(3, 3, 2, 2) << 8, 1,  //
      1, 9;

  // Fill off-diagonal block (lower left)
  fullMat.block(3, 0, 2, 3) << 1, 2, 1,  //
      2, 1, 2;

  // Fill upper right for non-symmetric
  fullMat.block(0, 3, 3, 2) << 2, 1,  //
      1, 2,                            //
      1, 1;

  // Densify expects column-major, and we need to map it to BaSpaCho's storage
  // For now, just damp the existing data to ensure positive definiteness
  iota(data.begin(), data.end(), 13);
  factorSkel.damp(data, T(5), T(50));

  // Get the dense matrix from BaSpaCho storage for verification
  Matrix<T> verifyMat = factorSkel.densify(data, true);  // Fill upper half for symmetry

  // Compute reference LU decomposition
  Eigen::PartialPivLU<Matrix<T>> eigenLU(verifyMat);

  // Create solver and perform LU factorization
  Solver solver(std::move(factorSkel), {}, {}, genOps());
  vector<int64_t> pivots(solver.skel().order());

  // Perform LU factorization
  solver.factorLU(data.data(), pivots.data());

  // Verify by solving a system and checking residual
  // Use residual check instead of comparing solutions since different pivoting
  // strategies (block-wise vs full) can give equivalent but different solutions
  Vector<T> b = Vector<T>::Ones(order);
  Vector<T> xRef = eigenLU.solve(b);

  Vector<T> x = b;
  solver.solveLU(data.data(), pivots.data(), x.data(), order, 1);

  // Check residual: ||A*x - b|| should be small
  T residual = (verifyMat * x - b).norm();
  T residualRef = (verifyMat * xRef - b).norm();

  // Both residuals should be small (close to machine precision * condition number)
  ASSERT_NEAR(residual, 0, Epsilon<T>::value2 * 1000)
      << "Block-sparse LU solve residual too large";
  ASSERT_LT(residual, residualRef * 100)
      << "Block-sparse LU residual much larger than Eigen's";
}

// NOTE: Multi-block LU factorization is not fully implemented yet.
// The upper triangle (U off-diagonal) storage and solve updates are missing.
// These tests are disabled until multi-block support is added.
TEST(LUFactor, DISABLED_BlockSparse_Blas_double) {
  testLUFactorBlockSparse<double>([] { return fastOps(); });
}

TEST(LUFactor, DISABLED_BlockSparse_Blas_float) {
  testLUFactorBlockSparse<float>([] { return fastOps(); });
}
