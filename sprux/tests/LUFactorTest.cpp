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
#include "sprux/sprux/CoalescedBlockMatrix.h"
#include "sprux/sprux/EliminationTree.h"
#include "sprux/sprux/Solver.h"
#include "sprux/sprux/SparseStructure.h"
#include "sprux/sprux/Utils.h"
#include "sprux/testing/TestingUtils.h"

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

  // Create a non-symmetric test matrix
  // Sprux stores dense blocks in ROW-MAJOR format
  vector<T> data(factorSkel.dataSize());
  int64_t n = 4;

  // Fill with a non-symmetric matrix that has a well-conditioned LU decomposition
  Matrix<T> testMat(n, n);
  testMat << 4, 1, 2, 1,   //
      1, 5, 1, 2,          //
      2, 1, 6, 1,          //
      1, 2, 1, 7;

  // Copy to data buffer (row-major - how Sprux stores blocks)
  for (int64_t row = 0; row < n; row++) {
    for (int64_t col = 0; col < n; col++) {
      data[row * n + col] = testMat(row, col);
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

  // Extract L and U from the factored data (row-major storage)
  // After getrf, L is below diagonal with unit diagonal, U is upper triangular
  Matrix<T> factored(n, n);
  for (int64_t row = 0; row < n; row++) {
    for (int64_t col = 0; col < n; col++) {
      factored(row, col) = data[row * n + col];
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

  // Copy to data buffer (row-major - how Sprux stores blocks)
  for (int64_t row = 0; row < n; row++) {
    for (int64_t col = 0; col < n; col++) {
      data[row * n + col] = A(row, col);
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

  // Initialize upper triangle storage for LU factorization
  factorSkel.initUpperTriangle();

  int64_t order = factorSkel.order();

  // Create a well-conditioned symmetric matrix for testing
  // Using symmetric data simplifies verification (L and U are related by transpose)
  Matrix<T> fullMat = Matrix<T>::Zero(order, order);

  // Fill diagonal blocks with diagonally dominant values
  fullMat.block(0, 0, 3, 3) << 10, 1, 2,  //
      1, 11, 1,                            //
      2, 1, 12;

  fullMat.block(3, 3, 2, 2) << 8, 1,  //
      1, 9;

  // Fill off-diagonal block (lower left) - symmetric data
  fullMat.block(3, 0, 2, 3) << 1, 2, 1,  //
      2, 1, 2;

  // Fill upper right as transpose of lower left (symmetric matrix)
  fullMat.block(0, 3, 3, 2) = fullMat.block(3, 0, 2, 3).transpose();

  // Allocate data for both lower and upper triangles
  vector<T> data(factorSkel.totalDataSize());

  // Fill lower triangle data from fullMat
  // Lower triangle has: diagonal blocks (with LU in-place) and below-diagonal blocks
  auto acc = factorSkel.accessor();
  int64_t numLumps = factorSkel.numLumps();

  for (int64_t l = 0; l < numLumps; l++) {
    int64_t chainStart = factorSkel.chainColPtr[l];
    int64_t chainEnd = factorSkel.chainColPtr[l + 1];
    int64_t lumpStart = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;

    for (int64_t c = chainStart; c < chainEnd; c++) {
      int64_t rowSpan = factorSkel.chainRowSpan[c];
      int64_t rowStart = factorSkel.spanStart[rowSpan];
      int64_t rowSize = factorSkel.spanStart[rowSpan + 1] - rowStart;
      int64_t dataOffset = factorSkel.chainData[c];

      // Copy from fullMat to data buffer (row-major storage)
      for (int64_t r = 0; r < rowSize; r++) {
        for (int64_t col = 0; col < lumpSize; col++) {
          data[dataOffset + r * lumpSize + col] = fullMat(rowStart + r, lumpStart + col);
        }
      }
    }
  }

  // Fill upper triangle data from fullMat (transpose of lower off-diagonal blocks)
  int64_t upperDataBase = factorSkel.dataSize();
  for (int64_t l = 0; l < numLumps; l++) {
    int64_t upperRowStart = factorSkel.upperChainRowPtr[l];
    int64_t upperRowEnd = factorSkel.upperChainRowPtr[l + 1];
    int64_t lumpStart = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;

    for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
      int64_t colSpan = factorSkel.upperChainColSpan[i];
      int64_t colStart = factorSkel.spanStart[colSpan];
      int64_t colSize = factorSkel.spanStart[colSpan + 1] - colStart;
      int64_t upperDataOffset = upperDataBase + factorSkel.upperChainData[i];

      // Copy from fullMat to upper data buffer (row-major: lumpSize rows x colSize cols)
      for (int64_t r = 0; r < lumpSize; r++) {
        for (int64_t c = 0; c < colSize; c++) {
          data[upperDataOffset + r * colSize + c] = fullMat(lumpStart + r, colStart + c);
        }
      }
    }
  }

  // Compute reference LU decomposition
  Eigen::PartialPivLU<Matrix<T>> eigenLU(fullMat);

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
  T residual = (fullMat * x - b).norm();
  T residualRef = (fullMat * xRef - b).norm();

  // Both residuals should be small (close to machine precision * condition number)
  ASSERT_NEAR(residual, 0, Epsilon<T>::value2 * 1000)
      << "Block-sparse LU solve residual too large";
  // Compare to Eigen's residual (with minimum threshold to handle Eigen getting exact 0)
  T residualThreshold = std::max(residualRef * 100, Epsilon<T>::value2);
  ASSERT_LT(residual, residualThreshold)
      << "Block-sparse LU residual much larger than Eigen's";
}

TEST(LUFactor, BlockSparse_Blas_double) {
  testLUFactorBlockSparse<double>([] { return fastOps(); });
}

// Debug test to verify block-sparse LU step by step (disabled by default)
TEST(LUFactor, DISABLED_DebugBlockSparse) {
  // Create a simple 2-block structure: Block 0: 3x3, Block 1: 2x2
  vector<set<int64_t>> colBlocks{{0, 1}, {1}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 3, 5};
  vector<int64_t> lumpToSpan{0, 1, 2};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  factorSkel.initUpperTriangle();

  // Create test matrix
  Matrix<double> fullMat = Matrix<double>::Zero(5, 5);
  fullMat.block(0, 0, 3, 3) << 10, 1, 2, 1, 11, 1, 2, 1, 12;
  fullMat.block(3, 3, 2, 2) << 8, 1, 1, 9;
  fullMat.block(3, 0, 2, 3) << 1, 2, 1, 2, 1, 2;
  fullMat.block(0, 3, 3, 2) = fullMat.block(3, 0, 2, 3).transpose();

  std::cout << "Original matrix:\n" << fullMat << "\n\n";

  // Compute reference LU
  Eigen::PartialPivLU<Matrix<double>> eigenLU(fullMat);
  std::cout << "Eigen P:\n" << eigenLU.permutationP().toDenseMatrix() << "\n\n";
  Matrix<double> L = Matrix<double>::Identity(5, 5);
  L.template triangularView<Eigen::StrictlyLower>() =
      eigenLU.matrixLU().template triangularView<Eigen::StrictlyLower>();
  std::cout << "Eigen L:\n" << L << "\n\n";
  Matrix<double> U = eigenLU.matrixLU().template triangularView<Eigen::Upper>();
  std::cout << "Eigen U:\n" << U << "\n\n";

  // Create solver and factor
  vector<double> data(factorSkel.totalDataSize());
  // Fill data from fullMat (same as test)
  for (int64_t l = 0; l < factorSkel.numLumps(); l++) {
    int64_t chainStart = factorSkel.chainColPtr[l];
    int64_t chainEnd = factorSkel.chainColPtr[l + 1];
    int64_t lumpStart = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
    for (int64_t c = chainStart; c < chainEnd; c++) {
      int64_t rowSpan = factorSkel.chainRowSpan[c];
      int64_t rowStart = factorSkel.spanStart[rowSpan];
      int64_t rowSize = factorSkel.spanStart[rowSpan + 1] - rowStart;
      int64_t dataOffset = factorSkel.chainData[c];
      for (int64_t r = 0; r < rowSize; r++) {
        for (int64_t col = 0; col < lumpSize; col++) {
          data[dataOffset + r * lumpSize + col] = fullMat(rowStart + r, lumpStart + col);
        }
      }
    }
  }
  // Fill upper triangle
  int64_t upperDataBase = factorSkel.dataSize();
  for (int64_t l = 0; l < factorSkel.numLumps(); l++) {
    int64_t upperRowStart = factorSkel.upperChainRowPtr[l];
    int64_t upperRowEnd = factorSkel.upperChainRowPtr[l + 1];
    int64_t lumpStartIdx = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStartIdx;
    for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
      int64_t colSpan = factorSkel.upperChainColSpan[i];
      int64_t colStart = factorSkel.spanStart[colSpan];
      int64_t colSize = factorSkel.spanStart[colSpan + 1] - colStart;
      int64_t upperDataOffset = upperDataBase + factorSkel.upperChainData[i];
      for (int64_t r = 0; r < lumpSize; r++) {
        for (int64_t c = 0; c < colSize; c++) {
          data[upperDataOffset + r * colSize + c] = fullMat(lumpStartIdx + r, colStart + c);
        }
      }
    }
  }

  Solver solver(std::move(factorSkel), {}, {}, fastOps());
  vector<int64_t> pivots(solver.skel().order());
  solver.factorLU(data.data(), pivots.data());

  std::cout << "Sprux pivots: ";
  for (auto p : pivots) std::cout << p << " ";
  std::cout << "\n\n";

  // Print the factored data manually
  const auto& skel = solver.skel();
  std::cout << "Lower triangle data (size=" << skel.dataSize() << "):\n";
  for (int64_t i = 0; i < skel.dataSize(); i++) {
    std::cout << data[i] << " ";
    if ((i + 1) % 6 == 0) std::cout << "\n";
  }
  std::cout << "\n\nUpper triangle data (size=" << skel.upperDataSize() << "):\n";
  for (int64_t i = skel.dataSize(); i < skel.totalDataSize(); i++) {
    std::cout << data[i] << " ";
  }
  std::cout << "\n\n";

  // Reconstruct L and U from Sprux format for debugging
  // Block 0 (3x3 diagonal): rows 0-2, data offset 0
  std::cout << "Block 0 diagonal (3x3 row-major at offset 0):\n";
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      std::cout << data[r * 3 + c] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Block L10 (2x3): rows 3-4, cols 0-2, data offset 9
  std::cout << "Block L10 (2x3 row-major at offset 9):\n";
  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 3; c++) {
      std::cout << data[9 + r * 3 + c] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Block 1 (2x2 diagonal): rows 3-4, data offset 15
  std::cout << "Block 1 diagonal (2x2 row-major at offset 15):\n";
  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 2; c++) {
      std::cout << data[15 + r * 2 + c] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Block U01 (3x2): rows 0-2, cols 3-4, upper triangle offset 0
  std::cout << "Block U01 (3x2 row-major at upper offset 0):\n";
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 2; c++) {
      std::cout << data[skel.dataSize() + r * 2 + c] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Solve and check
  Vector<double> b = Vector<double>::Ones(5);
  Vector<double> xRef = eigenLU.solve(b);
  Vector<double> x = b;
  solver.solveLU(data.data(), pivots.data(), x.data(), 5, 1);

  std::cout << "Eigen solution: " << xRef.transpose() << "\n";
  std::cout << "Sprux solution: " << x.transpose() << "\n";
  std::cout << "Residual (Eigen): " << (fullMat * xRef - b).norm() << "\n";
  std::cout << "Residual (Sprux): " << (fullMat * x - b).norm() << "\n";
}

TEST(LUFactor, BlockSparse_Blas_float) {
  testLUFactorBlockSparse<float>([] { return fastOps(); });
}

// Helper: build CSR row pointers and column indices for a dense n×n matrix
static void buildDenseCsr(int64_t n, vector<int64_t>& rowPtr, vector<int64_t>& colInd) {
  rowPtr.resize(n + 1);
  colInd.resize(n * n);
  for (int64_t i = 0; i < n; i++) {
    rowPtr[i] = i * n;
    for (int64_t j = 0; j < n; j++) {
      colInd[i * n + j] = j;
    }
  }
  rowPtr[n] = n * n;
}

// Test static pivoting: factor a singular matrix via createSolver
template <typename T>
void testStaticPivoting() {
  // Create a 4x4 singular matrix where row 1 = 2*row 0.
  // After Schur complement elimination, one pivot will be zero.
  // With static pivoting enabled, factorization should complete.
  int64_t n = 4;

  // Build lower-triangle CSR structure for createSolver
  vector<set<int64_t>> colBlocks;
  for (int64_t i = 0; i < n; i++) {
    set<int64_t> block;
    for (int64_t j = i; j < n; j++) block.insert(j);
    colBlocks.push_back(block);
  }
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> paramSizes(n, 1);

  // Create solver WITH static pivoting enabled (auto threshold)
  Settings settings;
  settings.backend = BackendFast;
  settings.matrixType = MTYPE_GENERAL;
  settings.staticPivotThreshold = 0.0;  // auto = sqrt(epsilon)
  auto solver = createSolver(settings, paramSizes, ss);

  // Singular matrix: row 1 = 2 * row 0
  Matrix<T> A(n, n);
  A << 1, 2, 3, 4,  //
      2, 4, 6, 8,   //
      1, 1, 1, 1,   //
      1, 3, 2, 5;

  // Build full CSR and load via loadFromCsr
  vector<int64_t> rowPtr, colInd;
  buildDenseCsr(n, rowPtr, colInd);
  vector<int64_t> blockSizes(n, 1);

  // Extract CSR values in row-major order (Eigen stores col-major by default)
  vector<T> csrValues(n * n);
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < n; j++) {
      csrValues[i * n + j] = A(i, j);
    }
  }

  vector<T> data(solver->skel().totalDataSize(), T(0));
  solver->loadFromCsr(rowPtr.data(), colInd.data(), blockSizes.data(), csrValues.data(),
                      data.data());

  vector<int64_t> pivots(n);

  // Without static pivoting, this would throw "getrf failed with info = ..."
  // With static pivoting, it should succeed
  ASSERT_NO_THROW(solver->factorLU(data.data(), pivots.data()));
  ASSERT_GT(solver->staticPivotPerturbCount(), 0)
      << "Expected at least one diagonal perturbation";

  cout << "Static pivoting perturbed " << solver->staticPivotPerturbCount() << " diagonal(s)"
       << endl;

  // Solve A*x = b; solution won't be exact due to perturbation, but should be finite
  Vector<T> b(n);
  b << 1, 2, 3, 4;

  const auto& perm = solver->paramToSpan();
  Vector<T> bp(n);
  for (int64_t j = 0; j < n; j++) bp(perm[j]) = b(j);

  solver->solveLU(data.data(), pivots.data(), bp.data(), n, 1);

  // Inverse permute
  Vector<T> x(n);
  for (int64_t j = 0; j < n; j++) x(j) = bp(perm[j]);

  // Solution should be finite (no NaN/Inf)
  for (int64_t j = 0; j < n; j++) {
    EXPECT_TRUE(std::isfinite(x(j))) << "Solution element " << j << " is not finite: " << x(j);
  }
}

TEST(LUFactor, StaticPivoting_double) { testStaticPivoting<double>(); }

TEST(LUFactor, StaticPivoting_float) { testStaticPivoting<float>(); }

// Test that static pivoting is disabled by default (negative threshold)
TEST(LUFactor, StaticPivotingDisabledByDefault) {
  int64_t n = 4;
  vector<set<int64_t>> colBlocks;
  for (int64_t i = 0; i < n; i++) {
    set<int64_t> block;
    for (int64_t j = i; j < n; j++) block.insert(j);
    colBlocks.push_back(block);
  }
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> paramSizes(n, 1);

  // Default settings: staticPivotThreshold = -1.0 (disabled)
  Settings settings;
  settings.backend = BackendFast;
  settings.matrixType = MTYPE_GENERAL;
  auto solver = createSolver(settings, paramSizes, ss);

  // Singular matrix: row 1 = 2 * row 0
  Matrix<double> A(n, n);
  A << 1, 2, 3, 4,  //
      2, 4, 6, 8,   //
      1, 1, 1, 1,   //
      1, 3, 2, 5;

  vector<int64_t> rowPtr, colInd;
  buildDenseCsr(n, rowPtr, colInd);
  vector<int64_t> blockSizes(n, 1);
  vector<double> csrValues(n * n);
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < n; j++) {
      csrValues[i * n + j] = A(i, j);
    }
  }

  vector<double> data(solver->skel().totalDataSize(), 0.0);
  solver->loadFromCsr(rowPtr.data(), colInd.data(), blockSizes.data(), csrValues.data(),
                      data.data());

  vector<int64_t> pivots(n);

  // Should throw since static pivoting is disabled
  EXPECT_THROW(solver->factorLU(data.data(), pivots.data()), std::runtime_error);
}
