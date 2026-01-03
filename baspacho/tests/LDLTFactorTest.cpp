/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
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
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

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

// Test LDL^T factorization on sparse SPD matrices using createSolver
template <typename T>
void testLDLTSparse_Many(BackendType backend) {
  Settings settings;
  settings.backend = backend;

  for (int i = 0; i < 10; i++) {
    auto colBlocks = randomCols(30, 0.1, 57 + i);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
    vector<int64_t> paramSize = randomVec(ss.ptrs.size() - 1, 1, 3, 47);

    // Use createSolver for proper setup
    SolverPtr solverPtr = createSolver(settings, paramSize, ss);

    // Create random SPD matrix
    vector<T> data = randomData<T>(solverPtr->dataSize(), -1.0, 1.0, 9 + i);
    solverPtr->skel().damp(data, T(0.0), T(solverPtr->order() * 2.0));

    // Store original dense matrix for verification
    Matrix<T> A = solverPtr->skel().densify(data);
    // Symmetrize A - densify only fills lower triangle
    for (int64_t row = 0; row < A.rows(); row++) {
      for (int64_t col = row + 1; col < A.cols(); col++) {
        A(row, col) = A(col, row);
      }
    }

    // Factor with LDL^T
    solverPtr->factorLDLT(data.data());

    // Verify by reconstruction: extract L and D, compute L * D * L^T
    Matrix<T> factored = solverPtr->skel().densify(data);

    // Extract L (unit lower) and D (diagonal) from factored matrix
    int64_t n = A.rows();
    Matrix<T> L = Matrix<T>::Identity(n, n);
    Vector<T> D(n);
    for (int64_t row = 0; row < n; row++) {
      D(row) = factored(row, row);
      for (int64_t col = 0; col < row; col++) {
        L(row, col) = factored(row, col);
      }
    }

    // Reconstruct and verify
    Matrix<T> reconstructed = L * D.asDiagonal() * L.transpose();
    ASSERT_NEAR((A - reconstructed).norm() / A.norm(), 0, Epsilon<T>::value2)
        << "Iteration " << i << " failed";
  }
}

TEST(LDLTFactor, Sparse_Many_Blas_double) { testLDLTSparse_Many<double>(BackendFast); }

TEST(LDLTFactor, Sparse_Many_Ref_double) { testLDLTSparse_Many<double>(BackendRef); }

TEST(LDLTFactor, Sparse_Many_Blas_float) { testLDLTSparse_Many<float>(BackendFast); }

TEST(LDLTFactor, Sparse_Many_Ref_float) { testLDLTSparse_Many<float>(BackendRef); }

// Test single-lump LDL^T (no Schur complement, just ldlt on diagonal)
template <typename T>
void testLDLTSingleLump(BackendType backend) {
  Settings settings;
  settings.backend = backend;

  // Create a single-block matrix (all parameters in one lump)
  int n = 5;
  vector<int64_t> paramSize(n, 1);  // n scalar parameters

  // Fully connected lower triangular structure -> single lump after ordering
  // BaSpaCho expects CSR lower triangular (row i has columns 0..i)
  vector<int64_t> ptrs(n + 1);
  vector<int64_t> inds;
  for (int i = 0; i < n; i++) {
    ptrs[i] = inds.size();
    for (int j = 0; j <= i; j++) {  // Lower triangle: j <= i
      inds.push_back(j);
    }
  }
  ptrs[n] = inds.size();
  SparseStructure ss(std::move(ptrs), std::move(inds));

  SolverPtr solverPtr = createSolver(settings, paramSize, ss);

  // Create random SPD matrix
  vector<T> data = randomData<T>(solverPtr->dataSize(), -1.0, 1.0, 42);
  solverPtr->skel().damp(data, T(0.0), T(n * 2.0));

  Matrix<T> A = solverPtr->skel().densify(data);
  // Symmetrize A - densify only fills lower triangle
  for (int row = 0; row < A.rows(); row++) {
    for (int col = row + 1; col < A.cols(); col++) {
      A(row, col) = A(col, row);
    }
  }

  // Factor with LDL^T
  solverPtr->factorLDLT(data.data());

  // Extract L (unit lower) and D (diagonal) from factored matrix
  Matrix<T> factored = solverPtr->skel().densify(data);
  Matrix<T> L = Matrix<T>::Identity(n, n);
  Vector<T> D(n);
  for (int row = 0; row < n; row++) {
    D(row) = factored(row, row);
    for (int col = 0; col < row; col++) {
      L(row, col) = factored(row, col);
    }
  }

  // Verify L * D * L^T = A
  Matrix<T> reconstructed = L * D.asDiagonal() * L.transpose();
  ASSERT_NEAR((A - reconstructed).norm() / A.norm(), 0, Epsilon<T>::value2)
      << "Single lump reconstruction failed";
}

TEST(LDLTFactor, SingleLump_Blas_double) { testLDLTSingleLump<double>(BackendFast); }
TEST(LDLTFactor, SingleLump_Ref_double) { testLDLTSingleLump<double>(BackendRef); }

// Test LDL^T solve (factor + solve)
template <typename T>
void testLDLTSolve(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 10; i++) {
    auto colBlocks = randomCols(25, 0.12, 77 + i);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss.symmetricPermutation(invPerm, false);

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 1, 3, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ false);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    // Create random SPD matrix
    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 2.0));

    // Store original dense matrix
    Matrix<T> A = factorSkel.densify(data);
    // Symmetrize A - densify only fills lower triangle
    for (int64_t row = 0; row < A.rows(); row++) {
      for (int64_t col = row + 1; col < A.cols(); col++) {
        A(row, col) = A(col, row);
      }
    }

    // Create random RHS
    int64_t n = A.rows();
    Vector<T> b = Vector<T>::Random(n);
    Vector<T> x_ref = A.ldlt().solve(b);

    // Factor and solve with BaSpaCho
    Solver solver(CoalescedBlockMatrixSkel(factorSkel), {}, {}, genOps());
    solver.factorLDLT(data.data());

    Vector<T> x = b;  // solve in place
    solver.solveLDLT(data.data(), x.data(), n, 1);

    // Verify solution
    ASSERT_NEAR((x - x_ref).norm() / x_ref.norm(), 0, Epsilon<T>::value2)
        << "Solve iteration " << i << " failed";

    // Also verify residual
    Vector<T> residual = A * x - b;
    ASSERT_NEAR(residual.norm() / b.norm(), 0, Epsilon<T>::value2)
        << "Residual iteration " << i << " too large";
  }
}

TEST(LDLTSolve, Solve_Blas_double) { testLDLTSolve<double>([] { return fastOps(); }); }

TEST(LDLTSolve, Solve_Ref_double) { testLDLTSolve<double>([] { return simpleOps(); }); }

TEST(LDLTSolve, Solve_Blas_float) { testLDLTSolve<float>([] { return fastOps(); }); }

TEST(LDLTSolve, Solve_Ref_float) { testLDLTSolve<float>([] { return simpleOps(); }); }

// Test LDL^T solve with multiple RHS
template <typename T>
void testLDLTSolveMultiRHS(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 5; i++) {
    auto colBlocks = randomCols(20, 0.15, 33 + i);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss.symmetricPermutation(invPerm, false);

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 1, 3, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(false);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 2.0));

    Matrix<T> A = factorSkel.densify(data);
    // Symmetrize A - densify only fills lower triangle
    for (int64_t row = 0; row < A.rows(); row++) {
      for (int64_t col = row + 1; col < A.cols(); col++) {
        A(row, col) = A(col, row);
      }
    }

    int64_t n = A.rows();
    int nRHS = 3;

    // Multiple RHS
    Matrix<T> B = Matrix<T>::Random(n, nRHS);
    Matrix<T> X_ref = A.ldlt().solve(B);

    // Factor and solve
    Solver solver(CoalescedBlockMatrixSkel(factorSkel), {}, {}, genOps());
    solver.factorLDLT(data.data());

    Matrix<T> X = B;
    solver.solveLDLT(data.data(), X.data(), n, nRHS);

    ASSERT_NEAR((X - X_ref).norm() / X_ref.norm(), 0, Epsilon<T>::value2)
        << "Multi-RHS solve iteration " << i << " failed";
  }
}

TEST(LDLTSolve, MultiRHS_Blas_double) { testLDLTSolveMultiRHS<double>([] { return fastOps(); }); }

TEST(LDLTSolve, MultiRHS_Ref_double) { testLDLTSolveMultiRHS<double>([] { return simpleOps(); }); }

TEST(LDLTSolve, MultiRHS_Blas_float) { testLDLTSolveMultiRHS<float>([] { return fastOps(); }); }

TEST(LDLTSolve, MultiRHS_Ref_float) { testLDLTSolveMultiRHS<float>([] { return simpleOps(); }); }
