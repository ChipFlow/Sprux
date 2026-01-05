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
#include "baspacho/baspacho/MetalDefs.h"
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

// Metal only supports float precision
template <typename T>
struct Epsilon;
template <>
struct Epsilon<float> {
  // Metal/Apple Silicon has slightly different float precision characteristics
  // Relaxed tolerance for sparse elim + solve operations
  static constexpr float value = 1e-4;
  static constexpr float value2 = 4e-5;
};

template <typename T>
void testSolveL(OpsPtr&& ops, int nRHS = 1) {
  vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
  vector<int64_t> lumpToSpan{0, 2, 4, 6};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  int64_t order = skel.order();

  vector<T> data(skel.dataSize());
  iota(data.begin(), data.end(), 13);
  skel.damp(data, T(5), T(50));

  vector<T> rhsData = randomData<T>(order * nRHS, -1.0, 1.0, 37);
  vector<T> rhsVerif(order * nRHS);
  Matrix<T> verifyMat = skel.densify(data);
  Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
      verifyMat.template triangularView<Eigen::Lower>().solve(
          Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS));

  Solver solver(std::move(skel), {}, {}, std::move(ops));

  // call solve on Metal GPU data
  {
    MetalMirror<T> dataGpu(data), rhsDataGpu(rhsData);
    solver.solveL(dataGpu.ptr(), rhsDataGpu.ptr(), order, nRHS);
    rhsDataGpu.get(rhsData);
  }

  ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
               Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS))
                  .norm(),
              0, Epsilon<T>::value);
}

TEST(MetalSolve, SolveL_float) { testSolveL<float>(metalOps(), 5); }

template <typename T>
void testSolveLt(OpsPtr&& ops, int nRHS = 1) {
  vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
  vector<int64_t> lumpToSpan{0, 2, 4, 6};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  int64_t order = skel.order();

  vector<T> data(skel.dataSize());
  iota(data.begin(), data.end(), 13);
  skel.damp(data, T(5), T(50));

  vector<T> rhsData = randomData<T>(order * nRHS, -1.0, 1.0, 37);
  vector<T> rhsVerif(order * nRHS);
  Matrix<T> verifyMat = skel.densify(data);
  Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
      verifyMat.template triangularView<Eigen::Lower>().adjoint().solve(
          Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS));

  Solver solver(std::move(skel), {}, {}, std::move(ops));

  // call solve on Metal GPU data
  {
    MetalMirror<T> dataGpu(data), rhsDataGpu(rhsData);
    solver.solveLt(dataGpu.ptr(), rhsDataGpu.ptr(), order, nRHS);
    rhsDataGpu.get(rhsData);
  }

  ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
               Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS))
                  .norm(),
              0, Epsilon<T>::value);
}

TEST(MetalSolve, SolveLt_float) { testSolveLt<float>(metalOps(), 5); }

template <typename T>
void testSolveLt_SparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps, int nRHS) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 1.5));

    int64_t order = factorSkel.order();
    vector<T> rhsData = randomData<T>(order * nRHS, -1.0, 1.0, 37 + i);
    vector<T> rhsVerif(order * nRHS);
    Matrix<T> verifyMat = factorSkel.densify(data);
    Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
        verifyMat.template triangularView<Eigen::Lower>().adjoint().solve(
            Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS));

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    // call solve on Metal GPU data
    {
      MetalMirror<T> dataGpu(data), rhsDataGpu(rhsData);
      solver.solveLt(dataGpu.ptr(), rhsDataGpu.ptr(), order, nRHS);
      rhsDataGpu.get(rhsData);
    }

    ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
                 Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS))
                    .norm(),
                0, Epsilon<T>::value);
  }
}

TEST(MetalSolve, SolveLt_SparseElimAndFactor_Many_float) {
  testSolveLt_SparseElimAndFactor_Many<float>([] { return metalOps(); }, 5);
}

template <typename T>
void testSolveL_SparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps, int nRHS) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 1.5));

    int64_t order = factorSkel.order();
    vector<T> rhsData = randomData<T>(order * nRHS, -1.0, 1.0, 37 + i);
    vector<T> rhsVerif(order * nRHS);
    Matrix<T> verifyMat = factorSkel.densify(data);
    Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
        verifyMat.template triangularView<Eigen::Lower>().solve(
            Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS));

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    // call solve on Metal GPU data
    {
      MetalMirror<T> dataGpu(data), rhsDataGpu(rhsData);
      solver.solveL(dataGpu.ptr(), rhsDataGpu.ptr(), order, nRHS);
      rhsDataGpu.get(rhsData);
    }

    ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
                 Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS))
                    .norm(),
                0, Epsilon<T>::value);
  }
}

TEST(MetalSolve, SolveL_SparseElimAndFactor_Many_float) {
  testSolveL_SparseElimAndFactor_Many<float>([] { return metalOps(); }, 5);
}

// Test combined solve() function (both solveL and solveLt) with sparse elimination.
// This is the function called by IREE's sparse solver integration.
template <typename T>
void testFullSolve_SparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps, int nRHS) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 1.5));

    int64_t order = factorSkel.order();
    vector<T> rhsData = randomData<T>(order * nRHS, -1.0, 1.0, 37 + i);
    vector<T> rhsVerif(order * nRHS);

    // For full solve (A*x = b where A = L*L^T), solution is:
    // x = (L^T)^-1 * L^-1 * b
    Matrix<T> verifyMat = factorSkel.densify(data);
    Matrix<T> rhsMat = Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS);
    // First solve L*y = b
    Matrix<T> y = verifyMat.template triangularView<Eigen::Lower>().solve(rhsMat);
    // Then solve L^T*x = y
    Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
        verifyMat.template triangularView<Eigen::Lower>().adjoint().solve(y);

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    // Call combined solve() on Metal GPU data - this is what IREE uses
    {
      MetalMirror<T> dataGpu(data), rhsDataGpu(rhsData);
      solver.solve(dataGpu.ptr(), rhsDataGpu.ptr(), order, nRHS);
      rhsDataGpu.get(rhsData);
    }

    T diff = (Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
              Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS))
                 .norm();
    ASSERT_NEAR(diff, 0, Epsilon<T>::value)
        << "Iteration " << i << ": solve() produced incorrect result, diff norm = " << diff;
  }
}

TEST(MetalSolve, FullSolve_SparseElimAndFactor_Many_float) {
  testFullSolve_SparseElimAndFactor_Many<float>([] { return metalOps(); }, 5);
}

// Test complete factor + solve workflow on Metal GPU.
// This matches the exact usage pattern in IREE: factor the matrix, then solve.
template <typename T>
void testFactorThenSolve_SparseElim_Many(const std::function<OpsPtr()>& genOps, int nRHS) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    // Generate random SPD matrix
    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 1.5));

    int64_t order = factorSkel.order();

    // Compute reference solution using dense Eigen
    Matrix<T> mat = factorSkel.densify(data);
    vector<T> rhsData = randomData<T>(order * nRHS, -1.0, 1.0, 37 + i);
    Matrix<T> rhsMat = Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS);

    // Eigen dense Cholesky solve for reference
    Eigen::LLT<Matrix<T>> llt(mat);
    Matrix<T> refSolution = llt.solve(rhsMat);

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    Solver solver(std::move(factorSkel), std::move(et.sparseElimRanges), {}, genOps());

    // Factor on Metal GPU, then solve on Metal GPU
    {
      MetalMirror<T> dataGpu(data), rhsDataGpu(rhsData);

      // Step 1: Factor
      solver.factor(dataGpu.ptr());

      // Step 2: Solve (using factored data)
      solver.solve(dataGpu.ptr(), rhsDataGpu.ptr(), order, nRHS);

      rhsDataGpu.get(rhsData);
    }

    T diff = (refSolution - Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS)).norm();
    T refNorm = refSolution.norm();
    T relError = diff / refNorm;

    // Note: Metal factor+solve has slightly higher numerical error than pure CPU
    // due to different operation ordering and use of CPU fallbacks in dense operations.
    // 3e-3 (0.3%) relative error is acceptable for float32 sparse Cholesky.
    ASSERT_LT(relError, 3e-3)
        << "Iteration " << i << ": factor+solve produced incorrect result"
        << ", relError = " << relError << ", diff = " << diff << ", refNorm = " << refNorm;
  }
}

TEST(MetalSolve, FactorThenSolve_SparseElim_Many_float) {
  testFactorThenSolve_SparseElim_Many<float>([] { return metalOps(); }, 5);
}

// Test complete factor + solve workflow on CPU (for comparison).
template <typename T>
void testFactorThenSolve_SparseElim_Many_CPU(int nRHS) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.03, 57 + i);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss;

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ true);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    // Generate random SPD matrix
    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0.0), T(factorSkel.order() * 1.5));

    int64_t order = factorSkel.order();

    // Compute reference solution using dense Eigen
    Matrix<T> mat = factorSkel.densify(data);
    vector<T> rhsData = randomData<T>(order * nRHS, -1.0, 1.0, 37 + i);
    Matrix<T> rhsMat = Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS);

    // Eigen dense Cholesky solve for reference
    Eigen::LLT<Matrix<T>> llt(mat);
    Matrix<T> refSolution = llt.solve(rhsMat);

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    Solver solver(std::move(factorSkel), std::move(et.sparseElimRanges), {}, fastOps());

    // Factor on CPU, then solve on CPU
    solver.factor(data.data());
    solver.solve(data.data(), rhsData.data(), order, nRHS);

    T diff = (refSolution - Eigen::Map<Matrix<T>>(rhsData.data(), order, nRHS)).norm();
    T refNorm = refSolution.norm();
    T relError = diff / refNorm;

    ASSERT_LT(relError, 1e-3)
        << "Iteration " << i << ": CPU factor+solve produced incorrect result"
        << ", relError = " << relError << ", diff = " << diff << ", refNorm = " << refNorm;
  }
}

TEST(MetalSolve, FactorThenSolve_SparseElim_Many_float_CPU) {
  testFactorThenSolve_SparseElim_Many_CPU<float>(5);
}
