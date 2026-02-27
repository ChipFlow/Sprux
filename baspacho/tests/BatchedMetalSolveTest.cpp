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
  // Relaxed tolerance: batched Metal solve with sparse elimination accumulates
  // more float rounding error than individual operations
  static constexpr float value = 1e-4;
  static constexpr float value2 = 1e-4;
};

template <typename T>
void testBatchedSolveL(OpsPtr&& ops, int nRHS = 1, int batchSize = 8) {
  vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
  vector<int64_t> lumpToSpan{0, 2, 4, 6};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  int64_t order = skel.order();

  Solver solver(std::move(skel), {}, {}, std::move(ops));

  // generate a batch of data
  vector<vector<T>> datas(batchSize);
  vector<vector<T>> rhsDatas(batchSize);
  for (int q = 0; q < batchSize; q++) {
    datas[q] = randomData<T>(solver.skel().dataSize(), -1.0, 1.0, 37 + q);
    solver.skel().damp(datas[q], T(0), T(solver.skel().order() * 1.3));
    rhsDatas[q] = randomData<T>(order * nRHS, -1.0, 1.0, 137 + q);
  }
  vector<vector<T>> rhsDatasBackup = rhsDatas;

  // call solve on Metal GPU data
  {
    vector<MetalMirror<T>> datasGpu(batchSize);
    vector<MetalMirror<T>> rhsDatasGpu(batchSize);
    vector<T*> datasPtr(batchSize);
    vector<T*> rhsDatasPtr(batchSize);
    for (int q = 0; q < batchSize; q++) {
      datasGpu[q].load(datas[q]);
      rhsDatasGpu[q].load(rhsDatas[q]);
      datasPtr[q] = datasGpu[q].ptr();
      rhsDatasPtr[q] = rhsDatasGpu[q].ptr();
    }
    solver.solveL(&datasPtr, &rhsDatasPtr, order, nRHS);
    for (int q = 0; q < batchSize; q++) {
      rhsDatasGpu[q].get(rhsDatas[q]);
    }
  }

  for (int q = 0; q < batchSize; q++) {
    vector<T> rhsVerif(order * nRHS);
    Matrix<T> verifyMat = solver.skel().densify(datas[q]);
    Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
        verifyMat.template triangularView<Eigen::Lower>().solve(
            Eigen::Map<Matrix<T>>(rhsDatasBackup[q].data(), order, nRHS));

    ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
                 Eigen::Map<Matrix<T>>(rhsDatas[q].data(), order, nRHS))
                    .norm(),
                0, Epsilon<T>::value);
  }
}

TEST(BatchedMetalSolve, SolveL_float) { testBatchedSolveL<float>(metalOps(), 5, 8); }

template <typename T>
void testBatchedSolveLt(OpsPtr&& ops, int nRHS = 1, int batchSize = 8) {
  vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
  vector<int64_t> lumpToSpan{0, 2, 4, 6};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  int64_t order = skel.order();

  Solver solver(std::move(skel), {}, {}, std::move(ops));

  // generate a batch of data
  vector<vector<T>> datas(batchSize);
  vector<vector<T>> rhsDatas(batchSize);
  for (int q = 0; q < batchSize; q++) {
    datas[q] = randomData<T>(solver.skel().dataSize(), -1.0, 1.0, 37 + q);
    solver.skel().damp(datas[q], T(0), T(solver.skel().order() * 1.3));
    rhsDatas[q] = randomData<T>(order * nRHS, -1.0, 1.0, 137 + q);
  }
  vector<vector<T>> rhsDatasBackup = rhsDatas;

  // call solve on Metal GPU data
  {
    vector<MetalMirror<T>> datasGpu(batchSize);
    vector<MetalMirror<T>> rhsDatasGpu(batchSize);
    vector<T*> datasPtr(batchSize);
    vector<T*> rhsDatasPtr(batchSize);
    for (int q = 0; q < batchSize; q++) {
      datasGpu[q].load(datas[q]);
      rhsDatasGpu[q].load(rhsDatas[q]);
      datasPtr[q] = datasGpu[q].ptr();
      rhsDatasPtr[q] = rhsDatasGpu[q].ptr();
    }
    solver.solveLt(&datasPtr, &rhsDatasPtr, order, nRHS);
    for (int q = 0; q < batchSize; q++) {
      rhsDatasGpu[q].get(rhsDatas[q]);
    }
  }

  for (int q = 0; q < batchSize; q++) {
    vector<T> rhsVerif(order * nRHS);
    Matrix<T> verifyMat = solver.skel().densify(datas[q]);
    Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
        verifyMat.template triangularView<Eigen::Lower>().adjoint().solve(
            Eigen::Map<Matrix<T>>(rhsDatasBackup[q].data(), order, nRHS));

    ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
                 Eigen::Map<Matrix<T>>(rhsDatas[q].data(), order, nRHS))
                    .norm(),
                0, Epsilon<T>::value);
  }
}

TEST(BatchedMetalSolve, SolveLt_float) { testBatchedSolveLt<float>(metalOps(), 5, 8); }

template <typename T>
void testBatchedSolveL_SparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps, int nRHS,
                                                int batchSize) {
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

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    // generate a batch of data
    int order = solver.order();
    vector<vector<T>> datas(batchSize);
    vector<vector<T>> rhsDatas(batchSize);
    for (int q = 0; q < batchSize; q++) {
      datas[q] = randomData<T>(solver.skel().dataSize(), -1.0, 1.0, 37 + q + i);
      solver.skel().damp(datas[q], T(0), T(order * 1.3));
      rhsDatas[q] = randomData<T>(order * nRHS, -1.0, 1.0, 137 + q + i);
    }
    vector<vector<T>> rhsDatasBackup = rhsDatas;

    // call solve on Metal GPU data
    {
      vector<MetalMirror<T>> datasGpu(batchSize);
      vector<MetalMirror<T>> rhsDatasGpu(batchSize);
      vector<T*> datasPtr(batchSize);
      vector<T*> rhsDatasPtr(batchSize);
      for (int q = 0; q < batchSize; q++) {
        datasGpu[q].load(datas[q]);
        rhsDatasGpu[q].load(rhsDatas[q]);
        datasPtr[q] = datasGpu[q].ptr();
        rhsDatasPtr[q] = rhsDatasGpu[q].ptr();
      }
      solver.solveL(&datasPtr, &rhsDatasPtr, order, nRHS);
      for (int q = 0; q < batchSize; q++) {
        rhsDatasGpu[q].get(rhsDatas[q]);
      }
    }

    for (int q = 0; q < batchSize; q++) {
      vector<T> rhsVerif(order * nRHS);
      Matrix<T> verifyMat = solver.skel().densify(datas[q]);
      Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
          verifyMat.template triangularView<Eigen::Lower>().solve(
              Eigen::Map<Matrix<T>>(rhsDatasBackup[q].data(), order, nRHS));

      ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
                   Eigen::Map<Matrix<T>>(rhsDatas[q].data(), order, nRHS))
                      .norm(),
                  0, Epsilon<T>::value);
    }
  }
}

TEST(BatchedMetalSolve, SolveL_SparseElimAndFactor_Many_float) {
  testBatchedSolveL_SparseElimAndFactor_Many<float>([] { return metalOps(); }, 5, 8);
}

template <typename T>
void testBatchedSolveLt_SparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps, int nRHS,
                                                 int batchSize) {
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

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    // generate a batch of data
    int order = solver.order();
    vector<vector<T>> datas(batchSize);
    vector<vector<T>> rhsDatas(batchSize);
    for (int q = 0; q < batchSize; q++) {
      datas[q] = randomData<T>(solver.skel().dataSize(), -1.0, 1.0, 37 + q + i);
      solver.skel().damp(datas[q], T(0), T(order * 1.3));
      rhsDatas[q] = randomData<T>(order * nRHS, -1.0, 1.0, 137 + q + i);
    }
    vector<vector<T>> rhsDatasBackup = rhsDatas;

    // call solve on Metal GPU data
    {
      vector<MetalMirror<T>> datasGpu(batchSize);
      vector<MetalMirror<T>> rhsDatasGpu(batchSize);
      vector<T*> datasPtr(batchSize);
      vector<T*> rhsDatasPtr(batchSize);
      for (int q = 0; q < batchSize; q++) {
        datasGpu[q].load(datas[q]);
        rhsDatasGpu[q].load(rhsDatas[q]);
        datasPtr[q] = datasGpu[q].ptr();
        rhsDatasPtr[q] = rhsDatasGpu[q].ptr();
      }
      solver.solveL(&datasPtr, &rhsDatasPtr, order, nRHS);
      for (int q = 0; q < batchSize; q++) {
        rhsDatasGpu[q].get(rhsDatas[q]);
      }
    }

    for (int q = 0; q < batchSize; q++) {
      vector<T> rhsVerif(order * nRHS);
      Matrix<T> verifyMat = solver.skel().densify(datas[q]);
      Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) =
          verifyMat.template triangularView<Eigen::Lower>().solve(
              Eigen::Map<Matrix<T>>(rhsDatasBackup[q].data(), order, nRHS));

      ASSERT_NEAR((Eigen::Map<Matrix<T>>(rhsVerif.data(), order, nRHS) -
                   Eigen::Map<Matrix<T>>(rhsDatas[q].data(), order, nRHS))
                      .norm(),
                  0, Epsilon<T>::value);
    }
  }
}

TEST(BatchedMetalSolve, SolveLt_SparseElimAndFactor_Many_float) {
  testBatchedSolveLt_SparseElimAndFactor_Many<float>([] { return metalOps(); }, 5, 8);
}
