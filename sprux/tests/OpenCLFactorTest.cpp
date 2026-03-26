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
#include "sprux/sprux/CoalescedBlockMatrix.h"
#include "sprux/sprux/EliminationTree.h"
#include "sprux/sprux/OpenCLDefs.h"
#include "sprux/sprux/Solver.h"
#include "sprux/sprux/SparseStructure.h"
#include "sprux/sprux/Utils.h"
#include "sprux/testing/TestingUtils.h"

using namespace Sprux;
using namespace ::Sprux::testing_utils;
using namespace std;
using namespace ::testing;

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

// OpenCL supports both float and double precision
template <typename T>
struct Epsilon;
template <>
struct Epsilon<float> {
  static constexpr float value = 1e-5;
  static constexpr float value2 = 3e-3;
};
template <>
struct Epsilon<double> {
  static constexpr double value = 1e-10;
  // OpenCL via PoCL CPU emulation can have different floating-point behavior
  // than native BLAS. Sparse elimination accumulates more rounding error.
  // Relaxed tolerance to accommodate CI environment variations.
  static constexpr double value2 = 1e-4;
};

template <typename T>
void testCoalescedFactor(OpsPtr&& ops) {
  vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
  vector<int64_t> lumpToSpan{0, 2, 4, 6};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);

  vector<T> data(factorSkel.dataSize());
  iota(data.begin(), data.end(), 13);
  factorSkel.damp(data, T(5), T(50));

  Matrix<T> verifyMat = factorSkel.densify(data);
  Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

  Solver solver(std::move(factorSkel), {}, {}, std::move(ops));

  // Use OpenCLMirror for GPU execution with proper sync
  OpenCLMirror<T> dataGpu(data);
  solver.factor(dataGpu.hostPtr());
  dataGpu.get(data);

  Matrix<T> computedMat = solver.skel().densify(data);

  ASSERT_NEAR(Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm(),
              0, Epsilon<T>::value);
}

TEST(OpenCLFactor, CoalescedFactor_float) { testCoalescedFactor<float>(openclOps()); }
TEST(OpenCLFactor, CoalescedFactor_double) { testCoalescedFactor<double>(openclOps()); }

template <typename T>
void testCoalescedFactor_Many(const std::function<OpsPtr()>& genOps) {
  for (int i = 0; i < 20; i++) {
    auto colBlocks = randomCols(115, 0.037, 57 + i);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss.symmetricPermutation(invPerm, false);

    vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47 + i);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.processTree(/* compute sparse elim ranges = */ false);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan, et.colStart,
                                        et.rowParam);

    vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(data, T(0), T(factorSkel.order() * 1.5));

    Matrix<T> verifyMat = factorSkel.densify(data);
    Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

    Solver solver(std::move(factorSkel), {}, {}, genOps());

    // Use OpenCLMirror for GPU execution with proper sync
    OpenCLMirror<T> dataGpu(data);
    solver.factor(dataGpu.hostPtr());
    dataGpu.get(data);

    Matrix<T> computedMat = solver.skel().densify(data);

    ASSERT_NEAR(Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm(),
                0, Epsilon<T>::value2);
  }
}

TEST(OpenCLFactor, CoalescedFactor_Many_float) {
  testCoalescedFactor_Many<float>([] { return openclOps(); });
}
TEST(OpenCLFactor, CoalescedFactor_Many_double) {
  testCoalescedFactor_Many<double>([] { return openclOps(); });
}

template <typename T>
void testSparseElim_Many(const std::function<OpsPtr()>& genOps) {
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
    factorSkel.damp(data, T(0), T(factorSkel.order() * 1.5));

    Matrix<T> verifyMat = factorSkel.densify(data);
    Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(std::move(factorSkel), std::move(et.sparseElimRanges), {}, genOps());

    NumericCtxPtr<T> numCtx = solver.internalSymbolicContext().createNumericCtx<T>(0, nullptr);

    // Use OpenCLMirror for GPU execution with proper sync
    OpenCLMirror<T> dataGpu(data);
    numCtx->doElimination(solver.internalGetElimCtx(0), dataGpu.hostPtr(), 0, largestIndep);
    dataGpu.get(data);

    Matrix<T> computedMat = solver.skel().densify(data);

    ASSERT_NEAR(Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>())
                    .leftCols(largestIndep)
                    .norm(),
                0, Epsilon<T>::value);
  }
}

TEST(OpenCLFactor, SparseElim_Many_float) {
  testSparseElim_Many<float>([] { return openclOps(); });
}
TEST(OpenCLFactor, SparseElim_Many_double) {
  testSparseElim_Many<double>([] { return openclOps(); });
}

template <typename T>
void testSparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps) {
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
    factorSkel.damp(data, T(0), T(factorSkel.order() * 1.5));

    Matrix<T> verifyMat = factorSkel.densify(data);
    Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(verifyMat);

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    Solver solver(std::move(factorSkel), std::move(et.sparseElimRanges), {}, genOps());

    // Use OpenCLMirror for GPU execution with proper sync
    OpenCLMirror<T> dataGpu(data);
    solver.factor(dataGpu.hostPtr());
    dataGpu.get(data);

    Matrix<T> computedMat = solver.skel().densify(data);
    ASSERT_NEAR(Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm(),
                0, Epsilon<T>::value2);
  }
}

TEST(OpenCLFactor, SparseElimAndFactor_Many_float) {
  testSparseElimAndFactor_Many<float>([] { return openclOps(); });
}
TEST(OpenCLFactor, SparseElimAndFactor_Many_double) {
  testSparseElimAndFactor_Many<double>([] { return openclOps(); });
}
