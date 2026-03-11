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
  static constexpr float value = 1e-5;
  // Relative tolerance for sparse elim + dense factor comparison.
  // Sparse elim is now deterministic (two-phase accumulation), but MPS
  // dense ops (potrf/trsm/GEMM) still differ on paravirtualized Metal GPU.
  // Real hardware: ~5e-8. CI paravirtualized: ~1.7e-5.
  static constexpr float value2 = 5e-5;
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

  // call factor on Metal GPU data
  {
    MetalMirror<T> dataGpu(data);
    solver.factor(dataGpu.ptr());
    dataGpu.get(data);
  }

  Matrix<T> computedMat = solver.skel().densify(data);

  ASSERT_NEAR(Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm(),
              0, Epsilon<T>::value);
}

TEST(MetalFactor, CoalescedFactor_float) { testCoalescedFactor<float>(metalOps()); }

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

    // call factor on Metal GPU data
    {
      MetalMirror<T> dataGpu(data);
      solver.factor(dataGpu.ptr());
      dataGpu.get(data);
    }

    Matrix<T> computedMat = solver.skel().densify(data);
    T absErr =
        Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm();
    T refNorm =
        Matrix<T>(verifyMat.template triangularView<Eigen::Lower>()).norm();
    T relErr = absErr / std::max(refNorm, T(1e-30));
    ASSERT_LT(relErr, Epsilon<T>::value2)
        << "iteration " << i << ": absErr=" << absErr << ", refNorm=" << refNorm
        << ", relErr=" << relErr;
  }
}

TEST(MetalFactor, CoalescedFactor_Many_float) {
  testCoalescedFactor_Many<float>([] { return metalOps(); });
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
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    NumericCtxPtr<T> numCtx = solver.internalSymbolicContext().createNumericCtx<T>(0, nullptr);

    // call doElimination with data on Metal GPU
    {
      MetalMirror<T> dataGpu(data);
      numCtx->doElimination(solver.internalGetElimCtx(0), dataGpu.ptr(), 0, largestIndep);
      dataGpu.get(data);
    }

    Matrix<T> computedMat = solver.skel().densify(data);

    ASSERT_NEAR(Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>())
                    .leftCols(largestIndep)
                    .norm(),
                0, 2e-5);
  }
}

TEST(MetalFactor, SparseElim_Many_float) {
  testSparseElim_Many<float>([] { return metalOps(); });
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
    int64_t largestIndep = et.sparseElimRanges[1];
    Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());

    // call factor on Metal GPU data
    {
      MetalMirror<T> dataGpu(data);
      solver.factor(dataGpu.ptr());
      dataGpu.get(data);
    }

    Matrix<T> computedMat = solver.skel().densify(data);
    T absErr =
        Matrix<T>((verifyMat - computedMat).template triangularView<Eigen::Lower>()).norm();
    T refNorm =
        Matrix<T>(verifyMat.template triangularView<Eigen::Lower>()).norm();
    T relErr = absErr / std::max(refNorm, T(1e-30));
    ASSERT_LT(relErr, Epsilon<T>::value2)
        << "iteration " << i << ": absErr=" << absErr << ", refNorm=" << refNorm
        << ", relErr=" << relErr;
  }
}

TEST(MetalFactor, SparseElimAndFactor_Many_float) {
  testSparseElimAndFactor_Many<float>([] { return metalOps(); });
}

// Diagnostic test: break down error sources between sparse elim and dense factor,
// comparing Metal GPU vs CPU BLAS (fastOps) vs Eigen LLT.
TEST(MetalFactor, DiagnoseErrorSources_float) {
  using T = float;
  T maxSparseElimErr = 0, maxDenseOnlyErr = 0, maxCombinedErr = 0;
  T maxMetalVsBlas = 0;
  int worstIteration = -1;

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
    int64_t order = factorSkel.order();
    int64_t numLumps = factorSkel.lumpStart.size() - 1;
    int64_t sparseElimEnd = et.sparseElimRanges.back();

    vector<T> origData = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
    factorSkel.damp(origData, T(0), T(order * 1.5));

    // Eigen LLT reference
    Matrix<T> eigenMat = factorSkel.densify(origData);
    Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(eigenMat);

    // CPU BLAS factor (same algorithm as Metal, but on CPU)
    vector<T> cpuData = origData;
    auto elimRanges = et.sparseElimRanges;  // copy for reuse
    {
      CoalescedBlockMatrixSkel cpuSkel = factorSkel;  // copy for separate solver
      auto cpuElimRanges = elimRanges;
      Solver cpuSolver(std::move(cpuSkel), std::move(cpuElimRanges), {}, fastOps());
      cpuSolver.factor(cpuData.data());
    }
    Matrix<T> cpuMat = factorSkel.densify(cpuData);

    // Metal GPU factor
    vector<T> gpuData = origData;
    {
      CoalescedBlockMatrixSkel gpuSkel = factorSkel;
      auto gpuElimRanges = elimRanges;
      Solver gpuSolver(std::move(gpuSkel), std::move(gpuElimRanges), {}, metalOps());
      MetalMirror<T> dataGpu(gpuData);
      gpuSolver.factor(dataGpu.ptr());
      dataGpu.get(gpuData);
    }
    Matrix<T> gpuMat = factorSkel.densify(gpuData);

    T refNorm = Matrix<T>(eigenMat.template triangularView<Eigen::Lower>()).norm();

    // Error: Metal GPU vs Eigen LLT
    T gpuVsEigenErr =
        Matrix<T>((eigenMat - gpuMat).template triangularView<Eigen::Lower>()).norm();
    T gpuVsEigenRel = gpuVsEigenErr / std::max(refNorm, T(1e-30));

    // Error: CPU BLAS vs Eigen LLT
    T cpuVsEigenErr =
        Matrix<T>((eigenMat - cpuMat).template triangularView<Eigen::Lower>()).norm();
    T cpuVsEigenRel = cpuVsEigenErr / std::max(refNorm, T(1e-30));

    // Error: Metal GPU vs CPU BLAS (same algorithm, different hardware)
    T gpuVsCpuErr =
        Matrix<T>((cpuMat - gpuMat).template triangularView<Eigen::Lower>()).norm();
    T gpuVsCpuRel = gpuVsCpuErr / std::max(refNorm, T(1e-30));

    // Find max element-wise difference between GPU and CPU BLAS
    T maxElemDiff = 0;
    int maxElemRow = 0, maxElemCol = 0;
    for (int r = 0; r < order; r++) {
      for (int c = 0; c <= r; c++) {
        T diff = std::abs(gpuMat(r, c) - cpuMat(r, c));
        if (diff > maxElemDiff) {
          maxElemDiff = diff;
          maxElemRow = r;
          maxElemCol = c;
        }
      }
    }

    std::cout << "iter " << i << ": order=" << order << " lumps=" << numLumps
              << " sparseElimEnd=" << sparseElimEnd
              << "\n  GPU vs Eigen:  relErr=" << gpuVsEigenRel
              << "\n  CPU vs Eigen:  relErr=" << cpuVsEigenRel
              << "\n  GPU vs CPU:    relErr=" << gpuVsCpuRel
              << " maxElem=" << maxElemDiff << " at (" << maxElemRow << "," << maxElemCol << ")"
              << "\n";

    if (gpuVsEigenRel > maxCombinedErr) {
      maxCombinedErr = gpuVsEigenRel;
      worstIteration = i;
    }
    maxMetalVsBlas = std::max(maxMetalVsBlas, gpuVsCpuRel);
  }

  std::cout << "\n=== SUMMARY ===\n"
            << "Worst GPU vs Eigen relErr: " << maxCombinedErr << " (iter " << worstIteration << ")\n"
            << "Worst GPU vs CPU relErr:   " << maxMetalVsBlas << "\n";

  // Don't assert — this is purely diagnostic
}
