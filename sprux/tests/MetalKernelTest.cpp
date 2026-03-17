/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include "sprux/sprux/CoalescedBlockMatrix.h"
#include "sprux/sprux/EliminationTree.h"
#include "sprux/sprux/MetalDefs.h"
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

static constexpr float kEps = 1e-4;

// Helper: build a simple 1-lump structure (single diagonal block, size n)
static CoalescedBlockMatrixSkel make1LumpSkel(int64_t n) {
  vector<set<int64_t>> colBlocks{{0}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> spanStart{0, n};
  vector<int64_t> lumpToSpan{0, 1};
  return CoalescedBlockMatrixSkel(spanStart, lumpToSpan, ss.ptrs, ss.inds);
}

// Helper: build a 2-lump structure with off-diagonal coupling
// Block 0: size n0, Block 1: size n1, with block (1,0) filled
static CoalescedBlockMatrixSkel make2LumpSkel(int64_t n0, int64_t n1) {
  vector<set<int64_t>> colBlocks{{0, 1}, {1}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, n0, n0 + n1};
  vector<int64_t> lumpToSpan{0, 1, 2};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  return CoalescedBlockMatrixSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
}

// ============================================================================
// Cholesky kernel tests
// ============================================================================

// Test potrf: Cholesky factorization of diagonal block
TEST(MetalKernel, Potrf) {
  int64_t n = 4;
  auto skel = make1LumpSkel(n);

  // Create SPD matrix
  Matrix<float> A = Matrix<float>::Random(n, n);
  A = A * A.transpose() + n * Matrix<float>::Identity(n, n);

  // Fill data from matrix (row-major in lump)
  vector<float> dataCpu(skel.dataSize()), dataMetal(skel.dataSize());
  for (int64_t r = 0; r < n; r++)
    for (int64_t c = 0; c < n; c++)
      dataCpu[r * n + c] = dataMetal[r * n + c] = A(r, c);

  // CPU reference
  Solver solverCpu(CoalescedBlockMatrixSkel(skel), {}, {}, fastOps());
  NumericCtxPtr<float> cpuCtx = solverCpu.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
  cpuCtx->potrf(n, dataCpu.data(), 0);

  // Metal
  Solver solverMetal(CoalescedBlockMatrixSkel(skel), {}, {}, metalOps());
  NumericCtxPtr<float> metalCtx = solverMetal.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
  {
    MetalMirror<float> dataGpu(dataMetal);
    metalCtx->potrf(n, dataGpu.ptr(), 0);
    metalCtx->flush();  // Sync GPU before reading back (MPS potrf uses deferred commit)
    dataGpu.get(dataMetal);
  }

  // Compare
  float diff = 0;
  for (size_t i = 0; i < dataCpu.size(); i++)
    diff += (dataCpu[i] - dataMetal[i]) * (dataCpu[i] - dataMetal[i]);
  ASSERT_NEAR(sqrt(diff), 0, kEps) << "potrf: Metal vs CPU mismatch";
}

// Test trsm: below-diagonal triangular solve
TEST(MetalKernel, Trsm) {
  int64_t n0 = 3, n1 = 2;
  auto skel = make2LumpSkel(n0, n1);

  // Create SPD data so trsm input is valid after potrf
  vector<float> data(skel.dataSize());
  iota(data.begin(), data.end(), 13);
  skel.damp(data, 5.0f, 50.0f);

  vector<float> dataCpu = data, dataMetal = data;

  // CPU: potrf diagonal, then trsm
  Solver solverCpu(CoalescedBlockMatrixSkel(skel), {}, {}, fastOps());
  NumericCtxPtr<float> cpuCtx = solverCpu.internalSymbolicContext().createNumericCtx<float>(n0 * n1, nullptr);
  cpuCtx->potrf(n0, dataCpu.data(), skel.chainData[0]);
  cpuCtx->trsm(n0, n1, dataCpu.data(), skel.chainData[0], skel.chainData[1]);

  // Metal: same operations
  Solver solverMetal(CoalescedBlockMatrixSkel(skel), {}, {}, metalOps());
  NumericCtxPtr<float> metalCtx = solverMetal.internalSymbolicContext().createNumericCtx<float>(n0 * n1, nullptr);
  {
    MetalMirror<float> dataGpu(dataMetal);
    metalCtx->potrf(n0, dataGpu.ptr(), skel.chainData[0]);
    metalCtx->trsm(n0, n1, dataGpu.ptr(), skel.chainData[0], skel.chainData[1]);
    metalCtx->flush();  // Sync GPU before reading back (MPS ops use deferred commit)
    dataGpu.get(dataMetal);
  }

  float diff = 0;
  for (size_t i = 0; i < dataCpu.size(); i++)
    diff += (dataCpu[i] - dataMetal[i]) * (dataCpu[i] - dataMetal[i]);
  ASSERT_NEAR(sqrt(diff), 0, kEps) << "trsm: Metal vs CPU mismatch";
}

// Test doElimination: factor_lumps + sparse_elim_straight
TEST(MetalKernel, DoElimination) {
  for (int iter = 0; iter < 5; iter++) {
    auto colBlocks = randomCols(115, 0.03, 57 + iter);
    colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<int64_t> paramSize = randomVec(ss.ptrs.size() - 1, 2, 5, 47 + iter);
    EliminationTree et(paramSize, ss);
    et.buildTree();
    et.processTree(true);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel skel(et.computeSpanStart(), et.lumpToSpan, et.colStart, et.rowParam);

    vector<float> data = randomData<float>(skel.dataSize(), -1.0, 1.0, 300 + iter);
    skel.damp(data, 0.0f, float(skel.order() * 1.5));

    ASSERT_GE(et.sparseElimRanges.size(), 2);
    int64_t largestIndep = et.sparseElimRanges[1];

    // CPU reference
    vector<float> dataCpu = data;
    Solver solverCpu(CoalescedBlockMatrixSkel(skel), vector<int64_t>(et.sparseElimRanges), {}, fastOps());
    NumericCtxPtr<float> cpuCtx = solverCpu.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
    cpuCtx->doElimination(solverCpu.internalGetElimCtx(0), dataCpu.data(), 0, largestIndep);

    // Metal
    vector<float> dataMetal = data;
    Solver solverMetal(CoalescedBlockMatrixSkel(skel), vector<int64_t>(et.sparseElimRanges), {}, metalOps());
    NumericCtxPtr<float> metalCtx = solverMetal.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
    {
      MetalMirror<float> dataGpu(dataMetal);
      metalCtx->doElimination(solverMetal.internalGetElimCtx(0), dataGpu.ptr(), 0, largestIndep);
      dataGpu.get(dataMetal);
    }

    // Compare via densification
    Matrix<float> cpuMat = solverCpu.skel().densify(dataCpu);
    Matrix<float> metalMat = solverMetal.skel().densify(dataMetal);
    float diff = Matrix<float>((cpuMat - metalMat).template triangularView<Eigen::Lower>())
                     .leftCols(largestIndep)
                     .norm();
    ASSERT_NEAR(diff, 0, 2e-4) << "doElimination: Metal vs CPU mismatch (iter=" << iter << ")";
  }
}

// ============================================================================
// LU kernel tests
// ============================================================================

// Test getrf: LU factorization with partial pivoting
TEST(MetalKernel, Getrf) {
  int64_t n = 5;
  auto skel = make1LumpSkel(n);

  Matrix<float> A = Matrix<float>::Random(n, n);
  A += n * Matrix<float>::Identity(n, n);

  vector<float> dataCpu(skel.dataSize()), dataMetal(skel.dataSize());
  for (int64_t r = 0; r < n; r++)
    for (int64_t c = 0; c < n; c++)
      dataCpu[r * n + c] = dataMetal[r * n + c] = A(r, c);

  vector<int64_t> pivotsCpu(n), pivotsMetal(n);

  // CPU reference
  Solver solverCpu(CoalescedBlockMatrixSkel(skel), {}, {}, fastOps());
  NumericCtxPtr<float> cpuCtx = solverCpu.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
  cpuCtx->getrf(n, n, dataCpu.data(), 0, pivotsCpu.data());

  // Metal
  Solver solverMetal(CoalescedBlockMatrixSkel(skel), {}, {}, metalOps());
  NumericCtxPtr<float> metalCtx = solverMetal.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
  {
    MetalMirror<float> dataGpu(dataMetal);
    metalCtx->getrf(n, n, dataGpu.ptr(), 0, pivotsMetal.data());
    dataGpu.get(dataMetal);
  }

  // Verify P*A = L*U for both
  auto verifyLU = [&](const vector<float>& data, const vector<int64_t>& pivots, const char* label) {
    Matrix<float> factored(n, n);
    for (int64_t r = 0; r < n; r++)
      for (int64_t c = 0; c < n; c++)
        factored(r, c) = data[r * n + c];

    Matrix<float> L = Matrix<float>::Identity(n, n);
    L.template triangularView<Eigen::StrictlyLower>() =
        factored.template triangularView<Eigen::StrictlyLower>();
    Matrix<float> U = factored.template triangularView<Eigen::Upper>();

    Matrix<float> P = Matrix<float>::Identity(n, n);
    for (int64_t i = 0; i < n; i++)
      if (pivots[i] != i)
        P.row(i).swap(P.row(pivots[i]));

    float error = (P * A - L * U).norm();
    ASSERT_NEAR(error, 0, kEps) << label << ": P*A != L*U";
  };

  verifyLU(dataCpu, pivotsCpu, "CPU getrf");
  verifyLU(dataMetal, pivotsMetal, "Metal getrf");
}

// Test trsmLowerUnit: unit lower triangular solve
TEST(MetalKernel, TrsmLowerUnit) {
  int64_t m = 4, n = 3;
  int64_t totalElems = m * m + m * n;
  auto skel = make1LumpSkel(m);

  // Create unit lower triangular L
  Matrix<float> L = Matrix<float>::Identity(m, m);
  L(1, 0) = 0.5f;
  L(2, 0) = -0.3f;
  L(2, 1) = 0.7f;
  L(3, 0) = 0.2f;
  L(3, 1) = -0.4f;
  L(3, 2) = 0.1f;

  // B is m×n row-major
  Matrix<float> B = Matrix<float>::Random(m, n);

  // Pack: L at offset 0 (m×m), B at offset m*m (m×n), both row-major
  vector<float> dataCpu(totalElems), dataMetal(totalElems);
  for (int64_t r = 0; r < m; r++)
    for (int64_t c = 0; c < m; c++)
      dataCpu[r * m + c] = dataMetal[r * m + c] = L(r, c);
  int64_t offB = m * m;
  for (int64_t r = 0; r < m; r++)
    for (int64_t c = 0; c < n; c++)
      dataCpu[offB + r * n + c] = dataMetal[offB + r * n + c] = B(r, c);

  // CPU reference
  Solver solverCpu(CoalescedBlockMatrixSkel(skel), {}, {}, fastOps());
  NumericCtxPtr<float> cpuCtx = solverCpu.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
  cpuCtx->trsmLowerUnit(m, n, dataCpu.data(), 0, dataCpu.data(), offB, n);

  // Metal
  Solver solverMetal(CoalescedBlockMatrixSkel(skel), {}, {}, metalOps());
  NumericCtxPtr<float> metalCtx = solverMetal.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
  {
    MetalMirror<float> dataGpu(dataMetal);
    metalCtx->trsmLowerUnit(m, n, dataGpu.ptr(), 0, dataGpu.ptr(), offB, n);
    metalCtx->flush();
    dataGpu.get(dataMetal);
  }

  float diff = 0;
  for (int64_t i = offB; i < (int64_t)dataCpu.size(); i++)
    diff += (dataCpu[i] - dataMetal[i]) * (dataCpu[i] - dataMetal[i]);
  ASSERT_NEAR(sqrt(diff), 0, kEps) << "trsmLowerUnit: Metal vs CPU mismatch";
}

// Test trsmUpperRight: right upper triangular solve
TEST(MetalKernel, TrsmUpperRight) {
  int64_t m = 3, n = 4;
  int64_t totalElems = n * n + m * n;
  auto skel = make1LumpSkel(n);

  // Create upper triangular U (n×n)
  Matrix<float> U = Matrix<float>::Zero(n, n);
  for (int64_t r = 0; r < n; r++) {
    U(r, r) = 5.0f + r;
    for (int64_t c = r + 1; c < n; c++)
      U(r, c) = 0.5f * (c - r);
  }

  // B is m×n row-major
  Matrix<float> B = Matrix<float>::Random(m, n);

  vector<float> dataCpu(totalElems), dataMetal(totalElems);
  for (int64_t r = 0; r < n; r++)
    for (int64_t c = 0; c < n; c++)
      dataCpu[r * n + c] = dataMetal[r * n + c] = U(r, c);
  int64_t offB = n * n;
  for (int64_t r = 0; r < m; r++)
    for (int64_t c = 0; c < n; c++)
      dataCpu[offB + r * n + c] = dataMetal[offB + r * n + c] = B(r, c);

  // CPU reference
  Solver solverCpu(CoalescedBlockMatrixSkel(skel), {}, {}, fastOps());
  NumericCtxPtr<float> cpuCtx = solverCpu.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
  cpuCtx->trsmUpperRight(m, n, dataCpu.data(), 0, dataCpu.data(), offB, n);

  // Metal
  Solver solverMetal(CoalescedBlockMatrixSkel(skel), {}, {}, metalOps());
  NumericCtxPtr<float> metalCtx = solverMetal.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
  {
    MetalMirror<float> dataGpu(dataMetal);
    metalCtx->trsmUpperRight(m, n, dataGpu.ptr(), 0, dataGpu.ptr(), offB, n);
    metalCtx->flush();
    dataGpu.get(dataMetal);
  }

  float diff = 0;
  for (int64_t i = offB; i < (int64_t)dataCpu.size(); i++)
    diff += (dataCpu[i] - dataMetal[i]) * (dataCpu[i] - dataMetal[i]);
  ASSERT_NEAR(sqrt(diff), 0, kEps) << "trsmUpperRight: Metal vs CPU mismatch";
}

// Test saveGemm: C -= L * U
TEST(MetalKernel, SaveGemm) {
  int64_t m = 3, n = 4, k = 2;
  // L is m×k, U is k×n, C is m×n, all row-major with same stride
  int64_t stride = max({m, n, k});
  stride = n;  // use n as stride for simplicity
  int64_t totalElems = (m + k + m) * stride;
  auto skel = make1LumpSkel(4);  // dummy skel

  vector<float> dataCpu(totalElems), dataMetal(totalElems);
  mt19937 rng(42);
  uniform_real_distribution<float> unif(-1.0f, 1.0f);
  for (auto& v : dataCpu) v = unif(rng);
  dataMetal = dataCpu;

  int64_t offL = 0, ldL = stride;
  int64_t offU = m * stride, ldU = stride;
  int64_t offC = (m + k) * stride, ldC = stride;

  // CPU reference
  Solver solverCpu(CoalescedBlockMatrixSkel(skel), {}, {}, fastOps());
  NumericCtxPtr<float> cpuCtx = solverCpu.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
  cpuCtx->saveGemm(m, n, k, dataCpu.data(), offL, ldL, dataCpu.data(), offU, ldU,
                    dataCpu.data(), offC, ldC);

  // Metal
  Solver solverMetal(CoalescedBlockMatrixSkel(skel), {}, {}, metalOps());
  NumericCtxPtr<float> metalCtx = solverMetal.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
  {
    MetalMirror<float> dataGpu(dataMetal);
    metalCtx->saveGemm(m, n, k, dataGpu.ptr(), offL, ldL, dataGpu.ptr(), offU, ldU,
                        dataGpu.ptr(), offC, ldC);
    metalCtx->flush();
    dataGpu.get(dataMetal);
  }

  float diff = 0;
  for (int64_t i = offC; i < offC + m * ldC; i++)
    diff += (dataCpu[i] - dataMetal[i]) * (dataCpu[i] - dataMetal[i]);
  ASSERT_NEAR(sqrt(diff), 0, kEps) << "saveGemm: Metal vs CPU mismatch";
}

// Test applyRowPerm: row permutation on matrix columns
TEST(MetalKernel, ApplyRowPerm) {
  int64_t n = 4;
  int64_t numCols = 3;
  auto skel = make1LumpSkel(n);

  // Create LU-factored data + extra columns for perm test
  vector<float> dataCpu(n * numCols), dataMetal(n * numCols);
  mt19937 rng(77);
  uniform_real_distribution<float> unif(-1.0f, 1.0f);
  for (auto& v : dataCpu) v = unif(rng);
  dataMetal = dataCpu;

  vector<int64_t> pivots = {2, 3, 2, 3};  // some permutation

  // CPU reference
  Solver solverCpu(CoalescedBlockMatrixSkel(skel), {}, {}, fastOps());
  NumericCtxPtr<float> cpuCtx = solverCpu.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
  cpuCtx->applyRowPerm(pivots.data(), n, dataCpu.data(), 0, n, numCols);

  // Metal
  Solver solverMetal(CoalescedBlockMatrixSkel(skel), {}, {}, metalOps());
  NumericCtxPtr<float> metalCtx = solverMetal.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
  {
    MetalMirror<float> dataGpu(dataMetal);
    metalCtx->applyRowPerm(pivots.data(), n, dataGpu.ptr(), 0, n, numCols);
    metalCtx->flush();
    dataGpu.get(dataMetal);
  }

  float diff = 0;
  for (size_t i = 0; i < dataCpu.size(); i++)
    diff += (dataCpu[i] - dataMetal[i]) * (dataCpu[i] - dataMetal[i]);
  ASSERT_NEAR(sqrt(diff), 0, kEps) << "applyRowPerm: Metal vs CPU mismatch";
}

// ============================================================================
// End-to-end single-kernel integration: LU factor + solve on 2-block system
// ============================================================================

TEST(MetalKernel, LUFactorSolve2Block) {
  int64_t n0 = 3, n1 = 2;
  auto skel = make2LumpSkel(n0, n1);
  skel.initUpperTriangle();
  int64_t order = skel.order();

  // Create non-symmetric diagonally-dominant matrix
  Matrix<float> fullMat = Matrix<float>::Zero(order, order);
  fullMat.block(0, 0, n0, n0) << 10, 1, 2, 1, 11, 1, 2, 1, 12;
  fullMat.block(n0, n0, n1, n1) << 8, 1, 1, 9;
  fullMat.block(n0, 0, n1, n0) << 1, 2, 1, 2, 1, 2;
  fullMat.block(0, n0, n0, n1) << 3, 1, 1, 2, 2, 1;

  // Fill block data
  auto fillData = [&](const CoalescedBlockMatrixSkel& s, vector<float>& data) {
    int64_t numLumps = s.numLumps();
    for (int64_t l = 0; l < numLumps; l++) {
      int64_t chainStart = s.chainColPtr[l];
      int64_t chainEnd = s.chainColPtr[l + 1];
      int64_t ls = s.lumpStart[l];
      int64_t lsz = s.lumpStart[l + 1] - ls;
      for (int64_t c = chainStart; c < chainEnd; c++) {
        int64_t rowSpan = s.chainRowSpan[c];
        int64_t rowStart = s.spanStart[rowSpan];
        int64_t rowSize = s.spanStart[rowSpan + 1] - rowStart;
        int64_t dataOffset = s.chainData[c];
        for (int64_t r = 0; r < rowSize; r++)
          for (int64_t col = 0; col < lsz; col++)
            data[dataOffset + r * lsz + col] = fullMat(rowStart + r, ls + col);
      }
    }
    int64_t upperDataBase = s.dataSize();
    for (int64_t l = 0; l < numLumps; l++) {
      int64_t upperRowStart = s.upperChainRowPtr[l];
      int64_t upperRowEnd = s.upperChainRowPtr[l + 1];
      int64_t lsIdx = s.lumpStart[l];
      int64_t lsz = s.lumpStart[l + 1] - lsIdx;
      for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
        int64_t colSpan = s.upperChainColSpan[i];
        int64_t colStart = s.spanStart[colSpan];
        int64_t colSize = s.spanStart[colSpan + 1] - colStart;
        int64_t upperDataOffset = upperDataBase + s.upperChainData[i];
        for (int64_t r = 0; r < lsz; r++)
          for (int64_t c = 0; c < colSize; c++)
            data[upperDataOffset + r * colSize + c] = fullMat(lsIdx + r, colStart + c);
      }
    }
  };

  vector<float> dataMetal(skel.totalDataSize());
  fillData(skel, dataMetal);

  Solver solverMetal(std::move(skel), {}, {}, metalOps());
  vector<int64_t> pivots(order);

  {
    MetalMirror<float> dataGpu(dataMetal);
    solverMetal.factorLU(dataGpu.ptr(), pivots.data());
    dataGpu.get(dataMetal);
  }

  Vector<float> b = Vector<float>::Ones(order);
  Vector<float> x = b;
  {
    MetalMirror<float> dataGpu(dataMetal);
    MetalMirror<float> xGpu(vector<float>(x.data(), x.data() + order));
    solverMetal.solveLU(dataGpu.ptr(), pivots.data(), xGpu.ptr(), order, 1);
    vector<float> xVec(order);
    xGpu.get(xVec);
    for (int64_t i = 0; i < order; i++) x(i) = xVec[i];
  }

  float residual = (fullMat * x - b).norm() / b.norm();
  ASSERT_LT(residual, 5e-3) << "2-block LU solve residual too large";
}

// ============================================================================
// MPS potrf/trsm threshold-crossing tests
// ============================================================================

// Test potrf at various sizes that cross the MPS threshold boundary (default 32)
class MetalPotrfSizeTest : public ::testing::TestWithParam<int64_t> {};

TEST_P(MetalPotrfSizeTest, PotrfVsCpuReference) {
  int64_t n = GetParam();
  auto skel = make1LumpSkel(n);

  // Create SPD matrix: A = R^T * R + n*I
  mt19937 rng(42 + n);
  uniform_real_distribution<float> unif(-1.0f, 1.0f);

  vector<float> dataCpu(skel.dataSize()), dataMetal(skel.dataSize());
  // Fill with random values, then make SPD via damping
  for (size_t i = 0; i < dataCpu.size(); i++) {
    float v = unif(rng);
    dataCpu[i] = dataMetal[i] = v;
  }

  // Make SPD: set diagonal to be dominant
  for (int64_t i = 0; i < n; i++) {
    dataCpu[i * n + i] = dataMetal[i * n + i] = float(n) * 2.0f + 1.0f;
  }

  // CPU reference
  Solver solverCpu(CoalescedBlockMatrixSkel(skel), {}, {}, fastOps());
  NumericCtxPtr<float> cpuCtx =
      solverCpu.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
  cpuCtx->potrf(n, dataCpu.data(), 0);

  // Metal
  Solver solverMetal(CoalescedBlockMatrixSkel(skel), {}, {}, metalOps());
  NumericCtxPtr<float> metalCtx =
      solverMetal.internalSymbolicContext().createNumericCtx<float>(0, nullptr);
  {
    MetalMirror<float> dataGpu(dataMetal);
    metalCtx->potrf(n, dataGpu.ptr(), 0);
    metalCtx->flush();  // Sync GPU before reading back (MPS potrf uses deferred commit)
    dataGpu.get(dataMetal);
  }

  // Compare lower triangle only (Cholesky output)
  float diff = 0;
  for (int64_t r = 0; r < n; r++)
    for (int64_t c = 0; c <= r; c++) {
      float d = dataCpu[r * n + c] - dataMetal[r * n + c];
      diff += d * d;
    }
  ASSERT_NEAR(sqrt(diff), 0, kEps * n)
      << "potrf (n=" << n << "): Metal vs CPU mismatch";
}

INSTANTIATE_TEST_SUITE_P(MetalKernel, MetalPotrfSizeTest,
                         ::testing::Values(1, 4, 16, 31, 32, 33, 64, 128));

// Test trsm at various sizes that cross the MPS threshold boundary
class MetalTrsmSizeTest : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {};

TEST_P(MetalTrsmSizeTest, TrsmVsCpuReference) {
  auto [n, k] = GetParam();

  auto skel = make2LumpSkel(n, k);

  // Random data in [-1,1] with damping proportional to order (same as factor tests)
  vector<float> data = randomData<float>(skel.dataSize(), -1.0f, 1.0f, 77 + n);
  skel.damp(data, 0.0f, float(skel.order()) * 1.5f);

  vector<float> dataCpu = data, dataMetal = data;

  // CPU: potrf diagonal, then trsm
  Solver solverCpu(CoalescedBlockMatrixSkel(skel), {}, {}, fastOps());
  NumericCtxPtr<float> cpuCtx =
      solverCpu.internalSymbolicContext().createNumericCtx<float>(n * k, nullptr);
  cpuCtx->potrf(n, dataCpu.data(), skel.chainData[0]);
  cpuCtx->trsm(n, k, dataCpu.data(), skel.chainData[0], skel.chainData[1]);

  // Metal: same operations
  Solver solverMetal(CoalescedBlockMatrixSkel(skel), {}, {}, metalOps());
  NumericCtxPtr<float> metalCtx =
      solverMetal.internalSymbolicContext().createNumericCtx<float>(n * k, nullptr);
  {
    MetalMirror<float> dataGpu(dataMetal);
    metalCtx->potrf(n, dataGpu.ptr(), skel.chainData[0]);
    metalCtx->trsm(n, k, dataGpu.ptr(), skel.chainData[0], skel.chainData[1]);
    metalCtx->flush();  // Sync GPU before reading back (MPS ops use deferred commit)
    dataGpu.get(dataMetal);
  }

  // Compare the B region (after potrf+trsm)
  int64_t offB = skel.chainData[1];
  float diff = 0;
  for (int64_t i = offB; i < offB + k * n; i++) {
    float d = dataCpu[i] - dataMetal[i];
    diff += d * d;
  }
  float tolerance = kEps * max(n, k);
  ASSERT_NEAR(sqrt(diff), 0, tolerance)
      << "trsm (n=" << n << ", k=" << k << "): Metal vs CPU mismatch";
}

INSTANTIATE_TEST_SUITE_P(MetalKernel, MetalTrsmSizeTest,
                         ::testing::Values(std::make_tuple(2, 1),
                                           std::make_tuple(4, 2),
                                           std::make_tuple(16, 8),
                                           std::make_tuple(32, 16),
                                           std::make_tuple(64, 32),
                                           std::make_tuple(64, 64),
                                           std::make_tuple(128, 64)));
