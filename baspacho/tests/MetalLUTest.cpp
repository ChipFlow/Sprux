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

template <typename T>
using Vector = Eigen::Vector<T, Eigen::Dynamic>;

// Metal only supports float precision
template <typename T>
struct Epsilon;
template <>
struct Epsilon<float> {
  static constexpr float value = 1e-4;
  static constexpr float value2 = 5e-3;
};

// Test LU factorization on a simple dense block (single lump)
TEST(MetalLU, FactorSimple_float) {
  // Single block of size 4
  vector<set<int64_t>> colBlocks{{0}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> spanStart{0, 4};
  vector<int64_t> lumpToSpan{0, 1};
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, ss.ptrs, ss.inds);

  int64_t n = 4;
  vector<float> data(factorSkel.dataSize());

  // Non-symmetric diagonally-dominant test matrix
  Matrix<float> testMat(n, n);
  testMat << 4, 1, 2, 1,  //
      1, 5, 1, 2,         //
      2, 1, 6, 1,         //
      1, 2, 1, 7;

  for (int64_t row = 0; row < n; row++)
    for (int64_t col = 0; col < n; col++)
      data[row * n + col] = testMat(row, col);

  Solver solver(std::move(factorSkel), {}, {}, metalOps());
  vector<int64_t> pivots(solver.skel().order());

  // Factor on Metal shared memory
  {
    MetalMirror<float> dataGpu(data);
    solver.factorLU(dataGpu.ptr(), pivots.data());
    dataGpu.get(data);
  }

  // Extract L and U, verify P*A = L*U
  Matrix<float> factored(n, n);
  for (int64_t row = 0; row < n; row++)
    for (int64_t col = 0; col < n; col++)
      factored(row, col) = data[row * n + col];

  Matrix<float> L = Matrix<float>::Identity(n, n);
  L.template triangularView<Eigen::StrictlyLower>() =
      factored.template triangularView<Eigen::StrictlyLower>();
  Matrix<float> U = factored.template triangularView<Eigen::Upper>();

  Matrix<float> P = Matrix<float>::Identity(n, n);
  for (int64_t i = 0; i < n; i++) {
    if (pivots[i] != i) {
      P.row(i).swap(P.row(pivots[i]));
    }
  }

  float error = (P * testMat - L * U).norm();
  ASSERT_NEAR(error, 0, Epsilon<float>::value2) << "LU factorization error: P*A != L*U";
}

// Test LU solve on a simple system
TEST(MetalLU, SolveSimple_float) {
  vector<set<int64_t>> colBlocks{{0}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> spanStart{0, 4};
  vector<int64_t> lumpToSpan{0, 1};
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, ss.ptrs, ss.inds);

  int64_t n = 4;
  vector<float> data(factorSkel.dataSize());

  Matrix<float> A(n, n);
  A << 4, 1, 2, 1,  //
      1, 5, 1, 2,   //
      2, 1, 6, 1,   //
      1, 2, 1, 7;

  for (int64_t row = 0; row < n; row++)
    for (int64_t col = 0; col < n; col++)
      data[row * n + col] = A(row, col);

  Vector<float> b(n);
  b << 1, 2, 3, 4;

  Vector<float> xRef = A.partialPivLu().solve(b);

  Solver solver(std::move(factorSkel), {}, {}, metalOps());
  vector<int64_t> pivots(solver.skel().order());

  // Factor and solve on Metal shared memory
  {
    MetalMirror<float> dataGpu(data);
    solver.factorLU(dataGpu.ptr(), pivots.data());
    dataGpu.get(data);
  }

  Vector<float> x = b;
  {
    MetalMirror<float> dataGpu(data);
    MetalMirror<float> xGpu(vector<float>(x.data(), x.data() + n));
    solver.solveLU(dataGpu.ptr(), pivots.data(), xGpu.ptr(), n, 1);
    vector<float> xVec(n);
    xGpu.get(xVec);
    for (int64_t i = 0; i < n; i++) x(i) = xVec[i];
  }

  float residual = (A * x - b).norm();
  ASSERT_NEAR(residual, 0, Epsilon<float>::value2) << "Residual error: A*x != b";
}

// Test LU on 2-block sparse matrix
TEST(MetalLU, BlockSparse_float) {
  vector<set<int64_t>> colBlocks{{0, 1}, {1}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 3, 5};
  vector<int64_t> lumpToSpan{0, 1, 2};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  factorSkel.initUpperTriangle();

  int64_t order = factorSkel.order();

  // Diagonally-dominant non-symmetric matrix
  Matrix<float> fullMat = Matrix<float>::Zero(order, order);
  fullMat.block(0, 0, 3, 3) << 10, 1, 2,  //
      1, 11, 1,                            //
      2, 1, 12;
  fullMat.block(3, 3, 2, 2) << 8, 1,  //
      1, 9;
  fullMat.block(3, 0, 2, 3) << 1, 2, 1,  //
      2, 1, 2;
  fullMat.block(0, 3, 3, 2) = fullMat.block(3, 0, 2, 3).transpose();

  // Fill block data
  vector<float> data(factorSkel.totalDataSize());

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

      for (int64_t r = 0; r < rowSize; r++)
        for (int64_t col = 0; col < lumpSize; col++)
          data[dataOffset + r * lumpSize + col] = fullMat(rowStart + r, lumpStart + col);
    }
  }

  // Fill upper triangle
  int64_t upperDataBase = factorSkel.dataSize();
  for (int64_t l = 0; l < numLumps; l++) {
    int64_t upperRowStart = factorSkel.upperChainRowPtr[l];
    int64_t upperRowEnd = factorSkel.upperChainRowPtr[l + 1];
    int64_t lumpStartIdx = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStartIdx;

    for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
      int64_t colSpan = factorSkel.upperChainColSpan[i];
      int64_t colStart = factorSkel.spanStart[colSpan];
      int64_t colSize = factorSkel.spanStart[colSpan + 1] - colStart;
      int64_t upperDataOffset = upperDataBase + factorSkel.upperChainData[i];

      for (int64_t r = 0; r < lumpSize; r++)
        for (int64_t c = 0; c < colSize; c++)
          data[upperDataOffset + r * colSize + c] = fullMat(lumpStartIdx + r, colStart + c);
    }
  }

  Solver solver(std::move(factorSkel), {}, {}, metalOps());
  vector<int64_t> pivots(order);

  // Factor on Metal
  {
    MetalMirror<float> dataGpu(data);
    solver.factorLU(dataGpu.ptr(), pivots.data());
    dataGpu.get(data);
  }

  // Solve
  Vector<float> b = Vector<float>::Ones(order);
  Vector<float> x = b;

  {
    MetalMirror<float> dataGpu(data);
    MetalMirror<float> xGpu(vector<float>(x.data(), x.data() + order));
    solver.solveLU(dataGpu.ptr(), pivots.data(), xGpu.ptr(), order, 1);
    vector<float> xVec(order);
    xGpu.get(xVec);
    for (int64_t i = 0; i < order; i++) x(i) = xVec[i];
  }

  float residual = (fullMat * x - b).norm() / b.norm();
  ASSERT_LT(residual, Epsilon<float>::value2) << "Block-sparse LU solve residual too large";
}

// Test with non-symmetric matrix (asymmetric coupling, SPICE-like)
TEST(MetalLU, NonSymmetric_float) {
  vector<set<int64_t>> colBlocks{{0, 1}, {1}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 3, 5};
  vector<int64_t> lumpToSpan{0, 1, 2};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  factorSkel.initUpperTriangle();

  int64_t order = factorSkel.order();

  // Truly non-symmetric matrix (asymmetric off-diagonal blocks)
  Matrix<float> fullMat = Matrix<float>::Zero(order, order);
  fullMat.block(0, 0, 3, 3) << 10, 1, 2,  //
      3, 11, 1,                            //
      1, 2, 12;
  fullMat.block(3, 3, 2, 2) << 8, 3,  //
      1, 9;
  // Asymmetric off-diagonal (lower != upper^T)
  fullMat.block(3, 0, 2, 3) << 1, 2, 1,  //
      2, 1, 3;
  fullMat.block(0, 3, 3, 2) << 3, 1,  //
      1, 2,                            //
      2, 1;

  vector<float> data(factorSkel.totalDataSize());

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

      for (int64_t r = 0; r < rowSize; r++)
        for (int64_t col = 0; col < lumpSize; col++)
          data[dataOffset + r * lumpSize + col] = fullMat(rowStart + r, lumpStart + col);
    }
  }

  int64_t upperDataBase = factorSkel.dataSize();
  for (int64_t l = 0; l < numLumps; l++) {
    int64_t upperRowStart = factorSkel.upperChainRowPtr[l];
    int64_t upperRowEnd = factorSkel.upperChainRowPtr[l + 1];
    int64_t lumpStartIdx = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStartIdx;

    for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
      int64_t colSpan = factorSkel.upperChainColSpan[i];
      int64_t colStart = factorSkel.spanStart[colSpan];
      int64_t colSize = factorSkel.spanStart[colSpan + 1] - colStart;
      int64_t upperDataOffset = upperDataBase + factorSkel.upperChainData[i];

      for (int64_t r = 0; r < lumpSize; r++)
        for (int64_t c = 0; c < colSize; c++)
          data[upperDataOffset + r * colSize + c] = fullMat(lumpStartIdx + r, colStart + c);
    }
  }

  Solver solver(std::move(factorSkel), {}, {}, metalOps());
  vector<int64_t> pivots(order);

  {
    MetalMirror<float> dataGpu(data);
    solver.factorLU(dataGpu.ptr(), pivots.data());
    dataGpu.get(data);
  }

  Vector<float> b = Vector<float>::Ones(order);
  Vector<float> x = b;

  {
    MetalMirror<float> dataGpu(data);
    MetalMirror<float> xGpu(vector<float>(x.data(), x.data() + order));
    solver.solveLU(dataGpu.ptr(), pivots.data(), xGpu.ptr(), order, 1);
    vector<float> xVec(order);
    xGpu.get(xVec);
    for (int64_t i = 0; i < order; i++) x(i) = xVec[i];
  }

  float residual = (fullMat * x - b).norm() / b.norm();
  ASSERT_LT(residual, Epsilon<float>::value2) << "Non-symmetric LU solve residual too large";

  // Compare with Eigen reference
  Vector<float> xRef = fullMat.partialPivLu().solve(b);
  float refResidual = (fullMat * xRef - b).norm() / b.norm();
  // Our residual should be in same ballpark as Eigen
  ASSERT_LT(residual, std::max(refResidual * 100.0f, Epsilon<float>::value2));
}

// Verify Metal matches CPU BackendFast for LU on a larger block-sparse matrix
TEST(MetalLU, VsCpuReference_float) {
  // 4-block structure
  vector<set<int64_t>> colBlocks{{0, 1, 2}, {1, 2, 3}, {2, 3}, {3}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> paramSize{2, 2, 2, 2};
  int64_t totalSize = 8;

  vector<int64_t> spanStart;
  spanStart.push_back(0);
  for (int64_t ps : paramSize) spanStart.push_back(spanStart.back() + ps);

  vector<int64_t> lumpToSpan(paramSize.size() + 1);
  iota(lumpToSpan.begin(), lumpToSpan.end(), 0);
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkelMetal(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  factorSkelMetal.initUpperTriangle();

  // Build non-symmetric diagonally-dominant matrix
  mt19937 rng(42);
  uniform_real_distribution<float> unif(-1.0f, 1.0f);
  Matrix<float> fullMat = Matrix<float>::Zero(totalSize, totalSize);

  for (int64_t rowBlock = 0; rowBlock < (int64_t)ss.ptrs.size() - 1; rowBlock++) {
    for (int64_t k = ss.ptrs[rowBlock]; k < ss.ptrs[rowBlock + 1]; k++) {
      int64_t colBlock = ss.inds[k];
      for (int64_t r = spanStart[rowBlock]; r < spanStart[rowBlock + 1]; r++)
        for (int64_t c = spanStart[colBlock]; c < spanStart[colBlock + 1]; c++)
          fullMat(r, c) = unif(rng);
    }
  }
  // Make NON-symmetric: generate separate upper triangle entries
  for (int64_t rowBlock = 0; rowBlock < (int64_t)ss.ptrs.size() - 1; rowBlock++) {
    for (int64_t k = ss.ptrs[rowBlock]; k < ss.ptrs[rowBlock + 1]; k++) {
      int64_t colBlock = ss.inds[k];
      if (colBlock != rowBlock) {
        // Fill upper (colBlock, rowBlock) separately
        for (int64_t r = spanStart[colBlock]; r < spanStart[colBlock + 1]; r++)
          for (int64_t c = spanStart[rowBlock]; c < spanStart[rowBlock + 1]; c++)
            fullMat(r, c) = unif(rng);
      }
    }
  }
  for (int64_t i = 0; i < totalSize; i++) fullMat(i, i) += totalSize * 4;

  // Helper lambda to fill data from fullMat
  auto fillData = [&](const CoalescedBlockMatrixSkel& skel, vector<float>& data) {
    int64_t numLumps = skel.numLumps();
    for (int64_t l = 0; l < numLumps; l++) {
      int64_t chainStart = skel.chainColPtr[l];
      int64_t chainEnd = skel.chainColPtr[l + 1];
      int64_t lumpStart = skel.lumpStart[l];
      int64_t lumpSize = skel.lumpStart[l + 1] - lumpStart;
      for (int64_t c = chainStart; c < chainEnd; c++) {
        int64_t rowSpan = skel.chainRowSpan[c];
        int64_t rowStart = skel.spanStart[rowSpan];
        int64_t rowSize = skel.spanStart[rowSpan + 1] - rowStart;
        int64_t dataOffset = skel.chainData[c];
        for (int64_t r = 0; r < rowSize; r++)
          for (int64_t col = 0; col < lumpSize; col++)
            data[dataOffset + r * lumpSize + col] = fullMat(rowStart + r, lumpStart + col);
      }
    }
    int64_t upperDataBase = skel.dataSize();
    for (int64_t l = 0; l < numLumps; l++) {
      int64_t upperRowStart = skel.upperChainRowPtr[l];
      int64_t upperRowEnd = skel.upperChainRowPtr[l + 1];
      int64_t lumpStartIdx = skel.lumpStart[l];
      int64_t lumpSize = skel.lumpStart[l + 1] - lumpStartIdx;
      for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
        int64_t colSpan = skel.upperChainColSpan[i];
        int64_t colStart = skel.spanStart[colSpan];
        int64_t colSize = skel.spanStart[colSpan + 1] - colStart;
        int64_t upperDataOffset = upperDataBase + skel.upperChainData[i];
        for (int64_t r = 0; r < lumpSize; r++)
          for (int64_t c = 0; c < colSize; c++)
            data[upperDataOffset + r * colSize + c] = fullMat(lumpStartIdx + r, colStart + c);
      }
    }
  };

  // Solve with CPU (BackendFast)
  CoalescedBlockMatrixSkel factorSkelCpu(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  factorSkelCpu.initUpperTriangle();
  vector<float> dataCpu(factorSkelCpu.totalDataSize());
  fillData(factorSkelCpu, dataCpu);

  Solver solverCpu(std::move(factorSkelCpu), {}, {}, fastOps());
  vector<int64_t> pivotsCpu(totalSize);
  solverCpu.factorLU(dataCpu.data(), pivotsCpu.data());

  Vector<float> b(totalSize);
  for (int64_t i = 0; i < totalSize; i++) b(i) = unif(rng);

  Vector<float> xCpu = b;
  solverCpu.solveLU(dataCpu.data(), pivotsCpu.data(), xCpu.data(), totalSize, 1);
  float cpuResidual = (fullMat * xCpu - b).norm() / b.norm();

  // Solve with Metal
  vector<float> dataMetal(factorSkelMetal.totalDataSize());
  fillData(factorSkelMetal, dataMetal);

  Solver solverMetal(std::move(factorSkelMetal), {}, {}, metalOps());
  vector<int64_t> pivotsMetal(totalSize);

  {
    MetalMirror<float> dataGpu(dataMetal);
    solverMetal.factorLU(dataGpu.ptr(), pivotsMetal.data());
    dataGpu.get(dataMetal);
  }

  Vector<float> xMetal = b;
  {
    MetalMirror<float> dataGpu(dataMetal);
    MetalMirror<float> xGpu(vector<float>(xMetal.data(), xMetal.data() + totalSize));
    solverMetal.solveLU(dataGpu.ptr(), pivotsMetal.data(), xGpu.ptr(), totalSize, 1);
    vector<float> xVec(totalSize);
    xGpu.get(xVec);
    for (int64_t i = 0; i < totalSize; i++) xMetal(i) = xVec[i];
  }

  float metalResidual = (fullMat * xMetal - b).norm() / b.norm();

  cout << "CPU residual: " << cpuResidual << ", Metal residual: " << metalResidual << endl;

  // Both should have small residuals
  ASSERT_LT(cpuResidual, Epsilon<float>::value2) << "CPU LU residual too large";
  ASSERT_LT(metalResidual, Epsilon<float>::value2) << "Metal LU residual too large";
}
