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
#include "sprux/sprux/CudaDefs.h"
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
  static constexpr float value2 = 5e-3;
};

// Helper: fill Sprux block data from dense matrix (lower + upper triangle)
template <typename T>
void fillBlockData(const CoalescedBlockMatrixSkel& skel, T* data, const Matrix<T>& fullMat) {
  int64_t numLumps = skel.numLumps();

  // Fill lower triangle (includes diagonal blocks)
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

  // Fill upper triangle
  if (!skel.upperChainData.empty()) {
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
  }
}

// Test LU factorization on a simple dense block (single lump)
template <typename T>
void testFactorSimple() {
  vector<set<int64_t>> colBlocks{{0}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> spanStart{0, 4};
  vector<int64_t> lumpToSpan{0, 1};
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, ss.ptrs, ss.inds);

  int64_t n = 4;
  vector<T> data(factorSkel.dataSize());

  Matrix<T> testMat(n, n);
  testMat << 4, 1, 2, 1,  //
      1, 5, 1, 2,         //
      2, 1, 6, 1,         //
      1, 2, 1, 7;

  for (int64_t row = 0; row < n; row++)
    for (int64_t col = 0; col < n; col++) data[row * n + col] = testMat(row, col);

  Solver solver(std::move(factorSkel), {}, {}, cudaOps());
  vector<int64_t> pivots(solver.skel().order());

  {
    DevMirror<T> dataGpu(data);
    solver.factorLU(dataGpu.ptr, pivots.data());
    dataGpu.get(data);
  }

  // Extract L and U, verify P*A = L*U
  Matrix<T> factored(n, n);
  for (int64_t row = 0; row < n; row++)
    for (int64_t col = 0; col < n; col++) factored(row, col) = data[row * n + col];

  Matrix<T> L = Matrix<T>::Identity(n, n);
  L.template triangularView<Eigen::StrictlyLower>() =
      factored.template triangularView<Eigen::StrictlyLower>();
  Matrix<T> U = factored.template triangularView<Eigen::Upper>();

  Matrix<T> P = Matrix<T>::Identity(n, n);
  for (int64_t i = 0; i < n; i++) {
    if (pivots[i] != i) P.row(i).swap(P.row(pivots[i]));
  }

  T error = (P * testMat - L * U).norm();
  ASSERT_NEAR(error, 0, Epsilon<T>::value2) << "LU factorization error: P*A != L*U";
}

TEST(CudaLU, FactorSimple_double) { testFactorSimple<double>(); }
TEST(CudaLU, FactorSimple_float) { testFactorSimple<float>(); }

// Test LU solve on a simple system
template <typename T>
void testSolveSimple() {
  vector<set<int64_t>> colBlocks{{0}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> spanStart{0, 4};
  vector<int64_t> lumpToSpan{0, 1};
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, ss.ptrs, ss.inds);

  int64_t n = 4;
  vector<T> data(factorSkel.dataSize());

  Matrix<T> A(n, n);
  A << 4, 1, 2, 1,  //
      1, 5, 1, 2,   //
      2, 1, 6, 1,   //
      1, 2, 1, 7;

  for (int64_t row = 0; row < n; row++)
    for (int64_t col = 0; col < n; col++) data[row * n + col] = A(row, col);

  Vector<T> b(n);
  b << 1, 2, 3, 4;

  Solver solver(std::move(factorSkel), {}, {}, cudaOps());
  vector<int64_t> pivots(solver.skel().order());

  {
    DevMirror<T> dataGpu(data);
    solver.factorLU(dataGpu.ptr, pivots.data());
    dataGpu.get(data);
  }

  Vector<T> x = b;
  {
    DevMirror<T> dataGpu(data);
    DevMirror<T> xGpu(vector<T>(x.data(), x.data() + n));
    solver.solveLU(dataGpu.ptr, pivots.data(), xGpu.ptr, n, 1);
    vector<T> xVec(n);
    xGpu.get(xVec);
    for (int64_t i = 0; i < n; i++) x(i) = xVec[i];
  }

  T residual = (A * x - b).norm();
  ASSERT_NEAR(residual, 0, Epsilon<T>::value2) << "Residual error: A*x != b";
}

TEST(CudaLU, SolveSimple_double) { testSolveSimple<double>(); }
TEST(CudaLU, SolveSimple_float) { testSolveSimple<float>(); }

// Test LU on 2-block sparse matrix
template <typename T>
void testBlockSparse() {
  vector<set<int64_t>> colBlocks{{0, 1}, {1}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 3, 5};
  vector<int64_t> lumpToSpan{0, 1, 2};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  factorSkel.initUpperTriangle();

  int64_t order = factorSkel.order();

  // Non-symmetric diagonally-dominant matrix
  Matrix<T> fullMat = Matrix<T>::Zero(order, order);
  fullMat.block(0, 0, 3, 3) << 10, 1, 2,  //
      3, 11, 1,                            //
      1, 2, 12;
  fullMat.block(3, 3, 2, 2) << 8, 3,  //
      1, 9;
  // Asymmetric off-diagonal
  fullMat.block(3, 0, 2, 3) << 1, 2, 1,  //
      2, 1, 3;
  fullMat.block(0, 3, 3, 2) << 3, 1,  //
      1, 2,                            //
      2, 1;

  vector<T> data(factorSkel.totalDataSize());
  fillBlockData(factorSkel, data.data(), fullMat);

  Solver solver(std::move(factorSkel), {}, {}, cudaOps());
  vector<int64_t> pivots(order);

  {
    DevMirror<T> dataGpu(data);
    solver.factorLU(dataGpu.ptr, pivots.data());
    dataGpu.get(data);
  }

  Vector<T> b = Vector<T>::Ones(order);
  Vector<T> x = b;

  {
    DevMirror<T> dataGpu(data);
    DevMirror<T> xGpu(vector<T>(x.data(), x.data() + order));
    solver.solveLU(dataGpu.ptr, pivots.data(), xGpu.ptr, order, 1);
    vector<T> xVec(order);
    xGpu.get(xVec);
    for (int64_t i = 0; i < order; i++) x(i) = xVec[i];
  }

  T residual = (fullMat * x - b).norm() / b.norm();
  ASSERT_LT(residual, Epsilon<T>::value2) << "Block-sparse LU solve residual too large";
}

TEST(CudaLU, BlockSparse_double) { testBlockSparse<double>(); }
TEST(CudaLU, BlockSparse_float) { testBlockSparse<float>(); }

// Verify CUDA matches CPU BackendFast for LU on a larger block-sparse matrix
template <typename T>
void testVsCpuReference() {
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

  // Build non-symmetric diagonally-dominant matrix
  mt19937 rng(42);
  uniform_real_distribution<T> unif(T(-1.0), T(1.0));
  Matrix<T> fullMat = Matrix<T>::Zero(totalSize, totalSize);

  for (int64_t rowBlock = 0; rowBlock < (int64_t)ss.ptrs.size() - 1; rowBlock++) {
    for (int64_t k = ss.ptrs[rowBlock]; k < ss.ptrs[rowBlock + 1]; k++) {
      int64_t colBlock = ss.inds[k];
      for (int64_t r = spanStart[rowBlock]; r < spanStart[rowBlock + 1]; r++)
        for (int64_t c = spanStart[colBlock]; c < spanStart[colBlock + 1]; c++)
          fullMat(r, c) = unif(rng);
    }
  }
  // Separate upper triangle entries
  for (int64_t rowBlock = 0; rowBlock < (int64_t)ss.ptrs.size() - 1; rowBlock++) {
    for (int64_t k = ss.ptrs[rowBlock]; k < ss.ptrs[rowBlock + 1]; k++) {
      int64_t colBlock = ss.inds[k];
      if (colBlock != rowBlock) {
        for (int64_t r = spanStart[colBlock]; r < spanStart[colBlock + 1]; r++)
          for (int64_t c = spanStart[rowBlock]; c < spanStart[rowBlock + 1]; c++)
            fullMat(r, c) = unif(rng);
      }
    }
  }
  for (int64_t i = 0; i < totalSize; i++) fullMat(i, i) += totalSize * 4;

  auto fillData = [&](const CoalescedBlockMatrixSkel& skel, vector<T>& data) {
    fillBlockData(skel, data.data(), fullMat);
  };

  // Solve with CPU
  CoalescedBlockMatrixSkel factorSkelCpu(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  factorSkelCpu.initUpperTriangle();
  vector<T> dataCpu(factorSkelCpu.totalDataSize());
  fillData(factorSkelCpu, dataCpu);

  Solver solverCpu(std::move(factorSkelCpu), {}, {}, fastOps());
  vector<int64_t> pivotsCpu(totalSize);
  solverCpu.factorLU(dataCpu.data(), pivotsCpu.data());

  Vector<T> b(totalSize);
  for (int64_t i = 0; i < totalSize; i++) b(i) = unif(rng);

  Vector<T> xCpu = b;
  solverCpu.solveLU(dataCpu.data(), pivotsCpu.data(), xCpu.data(), totalSize, 1);
  T cpuResidual = (fullMat * xCpu - b).norm() / b.norm();

  // Solve with CUDA
  CoalescedBlockMatrixSkel factorSkelCuda(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  factorSkelCuda.initUpperTriangle();
  vector<T> dataCuda(factorSkelCuda.totalDataSize());
  fillData(factorSkelCuda, dataCuda);

  Solver solverCuda(std::move(factorSkelCuda), {}, {}, cudaOps());
  vector<int64_t> pivotsCuda(totalSize);

  {
    DevMirror<T> dataGpu(dataCuda);
    solverCuda.factorLU(dataGpu.ptr, pivotsCuda.data());
    dataGpu.get(dataCuda);
  }

  Vector<T> xCuda = b;
  {
    DevMirror<T> dataGpu(dataCuda);
    DevMirror<T> xGpu(vector<T>(xCuda.data(), xCuda.data() + totalSize));
    solverCuda.solveLU(dataGpu.ptr, pivotsCuda.data(), xGpu.ptr, totalSize, 1);
    vector<T> xVec(totalSize);
    xGpu.get(xVec);
    for (int64_t i = 0; i < totalSize; i++) xCuda(i) = xVec[i];
  }

  T cudaResidual = (fullMat * xCuda - b).norm() / b.norm();

  cout << "CPU residual: " << cpuResidual << ", CUDA residual: " << cudaResidual << endl;

  ASSERT_LT(cpuResidual, Epsilon<T>::value2) << "CPU LU residual too large";
  ASSERT_LT(cudaResidual, Epsilon<T>::value2) << "CUDA LU residual too large";
}

TEST(CudaLU, VsCpuReference_double) { testVsCpuReference<double>(); }
TEST(CudaLU, VsCpuReference_float) { testVsCpuReference<float>(); }

// Test with multiple RHS
template <typename T>
void testMultipleRHS() {
  vector<set<int64_t>> colBlocks{{0, 1}, {1}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart{0, 3, 5};
  vector<int64_t> lumpToSpan{0, 1, 2};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  factorSkel.initUpperTriangle();

  int64_t order = factorSkel.order();
  int nRHS = 5;

  Matrix<T> fullMat = Matrix<T>::Zero(order, order);
  fullMat.block(0, 0, 3, 3) << 10, 1, 2,  //
      3, 11, 1,                            //
      1, 2, 12;
  fullMat.block(3, 3, 2, 2) << 8, 3,  //
      1, 9;
  fullMat.block(3, 0, 2, 3) << 1, 2, 1,  //
      2, 1, 3;
  fullMat.block(0, 3, 3, 2) << 3, 1,  //
      1, 2,                            //
      2, 1;

  vector<T> data(factorSkel.totalDataSize());
  fillBlockData(factorSkel, data.data(), fullMat);

  Solver solver(std::move(factorSkel), {}, {}, cudaOps());
  vector<int64_t> pivots(order);

  {
    DevMirror<T> dataGpu(data);
    solver.factorLU(dataGpu.ptr, pivots.data());
    dataGpu.get(data);
  }

  // Multiple RHS
  mt19937 rng(37);
  uniform_real_distribution<T> unif(T(-1.0), T(1.0));
  Matrix<T> B(order, nRHS);
  for (int64_t i = 0; i < order; i++)
    for (int j = 0; j < nRHS; j++) B(i, j) = unif(rng);

  Matrix<T> X = B;
  {
    DevMirror<T> dataGpu(data);
    DevMirror<T> xGpu(vector<T>(X.data(), X.data() + order * nRHS));
    solver.solveLU(dataGpu.ptr, pivots.data(), xGpu.ptr, order, nRHS);
    vector<T> xVec(order * nRHS);
    xGpu.get(xVec);
    for (int64_t i = 0; i < order * nRHS; i++) X.data()[i] = xVec[i];
  }

  T residual = (fullMat * X - B).norm() / B.norm();
  ASSERT_LT(residual, Epsilon<T>::value2) << "Multi-RHS LU solve residual too large";
}

TEST(CudaLU, MultipleRHS_double) { testMultipleRHS<double>(); }
TEST(CudaLU, MultipleRHS_float) { testMultipleRHS<float>(); }
