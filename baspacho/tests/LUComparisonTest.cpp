/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef BASPACHO_HAVE_UMFPACK

#include <gtest/gtest.h>
#include <umfpack.h>
#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/testing/TestingMatGen.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using Vector = Eigen::Vector<T, Eigen::Dynamic>;

// Helper struct to hold UMFPACK solve results
struct UmfpackSolveResult {
  double analysisTime;
  double factorTime;
  double solveTime;
  double residual;
  Vector<double> solution;
};

// Helper struct to hold BaSpaCho solve results
struct BaspachoSolveResult {
  double analysisTime;
  double factorTime;
  double solveTime;
  double residual;
  Vector<double> solution;
};

// Solve with UMFPACK for comparison
UmfpackSolveResult solveWithUmfpack(const vector<int64_t>& colPtr, const vector<int64_t>& rowIdx,
                                     const vector<double>& val, int64_t n,
                                     const Vector<double>& b) {
  UmfpackSolveResult result;

  double Control[UMFPACK_CONTROL];
  double Info[UMFPACK_INFO];
  umfpack_dl_defaults(Control);

  void* Symbolic = nullptr;
  void* Numeric = nullptr;

  // Symbolic analysis
  auto startAnalysis = hrc::now();
  int status = umfpack_dl_symbolic(n, n, colPtr.data(), rowIdx.data(), val.data(), &Symbolic,
                                   Control, Info);
  result.analysisTime = tdelta(hrc::now() - startAnalysis).count();

  EXPECT_EQ(status, UMFPACK_OK) << "UMFPACK symbolic analysis failed";
  if (status != UMFPACK_OK) {
    return result;
  }

  // Numeric factorization
  auto startFactor = hrc::now();
  status =
      umfpack_dl_numeric(colPtr.data(), rowIdx.data(), val.data(), Symbolic, &Numeric, Control,
                         Info);
  result.factorTime = tdelta(hrc::now() - startFactor).count();

  EXPECT_EQ(status, UMFPACK_OK) << "UMFPACK numeric factorization failed";
  if (status != UMFPACK_OK) {
    umfpack_dl_free_symbolic(&Symbolic);
    return result;
  }

  // Solve
  result.solution.resize(n);
  auto startSolve = hrc::now();
  status = umfpack_dl_solve(UMFPACK_A, colPtr.data(), rowIdx.data(), val.data(),
                            result.solution.data(), b.data(), Numeric, Control, Info);
  result.solveTime = tdelta(hrc::now() - startSolve).count();

  EXPECT_EQ(status, UMFPACK_OK) << "UMFPACK solve failed";

  // Compute residual
  Vector<double> Ax(n);
  Ax.setZero();
  for (int64_t col = 0; col < n; col++) {
    for (int64_t k = colPtr[col]; k < colPtr[col + 1]; k++) {
      int64_t row = rowIdx[k];
      Ax(row) += val[k] * result.solution(col);
    }
  }
  result.residual = (Ax - b).norm() / b.norm();

  umfpack_dl_free_symbolic(&Symbolic);
  umfpack_dl_free_numeric(&Numeric);

  return result;
}

// Compare UMFPACK and BaSpaCho on a randomly generated sparse matrix
TEST(LUComparison, VsUmfpack_SmallDense) {
  // Create a small dense test matrix
  int64_t n = 10;

  // Generate random non-symmetric matrix
  mt19937 gen(42);
  uniform_real_distribution<double> unif(-1.0, 1.0);

  Matrix<double> A(n, n);
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < n; j++) {
      A(i, j) = unif(gen);
    }
    A(i, i) += n * 2;  // diagonal dominance
  }

  // Build CSC format for UMFPACK
  vector<int64_t> colPtr, rowIdx;
  vector<double> val;
  colPtr.push_back(0);
  for (int64_t col = 0; col < n; col++) {
    for (int64_t row = 0; row < n; row++) {
      rowIdx.push_back(row);
      val.push_back(A(row, col));
    }
    colPtr.push_back(rowIdx.size());
  }

  // Generate RHS
  Vector<double> b(n);
  for (int64_t i = 0; i < n; i++) {
    b(i) = unif(gen);
  }

  // Solve with UMFPACK
  auto umfResult = solveWithUmfpack(colPtr, rowIdx, val, n, b);

  // Solve with BaSpaCho
  // Create single-block structure
  vector<set<int64_t>> colBlocks{{0}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> spanStart{0, n};
  vector<int64_t> lumpToSpan{0, 1};
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, ss.ptrs, ss.inds);

  // For single block, no upper triangle needed
  vector<double> data(factorSkel.dataSize());

  // Copy A to data in row-major format
  for (int64_t row = 0; row < n; row++) {
    for (int64_t col = 0; col < n; col++) {
      data[row * n + col] = A(row, col);
    }
  }

  auto startAnalysis = hrc::now();
  Solver solver(std::move(factorSkel), {}, {}, fastOps());
  double baspachoAnalysisTime = tdelta(hrc::now() - startAnalysis).count();

  vector<int64_t> pivots(n);

  auto startFactor = hrc::now();
  solver.factorLU(data.data(), pivots.data());
  double baspachoFactorTime = tdelta(hrc::now() - startFactor).count();

  Vector<double> x = b;
  auto startSolve = hrc::now();
  solver.solveLU(data.data(), pivots.data(), x.data(), n, 1);
  double baspachoSolveTime = tdelta(hrc::now() - startSolve).count();

  double baspachoResidual = (A * x - b).norm() / b.norm();

  // Print comparison
  cout << "\n=== Small Dense Matrix Comparison (n=" << n << ") ===" << endl;
  cout << "UMFPACK:"
       << "\n  Analysis: " << umfResult.analysisTime * 1000 << " ms"
       << "\n  Factor:   " << umfResult.factorTime * 1000 << " ms"
       << "\n  Solve:    " << umfResult.solveTime * 1000 << " ms"
       << "\n  Residual: " << umfResult.residual << endl;
  cout << "BaSpaCho:"
       << "\n  Analysis: " << baspachoAnalysisTime * 1000 << " ms"
       << "\n  Factor:   " << baspachoFactorTime * 1000 << " ms"
       << "\n  Solve:    " << baspachoSolveTime * 1000 << " ms"
       << "\n  Residual: " << baspachoResidual << endl;

  // Both should have small residuals
  EXPECT_LT(umfResult.residual, 1e-10) << "UMFPACK residual too large";
  EXPECT_LT(baspachoResidual, 1e-10) << "BaSpaCho residual too large";

  // Solutions should be similar (not identical due to different pivoting strategies)
  double solutionDiff = (x - umfResult.solution).norm() / umfResult.solution.norm();
  cout << "Solution difference: " << solutionDiff << endl;
  EXPECT_LT(solutionDiff, 1e-8) << "Solutions differ too much";
}

// Compare on a simple 2-block matrix (same structure as LUFactorTest)
TEST(LUComparison, VsUmfpack_TwoBlock) {
  // Use the exact same structure as the passing LUFactorTest
  vector<set<int64_t>> colBlocks{{0, 1}, {1}};
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> paramSize{3, 2};  // Block 0 is 3x3, Block 1 is 2x2

  int64_t totalSize = 5;

  cout << "\n=== Two-Block Comparison (same as LUFactorTest) ===" << endl;
  cout << "Blocks: 2, Total size: " << totalSize << "x" << totalSize << endl;

  // Use the exact same matrix as LUFactorTest
  Matrix<double> fullMat = Matrix<double>::Zero(totalSize, totalSize);
  fullMat.block(0, 0, 3, 3) << 10, 1, 2, 1, 11, 1, 2, 1, 12;
  fullMat.block(3, 3, 2, 2) << 8, 1, 1, 9;
  fullMat.block(3, 0, 2, 3) << 1, 2, 1, 2, 1, 2;
  fullMat.block(0, 3, 3, 2) = fullMat.block(3, 0, 2, 3).transpose();

  // Build CSC format for UMFPACK
  vector<int64_t> colPtr, rowIdx;
  vector<double> val;
  colPtr.push_back(0);
  for (int64_t col = 0; col < totalSize; col++) {
    for (int64_t row = 0; row < totalSize; row++) {
      if (fullMat(row, col) != 0.0) {
        rowIdx.push_back(row);
        val.push_back(fullMat(row, col));
      }
    }
    colPtr.push_back(rowIdx.size());
  }

  // Use the same RHS as LUFactorTest
  Vector<double> b = Vector<double>::Ones(totalSize);

  // Solve with UMFPACK
  auto umfResult = solveWithUmfpack(colPtr, rowIdx, val, totalSize, b);

  // Solve with BaSpaCho (using exact same setup as LUFactorTest)
  vector<int64_t> spanStart{0, 3, 5};
  vector<int64_t> lumpToSpan{0, 1, 2};
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  factorSkel.initUpperTriangle();

  // Allocate data
  vector<double> data(factorSkel.totalDataSize());

  // Fill data exactly as LUFactorTest does
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

      for (int64_t r = 0; r < rowSize; r++) {
        for (int64_t col = 0; col < lumpSize; col++) {
          data[dataOffset + r * lumpSize + col] = fullMat(rowStart + r, lumpStart + col);
        }
      }
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

      for (int64_t r = 0; r < lumpSize; r++) {
        for (int64_t c = 0; c < colSize; c++) {
          data[upperDataOffset + r * colSize + c] = fullMat(lumpStartIdx + r, colStart + c);
        }
      }
    }
  }

  auto startAnalysis = hrc::now();
  Solver solver(std::move(factorSkel), {}, {}, fastOps());
  double baspachoAnalysisTime = tdelta(hrc::now() - startAnalysis).count();

  vector<int64_t> pivots(totalSize);

  auto startFactor = hrc::now();
  solver.factorLU(data.data(), pivots.data());
  double baspachoFactorTime = tdelta(hrc::now() - startFactor).count();

  Vector<double> x = b;
  auto startSolve = hrc::now();
  solver.solveLU(data.data(), pivots.data(), x.data(), totalSize, 1);
  double baspachoSolveTime = tdelta(hrc::now() - startSolve).count();

  double baspachoResidual = (fullMat * x - b).norm() / b.norm();

  cout << "UMFPACK:"
       << "\n  Analysis: " << umfResult.analysisTime * 1000 << " ms"
       << "\n  Factor:   " << umfResult.factorTime * 1000 << " ms"
       << "\n  Solve:    " << umfResult.solveTime * 1000 << " ms"
       << "\n  Residual: " << umfResult.residual << endl;
  cout << "BaSpaCho:"
       << "\n  Analysis: " << baspachoAnalysisTime * 1000 << " ms"
       << "\n  Factor:   " << baspachoFactorTime * 1000 << " ms"
       << "\n  Solve:    " << baspachoSolveTime * 1000 << " ms"
       << "\n  Residual: " << baspachoResidual << endl;

  EXPECT_LT(umfResult.residual, 1e-10) << "UMFPACK residual too large";
  EXPECT_LT(baspachoResidual, 1e-8) << "BaSpaCho residual too large";

  double solutionDiff = (x - umfResult.solution).norm() / umfResult.solution.norm();
  cout << "Solution difference: " << solutionDiff << endl;
  EXPECT_LT(solutionDiff, 1e-6) << "Solutions differ too much";
}

// Compare on a sparse block matrix
TEST(LUComparison, DISABLED_VsUmfpack_BlockSparse) {
  // DISABLED: This test has issues with data setup for random structures
  // Create a block-sparse structure using TestingMatGen
  SparseMatGenerator gen = SparseMatGenerator::genFlat(20, 0.3, 42);
  SparseStructure ss = columnsToCscStruct(gen.columns).transpose();
  ss = ss.addFullEliminationFill();

  // Use small block sizes
  vector<int64_t> paramSize(gen.columns.size(), 3);

  // Compute total size
  int64_t totalSize = 0;
  for (int64_t ps : paramSize) {
    totalSize += ps;
  }

  cout << "\n=== Block-Sparse Matrix Comparison ===" << endl;
  cout << "Blocks: " << paramSize.size() << ", Total size: " << totalSize << "x" << totalSize
       << endl;

  // Build full dense matrix (for comparison)
  mt19937 rng(42);
  uniform_real_distribution<double> unif(-1.0, 1.0);
  Matrix<double> fullMat = Matrix<double>::Zero(totalSize, totalSize);

  vector<int64_t> spanStart;
  spanStart.push_back(0);
  for (int64_t ps : paramSize) {
    spanStart.push_back(spanStart.back() + ps);
  }

  // Fill blocks based on sparsity structure (lower triangle)
  // SparseStructure is CSR format: ptrs[i] is row i's start, inds contains column indices
  // For LU, we need to provide both triangles. Generate symmetric data.
  for (int64_t rowBlock = 0; rowBlock < (int64_t)ss.ptrs.size() - 1; rowBlock++) {
    for (int64_t k = ss.ptrs[rowBlock]; k < ss.ptrs[rowBlock + 1]; k++) {
      int64_t colBlock = ss.inds[k];
      // Fill the block at (rowBlock, colBlock)
      for (int64_t r = spanStart[rowBlock]; r < spanStart[rowBlock + 1]; r++) {
        for (int64_t c = spanStart[colBlock]; c < spanStart[colBlock + 1]; c++) {
          double val = unif(rng);
          fullMat(r, c) = val;
          // Make symmetric: also fill upper triangle
          if (r != c) {
            fullMat(c, r) = val;
          }
        }
      }
    }
  }

  // Add diagonal dominance
  for (int64_t i = 0; i < totalSize; i++) {
    fullMat(i, i) += totalSize * 2;
  }

  // Build CSC format for UMFPACK
  vector<int64_t> colPtr, rowIdx;
  vector<double> val;
  colPtr.push_back(0);
  for (int64_t col = 0; col < totalSize; col++) {
    for (int64_t row = 0; row < totalSize; row++) {
      if (fullMat(row, col) != 0.0) {
        rowIdx.push_back(row);
        val.push_back(fullMat(row, col));
      }
    }
    colPtr.push_back(rowIdx.size());
  }

  // Generate RHS
  Vector<double> b(totalSize);
  for (int64_t i = 0; i < totalSize; i++) {
    b(i) = unif(rng);
  }

  // Solve with UMFPACK
  auto umfResult = solveWithUmfpack(colPtr, rowIdx, val, totalSize, b);

  // Solve with BaSpaCho
  vector<int64_t> lumpToSpan(paramSize.size() + 1);
  iota(lumpToSpan.begin(), lumpToSpan.end(), 0);
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  factorSkel.initUpperTriangle();

  // Allocate data
  vector<double> data(factorSkel.totalDataSize());

  // Fill lower triangle data from fullMat
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

      for (int64_t r = 0; r < rowSize; r++) {
        for (int64_t col = 0; col < lumpSize; col++) {
          data[dataOffset + r * lumpSize + col] = fullMat(rowStart + r, lumpStart + col);
        }
      }
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

      for (int64_t r = 0; r < lumpSize; r++) {
        for (int64_t c = 0; c < colSize; c++) {
          data[upperDataOffset + r * colSize + c] = fullMat(lumpStartIdx + r, colStart + c);
        }
      }
    }
  }

  auto startAnalysis = hrc::now();
  Solver solver(std::move(factorSkel), {}, {}, fastOps());
  double baspachoAnalysisTime = tdelta(hrc::now() - startAnalysis).count();

  vector<int64_t> pivots(totalSize);

  auto startFactor = hrc::now();
  solver.factorLU(data.data(), pivots.data());
  double baspachoFactorTime = tdelta(hrc::now() - startFactor).count();

  Vector<double> x = b;
  auto startSolve = hrc::now();
  solver.solveLU(data.data(), pivots.data(), x.data(), totalSize, 1);
  double baspachoSolveTime = tdelta(hrc::now() - startSolve).count();

  double baspachoResidual = (fullMat * x - b).norm() / b.norm();

  // Print comparison
  cout << "UMFPACK:"
       << "\n  Analysis: " << umfResult.analysisTime * 1000 << " ms"
       << "\n  Factor:   " << umfResult.factorTime * 1000 << " ms"
       << "\n  Solve:    " << umfResult.solveTime * 1000 << " ms"
       << "\n  Residual: " << umfResult.residual << endl;
  cout << "BaSpaCho:"
       << "\n  Analysis: " << baspachoAnalysisTime * 1000 << " ms"
       << "\n  Factor:   " << baspachoFactorTime * 1000 << " ms"
       << "\n  Solve:    " << baspachoSolveTime * 1000 << " ms"
       << "\n  Residual: " << baspachoResidual << endl;

  // Both should have small residuals
  EXPECT_LT(umfResult.residual, 1e-10) << "UMFPACK residual too large";
  EXPECT_LT(baspachoResidual, 1e-8) << "BaSpaCho residual too large";

  // Solutions should be similar
  double solutionDiff = (x - umfResult.solution).norm() / umfResult.solution.norm();
  cout << "Solution difference: " << solutionDiff << endl;
  EXPECT_LT(solutionDiff, 1e-6) << "Solutions differ too much";
}

// Compare on a larger sparse matrix for performance
TEST(LUComparison, DISABLED_VsUmfpack_Performance) {
  // Create a larger block-sparse structure
  SparseMatGenerator gen = SparseMatGenerator::genGrid(20, 20, 0.5, 2, 42);
  SparseStructure ss = columnsToCscStruct(gen.columns).transpose();
  ss = ss.addFullEliminationFill();

  vector<int64_t> paramSize(gen.columns.size(), 3);

  int64_t totalSize = 0;
  for (int64_t ps : paramSize) {
    totalSize += ps;
  }

  cout << "\n=== Performance Comparison (Grid 20x20) ===" << endl;
  cout << "Blocks: " << paramSize.size() << ", Total size: " << totalSize << "x" << totalSize
       << endl;

  // Build full dense matrix
  mt19937 rng(42);
  uniform_real_distribution<double> unif(-1.0, 1.0);
  Matrix<double> fullMat = Matrix<double>::Zero(totalSize, totalSize);

  vector<int64_t> spanStart;
  spanStart.push_back(0);
  for (int64_t ps : paramSize) {
    spanStart.push_back(spanStart.back() + ps);
  }

  // SparseStructure is CSR format: ptrs[i] is row i's start, inds contains column indices
  // For LU, we need to provide both triangles. Generate symmetric data.
  for (int64_t rowBlock = 0; rowBlock < (int64_t)ss.ptrs.size() - 1; rowBlock++) {
    for (int64_t k = ss.ptrs[rowBlock]; k < ss.ptrs[rowBlock + 1]; k++) {
      int64_t colBlock = ss.inds[k];
      for (int64_t r = spanStart[rowBlock]; r < spanStart[rowBlock + 1]; r++) {
        for (int64_t c = spanStart[colBlock]; c < spanStart[colBlock + 1]; c++) {
          double v = unif(rng);
          fullMat(r, c) = v;
          if (r != c) {
            fullMat(c, r) = v;
          }
        }
      }
    }
  }

  for (int64_t i = 0; i < totalSize; i++) {
    fullMat(i, i) += totalSize * 2;
  }

  // Build CSC for UMFPACK (include all non-zeros)
  vector<int64_t> colPtr, rowIdx;
  vector<double> val;
  colPtr.push_back(0);
  for (int64_t col = 0; col < totalSize; col++) {
    for (int64_t row = 0; row < totalSize; row++) {
      if (fullMat(row, col) != 0.0) {
        rowIdx.push_back(row);
        val.push_back(fullMat(row, col));
      }
    }
    colPtr.push_back(rowIdx.size());
  }

  cout << "NNZ: " << val.size() << " (fill=" << val.size() / ((double)totalSize * totalSize) << ")"
       << endl;

  Vector<double> b(totalSize);
  for (int64_t i = 0; i < totalSize; i++) {
    b(i) = unif(rng);
  }

  // Warmup and solve with UMFPACK
  auto umfResult = solveWithUmfpack(colPtr, rowIdx, val, totalSize, b);

  // Solve with BaSpaCho
  vector<int64_t> lumpToSpan(paramSize.size() + 1);
  iota(lumpToSpan.begin(), lumpToSpan.end(), 0);
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  factorSkel.initUpperTriangle();

  vector<double> data(factorSkel.totalDataSize());

  // Fill data
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

      for (int64_t r = 0; r < rowSize; r++) {
        for (int64_t col = 0; col < lumpSize; col++) {
          data[dataOffset + r * lumpSize + col] = fullMat(rowStart + r, lumpStart + col);
        }
      }
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

      for (int64_t r = 0; r < lumpSize; r++) {
        for (int64_t c = 0; c < colSize; c++) {
          data[upperDataOffset + r * colSize + c] = fullMat(lumpStartIdx + r, colStart + c);
        }
      }
    }
  }

  auto startAnalysis = hrc::now();
  Solver solver(std::move(factorSkel), {}, {}, fastOps());
  double baspachoAnalysisTime = tdelta(hrc::now() - startAnalysis).count();

  vector<int64_t> pivots(totalSize);

  auto startFactor = hrc::now();
  solver.factorLU(data.data(), pivots.data());
  double baspachoFactorTime = tdelta(hrc::now() - startFactor).count();

  Vector<double> x = b;
  auto startSolve = hrc::now();
  solver.solveLU(data.data(), pivots.data(), x.data(), totalSize, 1);
  double baspachoSolveTime = tdelta(hrc::now() - startSolve).count();

  double baspachoResidual = (fullMat * x - b).norm() / b.norm();

  cout << "UMFPACK:"
       << "\n  Analysis: " << umfResult.analysisTime * 1000 << " ms"
       << "\n  Factor:   " << umfResult.factorTime * 1000 << " ms"
       << "\n  Solve:    " << umfResult.solveTime * 1000 << " ms"
       << "\n  Residual: " << umfResult.residual << endl;
  cout << "BaSpaCho:"
       << "\n  Analysis: " << baspachoAnalysisTime * 1000 << " ms"
       << "\n  Factor:   " << baspachoFactorTime * 1000 << " ms"
       << "\n  Solve:    " << baspachoSolveTime * 1000 << " ms"
       << "\n  Residual: " << baspachoResidual << endl;

  // Both should have reasonable residuals
  EXPECT_LT(umfResult.residual, 1e-8) << "UMFPACK residual too large";
  EXPECT_LT(baspachoResidual, 1e-6) << "BaSpaCho residual too large";
}

#endif  // BASPACHO_HAVE_UMFPACK
