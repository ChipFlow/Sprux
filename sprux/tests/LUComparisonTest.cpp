/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef SPRUX_HAVE_UMFPACK

#include <gtest/gtest.h>
#include <umfpack.h>
#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include "sprux/sprux/CoalescedBlockMatrix.h"
#include "sprux/sprux/EliminationTree.h"
#include "sprux/sprux/Solver.h"
#include "sprux/sprux/SparseStructure.h"
#include "sprux/sprux/Utils.h"
#include "sprux/testing/TestingMatGen.h"
#include "sprux/testing/TestingUtils.h"

using namespace Sprux;
using namespace ::Sprux::testing_utils;
using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using Vector = Eigen::Vector<T, Eigen::Dynamic>;

// ============================================================================
// Helper functions for filling block matrix data from dense matrices
// ============================================================================

// Fill Sprux block data from a dense matrix.
// This properly fills both lower triangle (chainData) and upper triangle
// (upperChainData) storage from the corresponding entries in fullMat.
template <typename T>
void fillDataFromDenseMatrix(const CoalescedBlockMatrixSkel& skel, T* data,
                             const Matrix<T>& fullMat, bool verbose = false) {
  int64_t numLumps = skel.numLumps();

  if (verbose) {
    cout << "fillDataFromDenseMatrix: numLumps=" << numLumps
         << ", dataSize=" << skel.dataSize() << ", upperDataSize=" << skel.upperDataSize()
         << ", totalDataSize=" << skel.totalDataSize() << endl;
  }

  // Fill lower triangle (includes diagonal blocks)
  for (int64_t l = 0; l < numLumps; l++) {
    int64_t chainStart = skel.chainColPtr[l];
    int64_t chainEnd = skel.chainColPtr[l + 1];
    int64_t lumpStartCol = skel.lumpStart[l];
    int64_t lumpSize = skel.lumpStart[l + 1] - lumpStartCol;

    if (verbose) {
      cout << "  Lump " << l << ": chains [" << chainStart << ", " << chainEnd << ")"
           << ", cols [" << lumpStartCol << ", " << lumpStartCol + lumpSize << ")" << endl;
    }

    for (int64_t c = chainStart; c < chainEnd; c++) {
      int64_t rowSpan = skel.chainRowSpan[c];
      int64_t rowStart = skel.spanStart[rowSpan];
      int64_t rowSize = skel.spanStart[rowSpan + 1] - rowStart;
      int64_t dataOffset = skel.chainData[c];

      if (verbose) {
        cout << "    Chain " << c << ": rowSpan=" << rowSpan << " rows [" << rowStart << ", "
             << rowStart + rowSize << "), dataOffset=" << dataOffset << endl;
      }

      // Copy from fullMat to data buffer (row-major storage)
      for (int64_t r = 0; r < rowSize; r++) {
        for (int64_t col = 0; col < lumpSize; col++) {
          data[dataOffset + r * lumpSize + col] = fullMat(rowStart + r, lumpStartCol + col);
        }
      }
    }
  }

  // Fill upper triangle (if initialized for LU)
  if (!skel.upperChainData.empty()) {
    int64_t upperDataBase = skel.dataSize();

    if (verbose) {
      cout << "  Upper triangle: upperDataBase=" << upperDataBase << endl;
    }

    for (int64_t l = 0; l < numLumps; l++) {
      int64_t upperRowStart = skel.upperChainRowPtr[l];
      int64_t upperRowEnd = skel.upperChainRowPtr[l + 1];
      int64_t lumpStartRow = skel.lumpStart[l];
      int64_t lumpSize = skel.lumpStart[l + 1] - lumpStartRow;

      if (verbose && upperRowStart < upperRowEnd) {
        cout << "  Upper lump " << l << ": entries [" << upperRowStart << ", " << upperRowEnd << ")"
             << ", rows [" << lumpStartRow << ", " << lumpStartRow + lumpSize << ")" << endl;
      }

      for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
        int64_t colSpan = skel.upperChainColSpan[i];
        int64_t colStart = skel.spanStart[colSpan];
        int64_t colSize = skel.spanStart[colSpan + 1] - colStart;
        int64_t upperDataOffset = upperDataBase + skel.upperChainData[i];

        if (verbose) {
          cout << "    Upper entry " << i << ": colSpan=" << colSpan << " cols [" << colStart
               << ", " << colStart + colSize << "), dataOffset=" << upperDataOffset << endl;
        }

        // Upper triangle stores: lumpSize rows x colSize cols
        // This is the block at matrix position (lumpStartRow, colStart)
        for (int64_t r = 0; r < lumpSize; r++) {
          for (int64_t c = 0; c < colSize; c++) {
            data[upperDataOffset + r * colSize + c] = fullMat(lumpStartRow + r, colStart + c);
          }
        }
      }
    }
  }
}

// Reconstruct a dense matrix from Sprux block data.
// This is useful for verifying that data was filled correctly.
template <typename T>
Matrix<T> reconstructDenseMatrix(const CoalescedBlockMatrixSkel& skel, const T* data,
                                 bool verbose = false) {
  int64_t order = skel.order();
  Matrix<T> result = Matrix<T>::Zero(order, order);
  int64_t numLumps = skel.numLumps();

  if (verbose) {
    cout << "reconstructDenseMatrix: order=" << order << ", numLumps=" << numLumps << endl;
  }

  // Reconstruct from lower triangle
  for (int64_t l = 0; l < numLumps; l++) {
    int64_t chainStart = skel.chainColPtr[l];
    int64_t chainEnd = skel.chainColPtr[l + 1];
    int64_t lumpStartCol = skel.lumpStart[l];
    int64_t lumpSize = skel.lumpStart[l + 1] - lumpStartCol;

    for (int64_t c = chainStart; c < chainEnd; c++) {
      int64_t rowSpan = skel.chainRowSpan[c];
      int64_t rowStart = skel.spanStart[rowSpan];
      int64_t rowSize = skel.spanStart[rowSpan + 1] - rowStart;
      int64_t dataOffset = skel.chainData[c];

      for (int64_t r = 0; r < rowSize; r++) {
        for (int64_t col = 0; col < lumpSize; col++) {
          result(rowStart + r, lumpStartCol + col) = data[dataOffset + r * lumpSize + col];
        }
      }
    }
  }

  // Reconstruct from upper triangle (if present)
  if (!skel.upperChainData.empty()) {
    int64_t upperDataBase = skel.dataSize();

    for (int64_t l = 0; l < numLumps; l++) {
      int64_t upperRowStart = skel.upperChainRowPtr[l];
      int64_t upperRowEnd = skel.upperChainRowPtr[l + 1];
      int64_t lumpStartRow = skel.lumpStart[l];
      int64_t lumpSize = skel.lumpStart[l + 1] - lumpStartRow;

      for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
        int64_t colSpan = skel.upperChainColSpan[i];
        int64_t colStart = skel.spanStart[colSpan];
        int64_t colSize = skel.spanStart[colSpan + 1] - colStart;
        int64_t upperDataOffset = upperDataBase + skel.upperChainData[i];

        for (int64_t r = 0; r < lumpSize; r++) {
          for (int64_t c = 0; c < colSize; c++) {
            result(lumpStartRow + r, colStart + c) = data[upperDataOffset + r * colSize + c];
          }
        }
      }
    }
  }

  return result;
}

// Print sparse structure info for debugging
void printSparseStructure(const string& name, const SparseStructure& ss) {
  cout << name << ": " << ss.ptrs.size() - 1 << " rows" << endl;
  for (size_t i = 0; i + 1 < ss.ptrs.size(); i++) {
    cout << "  Row " << i << ": cols [";
    for (int64_t k = ss.ptrs[i]; k < ss.ptrs[i + 1]; k++) {
      if (k > ss.ptrs[i]) cout << ", ";
      cout << ss.inds[k];
    }
    cout << "]" << endl;
  }
}

// Helper struct to hold UMFPACK solve results
struct UmfpackSolveResult {
  double analysisTime;
  double factorTime;
  double solveTime;
  double residual;
  Vector<double> solution;
};

// Helper struct to hold Sprux solve results
struct SpruxSolveResult {
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

// Compare UMFPACK and Sprux on a randomly generated sparse matrix
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

  // Solve with Sprux
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
  double spruxAnalysisTime = tdelta(hrc::now() - startAnalysis).count();

  vector<int64_t> pivots(n);

  auto startFactor = hrc::now();
  solver.factorLU(data.data(), pivots.data());
  double spruxFactorTime = tdelta(hrc::now() - startFactor).count();

  Vector<double> x = b;
  auto startSolve = hrc::now();
  solver.solveLU(data.data(), pivots.data(), x.data(), n, 1);
  double spruxSolveTime = tdelta(hrc::now() - startSolve).count();

  double spruxResidual = (A * x - b).norm() / b.norm();

  // Print comparison
  cout << "\n=== Small Dense Matrix Comparison (n=" << n << ") ===" << endl;
  cout << "UMFPACK:"
       << "\n  Analysis: " << umfResult.analysisTime * 1000 << " ms"
       << "\n  Factor:   " << umfResult.factorTime * 1000 << " ms"
       << "\n  Solve:    " << umfResult.solveTime * 1000 << " ms"
       << "\n  Residual: " << umfResult.residual << endl;
  cout << "Sprux:"
       << "\n  Analysis: " << spruxAnalysisTime * 1000 << " ms"
       << "\n  Factor:   " << spruxFactorTime * 1000 << " ms"
       << "\n  Solve:    " << spruxSolveTime * 1000 << " ms"
       << "\n  Residual: " << spruxResidual << endl;

  // Both should have small residuals
  EXPECT_LT(umfResult.residual, 1e-10) << "UMFPACK residual too large";
  EXPECT_LT(spruxResidual, 1e-10) << "Sprux residual too large";

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

  // Solve with Sprux (using exact same setup as LUFactorTest)
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
  double spruxAnalysisTime = tdelta(hrc::now() - startAnalysis).count();

  vector<int64_t> pivots(totalSize);

  auto startFactor = hrc::now();
  solver.factorLU(data.data(), pivots.data());
  double spruxFactorTime = tdelta(hrc::now() - startFactor).count();

  Vector<double> x = b;
  auto startSolve = hrc::now();
  solver.solveLU(data.data(), pivots.data(), x.data(), totalSize, 1);
  double spruxSolveTime = tdelta(hrc::now() - startSolve).count();

  double spruxResidual = (fullMat * x - b).norm() / b.norm();

  cout << "UMFPACK:"
       << "\n  Analysis: " << umfResult.analysisTime * 1000 << " ms"
       << "\n  Factor:   " << umfResult.factorTime * 1000 << " ms"
       << "\n  Solve:    " << umfResult.solveTime * 1000 << " ms"
       << "\n  Residual: " << umfResult.residual << endl;
  cout << "Sprux:"
       << "\n  Analysis: " << spruxAnalysisTime * 1000 << " ms"
       << "\n  Factor:   " << spruxFactorTime * 1000 << " ms"
       << "\n  Solve:    " << spruxSolveTime * 1000 << " ms"
       << "\n  Residual: " << spruxResidual << endl;

  EXPECT_LT(umfResult.residual, 1e-10) << "UMFPACK residual too large";
  EXPECT_LT(spruxResidual, 1e-8) << "Sprux residual too large";

  double solutionDiff = (x - umfResult.solution).norm() / umfResult.solution.norm();
  cout << "Solution difference: " << solutionDiff << endl;
  EXPECT_LT(solutionDiff, 1e-6) << "Solutions differ too much";
}

// Debug test to trace the BlockSparse data filling issue
TEST(LUComparison, DebugBlockSparse) {
  // Use a small structure for easier debugging: 4 blocks
  // Structure: block 0 connects to 0,1,2; block 1 connects to 1,2,3; etc.
  vector<set<int64_t>> colBlocks{{0, 1, 2}, {1, 2, 3}, {2, 3}, {3}};
  SparseStructure ssOrig = columnsToCscStruct(colBlocks).transpose();

  cout << "\n=== Debug BlockSparse Test ===" << endl;
  printSparseStructure("Original (before fill)", ssOrig);

  SparseStructure ss = ssOrig.addFullEliminationFill();
  printSparseStructure("After addFullEliminationFill", ss);

  // Use small block sizes: 2x2 each
  vector<int64_t> paramSize(colBlocks.size(), 2);
  int64_t totalSize = 8;

  cout << "Blocks: " << paramSize.size() << ", Total size: " << totalSize << "x" << totalSize
       << endl;

  // Build full dense matrix with ALL entries filled
  // Use a deterministic pattern for debugging
  Matrix<double> fullMat = Matrix<double>::Zero(totalSize, totalSize);

  vector<int64_t> spanStart;
  spanStart.push_back(0);
  for (int64_t ps : paramSize) {
    spanStart.push_back(spanStart.back() + ps);
  }

  // Fill based on sparsity structure
  // SparseStructure is CSR format: ptrs[row] gives start of row's column indices
  cout << "\nFilling blocks from SparseStructure (CSR format):" << endl;
  for (int64_t rowBlock = 0; rowBlock < (int64_t)ss.ptrs.size() - 1; rowBlock++) {
    for (int64_t k = ss.ptrs[rowBlock]; k < ss.ptrs[rowBlock + 1]; k++) {
      int64_t colBlock = ss.inds[k];
      cout << "  Block (" << rowBlock << ", " << colBlock << "): rows ["
           << spanStart[rowBlock] << ", " << spanStart[rowBlock + 1] << "), cols ["
           << spanStart[colBlock] << ", " << spanStart[colBlock + 1] << ")" << endl;

      // Fill the block with a recognizable pattern: 10*rowBlock + colBlock + 0.1*r + 0.01*c
      for (int64_t r = spanStart[rowBlock]; r < spanStart[rowBlock + 1]; r++) {
        for (int64_t c = spanStart[colBlock]; c < spanStart[colBlock + 1]; c++) {
          double val = 10 * rowBlock + colBlock + 0.1 * (r - spanStart[rowBlock]) +
                       0.01 * (c - spanStart[colBlock]);
          fullMat(r, c) = val;
          // Mirror to upper triangle for symmetric matrix
          if (r != c && rowBlock != colBlock) {
            fullMat(c, r) = val;
          }
        }
      }
    }
  }

  // Add diagonal dominance
  for (int64_t i = 0; i < totalSize; i++) {
    fullMat(i, i) += totalSize * 10;
  }

  cout << "\nFull matrix (before factorization):\n" << fullMat << endl;

  // Build Sprux skeleton
  vector<int64_t> lumpToSpan(paramSize.size() + 1);
  iota(lumpToSpan.begin(), lumpToSpan.end(), 0);
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);

  cout << "\nBefore initUpperTriangle:" << endl;
  cout << "  dataSize = " << factorSkel.dataSize() << endl;
  cout << "  upperDataSize = " << factorSkel.upperDataSize() << endl;

  factorSkel.initUpperTriangle();

  cout << "\nAfter initUpperTriangle:" << endl;
  cout << "  dataSize = " << factorSkel.dataSize() << endl;
  cout << "  upperDataSize = " << factorSkel.upperDataSize() << endl;
  cout << "  totalDataSize = " << factorSkel.totalDataSize() << endl;

  // Allocate and fill data using helper function
  vector<double> data(factorSkel.totalDataSize());
  cout << "\nFilling data using fillDataFromDenseMatrix (verbose):" << endl;
  fillDataFromDenseMatrix(factorSkel, data.data(), fullMat, true);

  // Verify by reconstructing
  Matrix<double> reconstructed = reconstructDenseMatrix(factorSkel, data.data());
  cout << "\nReconstructed matrix:\n" << reconstructed << endl;

  // Check for differences
  Matrix<double> diff = fullMat - reconstructed;
  double maxDiff = diff.cwiseAbs().maxCoeff();
  cout << "\nMax difference between original and reconstructed: " << maxDiff << endl;

  if (maxDiff > 1e-10) {
    cout << "Difference matrix (non-zeros indicate missing data):\n" << diff << endl;
    FAIL() << "Data reconstruction mismatch! maxDiff=" << maxDiff;
  }

  // Now try to solve
  Solver solver(std::move(factorSkel), {}, {}, fastOps());
  vector<int64_t> pivots(totalSize);

  // Make a copy of data before factorization for debugging
  vector<double> dataBeforeFactor = data;

  solver.factorLU(data.data(), pivots.data());

  cout << "\nPivots: ";
  for (auto p : pivots) cout << p << " ";
  cout << endl;

  // Reconstruct L and U from factored data to verify P*A = L*U
  cout << "\n=== Verifying P*A = L*U ===" << endl;
  const auto& skel = solver.skel();

  // Build L (unit lower triangular from diagonal and below-diagonal blocks)
  Matrix<double> L = Matrix<double>::Identity(totalSize, totalSize);
  Matrix<double> U = Matrix<double>::Zero(totalSize, totalSize);

  // Extract L and U from lower triangle (diagonal blocks have both L and U)
  for (int64_t l = 0; l < skel.numLumps(); l++) {
    int64_t lumpStartCol = skel.lumpStart[l];
    int64_t lumpSize = skel.lumpStart[l + 1] - lumpStartCol;
    int64_t chainStart = skel.chainColPtr[l];
    int64_t chainEnd = skel.chainColPtr[l + 1];

    for (int64_t c = chainStart; c < chainEnd; c++) {
      int64_t rowSpan = skel.chainRowSpan[c];
      int64_t rowStart = skel.spanStart[rowSpan];
      int64_t rowSize = skel.spanStart[rowSpan + 1] - rowStart;
      int64_t dataOffset = skel.chainData[c];

      bool isDiag = (rowSpan == skel.lumpToSpan[l]);

      for (int64_t r = 0; r < rowSize; r++) {
        for (int64_t col = 0; col < lumpSize; col++) {
          double val = data[dataOffset + r * lumpSize + col];
          if (isDiag) {
            // Diagonal block: L is strictly lower (unit diag), U is upper
            if (r > col) {
              L(rowStart + r, lumpStartCol + col) = val;  // Below diagonal -> L
            } else {
              U(rowStart + r, lumpStartCol + col) = val;  // On/above diagonal -> U
            }
          } else {
            // Below diagonal block: this is L
            L(rowStart + r, lumpStartCol + col) = val;
          }
        }
      }
    }
  }

  // Extract U from upper triangle storage
  int64_t upperDataBase = skel.dataSize();
  for (int64_t l = 0; l < skel.numLumps(); l++) {
    int64_t upperRowStart = skel.upperChainRowPtr[l];
    int64_t upperRowEnd = skel.upperChainRowPtr[l + 1];
    int64_t lumpStartRow = skel.lumpStart[l];
    int64_t lumpSize = skel.lumpStart[l + 1] - lumpStartRow;

    for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
      int64_t colSpan = skel.upperChainColSpan[i];
      int64_t colStart = skel.spanStart[colSpan];
      int64_t colSize = skel.spanStart[colSpan + 1] - colStart;
      int64_t upperDataOffset = upperDataBase + skel.upperChainData[i];

      for (int64_t r = 0; r < lumpSize; r++) {
        for (int64_t c = 0; c < colSize; c++) {
          U(lumpStartRow + r, colStart + c) = data[upperDataOffset + r * colSize + c];
        }
      }
    }
  }

  cout << "L (unit lower triangular):\n" << L << endl;
  cout << "\nU (upper triangular):\n" << U << endl;

  // Build P from pivots
  Matrix<double> P = Matrix<double>::Identity(totalSize, totalSize);
  for (int64_t l = 0; l < skel.numLumps(); l++) {
    int64_t lumpStart2 = skel.lumpStart[l];
    int64_t lumpSize = skel.lumpStart[l + 1] - lumpStart2;
    int64_t pivotOffset = lumpStart2;  // Row-based pivot index
    for (int64_t i = 0; i < lumpSize; i++) {
      int64_t pivotRow = pivots[pivotOffset + i];
      if (pivotRow != i) {
        P.row(lumpStart2 + i).swap(P.row(lumpStart2 + pivotRow));
      }
    }
  }
  cout << "\nP (permutation):\n" << P << endl;

  Matrix<double> PA = P * fullMat;
  Matrix<double> LU = L * U;
  cout << "\nP*A:\n" << PA << endl;
  cout << "\nL*U:\n" << LU << endl;

  double factorError = (PA - LU).norm();
  cout << "\n||P*A - L*U|| = " << factorError << endl;

  Vector<double> b = Vector<double>::Ones(totalSize);
  Vector<double> x = b;
  solver.solveLU(data.data(), pivots.data(), x.data(), totalSize, 1);

  double residual = (fullMat * x - b).norm() / b.norm();
  cout << "Sprux solution: " << x.transpose() << endl;
  cout << "Residual: " << residual << endl;

  // Compare with Eigen
  Vector<double> xRef = fullMat.partialPivLu().solve(b);
  double residualRef = (fullMat * xRef - b).norm() / b.norm();
  cout << "Eigen solution: " << xRef.transpose() << endl;
  cout << "Eigen residual: " << residualRef << endl;

  EXPECT_LT(residual, 1e-8) << "Sprux residual too large";
}

// Compare on a sparse block matrix
TEST(LUComparison, VsUmfpack_BlockSparse) {
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

  // Solve with Sprux
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
  double spruxAnalysisTime = tdelta(hrc::now() - startAnalysis).count();

  vector<int64_t> pivots(totalSize);

  auto startFactor = hrc::now();
  solver.factorLU(data.data(), pivots.data());
  double spruxFactorTime = tdelta(hrc::now() - startFactor).count();

  Vector<double> x = b;
  auto startSolve = hrc::now();
  solver.solveLU(data.data(), pivots.data(), x.data(), totalSize, 1);
  double spruxSolveTime = tdelta(hrc::now() - startSolve).count();

  double spruxResidual = (fullMat * x - b).norm() / b.norm();

  // Print comparison
  cout << "UMFPACK:"
       << "\n  Analysis: " << umfResult.analysisTime * 1000 << " ms"
       << "\n  Factor:   " << umfResult.factorTime * 1000 << " ms"
       << "\n  Solve:    " << umfResult.solveTime * 1000 << " ms"
       << "\n  Residual: " << umfResult.residual << endl;
  cout << "Sprux:"
       << "\n  Analysis: " << spruxAnalysisTime * 1000 << " ms"
       << "\n  Factor:   " << spruxFactorTime * 1000 << " ms"
       << "\n  Solve:    " << spruxSolveTime * 1000 << " ms"
       << "\n  Residual: " << spruxResidual << endl;

  // Both should have small residuals
  EXPECT_LT(umfResult.residual, 1e-10) << "UMFPACK residual too large";
  EXPECT_LT(spruxResidual, 1e-8) << "Sprux residual too large";

  // Solutions should be similar
  double solutionDiff = (x - umfResult.solution).norm() / umfResult.solution.norm();
  cout << "Solution difference: " << solutionDiff << endl;
  EXPECT_LT(solutionDiff, 1e-6) << "Solutions differ too much";
}

// Compare on a larger sparse matrix for performance
TEST(LUComparison, VsUmfpack_Performance) {
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

  // Solve with Sprux
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
  double spruxAnalysisTime = tdelta(hrc::now() - startAnalysis).count();

  vector<int64_t> pivots(totalSize);

  auto startFactor = hrc::now();
  solver.factorLU(data.data(), pivots.data());
  double spruxFactorTime = tdelta(hrc::now() - startFactor).count();

  Vector<double> x = b;
  auto startSolve = hrc::now();
  solver.solveLU(data.data(), pivots.data(), x.data(), totalSize, 1);
  double spruxSolveTime = tdelta(hrc::now() - startSolve).count();

  double spruxResidual = (fullMat * x - b).norm() / b.norm();

  cout << "UMFPACK:"
       << "\n  Analysis: " << umfResult.analysisTime * 1000 << " ms"
       << "\n  Factor:   " << umfResult.factorTime * 1000 << " ms"
       << "\n  Solve:    " << umfResult.solveTime * 1000 << " ms"
       << "\n  Residual: " << umfResult.residual << endl;
  cout << "Sprux:"
       << "\n  Analysis: " << spruxAnalysisTime * 1000 << " ms"
       << "\n  Factor:   " << spruxFactorTime * 1000 << " ms"
       << "\n  Solve:    " << spruxSolveTime * 1000 << " ms"
       << "\n  Residual: " << spruxResidual << endl;

  // Both should have reasonable residuals
  EXPECT_LT(umfResult.residual, 1e-8) << "UMFPACK residual too large";
  EXPECT_LT(spruxResidual, 1e-6) << "Sprux residual too large";
}

// ============================================================================
// Helper: Build non-symmetric block-sparse matrix and fill Sprux + UMFPACK data
// ============================================================================

struct LUTestData {
  Matrix<double> fullMat;
  vector<double> data;
  unique_ptr<CoalescedBlockMatrixSkel> factorSkel;
  int64_t totalSize;
  vector<int64_t> spanStart;
  // UMFPACK CSC format
  vector<int64_t> colPtr;
  vector<int64_t> rowIdx;
  vector<double> val;
};

// Build a non-symmetric block-sparse test matrix from a sparsity structure.
// paramSize gives the size of each block, rng is the random generator.
LUTestData buildNonSymmetricTestData(const SparseStructure& ss, const vector<int64_t>& paramSize,
                                      mt19937& rng) {
  LUTestData td;
  uniform_real_distribution<double> unif(-1.0, 1.0);

  td.totalSize = 0;
  for (int64_t ps : paramSize) td.totalSize += ps;

  td.spanStart.push_back(0);
  for (int64_t ps : paramSize) td.spanStart.push_back(td.spanStart.back() + ps);

  td.fullMat = Matrix<double>::Zero(td.totalSize, td.totalSize);

  // Fill lower triangle blocks from sparsity structure
  for (int64_t rowBlock = 0; rowBlock < (int64_t)ss.ptrs.size() - 1; rowBlock++) {
    for (int64_t k = ss.ptrs[rowBlock]; k < ss.ptrs[rowBlock + 1]; k++) {
      int64_t colBlock = ss.inds[k];
      for (int64_t r = td.spanStart[rowBlock]; r < td.spanStart[rowBlock + 1]; r++)
        for (int64_t c = td.spanStart[colBlock]; c < td.spanStart[colBlock + 1]; c++)
          td.fullMat(r, c) = unif(rng);
    }
  }

  // Fill upper triangle blocks INDEPENDENTLY (non-symmetric!)
  // The upper triangle has the transposed sparsity pattern
  for (int64_t rowBlock = 0; rowBlock < (int64_t)ss.ptrs.size() - 1; rowBlock++) {
    for (int64_t k = ss.ptrs[rowBlock]; k < ss.ptrs[rowBlock + 1]; k++) {
      int64_t colBlock = ss.inds[k];
      if (colBlock != rowBlock) {
        // Upper entry (colBlock, rowBlock) gets independent random values
        for (int64_t r = td.spanStart[colBlock]; r < td.spanStart[colBlock + 1]; r++)
          for (int64_t c = td.spanStart[rowBlock]; c < td.spanStart[rowBlock + 1]; c++)
            td.fullMat(r, c) = unif(rng);
      }
    }
  }

  // Diagonal dominance
  for (int64_t i = 0; i < td.totalSize; i++) td.fullMat(i, i) += td.totalSize * 3;

  // Build Sprux skeleton
  vector<int64_t> lumpToSpan(paramSize.size() + 1);
  iota(lumpToSpan.begin(), lumpToSpan.end(), 0);
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  td.factorSkel = make_unique<CoalescedBlockMatrixSkel>(td.spanStart, lumpToSpan, groupedSs.ptrs,
                                                        groupedSs.inds);
  td.factorSkel->initUpperTriangle();

  // Fill Sprux data
  td.data.resize(td.factorSkel->totalDataSize());
  fillDataFromDenseMatrix(*td.factorSkel, td.data.data(), td.fullMat);

  // Build CSC for UMFPACK
  td.colPtr.push_back(0);
  for (int64_t col = 0; col < td.totalSize; col++) {
    for (int64_t row = 0; row < td.totalSize; row++) {
      if (td.fullMat(row, col) != 0.0) {
        td.rowIdx.push_back(row);
        td.val.push_back(td.fullMat(row, col));
      }
    }
    td.colPtr.push_back(td.rowIdx.size());
  }

  return td;
}

// Helper to run Sprux LU solve and return residual
double solveSprux(LUTestData& td, const Vector<double>& b, Vector<double>& xOut) {
  Solver solver(std::move(*td.factorSkel), {}, {}, fastOps());
  vector<int64_t> pivots(td.totalSize);
  solver.factorLU(td.data.data(), pivots.data());
  xOut = b;
  solver.solveLU(td.data.data(), pivots.data(), xOut.data(), td.totalSize, 1);
  return (td.fullMat * xOut - b).norm() / b.norm();
}

// Compare with truly non-symmetric matrices (SPICE-like asymmetric coupling)
TEST(LUComparison, VsUmfpack_NonSymmetric) {
  SparseMatGenerator gen = SparseMatGenerator::genFlat(25, 0.25, 123);
  SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();

  vector<int64_t> paramSize(gen.columns.size(), 3);

  mt19937 rng(123);
  auto td = buildNonSymmetricTestData(ss, paramSize, rng);

  uniform_real_distribution<double> unif(-1.0, 1.0);
  Vector<double> b(td.totalSize);
  for (int64_t i = 0; i < td.totalSize; i++) b(i) = unif(rng);

  auto umfResult = solveWithUmfpack(td.colPtr, td.rowIdx, td.val, td.totalSize, b);

  Vector<double> xSprux;
  double spruxResidual = solveSprux(td, b, xSprux);

  cout << "\n=== Non-Symmetric Matrix Comparison ===" << endl;
  cout << "Size: " << td.totalSize << "x" << td.totalSize << endl;
  cout << "UMFPACK residual: " << umfResult.residual << endl;
  cout << "Sprux residual: " << spruxResidual << endl;

  EXPECT_LT(umfResult.residual, 1e-10) << "UMFPACK residual too large";
  EXPECT_LT(spruxResidual, 1e-8) << "Sprux residual too large";

  double solutionDiff = (xSprux - umfResult.solution).norm() / umfResult.solution.norm();
  cout << "Solution difference: " << solutionDiff << endl;
  EXPECT_LT(solutionDiff, 1e-6) << "Solutions differ too much";
}

// Larger block-sparse structure with mixed block sizes (2-15)
TEST(LUComparison, VsUmfpack_LargerMixedBlocks) {
  SparseMatGenerator gen = SparseMatGenerator::genFlat(50, 0.08, 77);
  SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();

  // Mixed block sizes 2-8 (typical of circuit blocks)
  mt19937 rng(77);
  uniform_int_distribution<int64_t> sizeDist(2, 8);
  vector<int64_t> paramSize;
  for (size_t i = 0; i < gen.columns.size(); i++) paramSize.push_back(sizeDist(rng));

  auto td = buildNonSymmetricTestData(ss, paramSize, rng);

  uniform_real_distribution<double> unif(-1.0, 1.0);
  Vector<double> b(td.totalSize);
  for (int64_t i = 0; i < td.totalSize; i++) b(i) = unif(rng);

  auto umfResult = solveWithUmfpack(td.colPtr, td.rowIdx, td.val, td.totalSize, b);

  Vector<double> xSprux;
  double spruxResidual = solveSprux(td, b, xSprux);

  cout << "\n=== Larger Mixed-Block Non-Symmetric Comparison ===" << endl;
  cout << "Blocks: " << paramSize.size() << ", Size: " << td.totalSize << "x" << td.totalSize
       << endl;
  cout << "UMFPACK residual: " << umfResult.residual << endl;
  cout << "Sprux residual: " << spruxResidual << endl;

  EXPECT_LT(umfResult.residual, 1e-10) << "UMFPACK residual too large";
  EXPECT_LT(spruxResidual, 1e-8) << "Sprux residual too large";

  double solutionDiff = (xSprux - umfResult.solution).norm() / umfResult.solution.norm();
  cout << "Solution difference: " << solutionDiff << endl;
  EXPECT_LT(solutionDiff, 1e-6) << "Solutions differ too much";
}

// Multiple RHS solve
TEST(LUComparison, VsUmfpack_MultipleRHS) {
  SparseMatGenerator gen = SparseMatGenerator::genFlat(20, 0.3, 99);
  SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();

  vector<int64_t> paramSize(gen.columns.size(), 3);

  mt19937 rng(99);
  auto td = buildNonSymmetricTestData(ss, paramSize, rng);

  int nRHS = 5;
  uniform_real_distribution<double> unif(-1.0, 1.0);
  Matrix<double> B(td.totalSize, nRHS);
  for (int64_t i = 0; i < td.totalSize; i++)
    for (int j = 0; j < nRHS; j++) B(i, j) = unif(rng);

  // Solve with Sprux (multiple RHS at once)
  Solver solver(std::move(*td.factorSkel), {}, {}, fastOps());
  vector<int64_t> pivots(td.totalSize);
  solver.factorLU(td.data.data(), pivots.data());

  Matrix<double> X = B;
  solver.solveLU(td.data.data(), pivots.data(), X.data(), td.totalSize, nRHS);

  // Solve each RHS with UMFPACK for comparison
  cout << "\n=== Multiple RHS Comparison (nRHS=" << nRHS << ") ===" << endl;
  cout << "Size: " << td.totalSize << "x" << td.totalSize << endl;

  for (int rhs = 0; rhs < nRHS; rhs++) {
    Vector<double> bCol = B.col(rhs);
    auto umfResult = solveWithUmfpack(td.colPtr, td.rowIdx, td.val, td.totalSize, bCol);

    Vector<double> xCol = X.col(rhs);
    double spruxResidual = (td.fullMat * xCol - bCol).norm() / bCol.norm();

    cout << "  RHS " << rhs << ": UMFPACK=" << umfResult.residual
         << ", Sprux=" << spruxResidual << endl;

    EXPECT_LT(umfResult.residual, 1e-10) << "UMFPACK residual too large for RHS " << rhs;
    EXPECT_LT(spruxResidual, 1e-8) << "Sprux residual too large for RHS " << rhs;

    double solutionDiff = (xCol - umfResult.solution).norm() / umfResult.solution.norm();
    EXPECT_LT(solutionDiff, 1e-6) << "Solutions differ too much for RHS " << rhs;
  }
}

// Grid topology (typical of 2D circuit mesh)
TEST(LUComparison, VsUmfpack_GridTopology) {
  SparseMatGenerator gen = SparseMatGenerator::genGrid(10, 10, 0.5, 2, 55);
  SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();

  mt19937 rng(55);
  uniform_int_distribution<int64_t> sizeDist(2, 5);
  vector<int64_t> paramSize;
  for (size_t i = 0; i < gen.columns.size(); i++) paramSize.push_back(sizeDist(rng));

  auto td = buildNonSymmetricTestData(ss, paramSize, rng);

  uniform_real_distribution<double> unif(-1.0, 1.0);
  Vector<double> b(td.totalSize);
  for (int64_t i = 0; i < td.totalSize; i++) b(i) = unif(rng);

  auto umfResult = solveWithUmfpack(td.colPtr, td.rowIdx, td.val, td.totalSize, b);

  Vector<double> xSprux;
  double spruxResidual = solveSprux(td, b, xSprux);

  cout << "\n=== Grid Topology (10x10) Non-Symmetric ===" << endl;
  cout << "Blocks: " << paramSize.size() << ", Size: " << td.totalSize << "x" << td.totalSize
       << endl;
  cout << "UMFPACK residual: " << umfResult.residual << endl;
  cout << "Sprux residual: " << spruxResidual << endl;

  EXPECT_LT(umfResult.residual, 1e-10) << "UMFPACK residual too large";
  EXPECT_LT(spruxResidual, 1e-8) << "Sprux residual too large";

  double solutionDiff = (xSprux - umfResult.solution).norm() / umfResult.solution.norm();
  cout << "Solution difference: " << solutionDiff << endl;
  EXPECT_LT(solutionDiff, 1e-6) << "Solutions differ too much";
}

// Meridian topology
TEST(LUComparison, VsUmfpack_MeridianTopology) {
  SparseMatGenerator gen = SparseMatGenerator::genMeridians(4, 10, 0.5, 3, 4, 2, 2, 88);
  SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();

  vector<int64_t> paramSize(gen.columns.size(), 3);

  mt19937 rng(88);
  auto td = buildNonSymmetricTestData(ss, paramSize, rng);

  uniform_real_distribution<double> unif(-1.0, 1.0);
  Vector<double> b(td.totalSize);
  for (int64_t i = 0; i < td.totalSize; i++) b(i) = unif(rng);

  auto umfResult = solveWithUmfpack(td.colPtr, td.rowIdx, td.val, td.totalSize, b);

  Vector<double> xSprux;
  double spruxResidual = solveSprux(td, b, xSprux);

  cout << "\n=== Meridian Topology Non-Symmetric ===" << endl;
  cout << "Blocks: " << paramSize.size() << ", Size: " << td.totalSize << "x" << td.totalSize
       << endl;
  cout << "UMFPACK residual: " << umfResult.residual << endl;
  cout << "Sprux residual: " << spruxResidual << endl;

  EXPECT_LT(umfResult.residual, 1e-10) << "UMFPACK residual too large";
  EXPECT_LT(spruxResidual, 1e-8) << "Sprux residual too large";

  double solutionDiff = (xSprux - umfResult.solution).norm() / umfResult.solution.norm();
  cout << "Solution difference: " << solutionDiff << endl;
  EXPECT_LT(solutionDiff, 1e-6) << "Solutions differ too much";
}

// ============================================================================
// GPU Backend vs UMFPACK comparison tests
// ============================================================================

#ifdef SPRUX_USE_METAL
#include "sprux/sprux/MetalDefs.h"

// Build float test data from the same sparsity structure (Metal only supports float)
struct LUTestDataFloat {
  Matrix<float> fullMat;
  vector<float> data;
  unique_ptr<CoalescedBlockMatrixSkel> factorSkel;
  int64_t totalSize;
  vector<int64_t> spanStart;
};

LUTestDataFloat buildNonSymmetricTestDataFloat(const SparseStructure& ss,
                                                const vector<int64_t>& paramSize, mt19937& rng) {
  LUTestDataFloat td;
  uniform_real_distribution<float> unif(-1.0f, 1.0f);

  td.totalSize = 0;
  for (int64_t ps : paramSize) td.totalSize += ps;

  td.spanStart.push_back(0);
  for (int64_t ps : paramSize) td.spanStart.push_back(td.spanStart.back() + ps);

  td.fullMat = Matrix<float>::Zero(td.totalSize, td.totalSize);

  for (int64_t rowBlock = 0; rowBlock < (int64_t)ss.ptrs.size() - 1; rowBlock++) {
    for (int64_t k = ss.ptrs[rowBlock]; k < ss.ptrs[rowBlock + 1]; k++) {
      int64_t colBlock = ss.inds[k];
      for (int64_t r = td.spanStart[rowBlock]; r < td.spanStart[rowBlock + 1]; r++)
        for (int64_t c = td.spanStart[colBlock]; c < td.spanStart[colBlock + 1]; c++)
          td.fullMat(r, c) = unif(rng);
    }
  }

  for (int64_t rowBlock = 0; rowBlock < (int64_t)ss.ptrs.size() - 1; rowBlock++) {
    for (int64_t k = ss.ptrs[rowBlock]; k < ss.ptrs[rowBlock + 1]; k++) {
      int64_t colBlock = ss.inds[k];
      if (colBlock != rowBlock) {
        for (int64_t r = td.spanStart[colBlock]; r < td.spanStart[colBlock + 1]; r++)
          for (int64_t c = td.spanStart[rowBlock]; c < td.spanStart[rowBlock + 1]; c++)
            td.fullMat(r, c) = unif(rng);
      }
    }
  }

  for (int64_t i = 0; i < td.totalSize; i++) td.fullMat(i, i) += td.totalSize * 3;

  vector<int64_t> lumpToSpan(paramSize.size() + 1);
  iota(lumpToSpan.begin(), lumpToSpan.end(), 0);
  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  td.factorSkel = make_unique<CoalescedBlockMatrixSkel>(td.spanStart, lumpToSpan, groupedSs.ptrs,
                                                        groupedSs.inds);
  td.factorSkel->initUpperTriangle();

  td.data.resize(td.factorSkel->totalDataSize());
  fillDataFromDenseMatrix(*td.factorSkel, td.data.data(), td.fullMat);

  return td;
}

// Helper: solve with Metal backend, return residual
float solveMetalLU(LUTestDataFloat& td, const Vector<float>& b, Vector<float>& xOut) {
  Solver solver(std::move(*td.factorSkel), {}, {}, metalOps());
  vector<int64_t> pivots(td.totalSize);

  // Factor on GPU
  {
    MetalMirror<float> dataGpu(td.data);
    solver.factorLU(dataGpu.ptr(), pivots.data());
    dataGpu.get(td.data);
  }

  // Solve on GPU
  xOut = b;
  {
    vector<float> xVec(xOut.data(), xOut.data() + td.totalSize);
    MetalMirror<float> dataGpu(td.data);
    MetalMirror<float> xGpu(xVec);
    solver.solveLU(dataGpu.ptr(), pivots.data(), xGpu.ptr(), td.totalSize, 1);
    xGpu.get(xVec);
    for (int64_t i = 0; i < td.totalSize; i++) xOut(i) = xVec[i];
  }

  return (td.fullMat * xOut - b).norm() / b.norm();
}

// Helper: build UMFPACK CSC data from float matrix (promoted to double for UMFPACK)
void buildUmfpackCSC(const Matrix<float>& fullMat, int64_t n, vector<int64_t>& colPtr,
                     vector<int64_t>& rowIdx, vector<double>& val) {
  colPtr.clear();
  rowIdx.clear();
  val.clear();
  colPtr.push_back(0);
  for (int64_t col = 0; col < n; col++) {
    for (int64_t row = 0; row < n; row++) {
      if (fullMat(row, col) != 0.0f) {
        rowIdx.push_back(row);
        val.push_back(static_cast<double>(fullMat(row, col)));
      }
    }
    colPtr.push_back(rowIdx.size());
  }
}

TEST(LUComparison, MetalVsUmfpack_BlockSparse) {
  SparseMatGenerator gen = SparseMatGenerator::genFlat(25, 0.25, 456);
  SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();
  vector<int64_t> paramSize(gen.columns.size(), 3);

  mt19937 rng(456);
  auto td = buildNonSymmetricTestDataFloat(ss, paramSize, rng);

  // Build UMFPACK data from the same matrix (promoted to double)
  vector<int64_t> colPtr;
  vector<int64_t> rowIdx;
  vector<double> val;
  buildUmfpackCSC(td.fullMat, td.totalSize, colPtr, rowIdx, val);

  uniform_real_distribution<float> unif(-1.0f, 1.0f);
  Vector<float> bf(td.totalSize);
  for (int64_t i = 0; i < td.totalSize; i++) bf(i) = unif(rng);

  // UMFPACK solve (double precision)
  Vector<double> bd = bf.cast<double>();
  auto umfResult = solveWithUmfpack(colPtr, rowIdx, val, td.totalSize, bd);

  // Metal solve (float precision)
  Vector<float> xMetal;
  float metalResidual = solveMetalLU(td, bf, xMetal);

  cout << "\n=== Metal vs UMFPACK: Block-Sparse ===" << endl;
  cout << "Size: " << td.totalSize << "x" << td.totalSize << endl;
  cout << "UMFPACK residual: " << umfResult.residual << endl;
  cout << "Metal residual: " << metalResidual << endl;

  EXPECT_LT(umfResult.residual, 1e-10) << "UMFPACK residual too large";
  EXPECT_LT(metalResidual, 5e-3) << "Metal residual too large (float precision)";

  // Compare solutions (float vs double, so tolerance is relaxed)
  double solutionDiff =
      (xMetal.cast<double>() - umfResult.solution).norm() / umfResult.solution.norm();
  cout << "Solution difference (float vs double): " << solutionDiff << endl;
  EXPECT_LT(solutionDiff, 1e-2) << "Metal and UMFPACK solutions differ too much";
}

TEST(LUComparison, MetalVsUmfpack_NonSymmetric) {
  SparseMatGenerator gen = SparseMatGenerator::genFlat(25, 0.25, 789);
  SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();
  vector<int64_t> paramSize(gen.columns.size(), 3);

  mt19937 rng(789);
  auto td = buildNonSymmetricTestDataFloat(ss, paramSize, rng);

  vector<int64_t> colPtr;
  vector<int64_t> rowIdx;
  vector<double> val;
  buildUmfpackCSC(td.fullMat, td.totalSize, colPtr, rowIdx, val);

  uniform_real_distribution<float> unif(-1.0f, 1.0f);
  Vector<float> bf(td.totalSize);
  for (int64_t i = 0; i < td.totalSize; i++) bf(i) = unif(rng);

  Vector<double> bd = bf.cast<double>();
  auto umfResult = solveWithUmfpack(colPtr, rowIdx, val, td.totalSize, bd);

  Vector<float> xMetal;
  float metalResidual = solveMetalLU(td, bf, xMetal);

  cout << "\n=== Metal vs UMFPACK: Non-Symmetric ===" << endl;
  cout << "Size: " << td.totalSize << "x" << td.totalSize << endl;
  cout << "UMFPACK residual: " << umfResult.residual << endl;
  cout << "Metal residual: " << metalResidual << endl;

  EXPECT_LT(umfResult.residual, 1e-10);
  EXPECT_LT(metalResidual, 5e-3);
}

TEST(LUComparison, MetalVsUmfpack_MixedBlocks) {
  SparseMatGenerator gen = SparseMatGenerator::genFlat(40, 0.1, 321);
  SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();

  mt19937 rng(321);
  uniform_int_distribution<int64_t> sizeDist(2, 8);
  vector<int64_t> paramSize;
  for (size_t i = 0; i < gen.columns.size(); i++) paramSize.push_back(sizeDist(rng));

  auto td = buildNonSymmetricTestDataFloat(ss, paramSize, rng);

  vector<int64_t> colPtr;
  vector<int64_t> rowIdx;
  vector<double> val;
  buildUmfpackCSC(td.fullMat, td.totalSize, colPtr, rowIdx, val);

  uniform_real_distribution<float> unif(-1.0f, 1.0f);
  Vector<float> bf(td.totalSize);
  for (int64_t i = 0; i < td.totalSize; i++) bf(i) = unif(rng);

  Vector<double> bd = bf.cast<double>();
  auto umfResult = solveWithUmfpack(colPtr, rowIdx, val, td.totalSize, bd);

  Vector<float> xMetal;
  float metalResidual = solveMetalLU(td, bf, xMetal);

  cout << "\n=== Metal vs UMFPACK: Mixed Block Sizes ===" << endl;
  cout << "Blocks: " << paramSize.size() << ", Size: " << td.totalSize << "x" << td.totalSize
       << endl;
  cout << "UMFPACK residual: " << umfResult.residual << endl;
  cout << "Metal residual: " << metalResidual << endl;

  EXPECT_LT(umfResult.residual, 1e-10);
  EXPECT_LT(metalResidual, 5e-3);
}

TEST(LUComparison, MetalVsUmfpack_GridTopology) {
  SparseMatGenerator gen = SparseMatGenerator::genGrid(5, 5, 0.5, 2, 654);
  SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();
  vector<int64_t> paramSize(gen.columns.size(), 3);

  mt19937 rng(654);
  auto td = buildNonSymmetricTestDataFloat(ss, paramSize, rng);

  vector<int64_t> colPtr;
  vector<int64_t> rowIdx;
  vector<double> val;
  buildUmfpackCSC(td.fullMat, td.totalSize, colPtr, rowIdx, val);

  uniform_real_distribution<float> unif(-1.0f, 1.0f);
  Vector<float> bf(td.totalSize);
  for (int64_t i = 0; i < td.totalSize; i++) bf(i) = unif(rng);

  Vector<double> bd = bf.cast<double>();
  auto umfResult = solveWithUmfpack(colPtr, rowIdx, val, td.totalSize, bd);

  Vector<float> xMetal;
  float metalResidual = solveMetalLU(td, bf, xMetal);

  cout << "\n=== Metal vs UMFPACK: Grid Topology ===" << endl;
  cout << "Blocks: " << paramSize.size() << ", Size: " << td.totalSize << "x" << td.totalSize
       << endl;
  cout << "UMFPACK residual: " << umfResult.residual << endl;
  cout << "Metal residual: " << metalResidual << endl;

  EXPECT_LT(umfResult.residual, 1e-10);
  EXPECT_LT(metalResidual, 5e-3);
}

// Performance comparison: Metal vs UMFPACK at different scales
TEST(LUComparison, MetalVsUmfpack_Performance) {
  struct TestCase {
    string name;
    int numBlocks;
    double density;
    int blockSize;
    int seed;
  };

  vector<TestCase> cases = {
      {"Small (50 blocks)", 50, 0.25, 3, 100},
      {"Medium (150 blocks)", 150, 0.08, 3, 200},
      {"Large (300 blocks)", 300, 0.03, 3, 300},
  };

  cout << "\n=== Metal vs UMFPACK Performance Comparison ===" << endl;
  cout << left << setw(25) << "Case" << setw(10) << "Size" << setw(15) << "UMFPACK(ms)"
       << setw(15) << "Metal(ms)" << setw(15) << "Speedup" << setw(15) << "UMF resid"
       << setw(15) << "MTL resid" << endl;
  cout << string(100, '-') << endl;

  for (const auto& tc : cases) {
    SparseMatGenerator gen = SparseMatGenerator::genFlat(tc.numBlocks, tc.density, tc.seed);
    SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();
    vector<int64_t> paramSize(gen.columns.size(), tc.blockSize);

    mt19937 rng(tc.seed);

    // Build float test data for Metal
    auto tdFloat = buildNonSymmetricTestDataFloat(ss, paramSize, rng);

    // Build double test data for UMFPACK
    rng.seed(tc.seed);  // Reset rng for identical matrix
    auto tdDouble = buildNonSymmetricTestData(ss, paramSize, rng);

    // RHS vector
    rng.seed(tc.seed + 1000);
    uniform_real_distribution<float> unif(-1.0f, 1.0f);
    Vector<float> bf(tdFloat.totalSize);
    for (int64_t i = 0; i < tdFloat.totalSize; i++) bf(i) = unif(rng);
    Vector<double> bd = bf.cast<double>();

    // UMFPACK timing (includes symbolic + numeric + solve)
    auto umfResult = solveWithUmfpack(tdDouble.colPtr, tdDouble.rowIdx, tdDouble.val,
                                      tdDouble.totalSize, bd);
    double umfFactorMs = (umfResult.analysisTime + umfResult.factorTime) * 1000;
    double umfSolveMs = umfResult.solveTime * 1000;

    // Metal: separate setup from factor+solve
    int64_t n = tdFloat.totalSize;
    Solver metalSolver(std::move(*tdFloat.factorSkel), {}, {}, metalOps());
    vector<int64_t> pivots(n);

    // Metal factor timing
    MetalMirror<float> dataGpu(tdFloat.data);
    auto mtlFactorStart = hrc::now();
    metalSolver.factorLU(dataGpu.ptr(), pivots.data());
    double mtlFactorMs = tdelta(hrc::now() - mtlFactorStart).count() * 1000;
    dataGpu.get(tdFloat.data);

    // Metal solve timing
    vector<float> xVec(bf.data(), bf.data() + n);
    MetalMirror<float> dataGpu2(tdFloat.data);
    MetalMirror<float> xGpu(xVec);
    auto mtlSolveStart = hrc::now();
    metalSolver.solveLU(dataGpu2.ptr(), pivots.data(), xGpu.ptr(), n, 1);
    double mtlSolveMs = tdelta(hrc::now() - mtlSolveStart).count() * 1000;
    xGpu.get(xVec);

    Vector<float> xMetal(n);
    for (int64_t i = 0; i < n; i++) xMetal(i) = xVec[i];
    float metalResidual = (tdFloat.fullMat * xMetal - bf).norm() / bf.norm();

    cout << left << setw(25) << tc.name << setw(10) << n
         << setw(15) << fixed << setprecision(2) << umfFactorMs + umfSolveMs
         << setw(15) << mtlFactorMs + mtlSolveMs
         << setw(15) << setprecision(2) << (umfFactorMs + umfSolveMs) / (mtlFactorMs + mtlSolveMs)
         << "x" << setw(15) << scientific << setprecision(2) << umfResult.residual
         << setw(15) << metalResidual << endl;

    // Correctness checks
    EXPECT_LT(umfResult.residual, 1e-10) << tc.name << ": UMFPACK residual too large";
    EXPECT_LT(metalResidual, 5e-3) << tc.name << ": Metal residual too large";
  }
}

// Performance comparison: Metal vs UMFPACK varying block sizes
TEST(LUComparison, MetalVsUmfpack_BlockSizeScaling) {
  struct TestCase {
    string name;
    int numBlocks;
    double density;
    int blockSize;
    int seed;
  };

  vector<TestCase> cases = {
      {"3x3 blocks (50)", 50, 0.25, 3, 100},
      {"6x6 blocks (50)", 50, 0.25, 6, 100},
      {"12x12 blocks (50)", 50, 0.25, 12, 100},
      {"24x24 blocks (50)", 50, 0.25, 24, 100},
      {"48x48 blocks (30)", 30, 0.30, 48, 100},
      {"6x6 blocks (100)", 100, 0.10, 6, 200},
      {"12x12 blocks (100)", 100, 0.10, 12, 200},
      {"24x24 blocks (100)", 100, 0.10, 24, 200},
  };

  cout << "\n=== Metal vs UMFPACK: Block Size Scaling ===" << endl;
  cout << left << setw(25) << "Case" << setw(10) << "Size" << setw(15) << "UMFPACK(ms)"
       << setw(15) << "Metal(ms)" << setw(15) << "Ratio" << setw(15) << "UMF resid"
       << setw(15) << "MTL resid" << endl;
  cout << string(110, '-') << endl;

  for (const auto& tc : cases) {
    SparseMatGenerator gen = SparseMatGenerator::genFlat(tc.numBlocks, tc.density, tc.seed);
    SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();
    vector<int64_t> paramSize(gen.columns.size(), tc.blockSize);

    mt19937 rng(tc.seed);

    // Build float test data for Metal
    auto tdFloat = buildNonSymmetricTestDataFloat(ss, paramSize, rng);

    // Build double test data for UMFPACK
    rng.seed(tc.seed);
    auto tdDouble = buildNonSymmetricTestData(ss, paramSize, rng);

    // RHS vector
    rng.seed(tc.seed + 1000);
    uniform_real_distribution<float> unif(-1.0f, 1.0f);
    Vector<float> bf(tdFloat.totalSize);
    for (int64_t i = 0; i < tdFloat.totalSize; i++) bf(i) = unif(rng);
    Vector<double> bd = bf.cast<double>();

    // UMFPACK timing
    auto umfResult = solveWithUmfpack(tdDouble.colPtr, tdDouble.rowIdx, tdDouble.val,
                                      tdDouble.totalSize, bd);
    double umfTotalMs = (umfResult.analysisTime + umfResult.factorTime + umfResult.solveTime) * 1000;

    // Metal timing
    int64_t n = tdFloat.totalSize;
    Solver metalSolver(std::move(*tdFloat.factorSkel), {}, {}, metalOps());
    vector<int64_t> pivots(n);

    MetalMirror<float> dataGpu(tdFloat.data);
    auto mtlStart = hrc::now();
    metalSolver.factorLU(dataGpu.ptr(), pivots.data());
    double mtlFactorMs = tdelta(hrc::now() - mtlStart).count() * 1000;
    dataGpu.get(tdFloat.data);

    vector<float> xVec(bf.data(), bf.data() + n);
    MetalMirror<float> dataGpu2(tdFloat.data);
    MetalMirror<float> xGpu(xVec);
    auto mtlSolveStart = hrc::now();
    metalSolver.solveLU(dataGpu2.ptr(), pivots.data(), xGpu.ptr(), n, 1);
    double mtlSolveMs = tdelta(hrc::now() - mtlSolveStart).count() * 1000;
    xGpu.get(xVec);

    double mtlTotalMs = mtlFactorMs + mtlSolveMs;

    Vector<float> xMetal(n);
    for (int64_t i = 0; i < n; i++) xMetal(i) = xVec[i];
    float metalResidual = (tdFloat.fullMat * xMetal - bf).norm() / bf.norm();

    string ratio;
    if (mtlTotalMs < umfTotalMs) {
      ratio = to_string(umfTotalMs / mtlTotalMs).substr(0, 5) + "x faster";
    } else {
      ratio = to_string(mtlTotalMs / umfTotalMs).substr(0, 5) + "x slower";
    }

    cout << left << setw(25) << tc.name << setw(10) << n
         << setw(15) << fixed << setprecision(2) << umfTotalMs
         << setw(15) << mtlTotalMs
         << setw(15) << ratio
         << setw(15) << scientific << setprecision(2) << umfResult.residual
         << setw(15) << metalResidual << endl;

    EXPECT_LT(metalResidual, 5e-3) << tc.name << ": Metal residual too large";
  }
}
#endif  // SPRUX_USE_METAL

#ifdef SPRUX_USE_CUBLAS
// CUDA GPU vs UMFPACK comparison
// CUDA supports both float and double; we test double for direct comparison with UMFPACK

double solveCudaLU(LUTestData& td, const Vector<double>& b, Vector<double>& xOut) {
  Solver solver(std::move(*td.factorSkel), {}, {}, cudaOps());
  vector<int64_t> pivots(td.totalSize);
  solver.factorLU(td.data.data(), pivots.data());
  xOut = b;
  solver.solveLU(td.data.data(), pivots.data(), xOut.data(), td.totalSize, 1);
  return (td.fullMat * xOut - b).norm() / b.norm();
}

TEST(LUComparison, CudaVsUmfpack_BlockSparse) {
  SparseMatGenerator gen = SparseMatGenerator::genFlat(25, 0.25, 456);
  SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();
  vector<int64_t> paramSize(gen.columns.size(), 3);

  mt19937 rng(456);
  auto td = buildNonSymmetricTestData(ss, paramSize, rng);

  uniform_real_distribution<double> unif(-1.0, 1.0);
  Vector<double> b(td.totalSize);
  for (int64_t i = 0; i < td.totalSize; i++) b(i) = unif(rng);

  auto umfResult = solveWithUmfpack(td.colPtr, td.rowIdx, td.val, td.totalSize, b);

  Vector<double> xCuda;
  double cudaResidual = solveCudaLU(td, b, xCuda);

  cout << "\n=== CUDA vs UMFPACK: Block-Sparse ===" << endl;
  cout << "Size: " << td.totalSize << "x" << td.totalSize << endl;
  cout << "UMFPACK residual: " << umfResult.residual << endl;
  cout << "CUDA residual: " << cudaResidual << endl;

  EXPECT_LT(umfResult.residual, 1e-10);
  EXPECT_LT(cudaResidual, 1e-8);

  double solutionDiff = (xCuda - umfResult.solution).norm() / umfResult.solution.norm();
  cout << "Solution difference: " << solutionDiff << endl;
  EXPECT_LT(solutionDiff, 1e-6);
}

TEST(LUComparison, CudaVsUmfpack_NonSymmetric) {
  SparseMatGenerator gen = SparseMatGenerator::genFlat(25, 0.25, 789);
  SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();
  vector<int64_t> paramSize(gen.columns.size(), 3);

  mt19937 rng(789);
  auto td = buildNonSymmetricTestData(ss, paramSize, rng);

  uniform_real_distribution<double> unif(-1.0, 1.0);
  Vector<double> b(td.totalSize);
  for (int64_t i = 0; i < td.totalSize; i++) b(i) = unif(rng);

  auto umfResult = solveWithUmfpack(td.colPtr, td.rowIdx, td.val, td.totalSize, b);

  Vector<double> xCuda;
  double cudaResidual = solveCudaLU(td, b, xCuda);

  cout << "\n=== CUDA vs UMFPACK: Non-Symmetric ===" << endl;
  cout << "Size: " << td.totalSize << "x" << td.totalSize << endl;
  cout << "UMFPACK residual: " << umfResult.residual << endl;
  cout << "CUDA residual: " << cudaResidual << endl;

  EXPECT_LT(umfResult.residual, 1e-10);
  EXPECT_LT(cudaResidual, 1e-8);
}

TEST(LUComparison, CudaVsUmfpack_MixedBlocks) {
  SparseMatGenerator gen = SparseMatGenerator::genFlat(40, 0.1, 321);
  SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();

  mt19937 rng(321);
  uniform_int_distribution<int64_t> sizeDist(2, 8);
  vector<int64_t> paramSize;
  for (size_t i = 0; i < gen.columns.size(); i++) paramSize.push_back(sizeDist(rng));

  auto td = buildNonSymmetricTestData(ss, paramSize, rng);

  uniform_real_distribution<double> unif(-1.0, 1.0);
  Vector<double> b(td.totalSize);
  for (int64_t i = 0; i < td.totalSize; i++) b(i) = unif(rng);

  auto umfResult = solveWithUmfpack(td.colPtr, td.rowIdx, td.val, td.totalSize, b);

  Vector<double> xCuda;
  double cudaResidual = solveCudaLU(td, b, xCuda);

  cout << "\n=== CUDA vs UMFPACK: Mixed Block Sizes ===" << endl;
  cout << "Blocks: " << paramSize.size() << ", Size: " << td.totalSize << "x" << td.totalSize
       << endl;
  cout << "UMFPACK residual: " << umfResult.residual << endl;
  cout << "CUDA residual: " << cudaResidual << endl;

  EXPECT_LT(umfResult.residual, 1e-10);
  EXPECT_LT(cudaResidual, 1e-8);
}

TEST(LUComparison, CudaVsUmfpack_Performance) {
  struct TestCase {
    string name;
    int numBlocks;
    double density;
    int blockSize;
    int seed;
  };

  vector<TestCase> cases = {
      {"Small (50 blocks)", 50, 0.25, 3, 100},
      {"Medium (150 blocks)", 150, 0.08, 3, 200},
      {"Large (300 blocks)", 300, 0.03, 3, 300},
  };

  cout << "\n=== CUDA vs UMFPACK Performance Comparison ===" << endl;
  cout << left << setw(25) << "Case" << setw(10) << "Size" << setw(15) << "UMFPACK(ms)"
       << setw(15) << "CUDA(ms)" << setw(15) << "Speedup" << setw(15) << "UMF resid"
       << setw(15) << "CUDA resid" << endl;
  cout << string(100, '-') << endl;

  for (const auto& tc : cases) {
    SparseMatGenerator gen = SparseMatGenerator::genFlat(tc.numBlocks, tc.density, tc.seed);
    SparseStructure ss = columnsToCscStruct(gen.columns).transpose().addFullEliminationFill();
    vector<int64_t> paramSize(gen.columns.size(), tc.blockSize);

    mt19937 rng(tc.seed);
    auto td = buildNonSymmetricTestData(ss, paramSize, rng);

    rng.seed(tc.seed + 1000);
    uniform_real_distribution<double> unif(-1.0, 1.0);
    Vector<double> b(td.totalSize);
    for (int64_t i = 0; i < td.totalSize; i++) b(i) = unif(rng);

    // UMFPACK timing (includes symbolic + numeric + solve)
    auto umfResult = solveWithUmfpack(td.colPtr, td.rowIdx, td.val, td.totalSize, b);
    double umfFactorMs = (umfResult.analysisTime + umfResult.factorTime) * 1000;
    double umfSolveMs = umfResult.solveTime * 1000;

    // CUDA: separate setup from factor+solve
    int64_t n = td.totalSize;
    Solver cudaSolver(std::move(*td.factorSkel), {}, {}, cudaOps());
    vector<int64_t> pivots(n);

    auto cudaFactorStart = hrc::now();
    cudaSolver.factorLU(td.data.data(), pivots.data());
    double cudaFactorMs = tdelta(hrc::now() - cudaFactorStart).count() * 1000;

    Vector<double> xCuda = b;
    auto cudaSolveStart = hrc::now();
    cudaSolver.solveLU(td.data.data(), pivots.data(), xCuda.data(), n, 1);
    double cudaSolveMs = tdelta(hrc::now() - cudaSolveStart).count() * 1000;

    double cudaResidual = (td.fullMat * xCuda - b).norm() / b.norm();

    cout << left << setw(25) << tc.name << setw(10) << n
         << setw(15) << fixed << setprecision(2) << umfFactorMs + umfSolveMs
         << setw(15) << cudaFactorMs + cudaSolveMs
         << setw(15) << setprecision(2)
         << (umfFactorMs + umfSolveMs) / (cudaFactorMs + cudaSolveMs)
         << "x" << setw(15) << scientific << setprecision(2) << umfResult.residual
         << setw(15) << cudaResidual << endl;

    EXPECT_LT(umfResult.residual, 1e-10) << tc.name << ": UMFPACK residual too large";
    EXPECT_LT(cudaResidual, 1e-8) << tc.name << ": CUDA residual too large";
  }
}
#endif  // SPRUX_USE_CUBLAS

#endif  // SPRUX_HAVE_UMFPACK
