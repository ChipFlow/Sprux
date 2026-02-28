/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests exercising the LU factorization path on sequences of real-world
// circuit Jacobians.  Each sequence shares a single sparsity pattern with
// different numerical values — the core circuit-simulator workflow:
// symbolic analysis once, repeated numerical factorization.

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/testing/MatrixMarketReader.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;
namespace fs = std::filesystem;

// Try to find a test_data subdirectory relative to common build locations.
static string findTestDataDir(const string& subdir) {
  const char* envDir = getenv("BASPACHO_TEST_DATA_DIR");
  if (envDir) {
    string candidate = string(envDir) + "/" + subdir;
    if (fs::is_directory(candidate)) return candidate;
  }

  vector<string> prefixes = {
      "test_data",
      "../test_data",
      "../../test_data",
      "../../../test_data",
  };

  for (const auto& prefix : prefixes) {
    string candidate = prefix + "/" + subdir;
    if (fs::is_directory(candidate)) return candidate;
  }

  return "";
}

// Discover all (jacobian, rhs) file pairs in a sequence directory.
// Returns sorted pairs: {jacobian_NNNN.mtx, rhs_NNNN.mtx}.
static vector<pair<string, string>> discoverSequenceFiles(const string& dir) {
  vector<pair<string, string>> pairs;

  for (int idx = 0;; idx++) {
    ostringstream jacName, rhsName;
    jacName << dir << "/jacobian_" << setw(4) << setfill('0') << idx << ".mtx";
    rhsName << dir << "/rhs_" << setw(4) << setfill('0') << idx << ".mtx";

    if (!fs::exists(jacName.str())) break;
    if (!fs::exists(rhsName.str())) break;
    pairs.push_back({jacName.str(), rhsName.str()});
  }

  return pairs;
}

// Build BaSpaCho solver structures from a CSR matrix.
// Returns: {CoalescedBlockMatrixSkel, SparseStructure (filled)}
// The skeleton has initUpperTriangle() already called.
static CoalescedBlockMatrixSkel buildSkeletonFromCsr(const CsrMatrix& A) {
  int64_t n = A.nRows;

  // Build column sets from CSR
  vector<set<int64_t>> colBlocks(n);
  for (int64_t i = 0; i < n; i++) {
    for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      colBlocks[A.colInd[k]].insert(i);
    }
  }

  // Create SparseStructure with fill
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();

  // Scalar blocks: each span/lump is size 1
  vector<int64_t> spanStart(n + 1);
  iota(spanStart.begin(), spanStart.end(), 0);
  vector<int64_t> lumpToSpan(n + 1);
  iota(lumpToSpan.begin(), lumpToSpan.end(), 0);

  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  skel.initUpperTriangle();

  return skel;
}

// Fill BaSpaCho block data from CSR values (lower + upper triangles).
template <typename T>
static void fillDataFromCsr(const CoalescedBlockMatrixSkel& skel, const CsrMatrix& A,
                            vector<T>& data) {
  fill(data.begin(), data.end(), T(0));
  int64_t numLumps = skel.numLumps();

  // Lower triangle (chain data)
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
        int64_t globalRow = rowStart + r;
        for (int64_t col = 0; col < lumpSize; col++) {
          int64_t globalCol = lumpStartCol + col;
          for (int64_t k = A.rowPtr[globalRow]; k < A.rowPtr[globalRow + 1]; k++) {
            if (A.colInd[k] == globalCol) {
              data[dataOffset + r * lumpSize + col] = T(A.values[k]);
              break;
            }
          }
        }
      }
    }
  }

  // Upper triangle
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
          int64_t globalRow = lumpStartRow + r;
          for (int64_t c = 0; c < colSize; c++) {
            int64_t globalCol = colStart + c;
            for (int64_t k = A.rowPtr[globalRow]; k < A.rowPtr[globalRow + 1]; k++) {
              if (A.colInd[k] == globalCol) {
                data[upperDataOffset + r * colSize + c] = T(A.values[k]);
                break;
              }
            }
          }
        }
      }
    }
  }
}

// ============================================================================
// Ring oscillator sequence: 50 tiny matrices (47x47), fast CI test
// ============================================================================

TEST(SequenceSolve, RingOscillator) {
  string dir = findTestDataDir("ring_sequence");
  if (dir.empty()) {
    GTEST_SKIP() << "test_data/ring_sequence/ not found";
  }

  auto files = discoverSequenceFiles(dir);
  ASSERT_GE(files.size(), 2u) << "Need at least 2 sequence files";

  cout << "Ring oscillator sequence: " << files.size() << " matrices in " << dir << endl;

  // Load first matrix for sparsity structure
  CsrMatrix A0 = readMatrixMarket(files[0].first);
  ASSERT_EQ(A0.nRows, A0.nCols) << "Matrix must be square";
  cout << "Matrix size: " << A0.nRows << " x " << A0.nCols << ", nnz=" << A0.nnz << endl;

  int64_t n = A0.nRows;

  // Build solver once from sparsity pattern
  CoalescedBlockMatrixSkel skel = buildSkeletonFromCsr(A0);
  vector<double> data(skel.totalDataSize());
  Solver solver(std::move(skel), {}, {}, fastOps());
  const auto& solverSkel = solver.skel();

  vector<int64_t> pivots(n);
  int passed = 0;
  int skipped = 0;

  for (size_t i = 0; i < files.size(); i++) {
    CsrMatrix A = readMatrixMarket(files[i].first);
    Eigen::VectorXd b = readRhsVector(files[i].second);

    ASSERT_EQ(A.nRows, n) << "Matrix #" << i << " has different dimensions";
    ASSERT_EQ(A.nCols, n) << "Matrix #" << i << " has different dimensions";
    ASSERT_EQ(b.size(), n) << "RHS #" << i << " has wrong size";

    // Skip near-zero RHS (initial conditions)
    if (b.norm() < 1e-15) {
      skipped++;
      continue;
    }

    // Fill data and factor
    fillDataFromCsr<double>(solverSkel, A, data);
    solver.factorLU(data.data(), pivots.data());

    // Solve
    Eigen::VectorXd x = b;
    solver.solveLU(data.data(), pivots.data(), x.data(), n, 1);

    // Check residual
    double residual = computeResidual(A, x, b);
    EXPECT_LT(residual, 1e-6) << "Matrix #" << i << " residual too large: " << residual;
    passed++;
  }

  cout << "Passed: " << passed << ", Skipped (zero RHS): " << skipped << endl;
  ASSERT_GT(passed, 0) << "No matrices were actually tested";
}

// ============================================================================
// C6288 sequence: 20 medium matrices (25380x25380), real-world integration
// ============================================================================

// DISABLED: Takes several minutes due to symbolic analysis on 25K scalar blocks.
// Run manually with: --gtest_also_run_disabled_tests --gtest_filter="*C6288*"
TEST(SequenceSolve, DISABLED_C6288) {
  string dir = findTestDataDir("c6288_sequence");
  if (dir.empty()) {
    GTEST_SKIP() << "test_data/c6288_sequence/ not found";
  }

  auto files = discoverSequenceFiles(dir);
  ASSERT_GE(files.size(), 2u) << "Need at least 2 sequence files";

  // Test a subset: first, middle, and last matrices.
  // Full sequence (20 matrices at 25K x 25K) takes too long in debug builds.
  vector<size_t> testIndices;
  testIndices.push_back(0);
  if (files.size() > 2) testIndices.push_back(files.size() / 2);
  testIndices.push_back(files.size() - 1);

  cout << "C6288 sequence: " << files.size() << " matrices in " << dir
       << ", testing " << testIndices.size() << endl;

  // Load first matrix for sparsity structure
  CsrMatrix A0 = readMatrixMarket(files[0].first);
  ASSERT_EQ(A0.nRows, A0.nCols) << "Matrix must be square";
  cout << "Matrix size: " << A0.nRows << " x " << A0.nCols << ", nnz=" << A0.nnz << endl;

  int64_t n = A0.nRows;

  // Build solver once from sparsity pattern
  CoalescedBlockMatrixSkel skel = buildSkeletonFromCsr(A0);
  vector<double> data(skel.totalDataSize());
  Solver solver(std::move(skel), {}, {}, fastOps());
  const auto& solverSkel = solver.skel();

  vector<int64_t> pivots(n);
  int passed = 0;
  int skipped = 0;

  for (size_t idx : testIndices) {
    CsrMatrix A = readMatrixMarket(files[idx].first);
    Eigen::VectorXd b = readRhsVector(files[idx].second);

    ASSERT_EQ(A.nRows, n) << "Matrix #" << idx << " has different dimensions";
    ASSERT_EQ(A.nCols, n) << "Matrix #" << idx << " has different dimensions";
    ASSERT_EQ(b.size(), n) << "RHS #" << idx << " has wrong size";

    // Skip near-zero RHS (initial conditions)
    if (b.norm() < 1e-15) {
      skipped++;
      continue;
    }

    // Fill data and factor
    fillDataFromCsr<double>(solverSkel, A, data);
    solver.factorLU(data.data(), pivots.data());

    // Solve
    Eigen::VectorXd x = b;
    solver.solveLU(data.data(), pivots.data(), x.data(), n, 1);

    // Check residual
    double residual = computeResidual(A, x, b);
    EXPECT_LT(residual, 1e-6) << "Matrix #" << idx << " residual too large: " << residual;

    cout << "  Matrix #" << idx << ": residual=" << scientific << setprecision(4) << residual
         << endl;

    passed++;
  }

  cout << "Passed: " << passed << ", Skipped (zero RHS): " << skipped << endl;
  ASSERT_GT(passed, 0) << "No matrices were actually tested";
}
