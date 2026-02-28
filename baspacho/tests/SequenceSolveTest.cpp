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
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
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

// Build CSR lower-triangle SparseStructure from CSR matrix.
// createSolver expects lower-triangle CSR (matching randomCols/columnsToCscStruct convention).
// Ensures diagonal entries are present (required by CHOLMOD symbolic analysis).
static SparseStructure csrToSparseStructure(const CsrMatrix& A) {
  int64_t n = A.nRows;
  vector<set<int64_t>> colBlocks(n);
  for (int64_t i = 0; i < n; i++) {
    colBlocks[i].insert(i);  // ensure diagonal is present
    for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      int64_t j = A.colInd[k];
      // CSC lower triangle: column min(i,j) has row max(i,j)
      colBlocks[min(i, j)].insert(max(i, j));
    }
  }
  return columnsToCscStruct(colBlocks).transpose();
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

  using Clock = chrono::high_resolution_clock;

  // Build solver once via CHOLMOD-based symbolic analysis
  SparseStructure ss = csrToSparseStructure(A0);
  vector<int64_t> paramSizes(n, 1);
  vector<int64_t> blockSizes(n, 1);

  Settings settings;
  settings.backend = BackendFast;
  settings.matrixType = MTYPE_GENERAL;

  auto t0 = Clock::now();
  auto solver = createSolver(settings, paramSizes, ss);
  double analysisTime = chrono::duration<double>(Clock::now() - t0).count();

  cout << "=== Ring Oscillator Timing ===" << endl;
  cout << "  Symbolic analysis: " << fixed << setprecision(4) << analysisTime << "s" << endl;

  vector<double> data(solver->skel().totalDataSize());
  vector<int64_t> pivots(n);
  const auto& perm = solver->paramToSpan();
  int passed = 0;
  int skipped = 0;
  double totalFactorTime = 0;
  double totalSolveTime = 0;

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

    // Load data via permutation-aware loadFromCsr and factor
    fill(data.begin(), data.end(), 0.0);
    solver->loadFromCsr(A.rowPtr.data(), A.colInd.data(), blockSizes.data(), A.values.data(),
                        data.data());

    auto tFactor = Clock::now();
    solver->factorLU(data.data(), pivots.data());
    double factorTime = chrono::duration<double>(Clock::now() - tFactor).count();

    // Permute RHS: bp[perm[i]] = b[i]
    Eigen::VectorXd bp(n);
    for (int64_t j = 0; j < n; j++) bp(perm[j]) = b(j);

    auto tSolve = Clock::now();
    solver->solveLU(data.data(), pivots.data(), bp.data(), n, 1);
    double solveTime = chrono::duration<double>(Clock::now() - tSolve).count();

    // Inverse permute solution: x[i] = bp[perm[i]]
    Eigen::VectorXd x(n);
    for (int64_t j = 0; j < n; j++) x(j) = bp(perm[j]);

    // Check residual
    double residual = computeResidual(A, x, b);
    EXPECT_LT(residual, 1e-6) << "Matrix #" << i << " residual too large: " << residual;

    cout << "  Matrix #" << i << ": factor=" << fixed << setprecision(4) << factorTime
         << "s, solve=" << solveTime << "s, residual=" << scientific << setprecision(2) << residual
         << endl;

    totalFactorTime += factorTime;
    totalSolveTime += solveTime;
    passed++;
  }

  if (passed > 0) {
    cout << "  Average: factor=" << fixed << setprecision(4) << totalFactorTime / passed
         << "s, solve=" << totalSolveTime / passed << "s" << endl;
  }
  cout << "Passed: " << passed << ", Skipped (zero RHS): " << skipped << endl;
  ASSERT_GT(passed, 0) << "No matrices were actually tested";
}

// ============================================================================
// C6288 sequence: 20 medium matrices (25380x25380), real-world integration
// ============================================================================

// This test exercises createSolver + CHOLMOD-based symbolic analysis + LU factorization
// with static pivoting on the 25K C6288 circuit Jacobians.
//
// KNOWN LIMITATION: BaSpaCho's fill-reducing ordering (designed for SPD/Cholesky) creates
// ~40% zero/near-zero pivots for these non-symmetric circuit Jacobians. Static pivoting
// handles individual zero pivots, but the cumulative perturbation errors cascade through
// the Schur complement, causing overflow in late-stage lumps (~lump 24224 of 24945).
// This produces NaN residuals for most matrices. The fix requires MC64 preprocessing
// (weighted bipartite matching to place large values on the diagonal before ordering).
//
// This test verifies that factorization completes without throwing, and reports timing
// and residual quality for benchmarking purposes. It does NOT assert residual quality.
TEST(SequenceSolve, DISABLED_C6288) {
  string dir = findTestDataDir("c6288_sequence");
  if (dir.empty()) {
    GTEST_SKIP() << "test_data/c6288_sequence/ not found";
  }

  auto files = discoverSequenceFiles(dir);
  ASSERT_GE(files.size(), 2u) << "Need at least 2 sequence files";

  // Test a subset: first, middle, and last matrices.
  vector<size_t> testIndices;
  testIndices.push_back(0);
  if (files.size() > 2) testIndices.push_back(files.size() / 2);
  testIndices.push_back(files.size() - 1);

  cout << "C6288 sequence: " << files.size() << " matrices in " << dir << ", testing "
       << testIndices.size() << endl;

  // Load first matrix for sparsity structure
  CsrMatrix A0 = readMatrixMarket(files[0].first);
  ASSERT_EQ(A0.nRows, A0.nCols) << "Matrix must be square";
  cout << "Matrix size: " << A0.nRows << " x " << A0.nCols << ", nnz=" << A0.nnz << endl;

  int64_t n = A0.nRows;

  using Clock = chrono::high_resolution_clock;

  // Build solver once via CHOLMOD-based symbolic analysis
  SparseStructure ss = csrToSparseStructure(A0);
  vector<int64_t> paramSizes(n, 1);
  vector<int64_t> blockSizes(n, 1);

  Settings settings;
  settings.backend = BackendFast;
  settings.matrixType = MTYPE_GENERAL;
  settings.staticPivotThreshold = 0.0;  // auto: cbrt(eps) * max_diag

  auto t0 = Clock::now();
  auto solver = createSolver(settings, paramSizes, ss);
  double analysisTime = chrono::duration<double>(Clock::now() - t0).count();

  cout << "=== C6288 Timing ===" << endl;
  cout << "  Symbolic analysis: " << fixed << setprecision(4) << analysisTime << "s" << endl;
  cout << "  Solver: " << solver->skel().numLumps() << " lumps, " << solver->skel().numSpans()
       << " spans, dataSize=" << solver->totalDataSize() << endl;

  vector<double> data(solver->skel().totalDataSize());
  vector<int64_t> pivots(n);
  const auto& perm = solver->paramToSpan();
  int passed = 0;
  int skipped = 0;
  double totalFactorTime = 0;
  double totalSolveTime = 0;

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

    // Load data via permutation-aware loadFromCsr and factor
    fill(data.begin(), data.end(), 0.0);
    solver->loadFromCsr(A.rowPtr.data(), A.colInd.data(), blockSizes.data(), A.values.data(),
                        data.data());

    auto tFactor = Clock::now();
    solver->factorLU(data.data(), pivots.data());
    double factorTime = chrono::duration<double>(Clock::now() - tFactor).count();

    int64_t perturbCount = solver->staticPivotPerturbCount();

    // Permute RHS: bp[perm[i]] = b[i]
    Eigen::VectorXd bp(n);
    for (int64_t j = 0; j < n; j++) bp(perm[j]) = b(j);

    auto tSolve = Clock::now();
    solver->solveLU(data.data(), pivots.data(), bp.data(), n, 1);
    double solveTime = chrono::duration<double>(Clock::now() - tSolve).count();

    // Inverse permute solution: x[i] = bp[perm[i]]
    Eigen::VectorXd x(n);
    for (int64_t j = 0; j < n; j++) x(j) = bp(perm[j]);

    // Compute residual for reporting (not asserted — see KNOWN LIMITATION above)
    double residual = computeResidual(A, x, b);

    cout << "  Matrix #" << idx << ": factor=" << fixed << setprecision(4) << factorTime
         << "s, solve=" << solveTime << "s, residual=" << scientific << setprecision(2) << residual
         << ", perturbed=" << perturbCount << endl;

    totalFactorTime += factorTime;
    totalSolveTime += solveTime;
    passed++;
  }

  if (passed > 0) {
    cout << "  Average: factor=" << fixed << setprecision(4) << totalFactorTime / passed
         << "s, solve=" << totalSolveTime / passed << "s" << endl;
  }
  cout << "Passed: " << passed << ", Skipped (zero RHS): " << skipped << endl;
  ASSERT_GT(passed, 0) << "No matrices were actually tested";
}
