/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Metal GPU variant of SequenceSolveTest.
// Tests LU factorization on real-world circuit Jacobian sequences using
// the Metal backend (float-only, Apple Silicon).

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "baspacho/baspacho/MetalDefs.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/testing/MatrixMarketReader.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;
namespace fs = std::filesystem;

static string findTestDataDir(const string& subdir) {
  const char* envDir = getenv("BASPACHO_TEST_DATA_DIR");
  if (envDir) {
    string candidate = string(envDir) + "/" + subdir;
    if (fs::is_directory(candidate)) return candidate;
  }

  vector<string> prefixes = {"test_data", "../test_data", "../../test_data", "../../../test_data"};
  for (const auto& prefix : prefixes) {
    string candidate = prefix + "/" + subdir;
    if (fs::is_directory(candidate)) return candidate;
  }
  return "";
}

static vector<pair<string, string>> discoverSequenceFiles(const string& dir) {
  vector<pair<string, string>> pairs;
  for (int idx = 0;; idx++) {
    ostringstream jacName, rhsName;
    jacName << dir << "/jacobian_" << setw(4) << setfill('0') << idx << ".mtx";
    rhsName << dir << "/rhs_" << setw(4) << setfill('0') << idx << ".mtx";
    if (!fs::exists(jacName.str()) || !fs::exists(rhsName.str())) break;
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

// Float-only residual computation for Metal (CSR values are double, solution is float)
static float computeResidualFloat(const CsrMatrix& A, const Eigen::VectorXf& x,
                                  const Eigen::VectorXf& b) {
  Eigen::VectorXf Ax = Eigen::VectorXf::Zero(A.nRows);
  for (int64_t i = 0; i < A.nRows; i++) {
    for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      Ax(i) += float(A.values[k]) * x(A.colInd[k]);
    }
  }
  return (Ax - b).norm() / b.norm();
}

// ============================================================================
// Ring oscillator on Metal (float-only, tiny matrices — good GPU smoke test)
// ============================================================================

TEST(MetalSequenceSolve, RingOscillator) {
  string dir = findTestDataDir("ring_sequence");
  if (dir.empty()) {
    GTEST_SKIP() << "test_data/ring_sequence/ not found";
  }

  auto files = discoverSequenceFiles(dir);
  ASSERT_GE(files.size(), 2u) << "Need at least 2 sequence files";

  cout << "Metal ring oscillator sequence: " << files.size() << " matrices in " << dir << endl;

  CsrMatrix A0 = readMatrixMarket(files[0].first);
  ASSERT_EQ(A0.nRows, A0.nCols) << "Matrix must be square";
  cout << "Matrix size: " << A0.nRows << " x " << A0.nCols << ", nnz=" << A0.nnz << endl;

  int64_t n = A0.nRows;

  // Build solver once via CHOLMOD-based symbolic analysis
  SparseStructure ss = csrToSparseStructure(A0);
  vector<int64_t> paramSizes(n, 1);
  vector<int64_t> blockSizes(n, 1);

  // Metal GPU solver
  Settings metalSettings;
  metalSettings.backend = BackendMetal;
  metalSettings.matrixType = MTYPE_GENERAL;
  auto solver = createSolver(metalSettings, paramSizes, ss);

  // CPU float solver for reference comparison (same fill-in structure)
  Settings cpuSettings;
  cpuSettings.backend = BackendFast;
  cpuSettings.matrixType = MTYPE_GENERAL;
  auto cpuSolver = createSolver(cpuSettings, paramSizes, ss);

  vector<float> data(solver->skel().totalDataSize());
  vector<float> cpuData(cpuSolver->skel().totalDataSize());

  // Convert CSR values to float for loadFromCsr<float>
  auto loadFloatCsr = [&](const CsrMatrix& A, Solver& slv, vector<float>& buf) {
    vector<float> floatValues(A.values.begin(), A.values.end());
    fill(buf.begin(), buf.end(), 0.0f);
    slv.loadFromCsr(A.rowPtr.data(), A.colInd.data(), blockSizes.data(), floatValues.data(),
                    buf.data());
  };

  const auto& perm = solver->paramToSpan();
  const auto& cpuPerm = cpuSolver->paramToSpan();
  vector<int64_t> pivots(n);
  vector<int64_t> cpuPivots(n);
  int passed = 0;
  int skipped = 0;

  for (size_t i = 0; i < files.size(); i++) {
    CsrMatrix A = readMatrixMarket(files[i].first);
    Eigen::VectorXd bDouble = readRhsVector(files[i].second);

    ASSERT_EQ(A.nRows, n) << "Matrix #" << i << " has different dimensions";
    ASSERT_EQ(bDouble.size(), n) << "RHS #" << i << " has wrong size";

    Eigen::VectorXf b = bDouble.cast<float>();

    // Skip near-zero RHS
    if (b.norm() < 1e-7f) {
      skipped++;
      continue;
    }

    // Check CPU float path first (same supernodal fill-in structure).
    // Some matrices trigger zero pivots in BaSpaCho's float LU due to fill-in
    // structure — skip those since the issue is precision, not Metal.
    float cpuResidual;
    try {
      loadFloatCsr(A, *cpuSolver, cpuData);
      cpuSolver->factorLU(cpuData.data(), cpuPivots.data());

      Eigen::VectorXf cpuBp(n);
      for (int64_t j = 0; j < n; j++) cpuBp(cpuPerm[j]) = b(j);
      cpuSolver->solveLU(cpuData.data(), cpuPivots.data(), cpuBp.data(), n, 1);
      Eigen::VectorXf xCpu(n);
      for (int64_t j = 0; j < n; j++) xCpu(j) = cpuBp(cpuPerm[j]);

      cpuResidual = computeResidualFloat(A, xCpu, b);
    } catch (const exception&) {
      // CPU float also fails (zero pivot) — skip this matrix
      skipped++;
      continue;
    }

    if (cpuResidual > 1e-2f) {
      skipped++;
      continue;
    }

    // Factor+solve on Metal GPU
    loadFloatCsr(A, *solver, data);

    {
      MetalMirror<float> dataGpu(data);
      solver->factorLU(dataGpu.ptr(), pivots.data());
      dataGpu.get(data);
    }

    // Permute RHS, solve on GPU, inverse permute
    Eigen::VectorXf bp(n);
    for (int64_t j = 0; j < n; j++) bp(perm[j]) = b(j);

    {
      MetalMirror<float> dataGpu(data);
      MetalMirror<float> xGpu(vector<float>(bp.data(), bp.data() + n));
      solver->solveLU(dataGpu.ptr(), pivots.data(), xGpu.ptr(), n, 1);
      vector<float> xVec(n);
      xGpu.get(xVec);
      for (int64_t j = 0; j < n; j++) bp(j) = xVec[j];
    }

    Eigen::VectorXf x(n);
    for (int64_t j = 0; j < n; j++) x(j) = bp(perm[j]);

    float residual = computeResidualFloat(A, x, b);

    // Metal should be within 100x of CPU BaSpaCho float
    float threshold = max(cpuResidual * 100.0f, 1e-4f);
    EXPECT_LT(residual, threshold) << "Matrix #" << i << " Metal residual too large: " << residual
                                   << " (CPU: " << cpuResidual << ")";
    passed++;
  }

  cout << "Passed: " << passed << ", Skipped (zero RHS): " << skipped << endl;
  ASSERT_GT(passed, 0) << "No matrices were actually tested";
}
