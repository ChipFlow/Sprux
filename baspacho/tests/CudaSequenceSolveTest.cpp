/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright (c) Robert Taylor, 2026. All rights reserved.
// Licensed under the MIT license found in the LICENSE file.

// CUDA GPU variant of SequenceSolveTest.
// Tests LU factorization on real-world circuit Jacobian sequences using
// the CUDA backend (float and double, NVIDIA GPU).

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/LU>
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
#include "baspacho/baspacho/CudaDefs.h"
#include "baspacho/baspacho/Preprocessing.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/testing/MatrixMarketReader.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;
namespace fs = std::filesystem;

static string findTestDataDir(const string& subdir) {
  const char* envDir = getenv("SPRUX_TEST_DATA_DIR");
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

static SparseStructure csrToSparseStructure(const CsrMatrix& A) {
  int64_t n = A.nRows;
  vector<set<int64_t>> colBlocks(n);
  for (int64_t i = 0; i < n; i++) {
    colBlocks[i].insert(i);
    for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      int64_t j = A.colInd[k];
      colBlocks[min(i, j)].insert(max(i, j));
    }
  }
  return columnsToCscStruct(colBlocks).transpose();
}

// ============================================================================
// Ring oscillator on CUDA (double precision, tiny matrices — GPU smoke test)
// ============================================================================

TEST(CudaSequenceSolve, RingOscillator) {
  string dir = findTestDataDir("ring_sequence");
  if (dir.empty()) {
    GTEST_SKIP() << "test_data/ring_sequence/ not found";
  }

  auto files = discoverSequenceFiles(dir);
  ASSERT_GE(files.size(), 2u) << "Need at least 2 sequence files";

  cout << "CUDA ring oscillator sequence: " << files.size() << " matrices in " << dir << endl;

  CsrMatrix A0 = readMatrixMarket(files[0].first);
  ASSERT_EQ(A0.nRows, A0.nCols) << "Matrix must be square";
  cout << "Matrix size: " << A0.nRows << " x " << A0.nCols << ", nnz=" << A0.nnz << endl;

  int64_t n = A0.nRows;

  using Clock = chrono::high_resolution_clock;

  SparseStructure ss = csrToSparseStructure(A0);
  vector<int64_t> paramSizes(n, 1);
  vector<int64_t> blockSizes(n, 1);

  Settings settings;
  settings.backend = BackendCuda;
  settings.matrixType = MTYPE_GENERAL;

  auto t0 = Clock::now();
  auto solver = createSolver(settings, paramSizes, ss);
  double analysisTime = chrono::duration<double>(Clock::now() - t0).count();

  cout << "=== CUDA Ring Oscillator Timing ===" << endl;
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

    if (b.norm() < 1e-15) {
      skipped++;
      continue;
    }

    fill(data.begin(), data.end(), 0.0);
    solver->loadFromCsr(A.rowPtr.data(), A.colInd.data(), blockSizes.data(), A.values.data(),
                        data.data());

    double factorTime;
    {
      DevMirror<double> dataGpu(data);
      auto tFactor = Clock::now();
      solver->factorLU(dataGpu.ptr, pivots.data());
      factorTime = chrono::duration<double>(Clock::now() - tFactor).count();
      dataGpu.get(data);
    }

    // Permute RHS, solve, inverse permute
    Eigen::VectorXd bp(n);
    for (int64_t j = 0; j < n; j++) bp(perm[j]) = b(j);

    double solveTime;
    {
      DevMirror<double> dataGpu(data);
      DevMirror<double> xGpu(vector<double>(bp.data(), bp.data() + n));
      auto tSolve = Clock::now();
      solver->solveLU(dataGpu.ptr, pivots.data(), xGpu.ptr, n, 1);
      solveTime = chrono::duration<double>(Clock::now() - tSolve).count();
      vector<double> xVec(n);
      xGpu.get(xVec);
      for (int64_t j = 0; j < n; j++) bp(j) = xVec[j];
    }

    Eigen::VectorXd x(n);
    for (int64_t j = 0; j < n; j++) x(j) = bp(perm[j]);

    double residual = computeResidual(A, x, b);
    EXPECT_LT(residual, 1e-6) << "Matrix #" << i << " residual too large: " << residual;

    // Dump solution for external comparison (e.g., scipy)
    if (getenv("SPRUX_DUMP_SOLUTIONS")) {
      cout << "SOLUTION_DUMP:" << i << ":";
      for (int64_t j = 0; j < n; j++) {
        if (j > 0) cout << " ";
        cout << setprecision(17) << x(j);
      }
      cout << endl;
    }

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
// C6288 sequence on CUDA (double precision) with iterative refinement.
// Preprocessing (BTF max transversal + equilibration) applied before factorization.
// ============================================================================

TEST(CudaSequenceSolve, C6288) {
  string dir = findTestDataDir("c6288_sequence");
  if (dir.empty()) {
    GTEST_SKIP() << "test_data/c6288_sequence/ not found";
  }

  auto files = discoverSequenceFiles(dir);
  ASSERT_GE(files.size(), 2u) << "Need at least 2 sequence files";

  // Test a subset: first, middle, last
  vector<size_t> testIndices;
  testIndices.push_back(0);
  if (files.size() > 2) testIndices.push_back(files.size() / 2);
  testIndices.push_back(files.size() - 1);

  cout << "CUDA C6288 sequence: " << files.size() << " matrices in " << dir << ", testing "
       << testIndices.size() << endl;

  CsrMatrix A0 = readMatrixMarket(files[0].first);
  ASSERT_EQ(A0.nRows, A0.nCols) << "Matrix must be square";
  cout << "Matrix size: " << A0.nRows << " x " << A0.nCols << ", nnz=" << A0.nnz << endl;

  int64_t n = A0.nRows;

  using Clock = chrono::high_resolution_clock;

  // BTF max transversal
  auto tPreproc = Clock::now();
  auto preproc = computeMaxTransversal(n, A0.rowPtr.data(), A0.colInd.data());
  double preprocTime = chrono::duration<double>(Clock::now() - tPreproc).count();

  cout << "=== CUDA C6288 Timing ===" << endl;
  cout << "  Preprocessing (max transversal): " << fixed << setprecision(4) << preprocTime
       << "s, structural rank=" << preproc.structuralRank << "/" << n << endl;
  ASSERT_EQ(preproc.structuralRank, n) << "Matrix is structurally singular";

  // Apply row perm to first matrix, build symmetric structure
  vector<int64_t> pRowPtr, pColInd;
  vector<double> pValues;
  applyRowPermToCsr<double>(n, A0.rowPtr.data(), A0.colInd.data(), A0.values.data(),
                            preproc.rowPerm.data(), pRowPtr, pColInd, pValues);

  SparseStructure ss = csrToSymmetricSparseStructure(n, pRowPtr.data(), pColInd.data());
  vector<int64_t> paramSizes(n, 1);
  vector<int64_t> blockSizes(n, 1);

  Settings settings;
  settings.backend = BackendCuda;
  settings.matrixType = MTYPE_GENERAL;
  settings.staticPivotThreshold = 0.0;

  auto t0 = Clock::now();
  auto solver = createSolver(settings, paramSizes, ss);
  double analysisTime = chrono::duration<double>(Clock::now() - t0).count();

  cout << "  Symbolic analysis: " << fixed << setprecision(4) << analysisTime << "s" << endl;
  cout << "  Solver: " << solver->skel().numLumps() << " lumps, " << solver->skel().numSpans()
       << " spans, dataSize=" << solver->totalDataSize() << endl;

  const auto& elimRanges = solver->sparseEliminationRanges();
  if (elimRanges.size() >= 2) {
    cout << "  Sparse elimination: " << elimRanges.back() << "/" << solver->skel().numLumps()
         << " lumps (" << (elimRanges.size() - 1) << " levels)" << endl;
  }

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

    if (b.norm() < 1e-15) {
      skipped++;
      continue;
    }

    // Compute per-matrix equilibration
    applyRowPermToCsr<double>(n, A.rowPtr.data(), A.colInd.data(), A.values.data(),
                              preproc.rowPerm.data(), pRowPtr, pColInd, pValues);
    vector<double> rowScale, colScale;
    computeEquilibration(n, pRowPtr.data(), pColInd.data(), pValues.data(), rowScale, colScale);

    // Apply scaling + perm
    applyRowPermAndScaleToCsr<double>(n, A.rowPtr.data(), A.colInd.data(), A.values.data(),
                                     preproc.rowPerm.data(), rowScale.data(), colScale.data(),
                                     pRowPtr, pColInd, pValues);

    // Load and factor on GPU
    fill(data.begin(), data.end(), 0.0);
    solver->loadFromCsr(pRowPtr.data(), pColInd.data(), blockSizes.data(), pValues.data(),
                        data.data());

    double factorTime;
    {
      DevMirror<double> dataGpu(data);
      auto tFactor = Clock::now();
      solver->factorLU(dataGpu.ptr, pivots.data());
      factorTime = chrono::duration<double>(Clock::now() - tFactor).count();
      dataGpu.get(data);
    }

    int64_t perturbCount = solver->staticPivotPerturbCount();

    // Initial solve
    Eigen::VectorXd bp(n);
    for (int64_t j = 0; j < n; j++) {
      bp(perm[j]) = rowScale[j] * b(preproc.rowPerm[j]);
    }

    auto tSolve = Clock::now();
    {
      DevMirror<double> dataGpu(data);
      DevMirror<double> xGpu(vector<double>(bp.data(), bp.data() + n));
      solver->solveLU(dataGpu.ptr, pivots.data(), xGpu.ptr, n, 1);
      vector<double> xVec(n);
      xGpu.get(xVec);
      for (int64_t j = 0; j < n; j++) bp(j) = xVec[j];
    }

    Eigen::VectorXd x(n);
    for (int64_t j = 0; j < n; j++) {
      x(j) = colScale[j] * bp(perm[j]);
    }

    double initialResidual = computeResidual(A, x, b);

    // Iterative refinement
    int refineSteps = 0;
    const int maxRefine = 30;
    double residual = initialResidual;
    for (int iter = 0; iter < maxRefine && residual > 1e-10; iter++) {
      Eigen::VectorXd r = Eigen::VectorXd::Zero(n);
      for (int64_t i = 0; i < n; i++) {
        for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
          r(i) += A.values[k] * x(A.colInd[k]);
        }
      }
      r = b - r;

      for (int64_t j = 0; j < n; j++) {
        bp(perm[j]) = rowScale[j] * r(preproc.rowPerm[j]);
      }

      {
        DevMirror<double> dataGpu(data);
        DevMirror<double> xGpu(vector<double>(bp.data(), bp.data() + n));
        solver->solveLU(dataGpu.ptr, pivots.data(), xGpu.ptr, n, 1);
        vector<double> xVec(n);
        xGpu.get(xVec);
        for (int64_t j = 0; j < n; j++) bp(j) = xVec[j];
      }

      for (int64_t j = 0; j < n; j++) {
        x(j) += colScale[j] * bp(perm[j]);
      }

      residual = computeResidual(A, x, b);
      refineSteps++;
    }
    double solveTime = chrono::duration<double>(Clock::now() - tSolve).count();

    cout << "  Matrix #" << idx << ": factor=" << fixed << setprecision(4) << factorTime
         << "s, solve=" << solveTime << "s" << ", initial_res=" << scientific << setprecision(2)
         << initialResidual << ", final_res=" << residual << ", refine=" << refineSteps
         << ", perturbed=" << perturbCount << endl;

    EXPECT_LT(residual, 1e-6) << "Matrix #" << idx << " residual too large after refinement";

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
