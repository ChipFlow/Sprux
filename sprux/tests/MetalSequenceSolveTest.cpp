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
#include "sprux/sprux/MetalDefs.h"
#include "sprux/sprux/Preprocessing.h"
#include "sprux/sprux/Solver.h"
#include "sprux/sprux/SparseStructure.h"
#include "sprux/testing/MatrixMarketReader.h"
#include "sprux/testing/TestingUtils.h"

using namespace Sprux;
using namespace ::Sprux::testing_utils;
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

// Double-precision residual computation (for mixed-precision iterative refinement)
static double computeResidualDouble(const CsrMatrix& A, const Eigen::VectorXd& x,
                                    const Eigen::VectorXd& b) {
  Eigen::VectorXd Ax = Eigen::VectorXd::Zero(A.nRows);
  for (int64_t i = 0; i < A.nRows; i++) {
    for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      Ax(i) += A.values[k] * x(A.colInd[k]);
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

  using Clock = chrono::high_resolution_clock;

  // Build solver once via CHOLMOD-based symbolic analysis
  SparseStructure ss = csrToSparseStructure(A0);
  vector<int64_t> paramSizes(n, 1);
  vector<int64_t> blockSizes(n, 1);

  // Metal GPU solver
  Settings metalSettings;
  metalSettings.backend = BackendMetal;
  metalSettings.matrixType = MTYPE_GENERAL;

  auto t0 = Clock::now();
  auto solver = createSolver(metalSettings, paramSizes, ss);
  double analysisTime = chrono::duration<double>(Clock::now() - t0).count();

  cout << "=== Metal Ring Oscillator Timing ===" << endl;
  cout << "  Symbolic analysis: " << fixed << setprecision(4) << analysisTime << "s" << endl;

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
  double totalFactorTime = 0;
  double totalSolveTime = 0;

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
    // Some matrices trigger zero pivots in Sprux's float LU due to fill-in
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

    double factorTime, solveTime;
    {
      MetalMirror<float> dataGpu(data);
      auto tFactor = Clock::now();
      solver->factorLU(dataGpu.ptr(), pivots.data());
      factorTime = chrono::duration<double>(Clock::now() - tFactor).count();
      dataGpu.get(data);
    }

    // Permute RHS, solve on GPU, inverse permute
    Eigen::VectorXf bp(n);
    for (int64_t j = 0; j < n; j++) bp(perm[j]) = b(j);

    {
      MetalMirror<float> dataGpu(data);
      MetalMirror<float> xGpu(vector<float>(bp.data(), bp.data() + n));
      auto tSolve = Clock::now();
      solver->solveLU(dataGpu.ptr(), pivots.data(), xGpu.ptr(), n, 1);
      solveTime = chrono::duration<double>(Clock::now() - tSolve).count();
      vector<float> xVec(n);
      xGpu.get(xVec);
      for (int64_t j = 0; j < n; j++) bp(j) = xVec[j];
    }

    Eigen::VectorXf x(n);
    for (int64_t j = 0; j < n; j++) x(j) = bp(perm[j]);

    float residual = computeResidualFloat(A, x, b);

    // Metal should be within 100x of CPU Sprux float
    float threshold = max(cpuResidual * 100.0f, 1e-4f);
    EXPECT_LT(residual, threshold) << "Matrix #" << i << " Metal residual too large: " << residual
                                   << " (CPU: " << cpuResidual << ")";

    // Dump solution for external comparison (e.g., scipy)
    if (getenv("SPRUX_DUMP_SOLUTIONS")) {
      cout << "SOLUTION_DUMP:" << i << ":";
      for (int64_t j = 0; j < n; j++) {
        if (j > 0) cout << " ";
        cout << setprecision(9) << x(j);
      }
      cout << endl;
    }

    cout << "  Matrix #" << i << ": factor=" << fixed << setprecision(4) << factorTime
         << "s, solve=" << solveTime << "s, residual=" << scientific << setprecision(2) << residual
         << " (CPU: " << cpuResidual << ")" << endl;

    totalFactorTime += factorTime;
    totalSolveTime += solveTime;
    passed++;
  }

  if (passed > 0) {
    cout << "  Average: factor=" << fixed << setprecision(4) << totalFactorTime / passed
         << "s, solve=" << totalSolveTime / passed << "s" << endl;
  }
  cout << "Passed: " << passed << ", Skipped: " << skipped << endl;
  ASSERT_GT(passed, 0) << "No matrices were actually tested";
}

// ============================================================================
// C6288 sequence on Metal with mixed-precision iterative refinement.
// Factor in float on GPU, residual in double on CPU, correction solve in float on GPU.
// Preprocessing (BTF max transversal + equilibration) applied before factorization.
// ============================================================================

TEST(MetalSequenceSolve, C6288) {
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

  cout << "Metal C6288 sequence: " << files.size() << " matrices in " << dir << ", testing "
       << testIndices.size() << endl;

  // Load first matrix for sparsity structure
  CsrMatrix A0 = readMatrixMarket(files[0].first);
  ASSERT_EQ(A0.nRows, A0.nCols) << "Matrix must be square";
  cout << "Matrix size: " << A0.nRows << " x " << A0.nCols << ", nnz=" << A0.nnz << endl;

  int64_t n = A0.nRows;

  using Clock = chrono::high_resolution_clock;

  // Step 1: BTF max transversal — find row permutation for zero-free diagonal (once per pattern)
  auto tPreproc = Clock::now();
  auto preproc = computeMaxTransversal(n, A0.rowPtr.data(), A0.colInd.data());
  double preprocTime = chrono::duration<double>(Clock::now() - tPreproc).count();

  cout << "=== Metal C6288 Timing ===" << endl;
  cout << "  Preprocessing (max transversal): " << fixed << setprecision(4) << preprocTime
       << "s, structural rank=" << preproc.structuralRank << "/" << n << endl;
  ASSERT_EQ(preproc.structuralRank, n) << "Matrix is structurally singular";

  // Step 2: Apply row perm to first matrix, build symmetric structure for solver
  vector<int64_t> pRowPtr, pColInd;
  vector<double> pValues;
  applyRowPermToCsr<double>(n, A0.rowPtr.data(), A0.colInd.data(), A0.values.data(),
                            preproc.rowPerm.data(), pRowPtr, pColInd, pValues);

  SparseStructure ss = csrToSymmetricSparseStructure(n, pRowPtr.data(), pColInd.data());
  vector<int64_t> paramSizes(n, 1);
  vector<int64_t> blockSizes(n, 1);

  // Metal GPU solver (float)
  Settings metalSettings;
  metalSettings.backend = BackendMetal;
  metalSettings.matrixType = MTYPE_GENERAL;
  metalSettings.staticPivotThreshold = 0.0;  // auto: cbrt(eps) * max_diag

  auto t0 = Clock::now();
  auto solver = createSolver(metalSettings, paramSizes, ss);
  double analysisTime = chrono::duration<double>(Clock::now() - t0).count();

  cout << "  Symbolic analysis: " << fixed << setprecision(4) << analysisTime << "s" << endl;
  cout << "  Solver: " << solver->skel().numLumps() << " lumps, " << solver->skel().numSpans()
       << " spans, dataSize=" << solver->totalDataSize() << endl;

  const auto& elimRanges = solver->sparseEliminationRanges();
  if (elimRanges.size() >= 2) {
    cout << "  Sparse elimination: " << elimRanges.back() << "/" << solver->skel().numLumps()
         << " lumps (" << (elimRanges.size() - 1) << " levels)" << endl;
  }

  vector<float> data(solver->skel().totalDataSize());
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

    // Compute per-matrix equilibration on Q*A (row perm applied, then scale)
    applyRowPermToCsr<double>(n, A.rowPtr.data(), A.colInd.data(), A.values.data(),
                              preproc.rowPerm.data(), pRowPtr, pColInd, pValues);
    vector<double> rowScale, colScale;
    computeEquilibration(n, pRowPtr.data(), pColInd.data(), pValues.data(), rowScale, colScale);

    // Apply scaling: A' = Dr * Q * A * Dc, then convert to float for GPU
    vector<int64_t> sRowPtr, sColInd;
    vector<float> sValues;
    applyRowPermAndScaleToCsr<double>(n, A.rowPtr.data(), A.colInd.data(), A.values.data(),
                                     preproc.rowPerm.data(), rowScale.data(), colScale.data(),
                                     pRowPtr, pColInd, pValues);
    // Convert scaled values to float
    sRowPtr = pRowPtr;
    sColInd = pColInd;
    sValues.assign(pValues.begin(), pValues.end());

    // Load scaled/permuted float data and factor on GPU
    fill(data.begin(), data.end(), 0.0f);
    solver->loadFromCsr(sRowPtr.data(), sColInd.data(), blockSizes.data(), sValues.data(),
                        data.data());

    double factorTime;
    {
      MetalMirror<float> dataGpu(data);
      auto tFactor = Clock::now();
      solver->factorLU(dataGpu.ptr(), pivots.data());
      factorTime = chrono::duration<double>(Clock::now() - tFactor).count();
      dataGpu.get(data);
    }

    int64_t perturbCount = solver->staticPivotPerturbCount();

    // Initial solve on GPU: Dr*Q*A*Dc * y = Dr*Q*b
    // RHS permutation + scaling in double, then cast to float for GPU solve
    Eigen::VectorXf bp(n);
    for (int64_t j = 0; j < n; j++) {
      bp(perm[j]) = float(rowScale[j] * b(preproc.rowPerm[j]));
    }

    auto tSolve = Clock::now();
    {
      MetalMirror<float> dataGpu(data);
      MetalMirror<float> xGpu(vector<float>(bp.data(), bp.data() + n));
      solver->solveLU(dataGpu.ptr(), pivots.data(), xGpu.ptr(), n, 1);
      vector<float> xVec(n);
      xGpu.get(xVec);
      for (int64_t j = 0; j < n; j++) bp(j) = xVec[j];
    }

    // Unscale solution: x = Dc * y (in double for accuracy)
    Eigen::VectorXd x(n);
    for (int64_t j = 0; j < n; j++) {
      x(j) = colScale[j] * double(bp(perm[j]));
    }

    double initialResidual = computeResidualDouble(A, x, b);

    // Mixed-precision iterative refinement:
    // Residual r = b - A*x computed in double (accurate)
    // Correction solve in float on GPU (fast)
    int refineSteps = 0;
    const int maxRefine = 30;
    double residual = initialResidual;
    for (int iter = 0; iter < maxRefine && residual > 1e-10; iter++) {
      // Compute r = b - A*x in double
      Eigen::VectorXd r = Eigen::VectorXd::Zero(n);
      for (int64_t i = 0; i < n; i++) {
        for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
          r(i) += A.values[k] * x(A.colInd[k]);
        }
      }
      r = b - r;

      // Solve for correction: Dr*Q*A*Dc * dy = Dr*Q*r (in float on GPU)
      for (int64_t j = 0; j < n; j++) {
        bp(perm[j]) = float(rowScale[j] * r(preproc.rowPerm[j]));
      }

      {
        MetalMirror<float> dataGpu(data);
        MetalMirror<float> xGpu(vector<float>(bp.data(), bp.data() + n));
        solver->solveLU(dataGpu.ptr(), pivots.data(), xGpu.ptr(), n, 1);
        vector<float> xVec(n);
        xGpu.get(xVec);
        for (int64_t j = 0; j < n; j++) bp(j) = xVec[j];
      }

      // Unscale and apply correction in double
      for (int64_t j = 0; j < n; j++) {
        x(j) += colScale[j] * double(bp(perm[j]));
      }

      residual = computeResidualDouble(A, x, b);
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
