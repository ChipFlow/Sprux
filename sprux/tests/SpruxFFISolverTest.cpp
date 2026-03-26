/*
 * Copyright (c) 2024-2026 ChipFlow
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests for SpruxFFISolver — the encapsulated FFI solver API.
// Verifies that it produces correct solutions on circuit Jacobian sequences.

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "sprux/sprux/SpruxFFISolver.h"
#include "sprux/testing/MatrixMarketReader.h"

using namespace Sprux;
using namespace Sprux::testing_utils;
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

static vector<pair<CsrMatrix, Eigen::VectorXd>> loadSequence(const string& dir, int maxCount = -1) {
  vector<pair<CsrMatrix, Eigen::VectorXd>> result;
  for (int idx = 0;; idx++) {
    if (maxCount >= 0 && idx >= maxCount) break;
    ostringstream jacName, rhsName;
    jacName << dir << "/jacobian_" << setw(4) << setfill('0') << idx << ".mtx";
    rhsName << dir << "/rhs_" << setw(4) << setfill('0') << idx << ".mtx";
    if (!fs::exists(jacName.str()) || !fs::exists(rhsName.str())) break;
    result.push_back({readMatrixMarket(jacName.str()), readRhsVector(rhsName.str())});
  }
  return result;
}

static double computeRelativeResidual(const CsrMatrix& A, const double* x,
                                      const double* b, int64_t n) {
  double resNormSq = 0.0, bNormSq = 0.0;
  for (int64_t i = 0; i < n; i++) {
    double Ax_i = 0.0;
    for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      Ax_i += A.values[k] * x[A.colInd[k]];
    }
    double r = Ax_i - b[i];
    resNormSq += r * r;
    bNormSq += b[i] * b[i];
  }
  return std::sqrt(resNormSq / std::max(bNormSq, 1e-300));
}

// Test: Solve a sequence of c6288 Jacobians and verify low relative residual
TEST(SpruxFFISolver, C6288Sequence) {
  string dir = findTestDataDir("c6288_sequence");
  if (dir.empty()) {
    GTEST_SKIP() << "c6288_sequence test data not found";
  }

  // Load first 5 matrices
  auto matrices = loadSequence(dir, 5);
  ASSERT_GE(matrices.size(), 2u) << "Need at least 2 matrices for testing";

  const CsrMatrix& A0 = matrices[0].first;
  int64_t n = A0.nRows;
  int32_t n32 = static_cast<int32_t>(n);
  int32_t nnz32 = static_cast<int32_t>(A0.nnz);

  // Convert int64 CSR to int32 (as VAJAX would provide)
  vector<int32_t> indptr32(A0.rowPtr.begin(), A0.rowPtr.end());
  vector<int32_t> indices32(A0.colInd.begin(), A0.colInd.end());

  // Test with 0 refinement steps first (pure f32)
  {
    SpruxFFISolver solver0(n32, nnz32, indptr32.data(), indices32.data(),
                           A0.values.data(), /*max_refine_steps=*/0);
    vector<double> x(n);
    const Eigen::VectorXd& b0 = matrices[0].second;
    solver0.solve(A0.values.data(), b0.data(), x.data());
    double relRes = computeRelativeResidual(A0, x.data(), b0.data(), n);
    cout << "  Matrix 0 (0 refine): relative residual = " << scientific << setprecision(3)
         << relRes << endl;

    // Check a few x values
    cout << "  x[0..4] = ";
    for (int i = 0; i < 5; i++) cout << x[i] << " ";
    cout << endl;

    // Check RHS stats
    double bNorm = b0.norm();
    int bNonzero = 0;
    for (int i = 0; i < n; i++) if (abs(b0(i)) > 1e-15) bNonzero++;
    cout << "  ||b|| = " << bNorm << ", non-zero entries: " << bNonzero << "/" << n << endl;
    cout << "  max(|x|) = " << *max_element(x.begin(), x.end(), [](double a, double b) {
      return abs(a) < abs(b);
    }) << endl;

    // Also try the existing MetalSequenceSolveTest approach for comparison:
    // Use the benchmark's residual approach directly.
    double Axnorm = 0, bnorm2 = 0;
    for (int64_t i = 0; i < n; i++) {
      double Ax_i = 0;
      for (int64_t k = A0.rowPtr[i]; k < A0.rowPtr[i+1]; k++)
        Ax_i += A0.values[k] * x[A0.colInd[k]];
      double ri = Ax_i - b0(i);
      Axnorm += ri*ri;
      bnorm2 += b0(i)*b0(i);
    }
    cout << "  ||Ax-b|| = " << sqrt(Axnorm) << ", ||b|| = " << sqrt(bnorm2) << endl;
  }

  // Create solver with first matrix
  SpruxFFISolver solver(n32, nnz32, indptr32.data(), indices32.data(),
                        A0.values.data(), /*max_refine_steps=*/10);

  EXPECT_EQ(solver.n(), n);

  // Solve each matrix in the sequence
  vector<double> x(n);
  for (size_t mi = 0; mi < matrices.size(); mi++) {
    const CsrMatrix& A = matrices[mi].first;
    const Eigen::VectorXd& b = matrices[mi].second;

    solver.solve(A.values.data(), b.data(), x.data());

    double relRes = computeRelativeResidual(A, x.data(), b.data(), n);
    cout << "  Matrix " << mi << ": relative residual = " << scientific << setprecision(3) << relRes
         << endl;

    // With f32 factorization + iterative refinement, expect near-f64 accuracy
    EXPECT_LT(relRes, 1e-10) << "Matrix " << mi << " has excessive residual";
  }
}

// Test: Profile convergence rate per refinement iteration
TEST(SpruxFFISolver, ConvergenceProfile) {
  string dir = findTestDataDir("c6288_sequence");
  if (dir.empty()) {
    GTEST_SKIP() << "c6288_sequence test data not found";
  }

  auto matrices = loadSequence(dir, 5);
  ASSERT_GE(matrices.size(), 2u);

  const CsrMatrix& A0 = matrices[0].first;
  int64_t n = A0.nRows;
  int32_t n32 = static_cast<int32_t>(n);
  int32_t nnz32 = static_cast<int32_t>(A0.nnz);
  vector<int32_t> indptr32(A0.rowPtr.begin(), A0.rowPtr.end());
  vector<int32_t> indices32(A0.colInd.begin(), A0.colInd.end());

  cout << "\n  Convergence profile (residual vs refine steps):\n";
  cout << "  refine";
  for (size_t mi = 0; mi < matrices.size(); mi++) cout << "    matrix_" << mi;
  cout << endl;

  for (int refine = 0; refine <= 10; refine++) {
    SpruxFFISolver solver(n32, nnz32, indptr32.data(), indices32.data(),
                          A0.values.data(), refine);
    cout << "  " << setw(6) << refine;
    for (size_t mi = 0; mi < matrices.size(); mi++) {
      const CsrMatrix& A = matrices[mi].first;
      const Eigen::VectorXd& b = matrices[mi].second;
      vector<double> x(n);
      solver.solve(A.values.data(), b.data(), x.data());
      double relRes = computeRelativeResidual(A, x.data(), b.data(), n);
      cout << "  " << scientific << setprecision(1) << setw(10) << relRes;
    }
    cout << endl;
  }
}

// Test: Small known matrix to verify the full solve pipeline
TEST(SpruxFFISolver, SmallMatrix) {
  // 3x3 matrix: A = [[4, 1, 0], [1, 3, 1], [0, 1, 4]]
  // CSR: indptr=[0,2,5,7], indices=[0,1, 0,1,2, 1,2], data=[4,1,1,3,1,1,4]
  int32_t n = 3, nnz = 7;
  vector<int32_t> indptr = {0, 2, 5, 7};
  vector<int32_t> indices = {0, 1, 0, 1, 2, 1, 2};
  vector<double> data = {4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 4.0};
  vector<double> rhs = {5.0, 5.0, 5.0};

  // Expected: x = A^-1 * b
  // Solve manually: x ≈ [1.0345, 0.8621, 1.0345]
  // (4x0 + 1x1 = 5, 1x0 + 3x1 + 1x2 = 5, 1x1 + 4x2 = 5)

  SpruxFFISolver solver(n, nnz, indptr.data(), indices.data(), data.data(), 0);
  vector<double> x(n);
  solver.solve(data.data(), rhs.data(), x.data());

  // Compute residual
  double relRes = 0, bNorm = 0;
  for (int i = 0; i < n; i++) {
    double Ax_i = 0;
    for (int k = indptr[i]; k < indptr[i + 1]; k++) {
      Ax_i += data[k] * x[indices[k]];
    }
    double r = Ax_i - rhs[i];
    relRes += r * r;
    bNorm += rhs[i] * rhs[i];
  }
  relRes = sqrt(relRes / bNorm);
  cout << "  Small 3x3: x = [" << x[0] << ", " << x[1] << ", " << x[2] << "]" << endl;
  cout << "  Small 3x3: relative residual = " << scientific << setprecision(3) << relRes << endl;
  EXPECT_LT(relRes, 1e-5) << "Small matrix solve failed";
}

// Test: solveOnly() reuses factored data for chord Newton
TEST(SpruxFFISolver, SolveOnly) {
  int32_t n = 3, nnz = 7;
  vector<int32_t> indptr = {0, 2, 5, 7};
  vector<int32_t> indices = {0, 1, 0, 1, 2, 1, 2};
  vector<double> data = {4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 4.0};

  SpruxFFISolver solver(n, nnz, indptr.data(), indices.data(), data.data(), 10);

  // First: full solve to establish factorization
  vector<double> rhs1 = {5.0, 5.0, 5.0};
  vector<double> x1(n);
  solver.solve(data.data(), rhs1.data(), x1.data());

  // Now: solveOnly with a different RHS but same matrix (chord Newton scenario)
  vector<double> rhs2 = {1.0, 2.0, 3.0};
  vector<double> x2(n);
  solver.solveOnly(data.data(), rhs2.data(), x2.data());

  // Verify x2 = A^-1 * rhs2
  double relRes = 0, bNorm = 0;
  for (int i = 0; i < n; i++) {
    double Ax_i = 0;
    for (int k = indptr[i]; k < indptr[i + 1]; k++) {
      Ax_i += data[k] * x2[indices[k]];
    }
    double r = Ax_i - rhs2[i];
    relRes += r * r;
    bNorm += rhs2[i] * rhs2[i];
  }
  relRes = sqrt(relRes / bNorm);
  cout << "  solveOnly: x = [" << x2[0] << ", " << x2[1] << ", " << x2[2] << "]" << endl;
  cout << "  solveOnly: relative residual = " << scientific << setprecision(3) << relRes << endl;
  EXPECT_LT(relRes, 1e-5) << "solveOnly failed";

  // Also test solveOnly vs fresh solve — should give same result
  vector<double> x2_ref(n);
  solver.solve(data.data(), rhs2.data(), x2_ref.data());
  double maxDiff = 0;
  for (int i = 0; i < n; i++) maxDiff = max(maxDiff, abs(x2[i] - x2_ref[i]));
  cout << "  solveOnly vs solve diff: " << scientific << setprecision(3) << maxDiff << endl;
  EXPECT_LT(maxDiff, 1e-10) << "solveOnly doesn't match solve";
}

// Test: dot() computes correct SpMV
TEST(SpruxFFISolver, DotProduct) {
  string dir = findTestDataDir("c6288_sequence");
  if (dir.empty()) {
    GTEST_SKIP() << "c6288_sequence test data not found";
  }

  auto matrices = loadSequence(dir, 1);
  ASSERT_GE(matrices.size(), 1u);

  const CsrMatrix& A = matrices[0].first;
  int64_t n = A.nRows;
  int32_t n32 = static_cast<int32_t>(n);
  int32_t nnz32 = static_cast<int32_t>(A.nnz);

  vector<int32_t> indptr32(A.rowPtr.begin(), A.rowPtr.end());
  vector<int32_t> indices32(A.colInd.begin(), A.colInd.end());

  SpruxFFISolver solver(n32, nnz32, indptr32.data(), indices32.data(),
                        A.values.data(), 1);

  // Test with known vector (ones)
  vector<double> x(n, 1.0);
  vector<double> b(n);
  solver.dot(A.values.data(), x.data(), b.data());

  // Verify against Eigen reference
  Eigen::VectorXd bRef = Eigen::VectorXd::Zero(n);
  for (int64_t i = 0; i < n; i++) {
    for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      bRef(i) += A.values[k] * x[A.colInd[k]];
    }
  }

  double maxDiff = 0;
  for (int64_t i = 0; i < n; i++) {
    maxDiff = max(maxDiff, abs(b[i] - bRef(i)));
  }
  EXPECT_LT(maxDiff, 1e-12) << "dot() SpMV mismatch";
}
