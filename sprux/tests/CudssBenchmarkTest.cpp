/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// cuDSS vs Sprux profiling benchmark.
// Loads c6288 Jacobian (Matrix Market) and solves with both solvers.
// Run under `nsys profile` to compare GPU execution patterns.

#ifdef SPRUX_HAVE_CUDSS

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cudss.h>
#include <nvtx3/nvToolsExt.h>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "sprux/sprux/CoalescedBlockMatrix.h"
#include "sprux/sprux/CudaDefs.h"
#include "sprux/sprux/EliminationTree.h"
#include "sprux/sprux/Solver.h"
#include "sprux/sprux/SparseStructure.h"
#include "sprux/sprux/Utils.h"
#include "sprux/testing/MatrixMarketReader.h"
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
// Get the test data directory
// ============================================================================

static string getMtxDir() {
  // Check --mtx command-line arg (gtest doesn't parse custom args, use env)
  const char* envDir = getenv("SPRUX_MTX_DIR");
  if (envDir) return string(envDir);

  // Default: test_data/c6288_jacobian relative to repo root
  // Try a few common locations
  vector<string> candidates = {
      "test_data/c6288_jacobian",
      "../test_data/c6288_jacobian",
      "../../test_data/c6288_jacobian",
  };

  for (const auto& dir : candidates) {
    ifstream test(dir + "/jacobian.mtx");
    if (test.good()) return dir;
  }

  // Last resort: absolute path
  return "test_data/c6288_jacobian";
}

// ============================================================================
// CSR -> CSC conversion (cuDSS uses CSR, UMFPACK-style needs CSC)
// ============================================================================

struct CscMatrix {
  int64_t nRows, nCols, nnz;
  vector<int64_t> colPtr;
  vector<int64_t> rowInd;
  vector<double> values;
};

CscMatrix csrToCsc(const CsrMatrix& csr) {
  CscMatrix csc;
  csc.nRows = csr.nRows;
  csc.nCols = csr.nCols;
  csc.nnz = csr.nnz;
  csc.colPtr.resize(csr.nCols + 1, 0);
  csc.rowInd.resize(csr.nnz);
  csc.values.resize(csr.nnz);

  // Count entries per column
  for (int64_t i = 0; i < csr.nnz; i++) {
    csc.colPtr[csr.colInd[i] + 1]++;
  }
  for (int64_t j = 0; j < csr.nCols; j++) {
    csc.colPtr[j + 1] += csc.colPtr[j];
  }

  // Fill values
  vector<int64_t> colCursor = csc.colPtr;
  for (int64_t i = 0; i < csr.nRows; i++) {
    for (int64_t k = csr.rowPtr[i]; k < csr.rowPtr[i + 1]; k++) {
      int64_t col = csr.colInd[k];
      int64_t pos = colCursor[col]++;
      csc.rowInd[pos] = i;
      csc.values[pos] = csr.values[k];
    }
  }

  return csc;
}

// ============================================================================
// cuDSS error checking macro
// ============================================================================

#define cudssCHECK(call)                                                          \
  do {                                                                            \
    cudssStatus_t status = (call);                                                \
    if (status != CUDSS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "[%s:%d] cuDSS Error: %d\n", __FILE__, __LINE__, (int)status); \
      FAIL() << "cuDSS call failed with status " << (int)status;                  \
    }                                                                             \
  } while (0)

// ============================================================================
// Sprux CUDA LU test: load c6288 Jacobian, solve on GPU
// ============================================================================

TEST(CudssBenchmark, Sprux_LU) {
  string mtxDir = getMtxDir();
  cout << "Loading matrix from: " << mtxDir << endl;

  CsrMatrix A = readMatrixMarket(mtxDir + "/jacobian.mtx");
  Vector<double> b = readRhsVector(mtxDir + "/rhs.mtx");

  cout << "Matrix: " << A.nRows << " x " << A.nCols << ", nnz=" << A.nnz << endl;
  ASSERT_EQ(A.nRows, A.nCols) << "Matrix must be square";
  ASSERT_EQ(b.size(), A.nRows) << "RHS size mismatch";

  int64_t n = A.nRows;

  nvtxRangePush("Sprux_Analysis");

  // Build CSR lower-triangle SparseStructure from CSR (ensure diagonal present)
  vector<set<int64_t>> colBlocks(n);
  for (int64_t i = 0; i < n; i++) {
    colBlocks[i].insert(i);  // ensure diagonal is present
    for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      int64_t j = A.colInd[k];
      colBlocks[min(i, j)].insert(max(i, j));
    }
  }
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

  vector<int64_t> paramSizes(n, 1);
  vector<int64_t> blockSizes(n, 1);

  Settings settings;
  settings.backend = BackendCuda;
  settings.matrixType = MTYPE_GENERAL;
  auto solver = createSolver(settings, paramSizes, ss);

  vector<double> data(solver->skel().totalDataSize(), 0.0);
  solver->loadFromCsr(A.rowPtr.data(), A.colInd.data(), blockSizes.data(), A.values.data(),
                      data.data());

  nvtxRangePop();  // Sprux_Analysis

  vector<int64_t> pivots(n);
  const auto& perm = solver->paramToSpan();

  nvtxRangePush("Sprux_Factor");
  auto factorStart = hrc::now();
  solver->factorLU(data.data(), pivots.data());
  double factorMs = tdelta(hrc::now() - factorStart).count() * 1000;
  nvtxRangePop();

  // Permute RHS, solve, inverse permute
  Vector<double> bp(n);
  for (int64_t i = 0; i < n; i++) bp(perm[i]) = b(i);

  nvtxRangePush("Sprux_Solve");
  auto solveStart = hrc::now();
  solver->solveLU(data.data(), pivots.data(), bp.data(), n, 1);
  double solveMs = tdelta(hrc::now() - solveStart).count() * 1000;
  nvtxRangePop();

  Vector<double> x(n);
  for (int64_t i = 0; i < n; i++) x(i) = bp(perm[i]);

  double residual = computeResidual(A, x, b);

  cout << "\n=== Sprux CUDA LU ===" << endl;
  cout << "  Factor: " << fixed << setprecision(2) << factorMs << " ms" << endl;
  cout << "  Solve:  " << solveMs << " ms" << endl;
  cout << "  Residual: " << scientific << setprecision(4) << residual << endl;

  EXPECT_LT(residual, 1e-8) << "Sprux residual too large";
}

// ============================================================================
// cuDSS LU test: load c6288 Jacobian, solve on GPU via cuDSS
// ============================================================================

TEST(CudssBenchmark, CuDSS_LU) {
  string mtxDir = getMtxDir();
  cout << "Loading matrix from: " << mtxDir << endl;

  CsrMatrix A = readMatrixMarket(mtxDir + "/jacobian.mtx");
  Vector<double> b = readRhsVector(mtxDir + "/rhs.mtx");

  cout << "Matrix: " << A.nRows << " x " << A.nCols << ", nnz=" << A.nnz << endl;
  ASSERT_EQ(A.nRows, A.nCols) << "Matrix must be square";
  ASSERT_EQ(b.size(), A.nRows) << "RHS size mismatch";

  int64_t n = A.nRows;

  // cuDSS uses CSR format with int32 indices
  // Convert int64_t -> int for cuDSS
  vector<int> csrRowPtr32(A.rowPtr.begin(), A.rowPtr.end());
  vector<int> csrColInd32(A.colInd.begin(), A.colInd.end());

  // Allocate GPU memory
  int* d_rowPtr = nullptr;
  int* d_colInd = nullptr;
  double* d_values = nullptr;
  double* d_b = nullptr;
  double* d_x = nullptr;

  cuCHECK(cudaMalloc(&d_rowPtr, (n + 1) * sizeof(int)));
  cuCHECK(cudaMalloc(&d_colInd, A.nnz * sizeof(int)));
  cuCHECK(cudaMalloc(&d_values, A.nnz * sizeof(double)));
  cuCHECK(cudaMalloc(&d_b, n * sizeof(double)));
  cuCHECK(cudaMalloc(&d_x, n * sizeof(double)));

  cuCHECK(cudaMemcpy(d_rowPtr, csrRowPtr32.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  cuCHECK(cudaMemcpy(d_colInd, csrColInd32.data(), A.nnz * sizeof(int), cudaMemcpyHostToDevice));
  cuCHECK(cudaMemcpy(d_values, A.values.data(), A.nnz * sizeof(double), cudaMemcpyHostToDevice));
  cuCHECK(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));

  // Create cuDSS handle and objects
  cudssHandle_t handle = nullptr;
  cudssCHECK(cudssCreate(&handle));

  cudssConfig_t config = nullptr;
  cudssCHECK(cudssConfigCreate(&config));

  cudssData_t data = nullptr;
  cudssCHECK(cudssDataCreate(handle, &data));

  // Create matrix descriptor (CSR, general/unsymmetric)
  cudssMatrix_t cudssA = nullptr;
  int64_t nRows64 = n;
  int64_t nCols64 = n;
  int64_t nnz64 = A.nnz;
  cudssCHECK(cudssMatrixCreateCsr(
      &cudssA, nRows64, nCols64, nnz64,
      d_rowPtr, nullptr,  // row end offsets (null = derive from rowPtr)
      d_colInd, d_values,
      CUDA_R_32I, CUDA_R_64F,
      CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
      CUDSS_BASE_ZERO));

  // Create dense RHS and solution vectors
  cudssMatrix_t cudssB = nullptr;
  cudssMatrix_t cudssX = nullptr;
  int64_t nrhs = 1;
  int64_t ldb = n;
  int64_t ldx = n;
  cudssCHECK(cudssMatrixCreateDn(&cudssB, nRows64, nrhs, ldb, d_b, CUDA_R_64F,
                                  CUDSS_LAYOUT_COL_MAJOR));
  cudssCHECK(cudssMatrixCreateDn(&cudssX, nRows64, nrhs, ldx, d_x, CUDA_R_64F,
                                  CUDSS_LAYOUT_COL_MAJOR));

  // Analysis phase
  nvtxRangePush("CuDSS_Analysis");
  auto analysisStart = hrc::now();
  cudssCHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, cudssA, cudssX, cudssB));
  cuCHECK(cudaDeviceSynchronize());
  double analysisMs = tdelta(hrc::now() - analysisStart).count() * 1000;
  nvtxRangePop();

  // Factorization phase
  nvtxRangePush("CuDSS_Factor");
  auto factorStart = hrc::now();
  cudssCHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, cudssA, cudssX, cudssB));
  cuCHECK(cudaDeviceSynchronize());
  double factorMs = tdelta(hrc::now() - factorStart).count() * 1000;
  nvtxRangePop();

  // Solve phase
  nvtxRangePush("CuDSS_Solve");
  auto solveStart = hrc::now();
  cudssCHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, cudssA, cudssX, cudssB));
  cuCHECK(cudaDeviceSynchronize());
  double solveMs = tdelta(hrc::now() - solveStart).count() * 1000;
  nvtxRangePop();

  // Copy solution back
  Vector<double> x(n);
  cuCHECK(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

  double residual = computeResidual(A, x, b);

  cout << "\n=== cuDSS LU ===" << endl;
  cout << "  Analysis: " << fixed << setprecision(2) << analysisMs << " ms" << endl;
  cout << "  Factor:   " << factorMs << " ms" << endl;
  cout << "  Solve:    " << solveMs << " ms" << endl;
  cout << "  Residual: " << scientific << setprecision(4) << residual << endl;

  EXPECT_LT(residual, 1e-8) << "cuDSS residual too large";

  // Cleanup
  cudssMatrixDestroy(cudssA);
  cudssMatrixDestroy(cudssB);
  cudssMatrixDestroy(cudssX);
  cudssDataDestroy(handle, data);
  cudssConfigDestroy(config);
  cudssDestroy(handle);

  cudaFree(d_rowPtr);
  cudaFree(d_colInd);
  cudaFree(d_values);
  cudaFree(d_b);
  cudaFree(d_x);
}

#endif  // SPRUX_HAVE_CUDSS
