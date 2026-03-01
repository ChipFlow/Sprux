/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// LU benchmark tool for non-symmetric matrices (circuit Jacobians).
// Supports Metal (float + mixed-precision iterative refinement),
// CPU (double), and CUDA (double) backends.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "BenchJson.h"
#include "baspacho/baspacho/Preprocessing.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/testing/MatrixMarketReader.h"
#include "baspacho/testing/TestingUtils.h"

#ifdef BASPACHO_USE_CUBLAS
#include "baspacho/baspacho/CudaDefs.h"
#endif

#ifdef BASPACHO_HAVE_CUDSS
#include <cuda_runtime.h>
#include <cudss.h>
#endif

#ifdef BASPACHO_USE_METAL
#include "baspacho/baspacho/MetalDefs.h"
#endif

using namespace BaSpaCho;
using namespace BaSpaCho::testing_utils;
using namespace std;
namespace fs = std::filesystem;

using Clock = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

// ============================================================================
// Shared helpers
// ============================================================================

// Build lower-triangle SparseStructure from CSR (ensures diagonal present)
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

// Double-precision residual (for CPU/CUDA and mixed-precision refinement)
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

// Discover sequence files in a directory
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

// ============================================================================
// Per-matrix timing results
// ============================================================================

struct LUTimingResult {
  double factorTime = 0;
  double solveTime = 0;   // includes iterative refinement if applicable
  double residual = 0;
  int refineSteps = 0;
  int64_t perturbCount = 0;
};

// ============================================================================
// CPU (double) benchmark
// ============================================================================

static vector<LUTimingResult> benchmarkLUCpu(
    const vector<pair<CsrMatrix, Eigen::VectorXd>>& matrices, bool verbose) {
  if (matrices.empty()) return {};

  const CsrMatrix& A0 = matrices[0].first;
  int64_t n = A0.nRows;

  // Preprocessing: BTF max transversal (once per pattern)
  auto preproc = computeMaxTransversal(n, A0.rowPtr.data(), A0.colInd.data());

  // Build symmetric structure from permuted matrix
  vector<int64_t> pRowPtr, pColInd;
  vector<double> pValues;
  applyRowPermToCsr<double>(n, A0.rowPtr.data(), A0.colInd.data(), A0.values.data(),
                            preproc.rowPerm.data(), pRowPtr, pColInd, pValues);
  SparseStructure ss = csrToSymmetricSparseStructure(n, pRowPtr.data(), pColInd.data());

  vector<int64_t> paramSizes(n, 1);
  vector<int64_t> blockSizes(n, 1);

  Settings settings;
  settings.backend = BackendFast;
  settings.matrixType = MTYPE_GENERAL;
  settings.staticPivotThreshold = 0.0;

  auto solver = createSolver(settings, paramSizes, ss);

  vector<double> data(solver->skel().totalDataSize());
  vector<int64_t> pivots(n);
  const auto& perm = solver->paramToSpan();

  vector<LUTimingResult> results;

  for (size_t mi = 0; mi < matrices.size(); mi++) {
    const CsrMatrix& A = matrices[mi].first;
    const Eigen::VectorXd& b = matrices[mi].second;
    LUTimingResult res;

    // Equilibration
    applyRowPermToCsr<double>(n, A.rowPtr.data(), A.colInd.data(), A.values.data(),
                              preproc.rowPerm.data(), pRowPtr, pColInd, pValues);
    vector<double> rowScale, colScale;
    computeEquilibration(n, pRowPtr.data(), pColInd.data(), pValues.data(), rowScale, colScale);

    // Apply scaling
    applyRowPermAndScaleToCsr<double>(n, A.rowPtr.data(), A.colInd.data(), A.values.data(),
                                     preproc.rowPerm.data(), rowScale.data(), colScale.data(),
                                     pRowPtr, pColInd, pValues);

    // Factor
    fill(data.begin(), data.end(), 0.0);
    solver->loadFromCsr(pRowPtr.data(), pColInd.data(), blockSizes.data(), pValues.data(),
                        data.data());

    auto tFactor = Clock::now();
    solver->factorLU(data.data(), pivots.data());
    res.factorTime = tdelta(Clock::now() - tFactor).count();
    res.perturbCount = solver->staticPivotPerturbCount();

    // Solve with iterative refinement
    Eigen::VectorXd bp(n);
    for (int64_t j = 0; j < n; j++) {
      bp(perm[j]) = rowScale[j] * b(preproc.rowPerm[j]);
    }

    auto tSolve = Clock::now();
    solver->solveLU(data.data(), pivots.data(), bp.data(), n, 1);

    Eigen::VectorXd x(n);
    for (int64_t j = 0; j < n; j++) {
      x(j) = colScale[j] * bp(perm[j]);
    }

    double residual = computeResidualDouble(A, x, b);
    res.refineSteps = 0;
    const int maxRefine = 30;
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
      solver->solveLU(data.data(), pivots.data(), bp.data(), n, 1);
      for (int64_t j = 0; j < n; j++) {
        x(j) += colScale[j] * bp(perm[j]);
      }
      residual = computeResidualDouble(A, x, b);
      res.refineSteps++;
    }
    res.solveTime = tdelta(Clock::now() - tSolve).count();
    res.residual = residual;

    if (verbose) {
      cout << "  [CPU] Matrix #" << mi << ": factor=" << fixed << setprecision(4) << res.factorTime
           << "s, solve=" << res.solveTime << "s, residual=" << scientific << setprecision(2)
           << res.residual << ", refine=" << res.refineSteps << ", perturbed=" << res.perturbCount
           << endl;
    }

    results.push_back(res);
  }

  return results;
}

// ============================================================================
// Metal (float + mixed-precision iterative refinement) benchmark
// ============================================================================

#ifdef BASPACHO_USE_METAL
static vector<LUTimingResult> benchmarkLUMetal(
    const vector<pair<CsrMatrix, Eigen::VectorXd>>& matrices, bool verbose) {
  if (matrices.empty()) return {};

  const CsrMatrix& A0 = matrices[0].first;
  int64_t n = A0.nRows;

  // Preprocessing
  auto preproc = computeMaxTransversal(n, A0.rowPtr.data(), A0.colInd.data());

  vector<int64_t> pRowPtr, pColInd;
  vector<double> pValues;
  applyRowPermToCsr<double>(n, A0.rowPtr.data(), A0.colInd.data(), A0.values.data(),
                            preproc.rowPerm.data(), pRowPtr, pColInd, pValues);
  SparseStructure ss = csrToSymmetricSparseStructure(n, pRowPtr.data(), pColInd.data());

  vector<int64_t> paramSizes(n, 1);
  vector<int64_t> blockSizes(n, 1);

  Settings metalSettings;
  metalSettings.backend = BackendMetal;
  metalSettings.matrixType = MTYPE_GENERAL;
  metalSettings.staticPivotThreshold = 0.0;

  auto solver = createSolver(metalSettings, paramSizes, ss);

  vector<float> data(solver->skel().totalDataSize());
  vector<int64_t> pivots(n);
  const auto& perm = solver->paramToSpan();

  vector<LUTimingResult> results;

  for (size_t mi = 0; mi < matrices.size(); mi++) {
    const CsrMatrix& A = matrices[mi].first;
    const Eigen::VectorXd& b = matrices[mi].second;
    LUTimingResult res;

    // Equilibration
    applyRowPermToCsr<double>(n, A.rowPtr.data(), A.colInd.data(), A.values.data(),
                              preproc.rowPerm.data(), pRowPtr, pColInd, pValues);
    vector<double> rowScale, colScale;
    computeEquilibration(n, pRowPtr.data(), pColInd.data(), pValues.data(), rowScale, colScale);

    // Apply scaling then convert to float
    applyRowPermAndScaleToCsr<double>(n, A.rowPtr.data(), A.colInd.data(), A.values.data(),
                                     preproc.rowPerm.data(), rowScale.data(), colScale.data(),
                                     pRowPtr, pColInd, pValues);
    vector<int64_t> sRowPtr = pRowPtr;
    vector<int64_t> sColInd = pColInd;
    vector<float> sValues(pValues.begin(), pValues.end());

    // Factor on GPU
    fill(data.begin(), data.end(), 0.0f);
    solver->loadFromCsr(sRowPtr.data(), sColInd.data(), blockSizes.data(), sValues.data(),
                        data.data());

    double factorTime;
    {
      MetalMirror<float> dataGpu(data);
      auto tFactor = Clock::now();
      solver->factorLU(dataGpu.ptr(), pivots.data());
      factorTime = tdelta(Clock::now() - tFactor).count();
      dataGpu.get(data);
    }
    res.factorTime = factorTime;
    res.perturbCount = solver->staticPivotPerturbCount();

    // Initial solve on GPU with mixed-precision iterative refinement
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

    Eigen::VectorXd x(n);
    for (int64_t j = 0; j < n; j++) {
      x(j) = colScale[j] * double(bp(perm[j]));
    }

    double residual = computeResidualDouble(A, x, b);
    res.refineSteps = 0;
    const int maxRefine = 30;
    for (int iter = 0; iter < maxRefine && residual > 1e-10; iter++) {
      Eigen::VectorXd r = Eigen::VectorXd::Zero(n);
      for (int64_t i = 0; i < n; i++) {
        for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
          r(i) += A.values[k] * x(A.colInd[k]);
        }
      }
      r = b - r;

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

      for (int64_t j = 0; j < n; j++) {
        x(j) += colScale[j] * double(bp(perm[j]));
      }
      residual = computeResidualDouble(A, x, b);
      res.refineSteps++;
    }
    res.solveTime = tdelta(Clock::now() - tSolve).count();
    res.residual = residual;

    if (verbose) {
      cout << "  [Metal] Matrix #" << mi << ": factor=" << fixed << setprecision(4)
           << res.factorTime << "s, solve=" << res.solveTime << "s, residual=" << scientific
           << setprecision(2) << res.residual << ", refine=" << res.refineSteps
           << ", perturbed=" << res.perturbCount << endl;
    }

    results.push_back(res);
  }

  return results;
}
#endif  // BASPACHO_USE_METAL

// ============================================================================
// CUDA (double) benchmark
// ============================================================================

#ifdef BASPACHO_USE_CUBLAS
static void bangGpu() {
  static bool doneBang = false;
  if (!doneBang) {
    void* ptr;
    vector<uint8_t> bytes(100000);
    cuCHECK(cudaMalloc(&ptr, bytes.size() * sizeof(uint8_t)));
    cuCHECK(cudaMemcpy(ptr, bytes.data(), bytes.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    cuCHECK(cudaMemcpy(bytes.data(), ptr, bytes.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    cuCHECK(cudaFree(ptr));
    cublasHandle_t cublasH = nullptr;
    cusolverDnHandle_t cusolverDnH = nullptr;
    cublasCHECK(cublasCreate(&cublasH));
    cusolverCHECK(cusolverDnCreate(&cusolverDnH));
    cublasCHECK(cublasDestroy(cublasH));
    cusolverCHECK(cusolverDnDestroy(cusolverDnH));
    doneBang = true;
  }
}

static vector<LUTimingResult> benchmarkLUCuda(
    const vector<pair<CsrMatrix, Eigen::VectorXd>>& matrices, bool verbose) {
  if (matrices.empty()) return {};

  bangGpu();

  const CsrMatrix& A0 = matrices[0].first;
  int64_t n = A0.nRows;

  auto preproc = computeMaxTransversal(n, A0.rowPtr.data(), A0.colInd.data());

  vector<int64_t> pRowPtr, pColInd;
  vector<double> pValues;
  applyRowPermToCsr<double>(n, A0.rowPtr.data(), A0.colInd.data(), A0.values.data(),
                            preproc.rowPerm.data(), pRowPtr, pColInd, pValues);
  SparseStructure ss = csrToSymmetricSparseStructure(n, pRowPtr.data(), pColInd.data());

  vector<int64_t> paramSizes(n, 1);
  vector<int64_t> blockSizes(n, 1);

  Settings cudaSettings;
  cudaSettings.backend = BackendCuda;
  cudaSettings.matrixType = MTYPE_GENERAL;
  cudaSettings.staticPivotThreshold = 0.0;
  cudaSettings.findSparseEliminationRanges = true;

  auto solver = createSolver(cudaSettings, paramSizes, ss);

  vector<double> data(solver->skel().totalDataSize());
  vector<int64_t> pivots(n);
  const auto& perm = solver->paramToSpan();

  vector<LUTimingResult> results;

  for (size_t mi = 0; mi < matrices.size(); mi++) {
    const CsrMatrix& A = matrices[mi].first;
    const Eigen::VectorXd& b = matrices[mi].second;
    LUTimingResult res;

    applyRowPermToCsr<double>(n, A.rowPtr.data(), A.colInd.data(), A.values.data(),
                              preproc.rowPerm.data(), pRowPtr, pColInd, pValues);
    vector<double> rowScale, colScale;
    computeEquilibration(n, pRowPtr.data(), pColInd.data(), pValues.data(), rowScale, colScale);

    applyRowPermAndScaleToCsr<double>(n, A.rowPtr.data(), A.colInd.data(), A.values.data(),
                                     preproc.rowPerm.data(), rowScale.data(), colScale.data(),
                                     pRowPtr, pColInd, pValues);

    fill(data.begin(), data.end(), 0.0);
    solver->loadFromCsr(pRowPtr.data(), pColInd.data(), blockSizes.data(), pValues.data(),
                        data.data());

    {
      DevMirror<double> dataGpu(data);
      auto tFactor = Clock::now();
      solver->factorLU(dataGpu.ptr, pivots.data());
      res.factorTime = tdelta(Clock::now() - tFactor).count();
      dataGpu.get(data);
    }
    res.perturbCount = solver->staticPivotPerturbCount();

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

    double residual = computeResidualDouble(A, x, b);
    res.refineSteps = 0;
    const int maxRefine = 30;
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
      residual = computeResidualDouble(A, x, b);
      res.refineSteps++;
    }
    res.solveTime = tdelta(Clock::now() - tSolve).count();
    res.residual = residual;

    if (verbose) {
      cout << "  [CUDA] Matrix #" << mi << ": factor=" << fixed << setprecision(4)
           << res.factorTime << "s, solve=" << res.solveTime << "s, residual=" << scientific
           << setprecision(2) << res.residual << ", refine=" << res.refineSteps
           << ", perturbed=" << res.perturbCount << endl;
    }

    results.push_back(res);
  }

  return results;
}
#endif  // BASPACHO_USE_CUBLAS

// ============================================================================
// cuDSS LU benchmark (NVIDIA's native sparse solver)
// ============================================================================

#ifdef BASPACHO_HAVE_CUDSS

#define cudssCHECK(call)                                                              \
  do {                                                                                \
    cudssStatus_t status_ = (call);                                                   \
    if (status_ != CUDSS_STATUS_SUCCESS) {                                            \
      fprintf(stderr, "[%s:%d] cuDSS Error: %d\n", __FILE__, __LINE__, (int)status_); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

static vector<LUTimingResult> benchmarkLUCudss(
    const vector<pair<CsrMatrix, Eigen::VectorXd>>& matrices, bool verbose) {
  if (matrices.empty()) return {};

  const CsrMatrix& A0 = matrices[0].first;
  int64_t n = A0.nRows;

  vector<LUTimingResult> results;

  for (size_t mi = 0; mi < matrices.size(); mi++) {
    const CsrMatrix& A = matrices[mi].first;
    const Eigen::VectorXd& b = matrices[mi].second;
    LUTimingResult res;

    int64_t nnz = A.nnz;

    // Convert int64_t indices to int32 for cuDSS
    vector<int> rowPtr32(A.rowPtr.begin(), A.rowPtr.end());
    vector<int> colInd32(A.colInd.begin(), A.colInd.end());

    // Upload to GPU
    int* d_rowPtr = nullptr;
    int* d_colInd = nullptr;
    double* d_values = nullptr;
    cuCHECK(cudaMalloc(&d_rowPtr, (n + 1) * sizeof(int)));
    cuCHECK(cudaMalloc(&d_colInd, nnz * sizeof(int)));
    cuCHECK(cudaMalloc(&d_values, nnz * sizeof(double)));
    cuCHECK(cudaMemcpy(d_rowPtr, rowPtr32.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cuCHECK(cudaMemcpy(d_colInd, colInd32.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    cuCHECK(cudaMemcpy(d_values, A.values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));

    // cuDSS setup
    cudssHandle_t handle = nullptr;
    cudssCHECK(cudssCreate(&handle));

    cudssConfig_t config = nullptr;
    cudssCHECK(cudssConfigCreate(&config));

    cudssData_t data = nullptr;
    cudssCHECK(cudssDataCreate(handle, &data));

    // General (non-symmetric) matrix, full CSR
    cudssMatrix_t cudssA = nullptr;
    cudssCHECK(cudssMatrixCreateCsr(&cudssA, n, n, nnz, d_rowPtr, nullptr, d_colInd, d_values,
                                     CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_GENERAL,
                                     CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

    // RHS and solution vectors on GPU
    double* d_b = nullptr;
    double* d_x = nullptr;
    cuCHECK(cudaMalloc(&d_b, n * sizeof(double)));
    cuCHECK(cudaMalloc(&d_x, n * sizeof(double)));
    cuCHECK(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    cuCHECK(cudaMemset(d_x, 0, n * sizeof(double)));

    cudssMatrix_t cudssB = nullptr;
    cudssMatrix_t cudssX = nullptr;
    cudssCHECK(
        cudssMatrixCreateDn(&cudssB, n, 1, n, d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    cudssCHECK(
        cudssMatrixCreateDn(&cudssX, n, 1, n, d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

    // Analysis
    cudssCHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, cudssA, cudssX, cudssB));
    cuCHECK(cudaDeviceSynchronize());

    // Factor (timed)
    auto tFactor = Clock::now();
    cudssCHECK(
        cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, cudssA, cudssX, cudssB));
    cuCHECK(cudaDeviceSynchronize());
    res.factorTime = tdelta(Clock::now() - tFactor).count();

    // Solve (timed)
    auto tSolve = Clock::now();
    cudssCHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, cudssA, cudssX, cudssB));
    cuCHECK(cudaDeviceSynchronize());
    res.solveTime = tdelta(Clock::now() - tSolve).count();

    // Copy solution back and compute residual
    Eigen::VectorXd x(n);
    cuCHECK(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
    res.residual = computeResidualDouble(A, x, b);
    res.refineSteps = 0;
    res.perturbCount = 0;

    if (verbose) {
      cout << "  [cuDSS] Matrix #" << mi << ": factor=" << fixed << setprecision(4)
           << res.factorTime << "s, solve=" << res.solveTime << "s, residual=" << scientific
           << setprecision(2) << res.residual << endl;
    }

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

    results.push_back(res);
  }

  return results;
}
#endif  // BASPACHO_HAVE_CUDSS

// ============================================================================
// Convert timing results to BenchRecords
// ============================================================================

static void resultToRecords(const string& problem, const string& solver,
                            const vector<LUTimingResult>& timings,
                            vector<BenchRecord>& records) {
  if (timings.empty()) return;

  // Factor times
  {
    BenchRecord rec;
    rec.problem = problem;
    rec.solver = solver;
    rec.operation = "factor";
    for (const auto& t : timings) rec.times_sec.push_back(t.factorTime);
    rec.median_sec = BaSpaCho::computeMedian(rec.times_sec);
    records.push_back(std::move(rec));
  }

  // Solve times (includes refinement)
  {
    BenchRecord rec;
    rec.problem = problem;
    rec.solver = solver;
    rec.operation = "solve";
    for (const auto& t : timings) rec.times_sec.push_back(t.solveTime);
    rec.median_sec = BaSpaCho::computeMedian(rec.times_sec);
    records.push_back(std::move(rec));
  }

  // Total (factor + solve)
  {
    BenchRecord rec;
    rec.problem = problem;
    rec.solver = solver;
    rec.operation = "total";
    for (const auto& t : timings) rec.times_sec.push_back(t.factorTime + t.solveTime);
    rec.median_sec = BaSpaCho::computeMedian(rec.times_sec);
    records.push_back(std::move(rec));
  }
}

// ============================================================================
// Print human-readable results
// ============================================================================

static void printResults(const string& solver, const vector<LUTimingResult>& timings) {
  if (timings.empty()) return;

  auto median = [](vector<double> v) -> double {
    if (v.empty()) return 0;
    sort(v.begin(), v.end());
    return v[v.size() / 2];
  };

  vector<double> factorTimes, solveTimes, totalTimes;
  for (const auto& t : timings) {
    factorTimes.push_back(t.factorTime);
    solveTimes.push_back(t.solveTime);
    totalTimes.push_back(t.factorTime + t.solveTime);
  }

  auto fmt = [](double t) -> string {
    ostringstream ss;
    if (t >= 1.0)
      ss << fixed << setprecision(3) << t << "s";
    else
      ss << fixed << setprecision(2) << t * 1000 << "ms";
    return ss.str();
  };

  cout << "  " << solver << ":" << endl;
  cout << "    factor:  median=" << fmt(median(factorTimes));
  if (timings.size() > 1) {
    cout << "  [";
    for (size_t i = 0; i < factorTimes.size(); i++) {
      if (i) cout << ", ";
      cout << fmt(factorTimes[i]);
    }
    cout << "]";
  }
  cout << endl;

  cout << "    solve:   median=" << fmt(median(solveTimes));
  if (timings.size() > 1) {
    cout << "  [";
    for (size_t i = 0; i < solveTimes.size(); i++) {
      if (i) cout << ", ";
      cout << fmt(solveTimes[i]);
    }
    cout << "]";
  }
  cout << endl;

  cout << "    total:   median=" << fmt(median(totalTimes)) << endl;

  // Average refinement stats
  double avgRefine = 0;
  for (const auto& t : timings) avgRefine += t.refineSteps;
  avgRefine /= timings.size();
  cout << "    avg refinement steps: " << fixed << setprecision(1) << avgRefine << endl;
}

// ============================================================================
// Main
// ============================================================================

void help() {
  cout << "lu_bench - LU factorization benchmark for non-symmetric matrices\n"
       << "\nUsage: lu_bench [options]\n"
       << "\nOptions:\n"
       << "  -i MTX_FILE    Single Matrix Market file (repeated factor+solve)\n"
       << "  -r RHS_FILE    RHS vector file (optional; generates b=A*ones if absent)\n"
       << "  -d SEQ_DIR     Sequence directory (jacobian_NNNN.mtx + rhs_NNNN.mtx)\n"
       << "  -m MAX         Max matrices from sequence (default: all)\n"
       << "  -n REPS        Repetitions for single-matrix mode (default: 5)\n"
       << "  -S REGEX       Select solvers (default: all available)\n"
       << "  -J             JSON output (same format as bench)\n"
       << "  -v             Verbose per-matrix output\n"
       << "  -h             Show this help\n"
       << "\nAvailable solvers:\n"
       << "  BaSpaCho_LU_CPU\n"
#ifdef BASPACHO_USE_METAL
       << "  BaSpaCho_LU_Metal\n"
#endif
#ifdef BASPACHO_USE_CUBLAS
       << "  BaSpaCho_LU_CUDA\n"
#endif
#ifdef BASPACHO_HAVE_CUDSS
       << "  cuDSS_LU\n"
#endif
       << endl;
}

int main(int argc, char* argv[]) {
  string mtxFile;
  string rhsFile;
  string seqDir;
  int maxMatrices = -1;
  int numReps = 5;
  regex selectSolvers(".");
  bool jsonOutput = false;
  bool verbose = false;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-h")) {
      help();
      return 0;
    } else if (!strcmp(argv[i], "-i") && i < argc - 1) {
      mtxFile = argv[++i];
    } else if (!strcmp(argv[i], "-r") && i < argc - 1) {
      rhsFile = argv[++i];
    } else if (!strcmp(argv[i], "-d") && i < argc - 1) {
      seqDir = argv[++i];
    } else if (!strcmp(argv[i], "-m") && i < argc - 1) {
      maxMatrices = stoi(argv[++i]);
    } else if (!strcmp(argv[i], "-n") && i < argc - 1) {
      numReps = stoi(argv[++i]);
    } else if (!strcmp(argv[i], "-S") && i < argc - 1) {
      selectSolvers = regex(argv[++i]);
    } else if (!strcmp(argv[i], "-J")) {
      jsonOutput = true;
    } else if (!strcmp(argv[i], "-v")) {
      verbose = true;
    } else {
      cerr << "Unknown option: " << argv[i] << " (use -h for help)" << endl;
      return 1;
    }
  }

  if (mtxFile.empty() && seqDir.empty()) {
    cerr << "Error: specify -i MTX_FILE or -d SEQ_DIR (use -h for help)" << endl;
    return 1;
  }

  // Load matrices
  vector<pair<CsrMatrix, Eigen::VectorXd>> matrices;
  string problemName;

  if (!seqDir.empty()) {
    // Sequence mode
    auto files = discoverSequenceFiles(seqDir);
    if (files.empty()) {
      cerr << "Error: no sequence files found in " << seqDir << endl;
      return 1;
    }

    int count = (maxMatrices > 0) ? min(maxMatrices, (int)files.size()) : (int)files.size();
    for (int i = 0; i < count; i++) {
      CsrMatrix A = readMatrixMarket(files[i].first);
      Eigen::VectorXd b = readRhsVector(files[i].second);

      // Skip near-zero RHS
      if (b.norm() < 1e-15) continue;

      matrices.push_back({std::move(A), std::move(b)});
    }

    // Extract directory name for problem label
    problemName = "LU_" + fs::path(seqDir).filename().string();

    if (!jsonOutput) {
      cout << "Sequence mode: " << matrices.size() << " matrices from " << seqDir << endl;
    }
  } else {
    // Single-matrix mode: repeat N times
    CsrMatrix A = readMatrixMarket(mtxFile);

    Eigen::VectorXd b;
    if (!rhsFile.empty()) {
      b = readRhsVector(rhsFile);
    } else {
      // Generate b = A * ones
      b = Eigen::VectorXd::Zero(A.nRows);
      for (int64_t i = 0; i < A.nRows; i++) {
        for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
          b(i) += A.values[k];
        }
      }
    }

    // Extract filename for problem label
    string basename = fs::path(mtxFile).stem().string();
    problemName = "LU_" + basename;

    // Repeat the same matrix numReps times (including 1 warmup)
    for (int i = 0; i < numReps + 1; i++) {
      matrices.push_back({A, b});
    }

    if (!jsonOutput) {
      cout << "Single-matrix mode: " << A.nRows << "x" << A.nCols << " nnz=" << A.nnz
           << ", " << numReps << " repetitions + 1 warmup" << endl;
    }
  }

  if (matrices.empty()) {
    cerr << "Error: no valid matrices to benchmark" << endl;
    return 1;
  }

  if (!jsonOutput) {
    cout << "\nProblem: " << problemName << ", matrices: " << matrices.size() << endl;
  }

  // Preprocessing timing (amortized, measured once)
  double preprocessTime = 0;
  double analysisTimeCpu = 0;
  {
    const CsrMatrix& A0 = matrices[0].first;
    int64_t n = A0.nRows;

    auto t0 = Clock::now();
    auto preproc = computeMaxTransversal(n, A0.rowPtr.data(), A0.colInd.data());
    preprocessTime = tdelta(Clock::now() - t0).count();

    vector<int64_t> pRowPtr, pColInd;
    vector<double> pValues;
    applyRowPermToCsr<double>(n, A0.rowPtr.data(), A0.colInd.data(), A0.values.data(),
                              preproc.rowPerm.data(), pRowPtr, pColInd, pValues);
    SparseStructure ss = csrToSymmetricSparseStructure(n, pRowPtr.data(), pColInd.data());
    vector<int64_t> paramSizes(n, 1);

    Settings settings;
    settings.backend = BackendFast;
    settings.matrixType = MTYPE_GENERAL;
    auto t1 = Clock::now();
    auto solver = createSolver(settings, paramSizes, ss);
    analysisTimeCpu = tdelta(Clock::now() - t1).count();

    if (!jsonOutput) {
      cout << "  Preprocessing (BTF max transversal): " << fixed << setprecision(4)
           << preprocessTime << "s" << endl;
      cout << "  Symbolic analysis: " << analysisTimeCpu << "s" << endl;
      cout << "  Matrix size: " << n << "x" << n << endl;
    }
  }

  vector<BenchRecord> allRecords;

  // Add preprocessing and analysis as single-value records
  {
    BenchRecord rec;
    rec.problem = problemName;
    rec.solver = "shared";
    rec.operation = "preprocess";
    rec.times_sec = {preprocessTime};
    rec.median_sec = preprocessTime;
    allRecords.push_back(std::move(rec));
  }
  {
    BenchRecord rec;
    rec.problem = problemName;
    rec.solver = "shared";
    rec.operation = "analysis";
    rec.times_sec = {analysisTimeCpu};
    rec.median_sec = analysisTimeCpu;
    allRecords.push_back(std::move(rec));
  }

  // Run selected solvers
  bool isWarmup = seqDir.empty();  // Single-matrix mode has warmup run

  // CPU solver
  if (regex_search(string("BaSpaCho_LU_CPU"), selectSolvers)) {
    if (!jsonOutput) cout << "\nRunning BaSpaCho_LU_CPU..." << endl;
    auto timings = benchmarkLUCpu(matrices, verbose);

    // Strip warmup run in single-matrix mode
    if (isWarmup && timings.size() > 1) {
      timings.erase(timings.begin());
    }

    resultToRecords(problemName, "BaSpaCho_LU_CPU", timings, allRecords);
    if (!jsonOutput) printResults("BaSpaCho_LU_CPU", timings);
  }

#ifdef BASPACHO_USE_METAL
  if (regex_search(string("BaSpaCho_LU_Metal"), selectSolvers)) {
    if (!jsonOutput) cout << "\nRunning BaSpaCho_LU_Metal..." << endl;
    auto timings = benchmarkLUMetal(matrices, verbose);

    if (isWarmup && timings.size() > 1) {
      timings.erase(timings.begin());
    }

    resultToRecords(problemName, "BaSpaCho_LU_Metal", timings, allRecords);
    if (!jsonOutput) printResults("BaSpaCho_LU_Metal", timings);
  }
#endif

#ifdef BASPACHO_USE_CUBLAS
  if (regex_search(string("BaSpaCho_LU_CUDA"), selectSolvers)) {
    if (!jsonOutput) cout << "\nRunning BaSpaCho_LU_CUDA..." << endl;
    auto timings = benchmarkLUCuda(matrices, verbose);

    if (isWarmup && timings.size() > 1) {
      timings.erase(timings.begin());
    }

    resultToRecords(problemName, "BaSpaCho_LU_CUDA", timings, allRecords);
    if (!jsonOutput) printResults("BaSpaCho_LU_CUDA", timings);
  }
#endif

#ifdef BASPACHO_HAVE_CUDSS
  if (regex_search(string("cuDSS_LU"), selectSolvers)) {
    if (!jsonOutput) cout << "\nRunning cuDSS_LU..." << endl;
    auto timings = benchmarkLUCudss(matrices, verbose);

    if (isWarmup && timings.size() > 1) {
      timings.erase(timings.begin());
    }

    resultToRecords(problemName, "cuDSS_LU", timings, allRecords);
    if (!jsonOutput) printResults("cuDSS_LU", timings);
  }
#endif

  // JSON output
  if (jsonOutput) {
    BaSpaCho::writeJson(cout, allRecords);
  }

  return 0;
}
