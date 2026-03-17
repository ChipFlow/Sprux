/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sprux/benchmarking/BenchCudss.h"
#include <cuda_runtime.h>
#include <cudss.h>
#include <chrono>
#include <iostream>
#include <random>
#include "sprux/sprux/CudaDefs.h"
#include "sprux/sprux/DebugMacros.h"
#include "sprux/sprux/Utils.h"

using namespace Sprux;
using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

#define cudssCHECK(call)                                                                 \
  do {                                                                                   \
    cudssStatus_t status_ = (call);                                                      \
    if (status_ != CUDSS_STATUS_SUCCESS) {                                               \
      fprintf(stderr, "[%s:%d] cuDSS Error: %d\n", __FILE__, __LINE__, (int)status_);    \
      exit(1);                                                                           \
    }                                                                                    \
  } while (0)

constexpr bool kPreApplyReordering = true;

CudssBenchResults benchmarkCudssSolve(const vector<int64_t>& paramSize,
                                       const SparseStructure& ssOrig,
                                       const vector<int64_t>& nRHSs, int verbose) {
  SparseStructure ss;
  if (kPreApplyReordering) {
    vector<int64_t> permutation = ssOrig.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    ss = ssOrig.symmetricPermutation(invPerm, false);
  } else {
    ss = ssOrig;
  }

  SPRUX_CHECK_EQ(paramSize.size(), ss.ptrs.size() - 1);
  vector<int64_t> rowPtr, colInd;
  vector<double> val;
  vector<int64_t> spanStart = paramSize;
  spanStart.push_back(0);
  int64_t totSize = cumSumVec(spanStart);
  rowPtr.push_back(0);

  mt19937 gen(37);
  uniform_real_distribution<double> unif(-1.0, 1.0);
  double diagBoost = totSize * 2;  // make positive definite

  if (verbose >= 2) {
    cout << "to csr... (order=" << totSize << ")" << endl;
  }
  for (int64_t rb = 0; rb < (int64_t)paramSize.size(); rb++) {
    for (int64_t ri = spanStart[rb]; ri < spanStart[rb + 1]; ri++) {
      for (int64_t q = ss.ptrs[rb]; q < ss.ptrs[rb + 1]; q++) {
        int64_t cb = ss.inds[q];
        for (int64_t ci = spanStart[cb]; ci < spanStart[cb + 1]; ci++) {
          if (ci > ri) {
            continue;
          }
          colInd.push_back(ci);
          val.push_back(unif(gen) + (ci == ri ? diagBoost : 0.0));
        }
      }
      rowPtr.push_back(colInd.size());
    }
  }
  if (verbose >= 2) {
    cout << "to csr done." << endl;
  }

  int64_t n = totSize;
  int64_t nnz = (int64_t)val.size();

  // Convert int64_t indices to int32 for cuDSS
  vector<int> rowPtr32(rowPtr.begin(), rowPtr.end());
  vector<int> colInd32(colInd.begin(), colInd.end());

  // Allocate GPU memory
  int* d_rowPtr = nullptr;
  int* d_colInd = nullptr;
  double* d_values = nullptr;

  cuCHECK(cudaMalloc(&d_rowPtr, (n + 1) * sizeof(int)));
  cuCHECK(cudaMalloc(&d_colInd, nnz * sizeof(int)));
  cuCHECK(cudaMalloc(&d_values, nnz * sizeof(double)));

  cuCHECK(cudaMemcpy(d_rowPtr, rowPtr32.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  cuCHECK(cudaMemcpy(d_colInd, colInd32.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
  cuCHECK(cudaMemcpy(d_values, val.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));

  // Create cuDSS handle and objects
  cudssHandle_t handle = nullptr;
  cudssCHECK(cudssCreate(&handle));

  cudssConfig_t config = nullptr;
  cudssCHECK(cudssConfigCreate(&config));

  cudssData_t data = nullptr;
  cudssCHECK(cudssDataCreate(handle, &data));

  // Create matrix descriptor (CSR, SPD, lower triangle)
  cudssMatrix_t cudssA = nullptr;
  cudssCHECK(cudssMatrixCreateCsr(
      &cudssA, n, n, nnz,
      d_rowPtr, nullptr,  // row end offsets (null = derive from rowPtr)
      d_colInd, d_values,
      CUDA_R_32I, CUDA_R_64F,
      CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER,
      CUDSS_BASE_ZERO));

  // Create dummy dense vectors for analysis/factorization (cuDSS requires them)
  double* d_dummy = nullptr;
  cuCHECK(cudaMalloc(&d_dummy, n * sizeof(double)));
  cuCHECK(cudaMemset(d_dummy, 0, n * sizeof(double)));

  cudssMatrix_t cudssB = nullptr;
  cudssMatrix_t cudssX = nullptr;
  int64_t nrhs = 1;
  cudssCHECK(cudssMatrixCreateDn(&cudssB, n, nrhs, n, d_dummy, CUDA_R_64F,
                                  CUDSS_LAYOUT_COL_MAJOR));
  cudssCHECK(cudssMatrixCreateDn(&cudssX, n, nrhs, n, d_dummy, CUDA_R_64F,
                                  CUDSS_LAYOUT_COL_MAJOR));

  // Analysis phase
  if (verbose >= 2) {
    cout << "cuDSS analyzing..." << endl;
  }
  auto startAnalysis = hrc::now();
  cudssCHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, cudssA, cudssX, cudssB));
  cuCHECK(cudaDeviceSynchronize());
  double analysisTime = tdelta(hrc::now() - startAnalysis).count();
  if (verbose >= 2) {
    cout << "Analysis time: " << analysisTime << "s" << endl;
  }

  // Factorization phase
  if (verbose >= 2) {
    cout << "cuDSS factoring..." << endl;
  }
  auto startFactor = hrc::now();
  cudssCHECK(
      cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, cudssA, cudssX, cudssB));
  cuCHECK(cudaDeviceSynchronize());
  double factorTime = tdelta(hrc::now() - startFactor).count();
  if (verbose >= 2) {
    cout << "Factor time: " << factorTime << "s" << endl;
  }

  if (verbose >= 1) {
    cout << "cuDSS stats:"
         << "\n  Matrix size: " << n << "\n  nnz (lower): " << nnz << endl;
  }

  // Destroy dummy dense objects (will re-create per solve)
  cudssMatrixDestroy(cudssB);
  cudssMatrixDestroy(cudssX);
  cudaFree(d_dummy);

  // Solve with different numbers of RHS
  map<int64_t, double> solveTimes;
  for (int64_t nRHS : nRHSs) {
    vector<double> bData(nRHS * n);
    for (auto& v : bData) {
      v = unif(gen);
    }
    vector<double> xData(nRHS * n, 0.0);

    double* d_b = nullptr;
    double* d_x = nullptr;
    cuCHECK(cudaMalloc(&d_b, nRHS * n * sizeof(double)));
    cuCHECK(cudaMalloc(&d_x, nRHS * n * sizeof(double)));
    cuCHECK(cudaMemcpy(d_b, bData.data(), nRHS * n * sizeof(double), cudaMemcpyHostToDevice));

    cudssMatrix_t solvB = nullptr;
    cudssMatrix_t solvX = nullptr;
    cudssCHECK(cudssMatrixCreateDn(&solvB, n, nRHS, n, d_b, CUDA_R_64F,
                                    CUDSS_LAYOUT_COL_MAJOR));
    cudssCHECK(cudssMatrixCreateDn(&solvX, n, nRHS, n, d_x, CUDA_R_64F,
                                    CUDSS_LAYOUT_COL_MAJOR));

    // Warm up
    cudssCHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, cudssA, solvX, solvB));
    cuCHECK(cudaDeviceSynchronize());

    // Reset x for timed solve
    cuCHECK(cudaMemset(d_x, 0, nRHS * n * sizeof(double)));

    auto startSolve = hrc::now();
    cudssCHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, cudssA, solvX, solvB));
    cuCHECK(cudaDeviceSynchronize());
    solveTimes[nRHS] = tdelta(hrc::now() - startSolve).count();

    if (verbose >= 2) {
      cout << "Solve time (nRHS=" << nRHS << "): " << solveTimes[nRHS] << "s" << endl;
    }

    cudssMatrixDestroy(solvB);
    cudssMatrixDestroy(solvX);
    cudaFree(d_b);
    cudaFree(d_x);
  }

  // Cleanup
  cudssMatrixDestroy(cudssA);
  cudssDataDestroy(handle, data);
  cudssConfigDestroy(config);
  cudssDestroy(handle);

  cudaFree(d_rowPtr);
  cudaFree(d_colInd);
  cudaFree(d_values);

  CudssBenchResults retv;
  retv.analysisTime = analysisTime;
  retv.factorTime = factorTime;
  retv.solveTimes = solveTimes;

  return retv;
}
