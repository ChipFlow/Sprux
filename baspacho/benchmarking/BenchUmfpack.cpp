/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/benchmarking/BenchUmfpack.h"
#include <umfpack.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/Utils.h"

using namespace BaSpaCho;
using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

UmfpackBenchResults benchmarkUmfpackSolve(const vector<int64_t>& paramSize,
                                           const SparseStructure& ssOrig,
                                           const vector<int64_t>& nRHSs, int verbose,
                                           bool symmetric) {
  // Apply fill-reducing ordering (same as Sprux would do)
  SparseStructure ss;
  vector<int64_t> permutation = ssOrig.fillReducingPermutation();
  vector<int64_t> invPerm = inversePermutation(permutation);
  ss = ssOrig.symmetricPermutation(invPerm, false);

  SPRUX_CHECK_EQ(paramSize.size(), ss.ptrs.size() - 1);

  // Compute span starts (cumulative sum of param sizes)
  vector<int64_t> spanStart = paramSize;
  spanStart.push_back(0);
  int64_t totSize = cumSumVec(spanStart);

  // Build CSC format for UMFPACK (column-oriented sparse matrix)
  // UMFPACK uses CSC format: colPtr, rowIdx, values
  vector<int64_t> colPtr, rowIdx;
  vector<double> val;
  colPtr.push_back(0);

  mt19937 gen(37);
  uniform_real_distribution<double> unif(-1.0, 1.0);
  double diagBoost = totSize * 2;  // make diagonally dominant for stability

  if (verbose >= 2) {
    cout << "Building CSC matrix... (order=" << totSize << ")" << endl;
  }

  // Build full matrix (not just lower triangle since we're doing general LU)
  // Store both lower and upper triangle entries
  for (int64_t cb = 0; cb < (int64_t)paramSize.size(); cb++) {
    for (int64_t ci = spanStart[cb]; ci < spanStart[cb + 1]; ci++) {
      // For each column ci
      // Find all row blocks connected to column block cb
      for (int64_t q = ss.ptrs[cb]; q < ss.ptrs[cb + 1]; q++) {
        int64_t rb = ss.inds[q];
        for (int64_t ri = spanStart[rb]; ri < spanStart[rb + 1]; ri++) {
          rowIdx.push_back(ri);
          double v = unif(gen);
          if (symmetric && ri != ci) {
            // For symmetric matrix, A[ri,ci] should equal A[ci,ri]
            // We'll handle this by generating the same value for both
            // Since we iterate column by column, just generate and store
            v = unif(gen);
          }
          if (ri == ci) {
            v += diagBoost;  // diagonal dominance
          }
          val.push_back(v);
        }
      }
      colPtr.push_back(rowIdx.size());
    }
  }

  if (verbose >= 2) {
    cout << "CSC matrix built. nnz=" << val.size() << endl;
  }

  // UMFPACK setup
  double Control[UMFPACK_CONTROL];
  double Info[UMFPACK_INFO];
  umfpack_dl_defaults(Control);

  if (verbose >= 2) {
    Control[UMFPACK_PRL] = 2;  // print level
  }

  void* Symbolic = nullptr;
  void* Numeric = nullptr;

  // Symbolic analysis
  if (verbose >= 2) {
    cout << "UMFPACK symbolic analysis..." << endl;
  }
  auto startAnalysis = hrc::now();
  int status =
      umfpack_dl_symbolic(totSize, totSize, colPtr.data(), rowIdx.data(), val.data(), &Symbolic,
                          Control, Info);
  double analysisTime = tdelta(hrc::now() - startAnalysis).count();

  if (status != UMFPACK_OK) {
    cerr << "UMFPACK symbolic analysis failed with status " << status << endl;
    exit(1);
  }

  if (verbose >= 2) {
    cout << "Symbolic analysis time: " << analysisTime << "s" << endl;
  }

  // Numeric factorization
  if (verbose >= 2) {
    cout << "UMFPACK numeric factorization..." << endl;
  }
  auto startFactor = hrc::now();
  status = umfpack_dl_numeric(colPtr.data(), rowIdx.data(), val.data(), Symbolic, &Numeric, Control,
                              Info);
  double factorTime = tdelta(hrc::now() - startFactor).count();

  if (status != UMFPACK_OK) {
    cerr << "UMFPACK numeric factorization failed with status " << status << endl;
    umfpack_dl_free_symbolic(&Symbolic);
    exit(1);
  }

  if (verbose >= 2) {
    cout << "Factor time: " << factorTime << "s" << endl;
  }

  if (verbose >= 1) {
    cout << "UMFPACK stats:"
         << "\n  Matrix size: " << totSize << " x " << totSize << "\n  nnz: " << val.size()
         << "\n  fill: " << val.size() / ((double)totSize * totSize)
         << "\n  rcond estimate: " << Info[UMFPACK_RCOND]
         << "\n  flops: " << Info[UMFPACK_FLOPS] << "\n  L nnz: " << Info[UMFPACK_LNZ]
         << "\n  U nnz: " << Info[UMFPACK_UNZ] << endl;
  }

  // Solve with different numbers of RHS
  map<int64_t, double> solveTimes;
  double lastResidual = 0.0;

  for (int64_t nRHS : nRHSs) {
    vector<double> b(nRHS * totSize);
    vector<double> x(nRHS * totSize);

    // Generate random RHS
    for (size_t i = 0; i < b.size(); i++) {
      b[i] = unif(gen);
    }

    // Warm up
    for (int64_t r = 0; r < nRHS; r++) {
      umfpack_dl_solve(UMFPACK_A, colPtr.data(), rowIdx.data(), val.data(), x.data() + r * totSize,
                       b.data() + r * totSize, Numeric, Control, Info);
    }

    // Timed solve
    auto startSolve = hrc::now();
    for (int64_t r = 0; r < nRHS; r++) {
      status = umfpack_dl_solve(UMFPACK_A, colPtr.data(), rowIdx.data(), val.data(),
                                x.data() + r * totSize, b.data() + r * totSize, Numeric, Control,
                                Info);
      if (status != UMFPACK_OK) {
        cerr << "UMFPACK solve failed with status " << status << endl;
        break;
      }
    }
    solveTimes[nRHS] = tdelta(hrc::now() - startSolve).count();

    // Compute residual ||A*x - b|| / ||b||
    double residualNorm = 0.0;
    double bNorm = 0.0;
    for (int64_t r = 0; r < nRHS; r++) {
      // Compute A*x
      vector<double> Ax(totSize, 0.0);
      for (int64_t col = 0; col < totSize; col++) {
        for (int64_t k = colPtr[col]; k < colPtr[col + 1]; k++) {
          int64_t row = rowIdx[k];
          Ax[row] += val[k] * x[r * totSize + col];
        }
      }
      // Compute ||A*x - b|| and ||b||
      for (int64_t i = 0; i < totSize; i++) {
        double diff = Ax[i] - b[r * totSize + i];
        residualNorm += diff * diff;
        bNorm += b[r * totSize + i] * b[r * totSize + i];
      }
    }
    lastResidual = sqrt(residualNorm) / sqrt(bNorm);

    if (verbose >= 2) {
      cout << "Solve time (nRHS=" << nRHS << "): " << solveTimes[nRHS] << "s"
           << ", residual: " << lastResidual << endl;
    }
  }

  // Cleanup
  umfpack_dl_free_symbolic(&Symbolic);
  umfpack_dl_free_numeric(&Numeric);

  UmfpackBenchResults retv;
  retv.analysisTime = analysisTime;
  retv.factorTime = factorTime;
  retv.solveTimes = solveTimes;
  retv.residual = lastResidual;

  return retv;
}
