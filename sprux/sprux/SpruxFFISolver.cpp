/*
 * Copyright (c) 2024-2026 ChipFlow
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SpruxFFISolver: encapsulated Metal LU solver for circuit simulation FFI.
// Reference implementation: benchmarking/LUBench.cpp:benchmarkLUMetalFFI

#include "sprux/sprux/SpruxFFISolver.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "sprux/sprux/Preprocessing.h"
#include "sprux/sprux/Solver.h"
#include "sprux/sprux/SparseStructure.h"

#ifdef SPRUX_USE_METAL
#include "sprux/sprux/MetalDefs.h"
#endif

namespace Sprux {

struct SpruxFFISolver::Impl {
  int64_t n;
  int64_t nnz;
  int maxRefine;

  // Original CSR structure (int64, converted from caller's int32)
  std::vector<int64_t> csrIndptr;
  std::vector<int64_t> csrIndices;

  // Preprocessing results (fixed for lifetime)
  LUPreprocessing preproc;

  // Permuted CSR structure (after BTF row perm — sparsity is fixed)
  std::vector<int64_t> permRowPtr;
  std::vector<int64_t> permColInd;

  // Solver
  std::unique_ptr<Solver> solver;
  std::vector<int64_t> blockSizes;  // all 1s for scalar MNA matrices

  // AMD permutation from solver (reference, not owned)
  const std::vector<int64_t>* perm = nullptr;

  // Factor data and pivots (CPU-side, loaded to MetalMirror per solve)
  std::vector<float> data;
  std::vector<int64_t> pivots;

  // Temporaries reused across solves to avoid allocation
  std::vector<double> permValues;    // permuted + equilibrated CSR values
  std::vector<double> rowScale;
  std::vector<double> colScale;
  std::vector<double> xAccum;  // f64 accumulator for refinement
};

// ---------------------------------------------------------------------------
// Constructor: one-time setup
// ---------------------------------------------------------------------------

SpruxFFISolver::SpruxFFISolver(int32_t n, int32_t nnz, const int32_t* csr_indptr,
                               const int32_t* csr_indices, const double* csr_data_init,
                               int max_refine_steps)
    : impl_(std::make_unique<Impl>()) {
  auto& d = *impl_;
  d.n = n;
  d.nnz = nnz;
  d.maxRefine = max_refine_steps;

  // Convert int32 CSR to int64 (Sprux uses int64 internally)
  d.csrIndptr.assign(csr_indptr, csr_indptr + n + 1);
  d.csrIndices.assign(csr_indices, csr_indices + nnz);

  // Block sizes: all 1x1 for scalar MNA matrices
  d.blockSizes.assign(n, 1);

  // Step 1: BTF max transversal — row permutation for diagonal quality
  d.preproc = computeMaxTransversal(n, d.csrIndptr.data(), d.csrIndices.data());

  // Apply row permutation to get permuted CSR structure
  applyRowPermToCsr<double>(n, d.csrIndptr.data(), d.csrIndices.data(), csr_data_init,
                            d.preproc.rowPerm.data(), d.permRowPtr, d.permColInd, d.permValues);

  // Step 2: Build symmetric SparseStructure for AMD ordering
  SparseStructure ss =
      csrToSymmetricSparseStructure(n, d.permRowPtr.data(), d.permColInd.data());

  // Step 3: Compute static pivot threshold from equilibrated first matrix diagonal.
  // This avoids GPU→CPU sync during factorization (sprux auto threshold=0 does
  // maxAbsDiag which requires a readback).
  double pivotThreshold;
  {
    std::vector<double> rowScale0, colScale0;
    computeEquilibration(n, d.permRowPtr.data(), d.permColInd.data(), d.permValues.data(),
                         rowScale0, colScale0);
    std::vector<int64_t> eqRowPtr, eqColInd;
    std::vector<double> eqValues;
    applyRowPermAndScaleToCsr<double>(n, d.csrIndptr.data(), d.csrIndices.data(), csr_data_init,
                                     d.preproc.rowPerm.data(), rowScale0.data(), colScale0.data(),
                                     eqRowPtr, eqColInd, eqValues);
    double maxDiag = 0;
    for (int64_t i = 0; i < n; i++) {
      for (int64_t k = eqRowPtr[i]; k < eqRowPtr[i + 1]; k++) {
        if (eqColInd[k] == i) {
          maxDiag = std::max(maxDiag, std::abs(eqValues[k]));
          break;
        }
      }
    }
    float epsScale = std::cbrt(std::numeric_limits<float>::epsilon());
    pivotThreshold = double(epsScale) * std::max(maxDiag, double(epsScale));
  }

  // Step 4: Create solver
  Settings settings;
#ifdef SPRUX_USE_METAL
  settings.backend = BackendMetal;
#else
  settings.backend = BackendFast;
#endif
  settings.matrixType = MTYPE_GENERAL;
  settings.numThreads = 1;
  settings.staticPivotThreshold = pivotThreshold;
  settings.findSparseEliminationRanges = true;

  std::vector<int64_t> paramSizes(n, 1);
  d.solver = createSolver(settings, paramSizes, ss);
  d.perm = &d.solver->paramToSpan();
  int64_t totalDataSz = d.solver->totalDataSize();

  // Allocate factor data and pivot buffers
  d.data.resize(totalDataSz, 0.0f);
  d.pivots.resize(n, 0);

  // Pre-allocate reusable temporaries
  d.permValues.resize(nnz);
  d.rowScale.resize(n);
  d.colScale.resize(n);
  d.xAccum.resize(n, 0.0);
}

SpruxFFISolver::~SpruxFFISolver() = default;

// ---------------------------------------------------------------------------
// solve(): per-NR-iteration Metal LU with iterative refinement
// ---------------------------------------------------------------------------

void SpruxFFISolver::solve(const double* csr_data, const double* rhs, double* x_out) {
  auto& d = *impl_;
  const int64_t n = d.n;
  const auto& perm = *d.perm;

  // Step 1: Apply row permutation and equilibrate
  applyRowPermToCsr<double>(n, d.csrIndptr.data(), d.csrIndices.data(), csr_data,
                            d.preproc.rowPerm.data(), d.permRowPtr, d.permColInd, d.permValues);
  computeEquilibration(n, d.permRowPtr.data(), d.permColInd.data(), d.permValues.data(),
                       d.rowScale, d.colScale);

  std::vector<int64_t> eqRowPtr, eqColInd;
  std::vector<double> eqValues;
  applyRowPermAndScaleToCsr<double>(n, d.csrIndptr.data(), d.csrIndices.data(), csr_data,
                                   d.preproc.rowPerm.data(), d.rowScale.data(), d.colScale.data(),
                                   eqRowPtr, eqColInd, eqValues);

  // Convert equilibrated values to f32
  std::vector<float> sValues(eqValues.begin(), eqValues.end());

  // Step 2: Load f32 values into solver's coalesced format
  std::fill(d.data.begin(), d.data.end(), 0.0f);
  d.solver->loadFromCsr(eqRowPtr.data(), eqColInd.data(), d.blockSizes.data(), sValues.data(),
                        d.data.data());

  // Step 3: Factor on GPU (or CPU), copy data back
#ifdef SPRUX_USE_METAL
  {
    MetalMirror<float> dataGpu(d.data);
    d.solver->factorLU(dataGpu.ptr(), d.pivots.data());
    dataGpu.get(d.data);
  }
#else
  d.solver->factorLU(d.data.data(), d.pivots.data());
#endif

  // Step 4: Initial solve — permute RHS, solve, unpermute
  std::vector<float> bp(n);
  for (int64_t j = 0; j < n; j++) {
    bp[perm[j]] = float(d.rowScale[j] * rhs[d.preproc.rowPerm[j]]);
  }

#ifdef SPRUX_USE_METAL
  {
    MetalMirror<float> dataGpu(d.data);
    MetalMirror<float> xGpu(bp);
    d.solver->solveLU(dataGpu.ptr(), d.pivots.data(), xGpu.ptr(), n, 1);
    xGpu.get(bp);
  }
#else
  d.solver->solveLU(d.data.data(), d.pivots.data(), bp.data(), n, 1);
#endif

  // Step 5: Unscale initial solution to double
  for (int64_t j = 0; j < n; j++) {
    x_out[j] = d.colScale[j] * double(bp[perm[j]]);
  }

  // Step 6: Iterative refinement — CPU f64 SpMV + GPU f32 correction solve
  for (int iter = 0; iter < d.maxRefine; iter++) {
    // Compute residual r = b - A*x in f64 with ORIGINAL matrix values
    std::vector<float> rp(n);
    for (int64_t j = 0; j < n; j++) {
      int64_t srcRow = d.preproc.rowPerm[j];
      double sum = 0.0;
      for (int64_t k = d.csrIndptr[srcRow]; k < d.csrIndptr[srcRow + 1]; k++) {
        sum += csr_data[k] * x_out[d.csrIndices[k]];
      }
      double residual = rhs[srcRow] - sum;
      rp[perm[j]] = float(d.rowScale[j] * residual);
    }

    // Solve for correction in f32
#ifdef SPRUX_USE_METAL
    {
      MetalMirror<float> dataGpu(d.data);
      MetalMirror<float> xGpu(rp);
      d.solver->solveLU(dataGpu.ptr(), d.pivots.data(), xGpu.ptr(), n, 1);
      xGpu.get(rp);
    }
#else
    d.solver->solveLU(d.data.data(), d.pivots.data(), rp.data(), n, 1);
#endif

    // Apply correction in f64
    for (int64_t j = 0; j < n; j++) {
      x_out[j] += d.colScale[j] * double(rp[perm[j]]);
    }
  }
}

// ---------------------------------------------------------------------------
// dot(): sparse matrix-vector multiply (CPU, f64, no permutation)
// ---------------------------------------------------------------------------

void SpruxFFISolver::dot(const double* csr_data, const double* x, double* b_out) {
  auto& d = *impl_;
  for (int64_t i = 0; i < d.n; i++) {
    double sum = 0.0;
    for (int64_t k = d.csrIndptr[i]; k < d.csrIndptr[i + 1]; k++) {
      sum += csr_data[k] * x[d.csrIndices[k]];
    }
    b_out[i] = sum;
  }
}

int64_t SpruxFFISolver::n() const {
  return impl_->n;
}

}  // namespace Sprux
