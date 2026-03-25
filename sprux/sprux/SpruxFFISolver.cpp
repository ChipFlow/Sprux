/*
 * Copyright (c) 2024-2026 ChipFlow
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SpruxFFISolver: encapsulated Metal LU solver for circuit simulation FFI.
// Reference implementation: benchmarking/LUBench.cpp:benchmarkLUMetalFFI
//
// Optimization for NR loops: equilibration scales are computed once from the
// initial matrix and reused. A pre-computed scatter map eliminates per-solve
// CSR reallocation — only values are scattered through the fixed map.

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

  // Pre-computed scatter map: for each NNZ position k in the ORIGINAL CSR,
  // scatterMap[k] = position in the permuted CSR where value[k] goes.
  // This eliminates per-solve applyRowPermToCsr + applyRowPermAndScaleToCsr allocations.
  std::vector<int64_t> scatterMap;

  // Per-row scale factors applied during scatter: combinedRowScale[permuted_row]
  // = rowScale[permuted_row] for the row that original_row maps to.
  // colScaleByCol[col] = colScale[col].
  // These are computed once from the init matrix.
  std::vector<double> rowScale;  // indexed by permuted row
  std::vector<double> colScale;  // indexed by column

  // Solver
  std::unique_ptr<Solver> solver;
  std::vector<int64_t> blockSizes;

  // AMD permutation from solver (reference, not owned)
  const std::vector<int64_t>* perm = nullptr;

  bool useMetal = false;

#ifdef SPRUX_USE_METAL
  MetalMirror<float> dataGpu;
  MetalMirror<float> xGpu;
  MetalMirror<int64_t> devPivots;
  NumericCtxPtr<float> numCtx;
  SolveCtxPtr<float> solveCtx;
#endif

  // CPU fallback buffers
  std::vector<float> dataCpu;
  std::vector<int64_t> pivotsCpu;

  // Reusable per-solve buffers (pre-allocated, no alloc in hot path)
  std::vector<float> permValuesF32;  // scaled permuted values for loadFromCsr
  std::vector<double> xAccum;
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

  d.csrIndptr.assign(csr_indptr, csr_indptr + n + 1);
  d.csrIndices.assign(csr_indices, csr_indices + nnz);
  d.blockSizes.assign(n, 1);

  // Step 1: BTF max transversal
  d.preproc = computeMaxTransversal(n, d.csrIndptr.data(), d.csrIndices.data());

  // Step 2: Apply row permutation to get permuted CSR structure (once)
  std::vector<double> permValues;
  applyRowPermToCsr<double>(n, d.csrIndptr.data(), d.csrIndices.data(), csr_data_init,
                            d.preproc.rowPerm.data(), d.permRowPtr, d.permColInd, permValues);

  // Step 3: Compute equilibration from initial matrix (reused for all solves)
  computeEquilibration(n, d.permRowPtr.data(), d.permColInd.data(), permValues.data(),
                       d.rowScale, d.colScale);

  // Step 4: Build scatter map — maps original CSR position k to permuted position
  // For each permuted row i (= position in output), the original row is rowPerm[i].
  // We iterate the permuted structure to build a map from (origRow, origK) -> permK.
  d.scatterMap.resize(nnz);
  {
    int64_t permPos = 0;
    for (int64_t i = 0; i < n; i++) {
      int64_t origRow = d.preproc.rowPerm[i];
      int64_t origStart = d.csrIndptr[origRow];
      int64_t origEnd = d.csrIndptr[origRow + 1];
      for (int64_t k = origStart; k < origEnd; k++) {
        d.scatterMap[k] = permPos++;
      }
    }
  }

  // Step 5: Build symmetric SparseStructure for AMD ordering
  SparseStructure ss =
      csrToSymmetricSparseStructure(n, d.permRowPtr.data(), d.permColInd.data());

  // Step 6: Compute static pivot threshold
  double pivotThreshold;
  {
    // Apply scaling to the init matrix to get equilibrated values
    std::vector<float> eqValues(nnz);
    for (int64_t i = 0; i < n; i++) {
      for (int64_t pk = d.permRowPtr[i]; pk < d.permRowPtr[i + 1]; pk++) {
        int64_t j = d.permColInd[pk];
        eqValues[pk] = float(d.rowScale[i] * permValues[pk] * d.colScale[j]);
      }
    }

    double maxDiag = 0;
    for (int64_t i = 0; i < n; i++) {
      for (int64_t pk = d.permRowPtr[i]; pk < d.permRowPtr[i + 1]; pk++) {
        if (d.permColInd[pk] == i) {
          maxDiag = std::max(maxDiag, double(std::abs(eqValues[pk])));
          break;
        }
      }
    }
    float epsScale = std::cbrt(std::numeric_limits<float>::epsilon());
    pivotThreshold = double(epsScale) * std::max(maxDiag, double(epsScale));
  }

  // Step 7: Create solver
  Settings settings;
#ifdef SPRUX_USE_METAL
  settings.backend = BackendMetal;
  d.useMetal = true;
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

#ifdef SPRUX_USE_METAL
  {
    auto& symCtx = d.solver->internalSymbolicContext();
    symCtx.disableAllStats();

    d.dataGpu.resizeToAtLeast(totalDataSz);
    d.xGpu.resizeToAtLeast(n);
    d.devPivots.resizeToAtLeast(n);

    d.numCtx = symCtx.createNumericCtx<float>(0, static_cast<float*>(nullptr));
    d.numCtx->beginRecording();
    d.numCtx->preAllocateForLU(1, n);
    d.solveCtx = symCtx.createSolveCtx<float>(1, static_cast<float*>(nullptr));

    // Recording pass
    d.solver->factorLU(d.dataGpu.ptr(), d.devPivots.ptr(), *d.numCtx, PivotLocation::Device);
    d.numCtx->endRecording();

    // MPS warmup with valid data
    {
      std::vector<float> initF32(nnz);
      for (int64_t i = 0; i < n; i++) {
        for (int64_t pk = d.permRowPtr[i]; pk < d.permRowPtr[i + 1]; pk++) {
          int64_t j = d.permColInd[pk];
          initF32[pk] = float(d.rowScale[i] * permValues[pk] * d.colScale[j]);
        }
      }
      std::memset(d.dataGpu.ptr(), 0, totalDataSz * sizeof(float));
      d.solver->loadFromCsr(d.permRowPtr.data(), d.permColInd.data(), d.blockSizes.data(),
                            initF32.data(), d.dataGpu.ptr());

      d.numCtx->reset();
      d.solver->factorLU(d.dataGpu.ptr(), d.devPivots.ptr(), *d.numCtx, PivotLocation::Device);
      d.numCtx->flush();
    }
  }
#endif

  d.dataCpu.resize(totalDataSz, 0.0f);
  d.pivotsCpu.resize(n, 0);
  d.permValuesF32.resize(nnz);
  d.xAccum.resize(n, 0.0);
}

SpruxFFISolver::~SpruxFFISolver() = default;

// ---------------------------------------------------------------------------
// solve(): per-NR-iteration — scatter values, factor, solve, refine
// ---------------------------------------------------------------------------

void SpruxFFISolver::solve(const double* csr_data, const double* rhs, double* x_out) {
  auto& d = *impl_;
  const int64_t n = d.n;
  const auto& perm = *d.perm;

  // Step 1: Scatter original CSR values through pre-computed map,
  // applying cached row/column equilibration scales.
  // No allocation — writes directly into pre-allocated permValuesF32.
  for (int64_t i = 0; i < n; i++) {
    int64_t origRow = d.preproc.rowPerm[i];
    double rs = d.rowScale[i];
    for (int64_t k = d.csrIndptr[origRow]; k < d.csrIndptr[origRow + 1]; k++) {
      int64_t permK = d.scatterMap[k];
      int64_t col = d.csrIndices[k];  // = permColInd[permK]
      d.permValuesF32[permK] = float(rs * csr_data[k] * d.colScale[col]);
    }
  }

#ifdef SPRUX_USE_METAL
  if (d.useMetal) {
    auto& symCtx = d.solver->internalSymbolicContext();
    auto& metalCtx = MetalContext::instance();
    int64_t totalDataSz = d.solver->totalDataSize();

    // Step 2: Load f32 values into persistent GPU buffer (zero-copy unified memory)
    std::memset(d.dataGpu.ptr(), 0, totalDataSz * sizeof(float));
    d.solver->loadFromCsr(d.permRowPtr.data(), d.permColInd.data(), d.blockSizes.data(),
                          d.permValuesF32.data(), d.dataGpu.ptr());

    // Step 3: Permute RHS into persistent GPU buffer
    for (int64_t j = 0; j < n; j++) {
      d.xGpu.ptr()[perm[j]] = float(d.rowScale[j] * rhs[d.preproc.rowPerm[j]]);
    }

    // Step 4: GPU factor + initial solve in one command buffer
    void* cmdBuf = metalCtx.createCommandBuffer();
    void* encoder = metalCtx.createComputeEncoder(cmdBuf);
    symCtx.setExternalEncoder(cmdBuf, encoder);

    d.numCtx->reset();
    d.solver->factorLU(d.dataGpu.ptr(), d.devPivots.ptr(), *d.numCtx, PivotLocation::Device);
    d.solver->solveLU(d.dataGpu.ptr(), d.devPivots.ptr(), d.xGpu.ptr(), n, 1, *d.solveCtx,
                      PivotLocation::Device);

    // Step 5: Iterative refinement with encoder cycling
    std::fill(d.xAccum.begin(), d.xAccum.end(), 0.0);

    for (int iter = 0; iter < d.maxRefine; iter++) {
      symCtx.clearExternalEncoder();

      for (int64_t j = 0; j < n; j++) {
        d.xAccum[j] += d.colScale[j] * double(d.xGpu.ptr()[perm[j]]);
      }

      for (int64_t j = 0; j < n; j++) {
        int64_t srcRow = d.preproc.rowPerm[j];
        double sum = 0.0;
        for (int64_t k = d.csrIndptr[srcRow]; k < d.csrIndptr[srcRow + 1]; k++) {
          sum += csr_data[k] * d.xAccum[d.csrIndices[k]];
        }
        d.xGpu.ptr()[perm[j]] = float(d.rowScale[j] * (rhs[srcRow] - sum));
      }

      void* newCmdBuf = metalCtx.createCommandBuffer();
      void* newEncoder = metalCtx.createComputeEncoder(newCmdBuf);
      symCtx.setExternalEncoder(newCmdBuf, newEncoder);

      d.solver->solveLU(d.dataGpu.ptr(), d.devPivots.ptr(), d.xGpu.ptr(), n, 1, *d.solveCtx,
                        PivotLocation::Device);
    }

    symCtx.clearExternalEncoder();

    for (int64_t j = 0; j < n; j++) {
      x_out[j] = d.xAccum[j] + d.colScale[j] * double(d.xGpu.ptr()[perm[j]]);
    }
    return;
  }
#endif

  // CPU fallback
  int64_t totalDataSz = d.solver->totalDataSize();
  std::fill(d.dataCpu.begin(), d.dataCpu.end(), 0.0f);
  d.solver->loadFromCsr(d.permRowPtr.data(), d.permColInd.data(), d.blockSizes.data(),
                        d.permValuesF32.data(), d.dataCpu.data());

  d.solver->factorLU(d.dataCpu.data(), d.pivotsCpu.data());

  std::vector<float> bp(n);
  for (int64_t j = 0; j < n; j++) {
    bp[perm[j]] = float(d.rowScale[j] * rhs[d.preproc.rowPerm[j]]);
  }
  d.solver->solveLU(d.dataCpu.data(), d.pivotsCpu.data(), bp.data(), n, 1);

  for (int64_t j = 0; j < n; j++) {
    x_out[j] = d.colScale[j] * double(bp[perm[j]]);
  }

  for (int iter = 0; iter < d.maxRefine; iter++) {
    std::vector<float> rp(n);
    for (int64_t j = 0; j < n; j++) {
      int64_t srcRow = d.preproc.rowPerm[j];
      double sum = 0.0;
      for (int64_t k = d.csrIndptr[srcRow]; k < d.csrIndptr[srcRow + 1]; k++) {
        sum += csr_data[k] * x_out[d.csrIndices[k]];
      }
      rp[perm[j]] = float(d.rowScale[j] * (rhs[srcRow] - sum));
    }

    d.solver->solveLU(d.dataCpu.data(), d.pivotsCpu.data(), rp.data(), n, 1);

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
