/*
 * Copyright (c) 2024-2026 ChipFlow
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SpruxFFISolver: encapsulated Metal LU solver for circuit simulation FFI.
// Reference implementation: benchmarking/LUBench.cpp:benchmarkLUMetalFFI
//
// Per-matrix equilibration (matching lu_bench) with pre-computed CSR→coalesced
// scatter map to eliminate loadFromCsr accessor lookups from the hot path.

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
  double refineTol;

  // Original CSR structure (int64, converted from caller's int32)
  std::vector<int64_t> csrIndptr;
  std::vector<int64_t> csrIndices;

  // Preprocessing results (fixed for lifetime)
  LUPreprocessing preproc;

  // Permuted CSR structure (after BTF row perm — sparsity is fixed)
  std::vector<int64_t> permRowPtr;
  std::vector<int64_t> permColInd;

  // Pre-computed CSR→coalesced scatter map: for each NNZ position k in the
  // ORIGINAL CSR, csrToDataMap[k] = offset in the coalesced data[] buffer.
  // Eliminates per-solve loadFromCsr accessor binary searches.
  std::vector<int64_t> csrToDataMap;

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
  std::vector<double> rowScale;
  std::vector<double> colScale;
  std::vector<double> xAccum;
};

// Compute equilibration inline from original CSR, iterating by permuted row.
// No intermediate buffer — computes row/col scales directly.
static void computeEquilibrationFromOrigCsr(
    int64_t n, const int64_t* csrIndptr, const int64_t* csrIndices,
    const double* csrData, const int64_t* rowPerm,
    std::vector<double>& rowScale, std::vector<double>& colScale) {
  // Row scaling: Dr[permRow] = 1 / max_col |A[origRow, col]|
  for (int64_t permRow = 0; permRow < n; permRow++) {
    int64_t origRow = rowPerm[permRow];
    double maxVal = 0;
    for (int64_t k = csrIndptr[origRow]; k < csrIndptr[origRow + 1]; k++) {
      maxVal = std::max(maxVal, std::abs(csrData[k]));
    }
    rowScale[permRow] = (maxVal > 0) ? 1.0 / maxVal : 1.0;
  }

  // Column scaling: Dc[col] = 1 / max_permRow |Dr[permRow] * A[origRow, col]|
  std::fill(colScale.begin(), colScale.end(), 0.0);
  for (int64_t permRow = 0; permRow < n; permRow++) {
    int64_t origRow = rowPerm[permRow];
    double rs = rowScale[permRow];
    for (int64_t k = csrIndptr[origRow]; k < csrIndptr[origRow + 1]; k++) {
      int64_t col = csrIndices[k];
      double scaled = std::abs(rs * csrData[k]);
      colScale[col] = std::max(colScale[col], scaled);
    }
  }
  for (int64_t j = 0; j < n; j++) {
    colScale[j] = (colScale[j] > 0) ? 1.0 / colScale[j] : 1.0;
  }
}

// Scatter original CSR values into coalesced data buffer using pre-computed map,
// applying per-matrix equilibration scales. Single pass, no intermediate buffers.
template <typename DataPtr>
static void scatterEquilibratedValues(
    int64_t n, int64_t /*nnz*/, const int64_t* csrIndptr, const int64_t* csrIndices,
    const double* csrData, const int64_t* rowPerm,
    const double* rowScale, const double* colScale,
    const int64_t* csrToDataMap, DataPtr data) {
  for (int64_t permRow = 0; permRow < n; permRow++) {
    int64_t origRow = rowPerm[permRow];
    double rs = rowScale[permRow];
    for (int64_t k = csrIndptr[origRow]; k < csrIndptr[origRow + 1]; k++) {
      int64_t col = csrIndices[k];
      data[csrToDataMap[k]] = float(rs * csrData[k] * colScale[col]);
    }
  }
}

// ---------------------------------------------------------------------------
// Constructor: one-time setup
// ---------------------------------------------------------------------------

SpruxFFISolver::SpruxFFISolver(int32_t n, int32_t nnz, const int32_t* csr_indptr,
                               const int32_t* csr_indices, const double* csr_data_init,
                               int max_refine_steps, double refine_tol)
    : impl_(std::make_unique<Impl>()) {
  auto& d = *impl_;
  d.n = n;
  d.nnz = nnz;
  d.maxRefine = max_refine_steps;
  d.refineTol = refine_tol;

  d.csrIndptr.assign(csr_indptr, csr_indptr + n + 1);
  d.csrIndices.assign(csr_indices, csr_indices + nnz);
  d.blockSizes.assign(n, 1);

  // Step 1: BTF max transversal
  d.preproc = computeMaxTransversal(n, d.csrIndptr.data(), d.csrIndices.data());

  // Step 2: Apply row permutation to get permuted CSR structure (once — sparsity is fixed)
  std::vector<double> permValues;
  applyRowPermToCsr<double>(n, d.csrIndptr.data(), d.csrIndices.data(), csr_data_init,
                            d.preproc.rowPerm.data(), d.permRowPtr, d.permColInd, permValues);

  // Step 3: Compute equilibration from initial matrix (for pivot threshold)
  d.rowScale.resize(n);
  d.colScale.resize(n);
  computeEquilibrationFromOrigCsr(n, d.csrIndptr.data(), d.csrIndices.data(), csr_data_init,
                                  d.preproc.rowPerm.data(), d.rowScale, d.colScale);

  // Step 4: Build symmetric SparseStructure for AMD ordering
  SparseStructure ss =
      csrToSymmetricSparseStructure(n, d.permRowPtr.data(), d.permColInd.data());

  // Step 5: Compute static pivot threshold from equilibrated initial diagonal
  double pivotThreshold;
  {
    double maxDiag = 0;
    for (int64_t permRow = 0; permRow < n; permRow++) {
      int64_t origRow = d.preproc.rowPerm[permRow];
      double rs = d.rowScale[permRow];
      for (int64_t k = d.csrIndptr[origRow]; k < d.csrIndptr[origRow + 1]; k++) {
        if (d.csrIndices[k] == permRow) {
          // This is the diagonal entry in permuted coordinates
          maxDiag = std::max(maxDiag, std::abs(rs * csr_data_init[k] * d.colScale[permRow]));
          break;
        }
      }
    }
    float epsScale = std::cbrt(std::numeric_limits<float>::epsilon());
    pivotThreshold = double(epsScale) * std::max(maxDiag, double(epsScale));
  }

  // Step 6: Create solver
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
  // Sparse elimination only for GPU backends — CPU SolveCtx doesn't implement
  // sparseElimSolveLUnit/sparseElimSolveU for LU.
  settings.findSparseEliminationRanges = d.useMetal;

  std::vector<int64_t> paramSizes(n, 1);
  d.solver = createSolver(settings, paramSizes, ss);
  d.perm = &d.solver->paramToSpan();
  int64_t totalDataSz = d.solver->totalDataSize();
  int64_t upperDataBase = d.solver->dataSize();

  // Step 7: Build CSR→coalesced scatter map — maps each original CSR entry k
  // to its offset in the coalesced data[] buffer. This precomputes the accessor
  // binary searches that loadFromCsr does per-call.
  d.csrToDataMap.resize(nnz);
  {
    auto acc = d.solver->accessor();

    for (int64_t permRow = 0; permRow < n; permRow++) {
      int64_t origRow = d.preproc.rowPerm[permRow];

      for (int64_t k = d.csrIndptr[origRow]; k < d.csrIndptr[origRow + 1]; k++) {
        int64_t col = d.csrIndices[k];

        // loadFromCsr passes (permRow, col) to accessor which applies AMD perm
        auto [offset, stride, flipped] = acc.blockOffset(permRow, col);

        if (flipped) {
          // Upper triangle (permRow < permCol after AMD permutation)
          int64_t amdRow = (*d.perm)[permRow];
          int64_t amdCol = (*d.perm)[col];
          int64_t rowLump = acc.plainAcc.spanToLump[amdRow];
          int64_t colLump = acc.plainAcc.spanToLump[amdCol];

          if (rowLump == colLump) {
            int64_t lumpSize =
                acc.plainAcc.lumpStart[rowLump + 1] - acc.plainAcc.lumpStart[rowLump];
            int64_t diagStart = acc.plainAcc.chainData[acc.plainAcc.chainColPtr[rowLump]];
            int64_t rowOff = acc.plainAcc.spanOffsetInLump[amdRow];
            int64_t colOff = acc.plainAcc.spanOffsetInLump[amdCol];
            d.csrToDataMap[k] = diagStart + rowOff * lumpSize + colOff;
          } else {
            auto [upperOff, upperStride] = acc.plainAcc.upperBlockOffset(amdRow, amdCol);
            d.csrToDataMap[k] = upperDataBase + upperOff;
          }
        } else {
          d.csrToDataMap[k] = offset;
        }
      }
    }
  }

#ifdef SPRUX_USE_METAL
  {
    auto& symCtx = d.solver->internalSymbolicContext();
    symCtx.disableAllStats();

    d.dataGpu.resizeToAtLeast(totalDataSz);
    d.xGpu.resizeToAtLeast(n);
    d.devPivots.resizeToAtLeast(n);

    d.numCtx = symCtx.createNumericCtx<float>(0, static_cast<float*>(nullptr));
    d.numCtx->preAllocateForLU(1, n);
    d.solveCtx = symCtx.createSolveCtx<float>(1, static_cast<float*>(nullptr));

    // Warmup with valid data (forces MPS shader JIT compilation)
    {
      std::memset(d.dataGpu.ptr(), 0, totalDataSz * sizeof(float));
      scatterEquilibratedValues(n, nnz, d.csrIndptr.data(), d.csrIndices.data(),
                                csr_data_init, d.preproc.rowPerm.data(),
                                d.rowScale.data(), d.colScale.data(),
                                d.csrToDataMap.data(), d.dataGpu.ptr());

      d.numCtx->reset();
      d.solver->factorLU(d.dataGpu.ptr(), d.devPivots.ptr(), *d.numCtx, PivotLocation::Device);
      d.numCtx->flush();
    }
  }
#endif

  d.dataCpu.resize(totalDataSz, 0.0f);
  d.pivotsCpu.resize(n, 0);
  d.xAccum.resize(n, 0.0);
}

SpruxFFISolver::~SpruxFFISolver() = default;

// ---------------------------------------------------------------------------
// solve(): per-NR-iteration — equilibrate, scatter, factor, solve, refine
// Matches lu_bench's benchmarkLUMetalFFI: per-matrix equilibration with
// pre-computed CSR→coalesced scatter map (no loadFromCsr).
// ---------------------------------------------------------------------------

int SpruxFFISolver::solve(const double* csr_data, const double* rhs, double* x_out) {
  auto& d = *impl_;
  const int64_t n = d.n;
  const auto& perm = *d.perm;

  // Step 1: Per-matrix equilibration (fresh scales, matching lu_bench)
  computeEquilibrationFromOrigCsr(n, d.csrIndptr.data(), d.csrIndices.data(), csr_data,
                                  d.preproc.rowPerm.data(), d.rowScale, d.colScale);

#ifdef SPRUX_USE_METAL
  if (d.useMetal) {
    auto& symCtx = d.solver->internalSymbolicContext();
    auto& metalCtx = MetalContext::instance();
    int64_t totalDataSz = d.solver->totalDataSize();

    // Step 2: Scatter equilibrated values directly into GPU buffer
    std::memset(d.dataGpu.ptr(), 0, totalDataSz * sizeof(float));
    scatterEquilibratedValues(n, d.nnz, d.csrIndptr.data(), d.csrIndices.data(),
                              csr_data, d.preproc.rowPerm.data(),
                              d.rowScale.data(), d.colScale.data(),
                              d.csrToDataMap.data(), d.dataGpu.ptr());

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

    // Step 5: Iterative refinement with encoder cycling and early termination.
    // Each iteration: flush GPU → CPU accumulate + SpMV residual → check convergence → GPU solve.
    // The residual norm is computed for free from the SpMV already needed for the correction.
    std::fill(d.xAccum.begin(), d.xAccum.end(), 0.0);
    double bNormSq = 0.0;
    for (int64_t j = 0; j < n; j++) bNormSq += rhs[j] * rhs[j];

    int itersUsed = 0;
    for (int iter = 0; iter < d.maxRefine; iter++) {
      symCtx.clearExternalEncoder();

      for (int64_t j = 0; j < n; j++) {
        d.xAccum[j] += d.colScale[j] * double(d.xGpu.ptr()[perm[j]]);
      }

      double resNormSq = 0.0;
      for (int64_t j = 0; j < n; j++) {
        int64_t srcRow = d.preproc.rowPerm[j];
        double sum = 0.0;
        for (int64_t k = d.csrIndptr[srcRow]; k < d.csrIndptr[srcRow + 1]; k++) {
          sum += csr_data[k] * d.xAccum[d.csrIndices[k]];
        }
        double residual = rhs[srcRow] - sum;
        resNormSq += residual * residual;
        d.xGpu.ptr()[perm[j]] = float(d.rowScale[j] * residual);
      }
      itersUsed = iter + 1;

      // Early termination: skip GPU solve if residual is below tolerance
      if (d.refineTol > 0 && resNormSq <= d.refineTol * d.refineTol * std::max(bNormSq, 1e-300)) {
        break;
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
    return itersUsed;
  }
#endif

  // CPU fallback — scatter directly into coalesced buffer
  std::fill(d.dataCpu.begin(), d.dataCpu.end(), 0.0f);
  scatterEquilibratedValues(n, d.nnz, d.csrIndptr.data(), d.csrIndices.data(),
                            csr_data, d.preproc.rowPerm.data(),
                            d.rowScale.data(), d.colScale.data(),
                            d.csrToDataMap.data(), d.dataCpu.data());

  d.solver->factorLU(d.dataCpu.data(), d.pivotsCpu.data());

  std::vector<float> bp(n);
  for (int64_t j = 0; j < n; j++) {
    bp[perm[j]] = float(d.rowScale[j] * rhs[d.preproc.rowPerm[j]]);
  }
  d.solver->solveLU(d.dataCpu.data(), d.pivotsCpu.data(), bp.data(), n, 1);

  for (int64_t j = 0; j < n; j++) {
    x_out[j] = d.colScale[j] * double(bp[perm[j]]);
  }

  double bNormSqCpu = 0.0;
  for (int64_t j = 0; j < n; j++) bNormSqCpu += rhs[j] * rhs[j];
  int itersUsedCpu = 0;

  for (int iter = 0; iter < d.maxRefine; iter++) {
    std::vector<float> rp(n);
    double resNormSq = 0.0;
    for (int64_t j = 0; j < n; j++) {
      int64_t srcRow = d.preproc.rowPerm[j];
      double sum = 0.0;
      for (int64_t k = d.csrIndptr[srcRow]; k < d.csrIndptr[srcRow + 1]; k++) {
        sum += csr_data[k] * x_out[d.csrIndices[k]];
      }
      double residual = rhs[srcRow] - sum;
      resNormSq += residual * residual;
      rp[perm[j]] = float(d.rowScale[j] * residual);
    }
    itersUsedCpu = iter + 1;

    if (d.refineTol > 0 && resNormSq <= d.refineTol * d.refineTol * std::max(bNormSqCpu, 1e-300)) {
      break;
    }

    d.solver->solveLU(d.dataCpu.data(), d.pivotsCpu.data(), rp.data(), n, 1);

    for (int64_t j = 0; j < n; j++) {
      x_out[j] += d.colScale[j] * double(rp[perm[j]]);
    }
  }
  return itersUsedCpu;
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
