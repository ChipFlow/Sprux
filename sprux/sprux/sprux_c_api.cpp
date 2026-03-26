/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) 2024-2026 ChipFlow
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sprux/sprux/sprux_c_api.h"

#include <cstring>
#include <memory>
#include <vector>

#include "sprux/sprux/Solver.h"
#include "sprux/sprux/SpruxFFISolver.h"

#ifdef SPRUX_USE_METAL
#include "sprux/sprux/MetalDefs.h"
#endif

using namespace Sprux;

struct sprux_solver {
  std::unique_ptr<Solver> solver;
};

static BackendType mapBackend(int backend) {
  switch (backend) {
    case SPRUX_BACKEND_CPU: return BackendFast;
    case SPRUX_BACKEND_CUDA: return BackendCuda;
    case SPRUX_BACKEND_METAL: return BackendMetal;
    case SPRUX_BACKEND_OPENCL: return BackendOpenCL;
    case SPRUX_BACKEND_AUTO: return BackendAuto;
    default: return BackendAuto;
  }
}

extern "C" {

sprux_solver_t sprux_create_lu_solver(const int64_t* paramSizes, int64_t numBlocks,
                                      const int64_t* ptrs, const int64_t* inds, int64_t nnzBlocks,
                                      int backend, double staticPivotThreshold) {
  try {
    std::vector<int64_t> paramSizeVec(paramSizes, paramSizes + numBlocks);
    SparseStructure ss;
    ss.ptrs.assign(ptrs, ptrs + numBlocks + 1);
    ss.inds.assign(inds, inds + nnzBlocks);

    Settings settings;
    settings.backend = mapBackend(backend);
    settings.matrixType = MTYPE_GENERAL;
    settings.staticPivotThreshold = staticPivotThreshold;

    auto solver = createSolver(settings, paramSizeVec, ss);

    auto h = new sprux_solver();
    h->solver = std::move(solver);
    return h;
  } catch (...) {
    return nullptr;
  }
}

void sprux_destroy(sprux_solver_t h) {
  delete h;
}

int64_t sprux_data_size(sprux_solver_t h) {
  if (!h) return 0;
  return h->solver->totalDataSize();
}

int64_t sprux_num_spans(sprux_solver_t h) {
  if (!h) return 0;
  return h->solver->skel().numSpans();
}

int sprux_load_from_csr_f32(sprux_solver_t h, const int64_t* ptrs, const int64_t* inds,
                            const int64_t* blockSizes, const float* values, float* data) {
  try {
    // Zero the full buffer (lower + upper) before loading
    std::memset(data, 0, h->solver->totalDataSize() * sizeof(float));
    h->solver->loadFromCsr(ptrs, inds, blockSizes, values, data);
    return 0;
  } catch (...) {
    return -1;
  }
}

int sprux_load_from_csr_f64_to_f32(sprux_solver_t h, const int64_t* ptrs, const int64_t* inds,
                                   const int64_t* blockSizes, const double* values, float* data) {
  try {
    // Compute total scalar values from CSR block structure
    int64_t numBlocks = static_cast<int64_t>(h->solver->paramToSpan().size());
    int64_t totalValues = 0;
    for (int64_t row = 0; row < numBlocks; row++) {
      for (int64_t p = ptrs[row]; p < ptrs[row + 1]; p++) {
        int64_t col = inds[p];
        totalValues += blockSizes[row] * blockSizes[col];
      }
    }

    // Convert f64 → f32
    std::vector<float> fvalues(totalValues);
    for (int64_t i = 0; i < totalValues; i++) {
      fvalues[i] = static_cast<float>(values[i]);
    }

    std::memset(data, 0, h->solver->totalDataSize() * sizeof(float));
    h->solver->loadFromCsr(ptrs, inds, blockSizes, fvalues.data(), data);
    return 0;
  } catch (...) {
    return -1;
  }
}

int sprux_factor_lu_f32(sprux_solver_t h, float* data, int64_t* pivots) {
  try {
    h->solver->factorLU(data, pivots);
    return 0;
  } catch (...) {
    return -1;
  }
}

int sprux_begin_factor_lu_f32(sprux_solver_t h, float* data, int64_t* pivots) {
  try {
    h->solver->beginFactorLU(data, pivots);
    return 0;
  } catch (...) {
    return -1;
  }
}

int sprux_finish_factor_lu_f32(sprux_solver_t h, float* data, int64_t* pivots) {
  try {
    h->solver->finishFactorLU(data, pivots);
    return 0;
  } catch (...) {
    return -1;
  }
}

int sprux_solve_lu_f32(sprux_solver_t h, const float* data, const int64_t* pivots, float* rhs,
                       int64_t stride, int nrhs) {
  try {
    h->solver->solveLU(data, pivots, rhs, stride, nrhs);
    return 0;
  } catch (...) {
    return -1;
  }
}

// =========================================================================
// FFI Solver API
// =========================================================================

struct sprux_ffi_solver {
  std::unique_ptr<SpruxFFISolver> solver;
};

sprux_ffi_solver_t sprux_ffi_create(int32_t n, int32_t nnz, const int32_t* csr_indptr,
                                    const int32_t* csr_indices, const double* csr_data_init,
                                    int max_refine_steps) {
  try {
    auto h = new sprux_ffi_solver();
    h->solver =
        std::make_unique<SpruxFFISolver>(n, nnz, csr_indptr, csr_indices, csr_data_init,
                                        max_refine_steps);
    return h;
  } catch (...) {
    return nullptr;
  }
}

void sprux_ffi_destroy(sprux_ffi_solver_t h) {
  delete h;
}

int sprux_ffi_solve(sprux_ffi_solver_t h, const double* csr_data, const double* rhs,
                    double* x_out) {
  try {
    h->solver->solve(csr_data, rhs, x_out);
    return 0;
  } catch (...) {
    return -1;
  }
}

int sprux_ffi_dot(sprux_ffi_solver_t h, const double* csr_data, const double* x, double* b_out) {
  try {
    h->solver->dot(csr_data, x, b_out);
    return 0;
  } catch (...) {
    return -1;
  }
}

int sprux_ffi_solve_only(sprux_ffi_solver_t h, const double* csr_data, const double* rhs,
                         double* x_out) {
  try {
    return h->solver->solveOnly(csr_data, rhs, x_out);
  } catch (...) {
    return -1;
  }
}

int sprux_ffi_begin_solve(sprux_ffi_solver_t h, const double* csr_data, const double* rhs) {
  try {
    h->solver->beginSolve(csr_data, rhs);
    return 0;
  } catch (...) {
    return -1;
  }
}

int sprux_ffi_end_solve(sprux_ffi_solver_t h, double* x_out) {
  try {
    return h->solver->endSolve(x_out);
  } catch (...) {
    return -1;
  }
}

int sprux_begin_capture(const char* output_path) {
#ifdef SPRUX_USE_METAL
  try {
    return MetalContext::instance().beginCapture(output_path) ? 1 : 0;
  } catch (...) {
    return 0;
  }
#else
  (void)output_path;
  return 0;
#endif
}

void sprux_end_capture(void) {
#ifdef SPRUX_USE_METAL
  try {
    MetalContext::instance().endCapture();
  } catch (...) {
  }
#endif
}

}  // extern "C"
