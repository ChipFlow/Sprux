/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) 2024-2026 ChipFlow
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/sprux_c_api.h"

#include <cstring>
#include <memory>
#include <vector>

#include "baspacho/baspacho/Solver.h"

using namespace BaSpaCho;

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

}  // extern "C"
