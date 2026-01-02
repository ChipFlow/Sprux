/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <map>
#include <vector>
#include "baspacho/baspacho/SparseStructure.h"

namespace BaSpaCho {

struct UmfpackBenchResults {
  double analysisTime;
  double factorTime;
  std::map<int64_t, double> solveTimes;
  double residual;  // ||A*x - b|| / ||b|| after solve
};

// Benchmark UMFPACK LU factorization on a (potentially non-symmetric) sparse matrix
// paramSize: sizes of each block parameter
// ss: sparse structure (CSR format, block indices)
// nRHSs: list of different numbers of right-hand sides to benchmark solve
// verbose: print detailed stats
// symmetric: if true, generate symmetric data (for comparison with Cholesky-friendly problems)
UmfpackBenchResults benchmarkUmfpackSolve(const std::vector<int64_t>& paramSize,
                                           const SparseStructure& ss,
                                           const std::vector<int64_t>& nRHSs = {1}, int verbose = 1,
                                           bool symmetric = false);

}  // namespace BaSpaCho
