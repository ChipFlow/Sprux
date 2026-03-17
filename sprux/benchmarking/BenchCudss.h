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
#include "sprux/sprux/SparseStructure.h"

struct CudssBenchResults {
  double analysisTime;
  double factorTime;
  std::map<int64_t, double> solveTimes;
};

CudssBenchResults benchmarkCudssSolve(const std::vector<int64_t>& paramSize,
                                       const BaSpaCho::SparseStructure& ss,
                                       const std::vector<int64_t>& nRHSs = {},
                                       int verbose = 0);
