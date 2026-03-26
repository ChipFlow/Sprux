/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace Sprux {

struct EliminationTree;

// Groups lumps by elimination tree level for parallel execution.
// All lumps at the same level are independent (no parent-child relationships).
// Level 0 = leaves, highest level = root(s).
struct LevelSetSchedule {
  std::vector<std::vector<int64_t>> levels;  // levels[l] = lump indices at level l

  int64_t numLevels() const { return (int64_t)levels.size(); }
  int64_t numLumps() const;

  // Build from lump parent array. lumpParent[l] = parent lump index, or -1 for roots.
  static LevelSetSchedule build(const std::vector<int64_t>& lumpParent);
};

// Compute the parent relationship between lumps (merged supernodes) in the
// elimination tree. Returns lumpParent[l] = parent lump of lump l, or -1 for roots.
// Must be called after processTree().
std::vector<int64_t> computeLumpParent(const EliminationTree& et);

// Perform aggressive fill-tolerance-based merging on an EliminationTree.
// Can be called either:
//   1. Between computeMerges() and collapseMergePointers() (additive merging), or
//   2. Standalone with initialized merge state (mergeWith=-1, numMergedNodes=1, numMerges=0)
// Requires buildTree() and computeNodeHeights() to have been called.
//
// - fillTolerance: maximum fraction of extra zeros allowed (0.25 = 25%)
// - maxSupernodeSize: maximum merged node size (sum of parameter sizes)
void computeRelaxedMerges(EliminationTree& et, double fillTolerance = 0.25,
                          int64_t maxSupernodeSize = 256);

}  // end namespace Sprux
