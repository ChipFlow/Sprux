/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/SupernodeMerger.h"

using namespace BaSpaCho;

// Build an EliminationTree and run through processTree (standard merges).
// Returns the tree ready for computeRelaxedMerges() or computeLumpParent().
static EliminationTree buildAndProcess(const std::vector<int64_t>& paramSize,
                                       SparseStructure& ss) {
  EliminationTree et(paramSize, ss);
  et.buildTree();
  et.processTree(/* detectSparseElimRanges= */ false);
  return et;
}

// ======================== LevelSetSchedule tests ========================

TEST(LevelSetSchedule, LinearChain) {
  // Linear chain: 0→1→2→3→4 (4 is root)
  // lumpParent = [1, 2, 3, 4, -1]
  std::vector<int64_t> lumpParent = {1, 2, 3, 4, -1};
  auto schedule = LevelSetSchedule::build(lumpParent);

  EXPECT_EQ(schedule.numLevels(), 5);
  EXPECT_EQ(schedule.numLumps(), 5);

  // Level 0 should contain only lump 0 (the leaf)
  EXPECT_EQ(schedule.levels[0].size(), 1u);
  EXPECT_EQ(schedule.levels[0][0], 0);

  // Level 4 should contain only lump 4 (the root)
  EXPECT_EQ(schedule.levels[4].size(), 1u);
  EXPECT_EQ(schedule.levels[4][0], 4);
}

TEST(LevelSetSchedule, WideBinaryTree) {
  // Binary tree:
  //       6
  //      / \
  //     4   5
  //    / \ / \
  //   0  1 2  3
  // lumpParent = [4, 4, 5, 5, 6, 6, -1]
  std::vector<int64_t> lumpParent = {4, 4, 5, 5, 6, 6, -1};
  auto schedule = LevelSetSchedule::build(lumpParent);

  EXPECT_EQ(schedule.numLevels(), 3);
  EXPECT_EQ(schedule.numLumps(), 7);

  // Level 0: leaves {0, 1, 2, 3}
  ASSERT_EQ(schedule.levels[0].size(), 4u);
  std::set<int64_t> leaves(schedule.levels[0].begin(), schedule.levels[0].end());
  EXPECT_EQ(leaves, (std::set<int64_t>{0, 1, 2, 3}));

  // Level 1: internal {4, 5}
  ASSERT_EQ(schedule.levels[1].size(), 2u);
  std::set<int64_t> internal(schedule.levels[1].begin(), schedule.levels[1].end());
  EXPECT_EQ(internal, (std::set<int64_t>{4, 5}));

  // Level 2: root {6}
  ASSERT_EQ(schedule.levels[2].size(), 1u);
  EXPECT_EQ(schedule.levels[2][0], 6);
}

TEST(LevelSetSchedule, SingleNode) {
  std::vector<int64_t> lumpParent = {-1};
  auto schedule = LevelSetSchedule::build(lumpParent);

  EXPECT_EQ(schedule.numLevels(), 1);
  EXPECT_EQ(schedule.numLumps(), 1);
  EXPECT_EQ(schedule.levels[0][0], 0);
}

TEST(LevelSetSchedule, Forest) {
  // Two independent roots (forest)
  // 0→2, 1→2, 3→4 (roots: 2, 4)
  std::vector<int64_t> lumpParent = {2, 2, -1, 4, -1};
  auto schedule = LevelSetSchedule::build(lumpParent);

  EXPECT_EQ(schedule.numLevels(), 2);
  EXPECT_EQ(schedule.numLumps(), 5);

  // Level 0: leaves {0, 1, 3}
  std::set<int64_t> leaves(schedule.levels[0].begin(), schedule.levels[0].end());
  EXPECT_EQ(leaves, (std::set<int64_t>{0, 1, 3}));

  // Level 1: roots {2, 4}
  std::set<int64_t> roots(schedule.levels[1].begin(), schedule.levels[1].end());
  EXPECT_EQ(roots, (std::set<int64_t>{2, 4}));
}

TEST(LevelSetSchedule, AllLumpsAppearExactlyOnce) {
  // Random-ish tree
  std::vector<int64_t> lumpParent = {3, 3, 4, 5, 5, 7, 7, -1};
  auto schedule = LevelSetSchedule::build(lumpParent);

  std::set<int64_t> seen;
  for (auto& level : schedule.levels) {
    for (int64_t l : level) {
      EXPECT_TRUE(seen.insert(l).second) << "Lump " << l << " appears in multiple levels";
    }
  }
  EXPECT_EQ((int64_t)seen.size(), (int64_t)lumpParent.size());
}

// ======================== computeLumpParent tests ========================

TEST(LumpParent, SmallSymmetricMatrix) {
  // 5x5 symmetric lower-triangular CSR (same as FastSymbolicTest)
  SparseStructure ss({0, 1, 3, 4, 7, 10}, {0, 0, 1, 2, 0, 2, 3, 1, 3, 4});
  std::vector<int64_t> paramSize = {1, 1, 1, 1, 1};

  EliminationTree et = buildAndProcess(paramSize, ss);

  auto lumpParent = computeLumpParent(et);
  int64_t numLumps = (int64_t)et.ss.order() - et.numMerges;
  EXPECT_EQ((int64_t)lumpParent.size(), numLumps);

  // Every non-root lump should have a valid parent
  int64_t numRoots = 0;
  for (int64_t l = 0; l < numLumps; l++) {
    if (lumpParent[l] == -1) {
      numRoots++;
    } else {
      EXPECT_GE(lumpParent[l], 0);
      EXPECT_LT(lumpParent[l], numLumps);
      EXPECT_NE(lumpParent[l], l) << "Lump " << l << " is its own parent";
    }
  }
  EXPECT_GE(numRoots, 1) << "Must have at least one root";
}

// ======================== computeRelaxedMerges tests ========================

// Helper: build tree, run ONLY relaxed merges (no standard cost-based merges),
// return lump count. This isolates the relaxed merge behavior.
static int64_t buildAndRelaxedMerge(const std::vector<int64_t>& paramSize, SparseStructure& ss,
                                    double fillTolerance, int64_t maxSize) {
  int64_t ord = (int64_t)paramSize.size();
  EliminationTree et(paramSize, ss);
  et.buildTree();
  et.computeNodeHeights({});

  // Initialize merge state with no merges (skip standard cost-based merges)
  et.mergeWith.assign(ord, -1);
  et.numMergedNodes.assign(ord, 1);
  et.numMerges = 0;

  computeRelaxedMerges(et, fillTolerance, maxSize);
  et.collapseMergePointers();

  return ord - et.numMerges;
}

TEST(RelaxedMerge, MaxSizeRespected) {
  // Create a chain-like structure where everything wants to merge,
  // but max size limits it.
  // Dense 8x8 lower triangle — all fill ratios are 1.0, everything should merge.
  int64_t n = 8;
  std::vector<int64_t> ptrs(n + 1);
  std::vector<int64_t> inds;
  for (int64_t k = 0; k < n; k++) {
    ptrs[k] = (int64_t)inds.size();
    for (int64_t j = 0; j <= k; j++) {
      inds.push_back(j);
    }
  }
  ptrs[n] = (int64_t)inds.size();

  SparseStructure ss(std::move(ptrs), std::move(inds));
  std::vector<int64_t> paramSize(n, 1);

  // With maxSize=4, should get at most 4 nodes per supernode → at least 2 lumps
  int64_t lumps = buildAndRelaxedMerge(paramSize, ss, 0.5, 4);
  EXPECT_GE(lumps, 2) << "Max size=4 should prevent merging all 8 nodes into one";

  // With maxSize=256, should merge aggressively (possibly all into one)
  int64_t lumpsLarge = buildAndRelaxedMerge(paramSize, ss, 0.5, 256);
  EXPECT_LE(lumpsLarge, lumps);
}

TEST(RelaxedMerge, FillToleranceRespected) {
  // 5x5 symmetric lower-triangular CSR
  SparseStructure ss({0, 1, 3, 4, 7, 10}, {0, 0, 1, 2, 0, 2, 3, 1, 3, 4});
  std::vector<int64_t> paramSize = {1, 1, 1, 1, 1};

  // Tight tolerance (0.0) — should not merge beyond what cost model already does
  int64_t lumpsTight = buildAndRelaxedMerge(paramSize, ss, 0.0, 256);

  // Loose tolerance (1.0) — should merge everything possible
  int64_t lumpsLoose = buildAndRelaxedMerge(paramSize, ss, 1.0, 256);

  EXPECT_LE(lumpsLoose, lumpsTight);
}

TEST(RelaxedMerge, RandomScalarMatrix) {
  // Generate a random sparse symmetric matrix with 100 scalar nodes.
  // Relaxed merging should significantly reduce lump count.
  int64_t n = 100;
  std::mt19937 rng(123);

  // Build random lower-triangular CSR
  std::vector<int64_t> ptrs(n + 1, 0);
  std::vector<int64_t> inds;

  for (int64_t k = 0; k < n; k++) {
    inds.push_back(k);  // diagonal
    ptrs[k]++;

    // Add ~3 random off-diagonal entries per row (below diagonal)
    for (int64_t attempt = 0; attempt < 3; attempt++) {
      int64_t col = rng() % k;  // column < k (below diagonal in CSR)
      if (k > 0) {
        inds.push_back(col);
        ptrs[k]++;
      }
    }
  }

  // Convert counts to cumulative
  int64_t total = 0;
  for (int64_t k = 0; k <= n; k++) {
    int64_t count = ptrs[k];
    ptrs[k] = total;
    total += count;
  }

  SparseStructure ss(std::move(ptrs), std::move(inds));
  ss.sortIndices();

  std::vector<int64_t> paramSize(n, 1);

  // With relaxed merging, expect significant reduction
  int64_t lumps = buildAndRelaxedMerge(paramSize, ss, 0.25, 256);
  EXPECT_LT(lumps, n) << "Relaxed merge should reduce lump count from " << n;
}

// ======================== End-to-end: merge + level-set ========================

TEST(SupernodeMerger, EndToEndSchedule) {
  // Build a tree with relaxed merges (maxSize=4) so we get multiple lumps,
  // then build a level-set schedule and verify structural properties.
  int64_t n = 30;
  std::vector<int64_t> ptrs(n + 1);
  std::vector<int64_t> inds;

  // Tridiagonal: each node connects to its neighbor → linear elimination tree
  for (int64_t k = 0; k < n; k++) {
    ptrs[k] = (int64_t)inds.size();
    if (k > 0) {
      inds.push_back(k - 1);  // sub-diagonal
    }
    inds.push_back(k);  // diagonal
  }
  ptrs[n] = (int64_t)inds.size();

  SparseStructure ss(std::move(ptrs), std::move(inds));
  std::vector<int64_t> paramSize(n, 1);

  EliminationTree et(paramSize, ss);
  et.buildTree();
  et.computeNodeHeights({});

  // No standard merges — use only relaxed merge with small max size
  et.mergeWith.assign(n, -1);
  et.numMergedNodes.assign(n, 1);
  et.numMerges = 0;

  computeRelaxedMerges(et, 0.5, /* maxSupernodeSize= */ 4);
  et.collapseMergePointers();

  int64_t numLumps = n - et.numMerges;
  EXPECT_GE(numLumps, 2) << "maxSize=4 should prevent merging all " << n << " nodes";

  // Run through processTree-like logic to assign lump indices
  // (processTree uses unmergedHeightNode ordering)
  std::vector<int64_t> rootToLump(n, -1);
  int64_t lumpIndex = 0;
  for (int64_t i = 0; i < n; i++) {
    int64_t k = std::get<2>(et.unmergedHeightNode[i]);
    if (et.mergeWith[k] != -1) continue;
    rootToLump[k] = lumpIndex++;
  }
  ASSERT_EQ(lumpIndex, numLumps);

  auto lumpParent = computeLumpParent(et);
  auto schedule = LevelSetSchedule::build(lumpParent);

  // All lumps appear exactly once
  EXPECT_EQ(schedule.numLumps(), numLumps);

  // Multiple levels expected (tridiagonal chain creates linear tree)
  EXPECT_GE(schedule.numLevels(), 2);

  // Verify parent-child level ordering: parent at strictly higher level than child
  std::vector<int64_t> lumpToLevel(numLumps, -1);
  for (int64_t lvl = 0; lvl < schedule.numLevels(); lvl++) {
    for (int64_t l : schedule.levels[lvl]) {
      lumpToLevel[l] = lvl;
    }
  }
  for (int64_t l = 0; l < numLumps; l++) {
    EXPECT_NE(lumpToLevel[l], -1) << "Lump " << l << " not in any level";
    if (lumpParent[l] != -1) {
      EXPECT_GT(lumpToLevel[lumpParent[l]], lumpToLevel[l])
          << "Parent lump " << lumpParent[l] << " (level " << lumpToLevel[lumpParent[l]]
          << ") must be at a higher level than child " << l << " (level " << lumpToLevel[l] << ")";
    }
  }
}
