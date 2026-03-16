/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/SupernodeMerger.h"

#include <algorithm>
#include <queue>
#include <tuple>

#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/EliminationTree.h"

namespace BaSpaCho {

using namespace std;

int64_t LevelSetSchedule::numLumps() const {
  int64_t total = 0;
  for (auto& level : levels) {
    total += (int64_t)level.size();
  }
  return total;
}

LevelSetSchedule LevelSetSchedule::build(const vector<int64_t>& lumpParent) {
  int64_t numLumps = (int64_t)lumpParent.size();
  LevelSetSchedule schedule;

  if (numLumps == 0) {
    return schedule;
  }

  // Count children per lump (for bottom-up traversal)
  vector<int64_t> childCount(numLumps, 0);
  for (int64_t l = 0; l < numLumps; l++) {
    if (lumpParent[l] != -1) {
      SPRUX_CHECK_LT(lumpParent[l], numLumps);
      childCount[lumpParent[l]]++;
    }
  }

  // BFS from leaves (childCount == 0), assigning levels bottom-up
  vector<int64_t> lumpLevel(numLumps, 0);
  queue<int64_t> q;
  for (int64_t l = 0; l < numLumps; l++) {
    if (childCount[l] == 0) {
      q.push(l);
    }
  }

  int64_t maxLevel = 0;
  while (!q.empty()) {
    int64_t l = q.front();
    q.pop();
    int64_t p = lumpParent[l];
    if (p != -1) {
      lumpLevel[p] = max(lumpLevel[p], lumpLevel[l] + 1);
      maxLevel = max(maxLevel, lumpLevel[p]);
      childCount[p]--;
      if (childCount[p] == 0) {
        q.push(p);
      }
    }
  }

  // Group lumps by level
  schedule.levels.resize(maxLevel + 1);
  for (int64_t l = 0; l < numLumps; l++) {
    schedule.levels[lumpLevel[l]].push_back(l);
  }

  return schedule;
}

vector<int64_t> computeLumpParent(const EliminationTree& et) {
  int64_t ord = et.ss.order();
  int64_t numLumps = ord - et.numMerges;

  // Map each root node (mergeWith == -1) to its lump index.
  // Lump indices are assigned in unmergedHeightNode order, matching processTree().
  vector<int64_t> rootToLump(ord, -1);
  int64_t lumpIndex = 0;
  for (int64_t i = 0; i < ord; i++) {
    int64_t k = get<2>(et.unmergedHeightNode[i]);
    if (et.mergeWith[k] != -1) {
      continue;
    }
    rootToLump[k] = lumpIndex++;
  }
  SPRUX_CHECK_EQ(lumpIndex, numLumps);

  // For each lump (represented by its root node k), find the parent lump.
  // parent[k] is k's tree parent. Follow merge chain to find parent's root.
  vector<int64_t> lumpPar(numLumps, -1);
  for (int64_t i = 0; i < ord; i++) {
    int64_t k = get<2>(et.unmergedHeightNode[i]);
    if (et.mergeWith[k] != -1) {
      continue;
    }

    int64_t p = et.parent[k];
    if (p == -1) {
      continue;  // tree root
    }

    // Follow merge chain to find p's root
    while (et.mergeWith[p] != -1) {
      p = et.mergeWith[p];
    }

    SPRUX_CHECK(rootToLump[p] != -1);
    lumpPar[rootToLump[k]] = rootToLump[p];
  }

  return lumpPar;
}

void computeRelaxedMerges(EliminationTree& et, double fillTolerance,
                          int64_t maxSupernodeSize) {
  int64_t ord = et.ss.order();

  // Follow merge chain to find the root of a merge group.
  auto findRoot = [&](int64_t k) -> int64_t {
    while (et.mergeWith[k] != -1) {
      k = et.mergeWith[k];
    }
    return k;
  };

  // Process nodes bottom-up using the height ordering from computeNodeHeights().
  // This ensures children are considered before parents.
  for (int64_t i = 0; i < ord; i++) {
    int64_t k = get<2>(et.unmergedHeightNode[i]);

    // Skip if already merged or forbidden (sparse elimination candidate)
    if (et.mergeWith[k] != -1 || et.forbidMerge[k]) {
      continue;
    }

    // Find tree parent
    int64_t treeParent = et.parent[k];
    if (treeParent == -1) {
      continue;  // tree root, nothing to merge into
    }

    // Find the effective parent (root of parent's merge group)
    int64_t p = findRoot(treeParent);
    if (p == k) {
      continue;  // shouldn't happen, but guard against cycles
    }

    // Size constraint: merged supernode must not exceed max size
    if (et.nodeSize[k] + et.nodeSize[p] > maxSupernodeSize) {
      continue;
    }

    // Fill constraint: fraction of extra zeros must be within tolerance.
    // fillRatio = nodeRows[k] / (nodeRows[p] + nodeSize[p])
    // extraZeros = 1 - fillRatio
    // Merge if extraZeros <= fillTolerance
    // Note: nodeRows uses original structural counts (not updated by prior merges),
    // consistent with computeMerges() in EliminationTree.cpp.
    double denom = et.nodeRows[p] + et.nodeSize[p];
    if (denom > 0) {  // denom==0 only for degenerate root; merge unconditionally
      double fillRatio = (double)et.nodeRows[k] / denom;
      if (1.0 - fillRatio > fillTolerance) {
        continue;
      }
    }

    // Merge k into p
    et.mergeWith[k] = p;
    et.nodeSize[p] += et.nodeSize[k];
    et.numMergedNodes[p] += et.numMergedNodes[k];
    et.numMerges++;
  }
}

}  // end namespace BaSpaCho
