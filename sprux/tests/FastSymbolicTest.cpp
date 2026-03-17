/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <random>
#include <set>

#include "sprux/sprux/SparseStructure.h"

using namespace BaSpaCho;

// Helper to create a small SYMMETRIC lower-triangular test matrix (CSR lower half).
// The input must be symmetric for Cholesky fill computation to be well-defined.
// This matches how EliminationTree::computeAggregateStruct() uses symmetricPermutation()
// to produce a symmetric half-matrix before calling addFullEliminationFill().
static SparseStructure makeSmallSymmetricLower() {
  // 5x5 symmetric matrix, lower triangle in CSR:
  //   row 0: [0]           (diagonal)
  //   row 1: [0, 1]        (A(1,0), diagonal)
  //   row 2: [2]           (diagonal)
  //   row 3: [0, 2, 3]     (A(3,0), A(3,2), diagonal)
  //   row 4: [1, 3, 4]     (A(4,1), A(4,3), diagonal)
  // Implied symmetric entries: A(0,1), A(0,3), A(2,3), A(1,4), A(3,4)
  return SparseStructure(
      {0, 1, 3, 4, 7, 10},                // ptrs
      {0, 0, 1, 2, 0, 2, 3, 1, 3, 4}     // inds
  );
}

TEST(FastSymbolic, CholmodMatchesOriginal_Small) {
  SparseStructure ss = makeSmallSymmetricLower();

  SparseStructure fillOrig = ss.addFullEliminationFill();
  SparseStructure fillCholmod = ss.addFullEliminationFillCholmod();

  // With natural ordering (no postordering), CHOLMOD with stype=+1 should produce
  // exactly the same fill pattern as the original algorithm on symmetric input.
  EXPECT_EQ(fillOrig.ptrs, fillCholmod.ptrs)
      << "Column pointers differ for small matrix";
  EXPECT_EQ(fillOrig.inds, fillCholmod.inds)
      << "Row indices differ for small matrix";
}

TEST(FastSymbolic, CholmodMatchesOriginal_Random) {
  // Generate a random sparse symmetric matrix and compare fill patterns
  int64_t n = 100;
  std::mt19937 rng(42);

  // Build random lower-triangular CSR with unique indices per row
  std::vector<int64_t> ptrs(n + 1, 0);
  std::vector<int64_t> inds;

  for (int64_t k = 0; k < n; k++) {
    std::set<int64_t> rowEntries;
    rowEntries.insert(k);  // diagonal always present

    // Add ~5 random entries below diagonal (deduplicated)
    for (int64_t attempt = 0; attempt < 5; attempt++) {
      int64_t row = k + 1 + (rng() % (n - k));
      if (row < n) {
        rowEntries.insert(row);
      }
    }

    ptrs[k] = (int64_t)rowEntries.size();
    for (int64_t idx : rowEntries) {
      inds.push_back(idx);
    }
  }

  // Convert counts to cumulative pointers
  int64_t total = 0;
  for (int64_t k = 0; k <= n; k++) {
    int64_t count = ptrs[k];
    ptrs[k] = total;
    total += count;
  }

  SparseStructure ss(std::move(ptrs), std::move(inds));
  ss.sortIndices();

  SparseStructure fillOrig = ss.addFullEliminationFill();
  SparseStructure fillCholmod = ss.addFullEliminationFillCholmod();

  EXPECT_EQ(fillOrig.ptrs, fillCholmod.ptrs)
      << "Column pointers differ for random matrix";
  EXPECT_EQ(fillOrig.inds, fillCholmod.inds)
      << "Row indices differ for random matrix";
}

TEST(FastSymbolic, CholmodMatchesOriginal_Diagonal) {
  // Pure diagonal matrix — no fill expected
  int64_t n = 10;
  std::vector<int64_t> ptrs(n + 1);
  std::vector<int64_t> inds(n);
  for (int64_t k = 0; k < n; k++) {
    ptrs[k] = k;
    inds[k] = k;
  }
  ptrs[n] = n;

  SparseStructure ss(std::move(ptrs), std::move(inds));

  SparseStructure fillOrig = ss.addFullEliminationFill();
  SparseStructure fillCholmod = ss.addFullEliminationFillCholmod();

  EXPECT_EQ(fillOrig.ptrs, fillCholmod.ptrs);
  EXPECT_EQ(fillOrig.inds, fillCholmod.inds);
}
