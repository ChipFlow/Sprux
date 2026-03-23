/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <set>
#include <vector>

namespace Sprux {

/**
 * @brief Block-level sparse matrix structure in CSR format.
 *
 * Represents the sparsity pattern of a block-structured matrix. Each entry
 * in the CSR structure corresponds to a *block* (not a scalar element).
 * For row `i`, the column indices of nonzero blocks are `inds[ptrs[i] : ptrs[i+1]]`.
 *
 * This is the primary input to createSolver(), which performs fill-reducing
 * reordering and symbolic factorization on this structure.
 *
 * Example (4-block matrix):
 * @code
 *   SparseStructure ss;
 *   ss.ptrs = {0, 2, 4, 7, 9};
 *   ss.inds = {0, 2, 1, 3, 0, 2, 3, 1, 3};
 * @endcode
 */
struct SparseStructure {
  std::vector<int64_t> ptrs;  ///< CSR row pointers. Size = order() + 1.
  std::vector<int64_t> inds;  ///< CSR column indices. Size = ptrs.back() (total nonzero blocks).

  SparseStructure() {}
  SparseStructure(std::vector<int64_t>&& ptrs_, std::vector<int64_t>&& inds_)
      : ptrs(std::move(ptrs_)), inds(std::move(inds_)) {}
  SparseStructure(const std::vector<int64_t>& ptrs_, const std::vector<int64_t>& inds_)
      : ptrs(ptrs_), inds(inds_) {}

  /// Number of block rows (and columns) in the structure.
  int64_t order() const { return ptrs.size() - 1; }

  /// Sort column indices within each row into ascending order.
  void sortIndices();

  /// Return the transpose of this structure (swap rows and columns).
  SparseStructure transpose() const;

  /**
   * @brief Remove the upper or lower triangle of the structure.
   *
   * @param clearLower If true (default), remove entries where row > col (keep upper).
   *                   If false, remove entries where row < col (keep lower).
   * @return New SparseStructure with the specified triangle cleared.
   */
  SparseStructure clear(bool clearLower = true) const;

  /**
   * @brief Apply a symmetric permutation and extract one triangle.
   *
   * Permutes both rows and columns: row `i` moves to row `mapPerm[i]`.
   * The input may contain only one triangle (upper or lower); the result
   * is the requested half after permutation.
   *
   * @param mapPerm   Permutation vector. `mapPerm[i]` is the new index for row/col `i`.
   * @param lowerHalf If true, extract the lower triangle of the permuted structure.
   * @param sortIndices If true, sort column indices within each row after permutation.
   * @return Permuted structure with only the requested triangle.
   */
  SparseStructure symmetricPermutation(const std::vector<int64_t>& mapPerm, bool lowerHalf = true,
                                       bool sortIndices = true) const;

  /**
   * @brief Add fill from independent elimination of a contiguous block range.
   *
   * For a lower-triangular CSR structure, eliminates rows/columns in [start, end)
   * and adds the resulting fill entries. Used for sparse elimination ranges where
   * blocks are independent and can be eliminated without affecting each other.
   *
   * @param start First row/column index to eliminate (inclusive).
   * @param end   Last row/column index to eliminate (exclusive).
   * @param sortIdx If true, sort column indices after adding fill.
   * @return New structure with fill entries added.
   */
  SparseStructure addIndependentEliminationFill(int64_t start, int64_t end,
                                                bool sortIdx = true) const;

  /// @internal Used by EliminationTree during createSolver().
  /// Users should call createSolver() instead.
  SparseStructure addFullEliminationFill() const;

  /// @internal Fast elimination fill using reach-set algorithm with path compression.
  /// Amortised O(nnz) vs O(nnz * tree_path_length) for addFullEliminationFill().
  /// Users should call createSolver() instead.
  SparseStructure addFullEliminationFillCholmod() const;

  /**
   * @brief Compute a fill-reducing permutation (AMD ordering).
   *
   * @return Permutation vector `perm` where `perm[i]` is the old index
   *         that should move to position `i` in the reordered matrix.
   */
  std::vector<int64_t> fillReducingPermutation() const;

  /**
   * @brief Extract the bottom-right submatrix starting from a given index.
   *
   * Returns the structure of rows/columns [start, order()), with indices
   * shifted so the submatrix starts at index 0.
   *
   * @param start First row/column to include.
   * @return Submatrix structure with shifted indices.
   */
  SparseStructure extractRightBottom(int64_t start);
};

}  // end namespace Sprux
