/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Geometry>
#include <cstdint>
#include <limits>
#include <vector>
#include "sprux/sprux/Accessor.h"
#include "sprux/sprux/CsrTypes.h"

namespace Sprux {

constexpr int64_t kInvalid = -1;

template <typename T>
using MatRMaj = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @brief Skeleton (sparsity pattern + layout) for a coalesced block matrix.
 *
 * ## Terminology
 *
 * - **span**: The basic parameter block grouping. Each span has a size
 *   (number of scalar rows/columns) defined by `spanStart`.
 * - **lump**: A supernode formed by merging consecutive spans. Lumps are
 *   the unit of dense BLAS operations during factorization.
 * - **block**: A `span_rows x span_cols` sub-matrix.
 * - **chain**: A `span_rows x lump_cols` sub-matrix (a column of blocks
 *   within one lump's columns).
 * - **board**: All spans belonging to one lump of rows, times one lump of
 *   columns. Boards are the unit of elimination.
 *
 * Numeric data within a column of chains is stored row-major, so an entire
 * chain column can be passed to BLAS as a single dense matrix.
 *
 * For LU factorization (`matrixType == MTYPE_GENERAL`), upper triangle
 * storage is initialised via initUpperTriangle().
 */
struct CoalescedBlockMatrixSkel {
  /**
   * @brief Construct skeleton from block-level structure.
   *
   * @param spanStart  Span boundary offsets (size = numSpans + 1). `spanStart[i]` is the
   *                   first scalar row/column of span `i`; `spanStart.back()` is the matrix order.
   * @param lumpToSpan Lump-to-span mapping (size = numLumps + 1). Lump `j` contains
   *                   spans `[lumpToSpan[j], lumpToSpan[j+1])`.
   * @param colPtr     Column pointers for chain structure (CSC-like, size = numLumps + 1).
   * @param rowInd     Row span indices for each chain entry.
   */
  CoalescedBlockMatrixSkel(const std::vector<int64_t>& spanStart,
                           const std::vector<int64_t>& lumpToSpan,
                           const std::vector<int64_t>& colPtr, const std::vector<int64_t>& rowInd);

  /**
   * @brief Convert internal sparse storage to a dense matrix.
   *
   * @param dense          Output dense matrix (resized internally).
   * @param data           Numeric data buffer.
   * @param fillUpperHalf  If true, fill both halves (for visualisation). Default: lower only.
   * @param startSpanIndex If nonzero, return only the bottom-right corner starting from
   *                       this span (must be on a supernode boundary).
   */
  template <typename T>
  void densify(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& dense, const T* data,
               bool fillUpperHalf = false, int64_t startSpanIndex = 0) const;

  /// Convenience overload that returns the dense matrix directly.
  template <typename T>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> densify(const std::vector<T>& data,
                                                           bool fillUpperHalf = false) const;

  /**
   * @brief Apply diagonal damping: `diag(data) = alpha * diag(data) + beta`.
   *
   * Useful for Levenberg-Marquardt or trust-region damping.
   *
   * @param data  Numeric data buffer (modified in place).
   * @param alpha Multiplicative factor for existing diagonal.
   * @param beta  Additive constant.
   */
  template <typename T>
  void damp(std::vector<T>& data, T alpha, T beta) const;

  // return number of `spans` (ie basic parameter blocks)
  int64_t numSpans() const { return spanStart.size() - 1; }

  // return number of `lumps` (ie aggregated parameter blocks)
  int64_t numLumps() const { return lumpStart.size() - 1; }

  // order of the matrix (sum of all parameter sizes)
  int64_t order() const { return spanStart.back(); }

  // storage of all matrix numeric data
  int64_t dataSize() const { return chainData.back(); }

  // vector storage start for given parameter
  int64_t spanVectorOffset(int64_t span) const { return spanStart[span]; }

  // matrix storage start for given parameter
  int64_t spanMatrixOffset(int64_t span) const {
    int64_t lump = spanToLump[span];
    SPRUX_CHECK_EQ(spanOffsetInLump[span], 0);
    return chainData[chainColPtr[lump]];
  }

  // return lightweight accessor
  CoalescedAccessor accessor() const {
    CoalescedAccessor retv;
    retv.init(spanStart.data(), spanToLump.data(), lumpStart.data(), spanOffsetInLump.data(),
              chainColPtr.data(), chainRowSpan.data(), chainData.data());
    // Initialize upper triangle pointers if available (for LU factorization)
    if (!upperChainRowPtr.empty()) {
      retv.initUpper(upperChainRowPtr.data(), upperChainColSpan.data(), upperChainData.data());
    }
    return retv;
  }

  std::vector<int64_t> spanStart;         // (with final el)
  std::vector<int64_t> spanToLump;        // (with final el)
  std::vector<int64_t> lumpStart;         // (with final el)
  std::vector<int64_t> lumpToSpan;        // (with final el)
  std::vector<int64_t> spanOffsetInLump;  // (with final el)

  // per-chain data, column-ordered
  std::vector<int64_t> chainColPtr;       // board col data start (with end)
  std::vector<int64_t> chainRowSpan;      // row-span id
  std::vector<int64_t> chainData;         // numeric data offset
  std::vector<int64_t> chainRowsTillEnd;  // num of rows till end

  // per-board data, column-ordered, colums have a final element
  std::vector<int64_t> boardColPtr;       // board col data start (with end)
  std::vector<int64_t> boardRowLump;      // row-lump id (end = invalid)
  std::vector<int64_t> boardChainColOrd;  // chain ord in col (end = #chains)

  // per-board data, row-ordered
  std::vector<int64_t> boardRowPtr;   // board row data start (with end)
  std::vector<int64_t> boardColLump;  // board's col lump
  std::vector<int64_t> boardColOrd;   // board order in col

  // ============ LU factorization support (MTYPE_GENERAL) ============
  // For general (non-symmetric) matrices, we need upper triangle storage.
  // Upper triangle uses CSR-like structure (row-ordered, column indices).

  MatrixType matrixType = MTYPE_SPD;  // Factorization type

  // Upper triangle chain data (for U factor in LU), row-ordered
  // Only populated when matrixType == MTYPE_GENERAL
  std::vector<int64_t> upperChainRowPtr;   // row pointers (CSR-style, with end)
  std::vector<int64_t> upperChainColSpan;  // column span indices
  std::vector<int64_t> upperChainData;     // numeric data offsets

  // Returns true if this skeleton supports general (non-symmetric) matrices
  bool isGeneral() const { return matrixType == MTYPE_GENERAL; }

  // Initialize upper triangle storage for LU factorization.
  // Must be called before factorLU() for multi-block matrices.
  // This derives the upper triangle pattern from the lower triangle
  // (assumes symmetric sparsity pattern).
  void initUpperTriangle();

  // Storage size for upper triangle data (0 for symmetric matrices)
  int64_t upperDataSize() const {
    return upperChainData.empty() ? 0 : upperChainData.back();
  }

  // Total data size (lower + upper for general, just lower for symmetric)
  int64_t totalDataSize() const { return dataSize() + upperDataSize(); }
};

}  // end namespace Sprux
