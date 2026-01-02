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
#include "baspacho/baspacho/Accessor.h"
#include "baspacho/baspacho/CsrTypes.h"

namespace BaSpaCho {

constexpr int64_t kInvalid = -1;

template <typename T>
using MatRMaj = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/*
    Notation for (symmetric) block matrix with coalesced columns:
    Linear data:
    * a `span` is the basic grouping of data (at creation)
    * an `lump` is formed by a few consecutive params
    Block data:
    * a `block` is `span rows` x `span cols`
    * a `chain` is `span rows` x `lump cols`
    * a `board` is (non-empty) and formed by:
      `all the spans belonging to an lump of rows` x `lump cols`

    Note that numeric data in a column of chains are a row-major
    matrix. In this way we can refer to the chain sub-matrix,
    or to the whole set of columns as the chain data are consecutive.
*/
struct CoalescedBlockMatrixSkel {
  CoalescedBlockMatrixSkel(const std::vector<int64_t>& spanStart,
                           const std::vector<int64_t>& lumpToSpan,
                           const std::vector<int64_t>& colPtr, const std::vector<int64_t>& rowInd);

  /* densify the data pointed to by `data`.
     by default only lower half is filled, unless `fillUpperHalf` is true
     if startSpanIndex is set it must be on supernode boundary, and then only the
     bottom right corner will be returned.
   */
  template <typename T>
  void densify(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& dense, const T* data,
               bool fillUpperHalf = false, int64_t startSpanIndex = 0) const;

  /* convenience overload */
  template <typename T>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> densify(const std::vector<T>& data,
                                                           bool fillUpperHalf = false) const;

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
    BASPACHO_CHECK_EQ(spanOffsetInLump[span], 0);
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

}  // end namespace BaSpaCho
