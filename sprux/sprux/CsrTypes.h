/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace Sprux {

/**
 * Matrix mathematical properties - determines factorization algorithm.
 * Modeled after cuDSS cudssMatrixType_t.
 */
enum MatrixType {
  MTYPE_GENERAL,    // General matrix (LDU factorization) - NOT YET SUPPORTED
  MTYPE_SYMMETRIC,  // Symmetric matrix (LDL^T factorization) - NOT YET SUPPORTED
  MTYPE_SPD         // Symmetric positive-definite (Cholesky LL^T factorization)
};

/**
 * Matrix data view - specifies which portion of matrix is provided.
 * For symmetric/SPD matrices, only one triangle needs to be stored.
 * Modeled after cuDSS cudssMatrixViewType_t.
 */
enum MatrixView {
  MVIEW_FULL,   // Full matrix stored (both triangles)
  MVIEW_LOWER,  // Only lower triangle stored (Sprux's native format)
  MVIEW_UPPER   // Only upper triangle stored
};

/**
 * Index base for sparse matrix arrays.
 * Modeled after cuDSS cudssIndexBase_t.
 */
enum IndexBase {
  BASE_ZERO,  // Zero-based indexing (C-style, default)
  BASE_ONE    // One-based indexing (Fortran-style)
};

/**
 * Index type for row/column pointer arrays.
 */
enum IndexType {
  INDEX_INT32,  // 32-bit indices
  INDEX_INT64   // 64-bit indices
};

/**
 * Convert MatrixType to string for error messages.
 */
inline const char* matrixTypeToString(MatrixType mtype) {
  switch (mtype) {
    case MTYPE_GENERAL:
      return "GENERAL";
    case MTYPE_SYMMETRIC:
      return "SYMMETRIC";
    case MTYPE_SPD:
      return "SPD";
    default:
      return "UNKNOWN";
  }
}

/**
 * Convert MatrixView to string for error messages.
 */
inline const char* matrixViewToString(MatrixView mview) {
  switch (mview) {
    case MVIEW_FULL:
      return "FULL";
    case MVIEW_LOWER:
      return "LOWER";
    case MVIEW_UPPER:
      return "UPPER";
    default:
      return "UNKNOWN";
  }
}

/**
 * Validate matrix type/view combination for Sprux.
 * Throws std::invalid_argument if unsupported.
 */
inline void validateMatrixTypeView(MatrixType mtype, MatrixView mview) {
  switch (mtype) {
    case MTYPE_SPD:
      // For SPD, FULL view is redundant (symmetric), we accept but will use lower
      // LOWER and UPPER are both acceptable
      (void)mview;
      break;

    case MTYPE_GENERAL:
      // General matrices require full storage (both L and U)
      if (mview != MVIEW_FULL) {
        throw std::invalid_argument(
            std::string("MTYPE_GENERAL requires MVIEW_FULL, got ") + matrixViewToString(mview));
      }
      break;

    case MTYPE_SYMMETRIC:
      // LDL^T for symmetric indefinite - not yet implemented
      throw std::invalid_argument("MTYPE_SYMMETRIC (LDL^T) not yet implemented");
  }
}

}  // namespace Sprux
