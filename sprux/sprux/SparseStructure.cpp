/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sprux/sprux/SparseStructure.h"

#include <amd.h>

#ifdef SPRUX_HAVE_CHOLMOD
#include <cholmod.h>
#endif

#include <algorithm>
#include "sprux/sprux/DebugMacros.h"
#include "sprux/sprux/Utils.h"

namespace BaSpaCho {

using namespace std;

void SparseStructure::sortIndices() {
  int64_t ord = order();
  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    std::sort(inds.begin() + start, inds.begin() + end);
  }
}

// assumed square matrix
SparseStructure SparseStructure::transpose() const {
  int64_t ord = order();
  SparseStructure retv;
  retv.ptrs.assign(ord + 1, 0);

  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      SPRUX_CHECK_LT(j, ord);
      retv.ptrs[j]++;
    }
  }

  int64_t tot = cumSumVec(retv.ptrs);
  retv.inds.resize(tot);

  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      SPRUX_CHECK_LT(j, ord);
      SPRUX_CHECK_LT(retv.ptrs[j], (int64_t)retv.inds.size());
      retv.inds[retv.ptrs[j]++] = i;
    }
  }

  rewindVec(retv.ptrs);

  return retv;
}

// assumed square matrix
SparseStructure SparseStructure::clear(bool lowerHalf) const {
  int64_t ord = order();
  SparseStructure retv;
  retv.ptrs.assign(ord + 1, 0);

  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      SPRUX_CHECK_LT(j, ord);
      if (i != j && (j > i) == lowerHalf) {
        continue;
      }
      retv.ptrs[i]++;
    }
  }

  int64_t tot = cumSumVec(retv.ptrs);
  retv.inds.resize(tot);

  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      SPRUX_CHECK_LT(j, ord);
      if (i != j && (j > i) == lowerHalf) {
        continue;
      }
      SPRUX_CHECK_LT(retv.ptrs[i], (int64_t)retv.inds.size());
      retv.inds[retv.ptrs[i]++] = j;
    }
  }

  rewindVec(retv.ptrs);

  return retv;
}

SparseStructure SparseStructure::symmetricPermutation(const std::vector<int64_t>& mapPerm,
                                                      bool lowerHalf, bool sortIndices) const {
  int64_t ord = order();
  SPRUX_CHECK_EQ(ord, (int64_t)mapPerm.size());

  SparseStructure retv;
  retv.ptrs.assign(ord + 1, 0);

  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    int64_t newI = mapPerm[i];
    SPRUX_CHECK_LT(newI, ord);
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      SPRUX_CHECK_LT(j, ord);
      int64_t newJ = mapPerm[j];
      SPRUX_CHECK_LT(newJ, ord);
      int64_t col = lowerHalf ? min(newI, newJ) : max(newI, newJ);
      retv.ptrs[col]++;
    }
  }

  int64_t tot = cumSumVec(retv.ptrs);
  retv.inds.resize(tot);

  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    int64_t newI = mapPerm[i];
    SPRUX_CHECK_LT(newI, ord);
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      SPRUX_CHECK_LT(j, ord);
      int64_t newJ = mapPerm[j];
      SPRUX_CHECK_LT(newJ, ord);
      int64_t col = lowerHalf ? min(newI, newJ) : max(newI, newJ);
      int64_t row = lowerHalf ? max(newI, newJ) : min(newI, newJ);
      SPRUX_CHECK_LT(retv.ptrs[col], (int64_t)retv.inds.size());
      retv.inds[retv.ptrs[col]++] = row;
    }
  }

  rewindVec(retv.ptrs);

  if (sortIndices) {
    retv.sortIndices();
  }

  return retv;
}

SparseStructure SparseStructure::addIndependentEliminationFill(int64_t elimStart, int64_t elimEnd,
                                                               bool sortIdx) const {
  int64_t ord = order();

  // nothing to do, no entries are added
  if (elimEnd == ord) {
    return *this;
  }

  SparseStructure tThis = transpose();
  for (int64_t i = elimStart; i < elimEnd; i++) {  // sort subset
    int64_t start = tThis.ptrs[i];
    int64_t end = tThis.ptrs[i + 1];
    std::sort(tThis.inds.begin() + start, tThis.inds.begin() + end);
  }

  SparseStructure retv;
  retv.ptrs.reserve(ptrs.size());
  retv.ptrs.assign(ptrs.begin(), ptrs.begin() + elimEnd + 1);
  retv.inds.assign(inds.begin(), inds.begin() + ptrs[elimEnd]);

  vector<int64_t> tags(ord, -1);  // mark added row entries
  for (int64_t k = elimEnd; k < ord; ++k) {
    int64_t start = ptrs[k];
    int64_t end = ptrs[k + 1];
    tags[k] = k;
    retv.inds.push_back(k);
    for (int64_t q = start; q < end; q++) {
      int64_t i = inds[q];
      if (i >= k) {
        continue;
      }
      if (tags[i] != k) {
        retv.inds.push_back(i); /* L(k,i) is nonzero */
        tags[i] = k;
      }

      // for i in elim lump, walk rows in same column
      if (i >= elimStart && i < elimEnd) {
        int64_t tStart = tThis.ptrs[i];
        int64_t tEnd = tThis.ptrs[i + 1];
        for (int64_t t = tStart; t < tEnd; t++) {
          int64_t w = tThis.inds[t];
          if (w >= k) {
            break;  // tThis rows are sorted
          }
          if (tags[w] < k) {
            tags[w] = k;
            retv.inds.push_back(w); /* L(k,q) is nonzero */
          }
        }
      }
    }
    retv.ptrs.push_back(retv.inds.size());
  }

  if (sortIdx) {
    retv.sortIndices();
  }

  return retv;
}

SparseStructure SparseStructure::addFullEliminationFill() const {
  int64_t ord = order();
  vector<int64_t> tags(ord), parent(ord, -1);

  // skeleton of the algo to iterate on fillup's nodes is from Eigen's
  // `SimplicialCholesky_impl.h` (by Gael Guennebaud),
  // in turn from LDL by Timothy A. Davis.
  SparseStructure retv;
  retv.ptrs.assign(ord + 1, 1);  // write sizes here initially
  for (int64_t k = 0; k < ord; ++k) {
    /* L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
    parent[k] = -1; /* parent of k is not yet known */
    tags[k] = k;    /* mark node k as visited, L(k,k) is nonzero */

    int64_t start = ptrs[k];
    int64_t end = ptrs[k + 1];
    for (int64_t q = start; q < end; q++) {
      int64_t i = inds[q];
      if (i >= k) {
        continue;
      }
      /* follow path from i to root of etree, stop at flagged node */
      for (; tags[i] != k; i = parent[i]) {
        /* find parent of i if not yet determined */
        if (parent[i] == -1) {
          parent[i] = k;
        }

        retv.ptrs[k]++; /* L(k,i) is nonzero */
        tags[i] = k;
      }
    }
  }

  // cumulate-sum ptrs: sizes -> pointers
  int64_t tot = cumSumVec(retv.ptrs);
  retv.inds.resize(tot);

  // walk again, saving entries in rows
  for (int64_t k = 0; k < ord; ++k) {
    /* L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
    parent[k] = -1;                /* parent of k is not yet known */
    tags[k] = k;                   /* mark node k as visited */
    retv.inds[retv.ptrs[k]++] = k; /* L(k,k) is nonzero */

    int64_t start = ptrs[k];
    int64_t end = ptrs[k + 1];
    for (int64_t q = start; q < end; q++) {
      int64_t i = inds[q];
      if (i < k) {
        /* follow path from i to root of etree, stop at flagged node
         */
        for (; tags[i] != k; i = parent[i]) {
          /* find parent of i if not yet determined */
          if (parent[i] == -1) {
            parent[i] = k;
          }
          retv.inds[retv.ptrs[k]++] = i; /* L(k,i) is nonzero */
          tags[i] = k;                   /* mark i as visited */
        }
      }
    }
  }

  rewindVec(retv.ptrs);

  retv.sortIndices();

  return retv;
}

#ifdef SPRUX_HAVE_CHOLMOD

SparseStructure SparseStructure::addFullEliminationFillCholmod() const {
  static_assert(sizeof(SuiteSparse_long) == sizeof(int64_t),
                "CHOLMOD long type must match int64_t");

  int64_t ord = order();
  int64_t nnz = inds.size();

  // The input is a CSR lower-triangular matrix: for row k, inds[ptrs[k]:ptrs[k+1]]
  // are column indices <= k. CSR lower = CSC upper. Pass to CHOLMOD with stype=+1
  // (upper triangle stored in CSC) so CHOLMOD sees the same symmetric matrix as the
  // original addFullEliminationFill() algorithm.
  //
  // CHOLMOD returns L in CSC lower triangle. We then transpose to get CSR lower
  // (matching the original function's output format).
  cholmod_common c;
  cholmod_l_start(&c);

  // RAII-style cleanup to prevent resource leaks if SPRUX_CHECK throws
  auto cleanup = [&](cholmod_factor** Lptr) {
    if (Lptr && *Lptr) {
      cholmod_l_free_factor(Lptr, &c);
    }
    cholmod_l_finish(&c);
  };

  // Don't print anything
  c.print = 0;

  // Force simplicial factorization (no supernodal — we only need the pattern)
  c.supernodal = CHOLMOD_SIMPLICIAL;

  // Force CHOLMOD to use natural ordering (identity permutation).
  // nmethods=1 with CHOLMOD_NATURAL disables fill-reducing reordering.
  // postorder=0 disables elimination tree postordering (which also permutes).
  c.nmethods = 1;
  c.method[0].ordering = CHOLMOD_NATURAL;
  c.postorder = 0;

  // Create dummy numeric values: large diagonal + 1.0 off-diagonal to ensure SPD.
  // We need numeric factorization to populate L->p/L->i (analyze only gives ColCount).
  vector<double> values(nnz);
  for (int64_t k = 0; k < ord; k++) {
    for (int64_t j = ptrs[k]; j < ptrs[k + 1]; j++) {
      values[j] = (inds[j] == k) ? (double)(ord + 1) : 1.0;
    }
  }

  // Create CHOLMOD sparse matrix (views our data directly, no copy).
  // CHOLMOD does not modify the input matrix during analyze/factorize.
  // Our CSR lower triangle data is CSC upper triangle, so use stype=+1.
  cholmod_sparse A;
  memset(&A, 0, sizeof(A));
  A.nrow = ord;
  A.ncol = ord;
  A.nzmax = nnz;
  A.p = const_cast<int64_t*>(ptrs.data());
  A.i = const_cast<int64_t*>(inds.data());
  A.x = values.data();
  A.z = nullptr;
  A.stype = 1;  // upper triangle stored (our CSR lower = CSC upper)
  A.itype = CHOLMOD_LONG;
  A.xtype = CHOLMOD_REAL;
  A.dtype = CHOLMOD_DOUBLE;
  A.sorted = 1;
  A.packed = 1;

  // Symbolic analysis using natural ordering (no additional reordering —
  // the input is already permuted by the fill-reducing permutation).
  cholmod_factor* L = cholmod_l_analyze(&A, &c);
  if (!L) {
    cleanup(&L);
    SPRUX_CHECK_NOTNULL(L);
  }

  // Numeric factorization populates L->p and L->i with the actual fill pattern
  int ok = cholmod_l_factorize(&A, L, &c);
  if (!ok || c.status != CHOLMOD_OK) {
    cleanup(&L);
    SPRUX_CHECK(ok);
    SPRUX_CHECK(c.status == CHOLMOD_OK);
  }

  // For simplicial LDL'/LL', L->p[k] gives start of column k, L->nz[k] gives count.
  // The pattern is stored in L->i with L->nz entries per column (not L->p[k+1]-L->p[k]).
  int64_t* Lp = (int64_t*)L->p;
  int64_t* Li = (int64_t*)L->i;
  int64_t* Lnz = (int64_t*)L->nz;

  // CHOLMOD returns L in CSC lower triangle. Build it, then transpose to get
  // CSR lower triangle (matching the original addFullEliminationFill() format).
  SparseStructure cholmodL;
  cholmodL.ptrs.resize(ord + 1);
  int64_t totalNnz = 0;
  for (int64_t k = 0; k < ord; k++) {
    totalNnz += Lnz[k];
  }
  cholmodL.ptrs[0] = 0;
  for (int64_t k = 0; k < ord; k++) {
    cholmodL.ptrs[k + 1] = cholmodL.ptrs[k] + Lnz[k];
  }
  cholmodL.inds.resize(totalNnz);
  for (int64_t k = 0; k < ord; k++) {
    for (int64_t j = 0; j < Lnz[k]; j++) {
      cholmodL.inds[cholmodL.ptrs[k] + j] = Li[Lp[k] + j];
    }
  }

  cleanup(&L);

  // Transpose CSC lower → CSR lower (matching original output format)
  SparseStructure retv = cholmodL.transpose();

  retv.sortIndices();
  return retv;
}

#else

SparseStructure SparseStructure::addFullEliminationFillCholmod() const {
  throw std::runtime_error(
      "addFullEliminationFillCholmod requires CHOLMOD (build with SPRUX_HAVE_CHOLMOD)");
}

#endif

std::vector<int64_t> SparseStructure::fillReducingPermutation() const {
  std::vector<int64_t> colPtr(ptrs.begin(), ptrs.end()), rowInd(inds.begin(), inds.end());
  std::vector<int64_t> P(colPtr.size() - 1);
  double Control[AMD_CONTROL], Info[AMD_INFO];

  amd_l_defaults(Control);

  int result = amd_l_order(P.size(), colPtr.data(), rowInd.data(), P.data(), Control, Info);
  SPRUX_CHECK_EQ(result, AMD_OK);

  return std::vector<int64_t>(P.begin(), P.end());
}

SparseStructure SparseStructure::extractRightBottom(int64_t startRow) {
  int64_t ord = order();
  SPRUX_CHECK_LE(startRow, ord);
  SPRUX_CHECK_GE(startRow, 0);
  int64_t newOrd = ord - startRow;

  SparseStructure retv;
  retv.ptrs.assign(newOrd + 1, 0);

  for (int64_t i = startRow; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      SPRUX_CHECK_LT(j, ord);
      if (j >= startRow) {
        retv.ptrs[i - startRow]++;
      }
    }
  }

  int64_t tot = cumSumVec(retv.ptrs);
  retv.inds.resize(tot);

  for (int64_t i = 0; i < ord; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      SPRUX_CHECK_LT(j, ord);
      if (j >= startRow) {
        SPRUX_CHECK_LT(retv.ptrs[i - startRow], (int64_t)retv.inds.size());
        retv.inds[retv.ptrs[i - startRow]++] = j - startRow;
      }
    }
  }

  rewindVec(retv.ptrs);
  return retv;
}

}  // end namespace BaSpaCho
