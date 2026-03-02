/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// ============================================================================
// Helper functions (equivalent to MathUtils.h device functions)
// ============================================================================

// Convert linear index to ordered pair (x, y) where 0 <= x <= y < n
// p varies in 0 <= p < n*(n+1)/2
inline int2 toOrderedPair(int64_t n, int64_t p) {
    int64_t odd = n & 1;
    int64_t m = n + 1 - odd;
    int64_t x = p % m;
    int64_t y = n - 1 - (p / m);
    if (x > y) {
        x = x - y - 1;
        y = n - 1 - odd - y;
    }
    return int2(int(x), int(y));
}

// Binary search: find largest i such that array[i] <= needle
inline int64_t bisect(constant int64_t* array, int64_t size, int64_t needle) {
    int64_t a = 0, b = size;
    while (b - a > 1) {
        int64_t mid = (a + b) / 2;
        if (needle >= array[mid]) {
            a = mid;
        } else {
            b = mid;
        }
    }
    return a;
}

// In-place Cholesky decomposition for small blocks
// A is row-major with stride lda
template <typename T>
inline void cholesky(device T* A, int lda, int n) {
    device T* b_ii = A;

    for (int i = 0; i < n; i++) {
        T d = sqrt(*b_ii);
        *b_ii = d;

        device T* b_ji = b_ii + lda;
        for (int j = i + 1; j < n; j++) {
            T c = *b_ji / d;
            *b_ji = c;

            device T* b_ki = b_ii + lda;
            device T* b_jk = b_ji + 1;
            for (int k = i + 1; k <= j; k++) {
                *b_jk -= c * (*b_ki);
                b_ki += lda;
                b_jk += 1;
            }

            b_ji += lda;
        }

        b_ii += lda + 1;
    }
}

// In-place solver for A^T (A built upper-diagonal col-major)
template <typename T>
inline void solveUpperT(constant T* A, int lda, int n, device T* v) {
    constant T* b_ii = A;
    for (int i = 0; i < n; i++) {
        T x = v[i];

        for (int j = 0; j < i; j++) {
            x -= b_ii[j] * v[j];
        }

        v[i] = x / b_ii[i];
        b_ii += lda;
    }
}

// In-place solver for A^T with device pointer for A
template <typename T>
inline void solveUpperT_dev(device T* A, int lda, int n, device T* v) {
    device T* b_ii = A;
    for (int i = 0; i < n; i++) {
        T x = v[i];

        for (int j = 0; j < i; j++) {
            x -= b_ii[j] * v[j];
        }

        v[i] = x / b_ii[i];
        b_ii += lda;
    }
}

// In-place solver for A (A built upper-diagonal col-major)
template <typename T>
inline void solveUpper(constant T* A, int lda, int n, device T* v) {
    constant T* b_ii = A + (lda + 1) * (n - 1);
    for (int i = n - 1; i >= 0; i--) {
        T x = v[i];

        constant T* b_ij = b_ii;
        for (int j = i + 1; j < n; j++) {
            b_ij += lda;
            x -= (*b_ij) * v[j];
        }

        v[i] = x / (*b_ii);
        b_ii -= lda + 1;
    }
}

// In-place solver for A with device pointer
template <typename T>
inline void solveUpper_dev(device T* A, int lda, int n, device T* v) {
    device T* b_ii = A + (lda + 1) * (n - 1);
    for (int i = n - 1; i >= 0; i--) {
        T x = v[i];

        device T* b_ij = b_ii;
        for (int j = i + 1; j < n; j++) {
            b_ij += lda;
            x -= (*b_ij) * v[j];
        }

        v[i] = x / (*b_ii);
        b_ii -= lda + 1;
    }
}

// In-place solver for U*x = b where U is row-major upper triangular
// (as stored by getrf: U is upper triangle of the LU-factored diagonal block)
template <typename T>
inline void solveUpperRowMajor_dev(device T* A, int lda, int n, device T* v) {
  for (int i = n - 1; i >= 0; i--) {
    T x = v[i];
    device T* row = A + i * lda;
    for (int j = i + 1; j < n; j++) {
      x -= row[j] * v[j];
    }
    v[i] = x / row[i];
  }
}

// ============================================================================
// Atomic operations for sparse elimination
// Metal 2.4 lacks native atomic_float, so we use CAS-based emulation
// ============================================================================

// Atomic subtract for float using compare-and-swap
inline void atomicSubFloat(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    do {
        float current = as_type<float>(expected);
        float newVal = current - val;
        desired = as_type<uint>(newVal);
    } while (!atomic_compare_exchange_weak_explicit(addr, &expected, desired,
                                                     memory_order_relaxed,
                                                     memory_order_relaxed));
}

// Atomic add for float using compare-and-swap
inline void atomicAddFloat(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    do {
        float current = as_type<float>(expected);
        float newVal = current + val;
        desired = as_type<uint>(newVal);
    } while (!atomic_compare_exchange_weak_explicit(addr, &expected, desired,
                                                     memory_order_relaxed,
                                                     memory_order_relaxed));
}

// Strided matrix subtraction with atomics: A -= B * C^T
// A is aRows x cRows, B is aRows x bCols, C is cRows x bCols (transposed)
inline void locked_sub_product_float(device float* aMat, int aStride,
                                     device float* bMat, int bRows, int bCols, int bStride,
                                     device float* cMatT, int cRows, int cStride) {
    for (int i = 0; i < bRows; i++) {
        for (int j = 0; j < cRows; j++) {
            float val = 0.0f;
            for (int k = 0; k < bCols; k++) {
                val += bMat[i * bStride + k] * cMatT[j * cStride + k];
            }
            device atomic_uint* addr = (device atomic_uint*)&aMat[i * aStride + j];
            atomicSubFloat(addr, val);
        }
    }
}

// ============================================================================
// Kernel 1: factor_lumps_kernel (Cholesky on diagonal blocks)
// One thread per lump
// ============================================================================
kernel void factor_lumps_kernel_float(
    constant int64_t* lumpStart [[buffer(0)]],
    constant int64_t* chainColPtr [[buffer(1)]],
    constant int64_t* chainData [[buffer(2)]],
    constant int64_t* boardColPtr [[buffer(3)]],
    constant int64_t* boardChainColOrd [[buffer(4)]],
    constant int64_t* chainRowsTillEnd [[buffer(5)]],
    device float* data [[buffer(6)]],
    constant int64_t& lumpIndexStart [[buffer(7)]],
    constant int64_t& lumpIndexEnd [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpSize = lumpStart[lump + 1] - lumpStart[lump];
    int64_t colStart = chainColPtr[lump];
    int64_t dataPtr = chainData[colStart];

    // In-place lower diag Cholesky on diagonal block
    device float* diagBlockPtr = data + dataPtr;
    cholesky(diagBlockPtr, int(lumpSize), int(lumpSize));

    // Below-diagonal solve
    int64_t gatheredStart = boardColPtr[lump];
    int64_t gatheredEnd = boardColPtr[lump + 1];
    int64_t rowDataStart = boardChainColOrd[gatheredStart + 1];
    int64_t rowDataEnd = boardChainColOrd[gatheredEnd - 1];
    int64_t belowDiagStart = chainData[colStart + rowDataStart];
    int64_t numRows = chainRowsTillEnd[colStart + rowDataEnd - 1]
                    - chainRowsTillEnd[colStart + rowDataStart - 1];

    device float* belowDiagBlockPtr = data + belowDiagStart;
    for (int64_t i = 0; i < numRows; i++) {
        solveUpperT_dev(diagBlockPtr, int(lumpSize), int(lumpSize), belowDiagBlockPtr);
        belowDiagBlockPtr += lumpSize;
    }
}

// ============================================================================
// Kernel 2: factor_spans_kernel (Partial block Cholesky)
// One thread per span
// ============================================================================
kernel void factor_spans_kernel_float(
    constant int64_t* spanToLump [[buffer(0)]],
    constant int64_t* spanOffsetInLump [[buffer(1)]],
    constant int64_t* lumpToSpan [[buffer(2)]],
    constant int64_t* spanStart [[buffer(3)]],
    constant int64_t* lumpStart [[buffer(4)]],
    constant int64_t* chainColPtr [[buffer(5)]],
    constant int64_t* chainData [[buffer(6)]],
    constant int64_t* boardColPtr [[buffer(7)]],
    constant int64_t* boardChainColOrd [[buffer(8)]],
    constant int64_t* chainRowsTillEnd [[buffer(9)]],
    device float* data [[buffer(10)]],
    constant int64_t& spanIndexStart [[buffer(11)]],
    constant int64_t& spanIndexEnd [[buffer(12)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t span = spanIndexStart + tid;
    if (span >= spanIndexEnd) {
        return;
    }

    int64_t lump = spanToLump[span];
    int64_t spanOffInLump = spanOffsetInLump[span];
    int64_t lumpSize = lumpStart[lump + 1] - lumpStart[lump];
    int64_t spanSize = spanStart[span + 1] - spanStart[span];
    int64_t colStart = chainColPtr[lump];
    int64_t dataPtr = chainData[colStart];

    // Pointer to start of this span within the diagonal block
    device float* spanDiagPtr = data + dataPtr + spanOffInLump * (lumpSize + 1);
    cholesky(spanDiagPtr, int(lumpSize), int(spanSize));

    // Below-diagonal solve for this span
    int64_t gatheredStart = boardColPtr[lump];
    int64_t gatheredEnd = boardColPtr[lump + 1];
    int64_t rowDataStart = boardChainColOrd[gatheredStart + 1];
    int64_t rowDataEnd = boardChainColOrd[gatheredEnd - 1];
    int64_t belowDiagStart = chainData[colStart + rowDataStart];
    int64_t numRows = chainRowsTillEnd[colStart + rowDataEnd - 1]
                    - chainRowsTillEnd[colStart + rowDataStart - 1];

    device float* belowDiagBlockPtr = data + belowDiagStart + spanOffInLump;
    for (int64_t i = 0; i < numRows; i++) {
        solveUpperT_dev(spanDiagPtr, int(lumpSize), int(spanSize), belowDiagBlockPtr);
        belowDiagBlockPtr += lumpSize;
    }
}

// ============================================================================
// Kernel 3: sparse_elim_straight_kernel (Sparse elimination)
// One thread per block pair
// ============================================================================
kernel void sparse_elim_straight_kernel_float(
    constant int64_t* chainColPtr [[buffer(0)]],
    constant int64_t* lumpStart [[buffer(1)]],
    constant int64_t* chainRowSpan [[buffer(2)]],
    constant int64_t* spanStart [[buffer(3)]],
    constant int64_t* chainData [[buffer(4)]],
    constant int64_t* spanToLump [[buffer(5)]],
    constant int64_t* spanOffsetInLump [[buffer(6)]],
    device float* data [[buffer(7)]],
    constant int64_t& lumpIndexStart [[buffer(8)]],
    constant int64_t& lumpIndexEnd [[buffer(9)]],
    constant int64_t* makeBlockPairEnumStraight [[buffer(10)]],
    constant int64_t& numBlockPairs [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= numBlockPairs) {
        return;
    }

    // Find which lump this block pair belongs to
    int64_t pos = bisect(makeBlockPairEnumStraight, lumpIndexEnd - lumpIndexStart, int64_t(tid));
    int64_t l = lumpIndexStart + pos;

    // Get the number of below-diagonal blocks in this column
    int64_t colStart = chainColPtr[l] + 1;  // skip diagonal
    int64_t colEnd = chainColPtr[l + 1];
    int64_t n = colEnd - colStart;

    // Convert linear index to block pair (di, dj)
    int2 di_dj = toOrderedPair(n, int64_t(tid) - makeBlockPairEnumStraight[pos]);
    int64_t di = di_dj.x;
    int64_t dj = di_dj.y;

    // Get block information
    int64_t lumpSize = lumpStart[l + 1] - lumpStart[l];
    int64_t iSpan = chainRowSpan[colStart + di];
    int64_t jSpan = chainRowSpan[colStart + dj];
    int64_t iSize = spanStart[iSpan + 1] - spanStart[iSpan];
    int64_t jSize = spanStart[jSpan + 1] - spanStart[jSpan];
    int64_t iDataPtr = chainData[colStart + di];
    int64_t jDataPtr = chainData[colStart + dj];

    // Find target block in factored matrix
    int64_t iLump = spanToLump[iSpan];
    int64_t iSpanOff = spanOffsetInLump[iSpan];
    int64_t targetLumpSize = lumpStart[iLump + 1] - lumpStart[iLump];

    // Find target chain entry: bisect chainRowSpan in iLump's chain to find jSpan
    int64_t targetStartPtr = chainColPtr[iLump];
    int64_t targetEndPtr = chainColPtr[iLump + 1];
    int64_t targetPos = bisect(chainRowSpan + targetStartPtr, targetEndPtr - targetStartPtr, jSpan);
    int64_t jiDataPtr = chainData[targetStartPtr + targetPos];

    // Target block pointer: offset by iSpanOff within the lump
    device float* target = data + jiDataPtr + iSpanOff;

    // Perform elimination: target -= srcJ * srcI^T (with atomics)
    device float* srcI = data + iDataPtr;
    device float* srcJ = data + jDataPtr;
    locked_sub_product_float(target, int(targetLumpSize),
                             srcJ, int(jSize), int(lumpSize), int(lumpSize),
                             srcI, int(iSize), int(lumpSize));
}

// ============================================================================
// Kernel: lu_factor_lumps_kernel (LU factorization for 1x1 scalar lumps)
// Divides below-diagonal entries by diagonal (no Cholesky sqrt).
// One thread per lump.
// ============================================================================
kernel void lu_factor_lumps_kernel_float(
    constant int64_t* lumpStart [[buffer(0)]],
    constant int64_t* chainColPtr [[buffer(1)]],
    constant int64_t* chainData [[buffer(2)]],
    constant int64_t* boardColPtr [[buffer(3)]],
    constant int64_t* boardChainColOrd [[buffer(4)]],
    constant int64_t* chainRowsTillEnd [[buffer(5)]],
    device float* data [[buffer(6)]],
    constant int64_t& lumpIndexStart [[buffer(7)]],
    constant int64_t& lumpIndexEnd [[buffer(8)]],
    constant float& staticPivotThreshold [[buffer(9)]],
    device atomic_uint* perturbCount [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) return;

    int64_t lumpSize = lumpStart[lump + 1] - lumpStart[lump];
    int64_t colStart = chainColPtr[lump];
    int64_t dataPtr = chainData[colStart];

    // Read diagonal value
    device float* diagPtr = data + dataPtr;
    float diag = *diagPtr;

    // Static pivoting: perturb near-zero or non-finite diagonals
    if (staticPivotThreshold >= 0.0f) {
        if (!isfinite(diag) || abs(diag) < staticPivotThreshold) {
            diag = (diag >= 0.0f) ? staticPivotThreshold : -staticPivotThreshold;
            *diagPtr = diag;
            atomic_fetch_add_explicit(perturbCount, 1u, memory_order_relaxed);
        }
    }

    // Divide below-diagonal entries by diagonal to get L column
    int64_t gatheredStart = boardColPtr[lump];
    int64_t gatheredEnd = boardColPtr[lump + 1];
    int64_t rowDataStart = boardChainColOrd[gatheredStart + 1];
    int64_t rowDataEnd = boardChainColOrd[gatheredEnd - 1];
    int64_t belowDiagStart = chainData[colStart + rowDataStart];
    int64_t numRows = chainRowsTillEnd[colStart + rowDataEnd - 1]
                    - chainRowsTillEnd[colStart + rowDataStart - 1];

    device float* belowDiagPtr = data + belowDiagStart;
    float invDiag = 1.0f / diag;
    for (int64_t i = 0; i < numRows * lumpSize; i++) {
        belowDiagPtr[i] *= invDiag;
    }
}

// ============================================================================
// Kernel: lu_sparse_elim_kernel (LU Schur complement for scalar lumps)
// Updates both lower and upper triangle: target -= L[a,k] * U[k,b]
// One thread per (L_row, U_col) pair. n^2 pairs per lump.
// ============================================================================
kernel void lu_sparse_elim_kernel_float(
    constant int64_t* chainColPtr [[buffer(0)]],
    constant int64_t* lumpStart [[buffer(1)]],
    constant int64_t* chainRowSpan [[buffer(2)]],
    constant int64_t* spanStart [[buffer(3)]],
    constant int64_t* chainData [[buffer(4)]],
    constant int64_t* spanToLump [[buffer(5)]],
    constant int64_t* spanOffsetInLump [[buffer(6)]],
    device float* data [[buffer(7)]],
    constant int64_t& lumpIndexStart [[buffer(8)]],
    constant int64_t& lumpIndexEnd [[buffer(9)]],
    constant int64_t* blockPairEnum [[buffer(10)]],
    constant int64_t& numBlockPairs [[buffer(11)]],
    constant int64_t* upperChainRowPtr [[buffer(12)]],
    constant int64_t* upperChainColSpan [[buffer(13)]],
    constant int64_t* upperChainData [[buffer(14)]],
    constant int64_t& upperDataBase [[buffer(15)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= numBlockPairs) return;

    // Find which lump this pair belongs to
    int64_t pos = bisect(blockPairEnum, lumpIndexEnd - lumpIndexStart, int64_t(tid));
    int64_t l = lumpIndexStart + pos;

    // Number of below-diagonal chain entries
    int64_t colStart = chainColPtr[l] + 1;  // skip diagonal
    int64_t colEnd = chainColPtr[l + 1];
    int64_t n = colEnd - colStart;

    // Map linear pair index to (row_idx, col_idx) in [0,n) x [0,n)
    int64_t localPair = int64_t(tid) - blockPairEnum[pos];
    int64_t row_idx = localPair / n;
    int64_t col_idx = localPair % n;

    // Get spans for this pair
    int64_t aSpan = chainRowSpan[colStart + row_idx];  // row (L entry)
    int64_t bSpan = chainRowSpan[colStart + col_idx];  // col (U entry)

    // L value from lower chain
    float L_val = data[chainData[colStart + row_idx]];

    // U value from upper chain
    int64_t uRowStart = upperChainRowPtr[l];
    float U_val = data[upperDataBase + upperChainData[uRowStart + col_idx]];

    float product = L_val * U_val;

    if (aSpan >= bSpan) {
        // Target in lower triangle at (row=aSpan, col=bSpan)
        // Find in column chain of bSpan's lump
        int64_t bLump = spanToLump[bSpan];
        int64_t bSpanOff = spanOffsetInLump[bSpan];
        int64_t targetStartPtr = chainColPtr[bLump];
        int64_t targetEndPtr = chainColPtr[bLump + 1];
        int64_t targetPos = bisect(chainRowSpan + targetStartPtr,
                                   targetEndPtr - targetStartPtr, aSpan);
        int64_t targetDataPtr = chainData[targetStartPtr + targetPos];

        device atomic_uint* addr =
            (device atomic_uint*)&data[targetDataPtr + bSpanOff];
        atomicSubFloat(addr, product);
    } else {
        // Target in upper triangle at (row=aSpan, col=bSpan)
        // Find in upper chain of aSpan's lump
        int64_t aLump = spanToLump[aSpan];
        int64_t targetURowStart = upperChainRowPtr[aLump];
        int64_t targetURowEnd = upperChainRowPtr[aLump + 1];
        int64_t targetPos = bisect(upperChainColSpan + targetURowStart,
                                   targetURowEnd - targetURowStart, bSpan);
        int64_t targetDataPtr = upperDataBase +
            upperChainData[targetURowStart + targetPos];

        device atomic_uint* addr = (device atomic_uint*)&data[targetDataPtr];
        atomicSubFloat(addr, product);
    }
}

// ============================================================================
// Kernel 4: assemble_kernel (Assemble rectangular sections)
// ============================================================================
kernel void assemble_kernel_float(
    constant int64_t& numBlockRows [[buffer(0)]],
    constant int64_t& numBlockCols [[buffer(1)]],
    constant int64_t& startRow [[buffer(2)]],
    constant int64_t& srcRectWidth [[buffer(3)]],
    constant int64_t& dstStride [[buffer(4)]],
    constant int64_t* pChainRowsTillEnd [[buffer(5)]],
    constant int64_t* pToSpan [[buffer(6)]],
    constant int64_t* pSpanToChainOffset [[buffer(7)]],
    constant int64_t* pSpanOffsetInLump [[buffer(8)]],
    constant float* matRectPtr [[buffer(9)]],
    device float* data [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= numBlockRows * numBlockCols) {
        return;
    }

    int64_t r = tid % numBlockRows;
    int64_t c = tid / numBlockRows;

    // Only process lower triangle
    if (c > r) {
        return;
    }

    // CPU ref: rBegin = pChainRowsTillEnd[r - 1] - startRow
    //          rSize = pChainRowsTillEnd[r] - startRow - rBegin
    // where startRow = pChainRowsTillEnd[-1] (element before chain start)
    // Note: When r=0, pChainRowsTillEnd[-1] - startRow = startRow - startRow = 0
    // Handle r=0 explicitly to avoid negative indexing
    int64_t rBegin = (r > 0) ? (pChainRowsTillEnd[r - 1] - startRow) : 0;
    int64_t rEnd = pChainRowsTillEnd[r] - startRow;
    int64_t rSize = rEnd - rBegin;
    int64_t rParam = pToSpan[r];
    int64_t rOffset = pSpanToChainOffset[rParam];

    // Handle c=0 explicitly to avoid negative indexing
    int64_t cStart = (c > 0) ? (pChainRowsTillEnd[c - 1] - startRow) : 0;
    int64_t cEnd = pChainRowsTillEnd[c] - startRow;
    int64_t cSize = cEnd - cStart;
    int64_t offset = rOffset + pSpanOffsetInLump[pToSpan[c]];

    // Source pointer in temporary rectangle (row-major layout)
    constant float* matRowPtr = matRectPtr + rBegin * srcRectWidth;
    constant float* src = matRowPtr + cStart;

    // Destination pointer in block matrix (row-major layout)
    device float* dst = data + offset;

    // Subtract source from destination (stridedMatSub)
    for (int64_t i = 0; i < rSize; i++) {
        for (int64_t j = 0; j < cSize; j++) {
            device atomic_uint* addr = (device atomic_uint*)&dst[i * dstStride + j];
            atomicSubFloat(addr, src[i * srcRectWidth + j]);
        }
    }
}

// ============================================================================
// Kernel 5: assembleVec_kernel (Vector assembly during solve)
// ============================================================================
kernel void assembleVec_kernel_float(
    constant int64_t* chainRowsTillEnd [[buffer(0)]],
    constant int64_t* toSpan [[buffer(1)]],
    constant int64_t* spanStarts [[buffer(2)]],
    constant float* A [[buffer(3)]],
    constant int64_t& numColItems [[buffer(4)]],
    device float* C [[buffer(5)]],
    constant int64_t& ldc [[buffer(6)]],
    constant int64_t& nRHS [[buffer(7)]],
    constant int64_t& startRow [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= numColItems) {
        return;
    }

    // CPU ref: rowOffset = chainRowsTillEnd[i - 1] - startRow
    // where startRow = chainRowsTillEnd[-1] (element before chain start)
    int64_t rowsBefore = (tid > 0) ? (chainRowsTillEnd[tid - 1] - startRow) : 0;
    int64_t rowsAfter = chainRowsTillEnd[tid] - startRow;
    int64_t blockRows = rowsAfter - rowsBefore;

    int64_t span = toSpan[tid];
    int64_t spanStart = spanStarts[span];

    // A (temp buffer) is row-major with stride nRHS
    // C (output vector) is column-major with stride ldc
    // stridedTransAdd: dst += src (transposes from row-major to col-major)
    constant float* srcPtr = A + rowsBefore * nRHS;
    device float* dstPtr = C + spanStart;

    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        for (int64_t i = 0; i < blockRows; i++) {
            dstPtr[i + rhs * ldc] += srcPtr[i * nRHS + rhs];
        }
    }
}

// ============================================================================
// Kernel 6: assembleVecT_kernel (Transposed vector assembly)
// ============================================================================
kernel void assembleVecT_kernel_float(
    constant int64_t* chainRowsTillEnd [[buffer(0)]],
    constant int64_t* toSpan [[buffer(1)]],
    constant int64_t* spanStarts [[buffer(2)]],
    constant float* C [[buffer(3)]],
    constant int64_t& numColItems [[buffer(4)]],
    device float* A [[buffer(5)]],
    constant int64_t& ldc [[buffer(6)]],
    constant int64_t& nRHS [[buffer(7)]],
    constant int64_t& startRow [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= numColItems) {
        return;
    }

    // CPU ref: rowOffset = chainRowsTillEnd[i - 1] - startRow
    // where startRow = chainRowsTillEnd[-1] (element before chain start)
    int64_t rowsBefore = (tid > 0) ? (chainRowsTillEnd[tid - 1] - startRow) : 0;
    int64_t rowsAfter = chainRowsTillEnd[tid] - startRow;
    int64_t blockRows = rowsAfter - rowsBefore;

    int64_t span = toSpan[tid];
    int64_t spanStart = spanStarts[span];

    // A (temp buffer) is row-major with stride nRHS
    // C (input vector) is column-major with stride ldc
    device float* dstPtr = A + rowsBefore * nRHS;
    constant float* srcPtr = C + spanStart;

    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        for (int64_t i = 0; i < blockRows; i++) {
            dstPtr[i * nRHS + rhs] = srcPtr[i + rhs * ldc];
        }
    }
}

// ============================================================================
// Solve kernels: sparseElim_diagSolveL
// ============================================================================
kernel void sparseElim_diagSolveL_float(
    constant int64_t* lumpStarts [[buffer(0)]],
    constant int64_t* chainColPtr [[buffer(1)]],
    constant int64_t* chainData [[buffer(2)]],
    device float* data [[buffer(3)]],
    device float* v [[buffer(4)]],
    constant int64_t& ldc [[buffer(5)]],
    constant int64_t& nRHS [[buffer(6)]],
    constant int64_t& lumpIndexStart [[buffer(7)]],
    constant int64_t& lumpIndexEnd [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpStart = lumpStarts[lump];
    int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
    int64_t colStart = chainColPtr[lump];
    int64_t diagDataPtr = chainData[colStart];

    device float* diagBlock = data + diagDataPtr;

    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        solveUpperT_dev(diagBlock, int(lumpSize), int(lumpSize), v + lumpStart + ldc * rhs);
    }
}

// ============================================================================
// Solve kernels: sparseElim_diagSolveLt
// ============================================================================
kernel void sparseElim_diagSolveLt_float(
    constant int64_t* lumpStarts [[buffer(0)]],
    constant int64_t* chainColPtr [[buffer(1)]],
    constant int64_t* chainData [[buffer(2)]],
    device float* data [[buffer(3)]],
    device float* v [[buffer(4)]],
    constant int64_t& ldc [[buffer(5)]],
    constant int64_t& nRHS [[buffer(6)]],
    constant int64_t& lumpIndexStart [[buffer(7)]],
    constant int64_t& lumpIndexEnd [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpStart = lumpStarts[lump];
    int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
    int64_t colStart = chainColPtr[lump];
    int64_t diagDataPtr = chainData[colStart];

    device float* diagBlock = data + diagDataPtr;

    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        solveUpper_dev(diagBlock, int(lumpSize), int(lumpSize), v + lumpStart + ldc * rhs);
    }
}

// ============================================================================
// LU factorization helper functions
// ============================================================================

// In-place LU factorization with partial pivoting for small blocks
// A is row-major with stride lda, pivots output array (0-based swap indices)
template <typename T>
inline void lu_factor(device T* A, int lda, int n, device int64_t* pivots) {
    for (int i = 0; i < n; i++) {
        // Find pivot (max abs in column i, rows i..n-1)
        int maxRow = i;
        T maxVal = abs(A[i * lda + i]);
        for (int k = i + 1; k < n; k++) {
            T val = abs(A[k * lda + i]);
            if (val > maxVal) {
                maxVal = val;
                maxRow = k;
            }
        }
        pivots[i] = maxRow;

        // Swap rows i and maxRow
        if (maxRow != i) {
            for (int j = 0; j < n; j++) {
                T tmp = A[i * lda + j];
                A[i * lda + j] = A[maxRow * lda + j];
                A[maxRow * lda + j] = tmp;
            }
        }

        // Eliminate below diagonal
        T diag = A[i * lda + i];
        for (int k = i + 1; k < n; k++) {
            A[k * lda + i] /= diag;
            for (int j = i + 1; j < n; j++) {
                A[k * lda + j] -= A[k * lda + i] * A[i * lda + j];
            }
        }
    }
}

// Forward substitution with unit lower triangular matrix (row-major)
// Solves L * x = b in-place where L has unit diagonal
template <typename T>
inline void solveLowerUnit_rm(device T* L, int ldl, int n, device T* v) {
    for (int i = 0; i < n; i++) {
        T x = v[i];
        for (int j = 0; j < i; j++) {
            x -= L[i * ldl + j] * v[j];
        }
        v[i] = x;  // Unit diagonal, no division
    }
}

// Backward substitution for upper triangular matrix (row-major)
// Solves U * x = b in-place
template <typename T>
inline void solveUpperRM(device T* U, int ldu, int n, device T* v) {
    for (int i = n - 1; i >= 0; i--) {
        T x = v[i];
        for (int j = i + 1; j < n; j++) {
            x -= U[i * ldu + j] * v[j];
        }
        v[i] = x / U[i * ldu + i];
    }
}

// Solve L * X = B where L is unit lower triangular (row-major), B is m×n col-major with stride ldb
// This is "left side" triangular solve for multiple RHS
template <typename T>
inline void solveLowerUnit_rm_colmaj(device T* L, int ldl, int m,
                                      device T* B, int ldb, int nRHS) {
    for (int rhs = 0; rhs < nRHS; rhs++) {
        solveLowerUnit_rm(L, ldl, m, B + rhs * ldb);
    }
}

// Solve U * X = B where U is upper triangular (row-major), B is m×n col-major with stride ldb
template <typename T>
inline void solveUpperRM_colmaj(device T* U, int ldu, int m,
                                 device T* B, int ldb, int nRHS) {
    for (int rhs = 0; rhs < nRHS; rhs++) {
        solveUpperRM(U, ldu, m, B + rhs * ldb);
    }
}

// Solve L * X = B where L is m×m unit lower triangular (row-major)
// B is m×n row-major with stride ldb
template <typename T>
inline void trsmLowerUnit_rm(device T* L, int ldl, int m,
                              device T* B, int ldb, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            T val = B[i * ldb + j];
            for (int k = 0; k < i; k++) {
                val -= L[i * ldl + k] * B[k * ldb + j];
            }
            B[i * ldb + j] = val;
        }
    }
}

// Solve X * U = B where U is n×n upper triangular (row-major)
// B is m×n row-major with stride ldb
template <typename T>
inline void trsmUpperRight_rm(device T* U, int ldu, int n,
                               device T* B, int ldb, int m) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            T val = B[i * ldb + j];
            for (int k = 0; k < j; k++) {
                val -= B[i * ldb + k] * U[k * ldu + j];
            }
            B[i * ldb + j] = val / U[j * ldu + j];
        }
    }
}

// ============================================================================
// LU Factorization Kernels
// ============================================================================

// LU factorize diagonal block + apply pivots + TRSM for below-diagonal blocks
// One thread per lump (matches factor_lumps_kernel_float pattern)
kernel void lu_factor_lump_kernel_float(
    constant int64_t* lumpStart [[buffer(0)]],
    constant int64_t* chainColPtr [[buffer(1)]],
    constant int64_t* chainData [[buffer(2)]],
    constant int64_t* boardColPtr [[buffer(3)]],
    constant int64_t* boardChainColOrd [[buffer(4)]],
    constant int64_t* chainRowsTillEnd [[buffer(5)]],
    device float* data [[buffer(6)]],
    device int64_t* pivots [[buffer(7)]],
    constant int64_t* pivotOffsets [[buffer(8)]],
    constant int64_t& lumpIndexStart [[buffer(9)]],
    constant int64_t& lumpIndexEnd [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpSize = lumpStart[lump + 1] - lumpStart[lump];
    int64_t colStart = chainColPtr[lump];
    int64_t dataPtr = chainData[colStart];

    // Step 1: In-place LU factorization on diagonal block
    device float* diagBlockPtr = data + dataPtr;
    device int64_t* lumpPivots = pivots + pivotOffsets[lump - lumpIndexStart];
    lu_factor(diagBlockPtr, int(lumpSize), int(lumpSize), lumpPivots);

    // Step 2: Get below-diagonal block info
    int64_t gatheredStart = boardColPtr[lump];
    int64_t gatheredEnd = boardColPtr[lump + 1];
    int64_t rowDataStart = boardChainColOrd[gatheredStart + 1];
    int64_t rowDataEnd = boardChainColOrd[gatheredEnd - 1];
    int64_t belowDiagStart = chainData[colStart + rowDataStart];
    int64_t numRows = chainRowsTillEnd[colStart + rowDataEnd - 1]
                    - chainRowsTillEnd[colStart + rowDataStart - 1];

    if (numRows <= 0) return;

    device float* belowDiagBlockPtr = data + belowDiagStart;

    // Step 3: Apply row permutation to below-diagonal block (column-major with stride lumpSize)
    for (int64_t i = 0; i < lumpSize; i++) {
        int64_t swapRow = lumpPivots[i];
        if (swapRow != i) {
            // Swap rows i and swapRow in the below-diagonal block
            // Below-diagonal is stored as numRows rows × lumpSize cols (row-major)
            // But we need to apply the permutation to the "column" block below the diagonal
            // which is stored column-major from the perspective of the LU
            for (int64_t r = 0; r < numRows; r++) {
                float tmp = belowDiagBlockPtr[r * lumpSize + i];
                belowDiagBlockPtr[r * lumpSize + i] = belowDiagBlockPtr[r * lumpSize + swapRow];
                belowDiagBlockPtr[r * lumpSize + swapRow] = tmp;
            }
        }
    }

    // Step 4: Solve L * X = B for below-diagonal rows (X * U = B for right columns)
    // Below-diagonal block: numRows × lumpSize (row-major, stride = lumpSize)
    // U is the upper triangle of diagBlockPtr (lumpSize × lumpSize, row-major)
    // We need: belowDiag = belowDiag * U^{-1}
    trsmUpperRight_rm(diagBlockPtr, int(lumpSize), int(lumpSize),
                      belowDiagBlockPtr, int(lumpSize), int(numRows));
}

// Apply row permutation to factored matrix columns (for the block above diagonal in LU)
// pivots[i] indicates row i should be swapped with row pivots[i]
// Data is column-major with stride ld
// Parallelized across columns: each thread handles a subset of columns.
// Must dispatch as a SINGLE threadgroup (threadgroup_barrier only syncs within one group).
kernel void lu_applyRowPerm_kernel_float(
    device int64_t* pivots [[buffer(0)]],
    constant int64_t& n [[buffer(1)]],
    device float* data [[buffer(2)]],
    constant int64_t& offData [[buffer(3)]],
    constant int64_t& ld [[buffer(4)]],
    constant int64_t& numCols [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]])
{
    device float* d = data + offData;
    for (int64_t i = 0; i < n; i++) {
        int64_t swapRow = pivots[i];
        if (swapRow != i) {
            for (int64_t c = tid; c < numCols; c += tcount) {
                float tmp = d[i + c * ld];
                d[i + c * ld] = d[swapRow + c * ld];
                d[swapRow + c * ld] = tmp;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// TRSM: Solve L * X = B where L is m×m unit lower triangular (row-major)
// B is m×n row-major with stride ldb
kernel void lu_trsmLowerUnit_kernel_float(
    constant float* L [[buffer(0)]],
    constant int64_t& offL [[buffer(1)]],
    device float* B [[buffer(2)]],
    constant int64_t& offB [[buffer(3)]],
    constant int64_t& m [[buffer(4)]],
    constant int64_t& n [[buffer(5)]],
    constant int64_t& ldb [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    // Single-threaded (sequential data dependencies)
    if (tid != 0) return;

    device float* Bp = B + offB;
    // Copy L from constant to work with - we need device pointer for template
    // Actually L is in the same data buffer, just const. Use direct access.
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float val = Bp[i * ldb + j];
            for (int64_t k = 0; k < i; k++) {
                val -= L[offL + i * m + k] * Bp[k * ldb + j];
            }
            Bp[i * ldb + j] = val;
        }
    }
}

// TRSM: Solve X * U = B where U is n×n upper triangular (row-major)
// B is m×n row-major with stride ldb
kernel void lu_trsmUpperRight_kernel_float(
    constant float* U [[buffer(0)]],
    constant int64_t& offU [[buffer(1)]],
    device float* B [[buffer(2)]],
    constant int64_t& offB [[buffer(3)]],
    constant int64_t& m [[buffer(4)]],
    constant int64_t& n [[buffer(5)]],
    constant int64_t& ldb [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;

    device float* Bp = B + offB;
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i < m; i++) {
            float val = Bp[i * ldb + j];
            for (int64_t k = 0; k < j; k++) {
                val -= Bp[i * ldb + k] * U[offU + k * n + j];
            }
            Bp[i * ldb + j] = val / U[offU + j * n + j];
        }
    }
}

// Work item for batched saveGemm: one thread computes one full C -= L * U block
struct LUGemmWorkItem {
    int64_t offL, ldL;   // L block element offset and row stride
    int64_t offU, ldU;   // U block element offset and row stride
    int64_t offC, ldC;   // C block element offset and row stride
    int64_t m, n, k;     // rows of C, cols of C, inner dimension
};

// Batched saveGemm: one thread per work item, each computing a full m×n GEMM
kernel void lu_batchedSaveGemm_kernel_float(
    device float* data [[buffer(0)]],
    constant LUGemmWorkItem* workItems [[buffer(1)]],
    constant int64_t& workCount [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= workCount) return;

    LUGemmWorkItem item = workItems[tid];
    for (int64_t row = 0; row < item.m; row++) {
        for (int64_t col = 0; col < item.n; col++) {
            float sum = 0.0f;
            for (int64_t p = 0; p < item.k; p++) {
                sum += data[item.offL + row * item.ldL + p]
                     * data[item.offU + p * item.ldU + col];
            }
            device atomic_uint* addr =
                (device atomic_uint*)&data[item.offC + row * item.ldC + col];
            atomicSubFloat(addr, sum);
        }
    }
}

// saveGemm: C -= L * U (all row-major with strides)
kernel void lu_saveGemm_kernel_float(
    constant float* L [[buffer(0)]],
    constant int64_t& offL [[buffer(1)]],
    constant int64_t& ldL [[buffer(2)]],
    constant float* U [[buffer(3)]],
    constant int64_t& offU [[buffer(4)]],
    constant int64_t& ldU [[buffer(5)]],
    device float* C [[buffer(6)]],
    constant int64_t& offC [[buffer(7)]],
    constant int64_t& ldC [[buffer(8)]],
    constant int64_t& m [[buffer(9)]],
    constant int64_t& n [[buffer(10)]],
    constant int64_t& k [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t totalElements = m * n;
    if (int64_t(tid) >= totalElements) return;

    int64_t row = tid / n;
    int64_t col = tid % n;

    float sum = 0.0f;
    for (int64_t p = 0; p < k; p++) {
        sum += L[offL + row * ldL + p] * U[offU + p * ldU + col];
    }

    // Use atomic subtract for thread safety
    device atomic_uint* addr = (device atomic_uint*)&C[offC + row * ldC + col];
    atomicSubFloat(addr, sum);
}

// ============================================================================
// LU Solve Kernels
// ============================================================================

// Apply forward row permutation to solve vector
// For each i from 0..n-1: swap vec[i] and vec[pivots[i]] across all RHS
kernel void lu_applyRowPermVec_kernel_float(
    constant int64_t* pivots [[buffer(0)]],
    constant int64_t& n [[buffer(1)]],
    device float* vec [[buffer(2)]],
    constant int64_t& ldVec [[buffer(3)]],
    constant int64_t& nRHS [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;

    for (int64_t i = 0; i < n; i++) {
        int64_t swapRow = pivots[i];
        if (swapRow != i) {
            for (int64_t rhs = 0; rhs < nRHS; rhs++) {
                float tmp = vec[i + rhs * ldVec];
                vec[i + rhs * ldVec] = vec[swapRow + rhs * ldVec];
                vec[swapRow + rhs * ldVec] = tmp;
            }
        }
    }
}

// Apply inverse row permutation (reverse order) to solve vector
kernel void lu_applyRowPermVecInv_kernel_float(
    constant int64_t* pivots [[buffer(0)]],
    constant int64_t& n [[buffer(1)]],
    device float* vec [[buffer(2)]],
    constant int64_t& ldVec [[buffer(3)]],
    constant int64_t& nRHS [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;

    for (int64_t i = n - 1; i >= 0; i--) {
        int64_t swapRow = pivots[i];
        if (swapRow != i) {
            for (int64_t rhs = 0; rhs < nRHS; rhs++) {
                float tmp = vec[i + rhs * ldVec];
                vec[i + rhs * ldVec] = vec[swapRow + rhs * ldVec];
                vec[swapRow + rhs * ldVec] = tmp;
            }
        }
    }
}

// Solve L * x = b where L is unit lower triangular (row-major), x is col-major
// One thread per lump
kernel void lu_solveLUnit_kernel_float(
    constant int64_t* lumpStarts [[buffer(0)]],
    constant int64_t* chainColPtr [[buffer(1)]],
    constant int64_t* chainData [[buffer(2)]],
    device float* data [[buffer(3)]],
    device float* v [[buffer(4)]],
    constant int64_t& ldc [[buffer(5)]],
    constant int64_t& nRHS [[buffer(6)]],
    constant int64_t& lumpIndexStart [[buffer(7)]],
    constant int64_t& lumpIndexEnd [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpStart = lumpStarts[lump];
    int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
    int64_t colStart = chainColPtr[lump];
    int64_t diagDataPtr = chainData[colStart];

    device float* diagBlock = data + diagDataPtr;

    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        solveLowerUnit_rm(diagBlock, int(lumpSize), int(lumpSize), v + lumpStart + ldc * rhs);
    }
}

// Solve U * x = b where U is upper triangular (row-major), x is col-major
// One thread per lump
kernel void lu_solveU_kernel_float(
    constant int64_t* lumpStarts [[buffer(0)]],
    constant int64_t* chainColPtr [[buffer(1)]],
    constant int64_t* chainData [[buffer(2)]],
    device float* data [[buffer(3)]],
    device float* v [[buffer(4)]],
    constant int64_t& ldc [[buffer(5)]],
    constant int64_t& nRHS [[buffer(6)]],
    constant int64_t& lumpIndexStart [[buffer(7)]],
    constant int64_t& lumpIndexEnd [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpStart = lumpStarts[lump];
    int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
    int64_t colStart = chainColPtr[lump];
    int64_t diagDataPtr = chainData[colStart];

    device float* diagBlock = data + diagDataPtr;

    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        solveUpperRM(diagBlock, int(lumpSize), int(lumpSize), v + lumpStart + ldc * rhs);
    }
}

// gemvDirect: result += alpha * M * x where M is row-major (nRows × nCols)
// x is col-major at vec+srcOff with stride ldVec, result at vec+dstOff
kernel void lu_gemvDirect_kernel_float(
    constant float* data [[buffer(0)]],
    constant int64_t& offset [[buffer(1)]],
    constant int64_t& nRows [[buffer(2)]],
    constant int64_t& nCols [[buffer(3)]],
    device float* vec [[buffer(4)]],
    constant int64_t& srcOff [[buffer(5)]],
    constant int64_t& dstOff [[buffer(6)]],
    constant int64_t& ldVec [[buffer(7)]],
    constant float& alpha [[buffer(8)]],
    constant int64_t& nRHS [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    // One thread per output row
    if (int64_t(tid) >= nRows) return;

    int64_t row = tid;
    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        float sum = 0.0f;
        for (int64_t col = 0; col < nCols; col++) {
            sum += data[offset + row * nCols + col] * vec[srcOff + col + rhs * ldVec];
        }
        vec[dstOff + row + rhs * ldVec] += alpha * sum;
    }
}

// LU getrf kernel: standalone (for when not using fused lump kernel)
// In-place LU with partial pivoting on a single block
kernel void lu_getrf_kernel_float(
    device float* data [[buffer(0)]],
    constant int64_t& offA [[buffer(1)]],
    constant int64_t& m [[buffer(2)]],
    constant int64_t& n [[buffer(3)]],
    device int64_t* pivots [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;

    int64_t minMN = m < n ? m : n;
    device float* A = data + offA;
    lu_factor(A, int(n), int(minMN), pivots);
}

// Convert MPS uint32_t pivots to int64_t (MPS outputs 0-based uint32, BaSpaCho uses int64)
kernel void lu_convertPivots_kernel_float(
    device const uint32_t* src [[buffer(0)]],
    device int64_t* dst [[buffer(1)]],
    constant int64_t& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < uint(count)) {
        dst[tid] = int64_t(src[tid]);
    }
}

// GPU-side perturbSmallDiagonals: scan diagonal elements after getrf and perturb
// near-zero or non-finite values. Data is row-major with stride 'stride'.
// perturbCount is an atomic counter incremented for each perturbed diagonal.
kernel void lu_perturbDiag_kernel_float(
    device float* data [[buffer(0)]],
    constant int64_t& offset [[buffer(1)]],
    constant int64_t& stride [[buffer(2)]],
    constant int64_t& n [[buffer(3)]],
    constant float& threshold [[buffer(4)]],
    device atomic_uint* perturbCount [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uint(n)) return;
    int64_t idx = offset + int64_t(tid) * stride + int64_t(tid);
    float diag = data[idx];
    if (!isfinite(diag) || abs(diag) < threshold) {
        data[idx] = (diag >= 0.0f) ? threshold : -threshold;
        atomic_fetch_add_explicit(perturbCount, 1u, memory_order_relaxed);
    }
}

// ============================================================================
// Direct-offset LU solve kernels (for per-lump calls with explicit offsets)
// ============================================================================

// Solve L * x = b where L is unit lower triangular (row-major at data+offM, n×n)
// x is col-major at C+offC with stride ldc
kernel void lu_solveLUnit_direct_kernel_float(
    device float* data [[buffer(0)]],
    constant int64_t& offM [[buffer(1)]],
    constant int64_t& n [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant int64_t& offC [[buffer(4)]],
    constant int64_t& ldc [[buffer(5)]],
    constant int64_t& nRHS [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;

    device float* L = data + offM;
    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        solveLowerUnit_rm(L, int(n), int(n), C + offC + rhs * ldc);
    }
}

// Solve U * x = b where U is upper triangular (row-major at data+offM, n×n)
// x is col-major at C+offC with stride ldc
kernel void lu_solveU_direct_kernel_float(
    device float* data [[buffer(0)]],
    constant int64_t& offM [[buffer(1)]],
    constant int64_t& n [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant int64_t& offC [[buffer(4)]],
    constant int64_t& ldc [[buffer(5)]],
    constant int64_t& nRHS [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;

    device float* U = data + offM;
    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        solveUpperRM(U, int(n), int(n), C + offC + rhs * ldc);
    }
}

// ============================================================================
// Sparse elimination solve kernels: below-diagonal block multiply
// ============================================================================

// Forward solve: matQ -= block * matC for below-diagonal blocks
// One thread per lump. For each lump, iterates over below-diagonal rows.
kernel void sparseElim_subDiagMult_float(
    constant int64_t* lumpStarts [[buffer(0)]],
    constant int64_t* spanStarts [[buffer(1)]],
    constant int64_t* chainColPtr [[buffer(2)]],
    constant int64_t* chainRowSpan [[buffer(3)]],
    constant int64_t* chainData [[buffer(4)]],
    constant float* data [[buffer(5)]],
    device float* v [[buffer(6)]],
    constant int64_t& ldc [[buffer(7)]],
    constant int64_t& nRHS [[buffer(8)]],
    constant int64_t& lumpIndexStart [[buffer(9)]],
    constant int64_t& lumpIndexEnd [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpStart = lumpStarts[lump];
    int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
    int64_t colStart = chainColPtr[lump];
    int64_t colEnd = chainColPtr[lump + 1];

    // matC is at v + lumpStart, col-major with stride ldc
    for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
        int64_t rowSpan = chainRowSpan[colPtr];
        int64_t rowSpanStart = spanStarts[rowSpan];
        int64_t rowSpanSize = spanStarts[rowSpan + 1] - rowSpanStart;
        int64_t blockPtr = chainData[colPtr];

        // block is rowSpanSize × lumpSize (row-major)
        // matQ -= block * matC
        for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            for (int64_t r = 0; r < rowSpanSize; r++) {
                float sum = 0.0f;
                for (int64_t k = 0; k < lumpSize; k++) {
                    sum += data[blockPtr + r * lumpSize + k] * v[lumpStart + k + rhs * ldc];
                }
                v[rowSpanStart + r + rhs * ldc] -= sum;
            }
        }
    }
}

// Backward solve: matC -= block^T * matQ for below-diagonal blocks
// One thread per lump.
kernel void sparseElim_subDiagMultT_float(
    constant int64_t* lumpStarts [[buffer(0)]],
    constant int64_t* spanStarts [[buffer(1)]],
    constant int64_t* chainColPtr [[buffer(2)]],
    constant int64_t* chainRowSpan [[buffer(3)]],
    constant int64_t* chainData [[buffer(4)]],
    constant float* data [[buffer(5)]],
    device float* v [[buffer(6)]],
    constant int64_t& ldc [[buffer(7)]],
    constant int64_t& nRHS [[buffer(8)]],
    constant int64_t& lumpIndexStart [[buffer(9)]],
    constant int64_t& lumpIndexEnd [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpStart = lumpStarts[lump];
    int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
    int64_t colStart = chainColPtr[lump];
    int64_t colEnd = chainColPtr[lump + 1];

    // matC is at v + lumpStart, col-major with stride ldc
    for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
        int64_t rowSpan = chainRowSpan[colPtr];
        int64_t rowSpanStart = spanStarts[rowSpan];
        int64_t rowSpanSize = spanStarts[rowSpan + 1] - rowSpanStart;
        int64_t blockPtr = chainData[colPtr];

        // block is rowSpanSize × lumpSize (row-major)
        // matC -= block^T * matQ
        for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            for (int64_t c = 0; c < lumpSize; c++) {
                float sum = 0.0f;
                for (int64_t r = 0; r < rowSpanSize; r++) {
                    sum += data[blockPtr + r * lumpSize + c] * v[rowSpanStart + r + rhs * ldc];
                }
                v[lumpStart + c + rhs * ldc] -= sum;
            }
        }
    }
}

// ============================================================================
// LU solve kernels: upper triangle gather and diagonal divide
// ============================================================================

// Backward U solve: gather from upper triangle entries
// For each lump, computes v[lump] -= U[lump, colSpan] * v[colSpan]
// One thread per lump.
kernel void sparseElim_upperGather_float(
    constant int64_t* lumpStarts [[buffer(0)]],
    constant int64_t* spanStarts [[buffer(1)]],
    constant int64_t* upperChainRowPtr [[buffer(2)]],
    constant int64_t* upperChainColSpan [[buffer(3)]],
    constant int64_t* upperChainData [[buffer(4)]],
    constant float* data [[buffer(5)]],
    device float* v [[buffer(6)]],
    constant int64_t& ldc [[buffer(7)]],
    constant int64_t& nRHS [[buffer(8)]],
    constant int64_t& lumpIndexStart [[buffer(9)]],
    constant int64_t& lumpIndexEnd [[buffer(10)]],
    constant int64_t& upperDataBase [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpStart = lumpStarts[lump];
    int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
    int64_t uRowStart = upperChainRowPtr[lump];
    int64_t uRowEnd = upperChainRowPtr[lump + 1];

    for (int64_t uIdx = uRowStart; uIdx < uRowEnd; uIdx++) {
        int64_t colSpan = upperChainColSpan[uIdx];
        int64_t colStart = spanStarts[colSpan];
        int64_t colSize = spanStarts[colSpan + 1] - colStart;
        int64_t uDataOffset = upperDataBase + upperChainData[uIdx];

        // U block: lumpSize x colSize, row-major
        // v[lump] -= U * v[colSpan]
        for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            for (int64_t r = 0; r < lumpSize; r++) {
                float sum = 0.0f;
                for (int64_t c = 0; c < colSize; c++) {
                    sum += data[uDataOffset + r * colSize + c] * v[colStart + c + rhs * ldc];
                }
                v[lumpStart + r + rhs * ldc] -= sum;
            }
        }
    }
}

// Backward U solve: divide by U diagonal
// For each lump, computes v[lump] /= U_diagonal
// One thread per lump.
kernel void sparseElim_diagDivU_float(
    constant int64_t* lumpStarts [[buffer(0)]],
    constant int64_t* chainColPtr [[buffer(1)]],
    constant int64_t* chainData [[buffer(2)]],
    device float* data [[buffer(3)]],
    device float* v [[buffer(4)]],
    constant int64_t& ldc [[buffer(5)]],
    constant int64_t& nRHS [[buffer(6)]],
    constant int64_t& lumpIndexStart [[buffer(7)]],
    constant int64_t& lumpIndexEnd [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + tid;
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpStart = lumpStarts[lump];
    int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
    int64_t colStart = chainColPtr[lump];
    int64_t diagDataPtr = chainData[colStart];

    device float* diagBlock = data + diagDataPtr;

    // For LU, diagonal block has U in upper triangle (row-major, from getrf)
    // Solve U * x = b: back-substitution with row-major U
    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        solveUpperRowMajor_dev(diagBlock, int(lumpSize), int(lumpSize), v + lumpStart + ldc * rhs);
    }
}

// ============================================================================
// Utility kernel: transpose square matrix in-place
// ============================================================================

// One thread per off-diagonal pair (i, j) where j > i.
// Swaps mat[i*n+j] with mat[j*n+i].
kernel void transposeSquareInPlace_kernel_float(
    device float* mat [[buffer(0)]],
    constant int64_t& n [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t total = n * (n - 1) / 2;
    if (int64_t(tid) >= total) return;

    // Map linear index to upper triangle (i, j) where j > i
    int64_t idx = int64_t(tid);
    int64_t i = 0;
    int64_t count = 0;
    for (i = 0; i < n - 1; i++) {
        int64_t rowElems = n - 1 - i;
        if (count + rowElems > idx) {
            break;
        }
        count += rowElems;
    }
    int64_t j = i + 1 + (idx - count);

    float tmp = mat[i * n + j];
    mat[i * n + j] = mat[j * n + i];
    mat[j * n + i] = tmp;
}

