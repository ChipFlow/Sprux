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
// Kernel: lu_sparse_elim_precomputed (Pre-computed work list version)
// Each thread loads one LUWorkItem: target -= L[L_offset] * U[U_offset]
// No binary searches, 3 buffer bindings, uniform SIMD execution.
// ============================================================================
struct LUSparseWorkItem {
  int32_t L_offset;
  int32_t U_offset;
  int32_t target_offset;
};

kernel void lu_sparse_elim_precomputed_float(
    device float* data [[buffer(0)]],
    constant LUSparseWorkItem* items [[buffer(1)]],
    constant int64_t& numItems [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= numItems) return;
    LUSparseWorkItem w = items[tid];
    float product = data[w.L_offset] * data[w.U_offset];
    device atomic_uint* addr = (device atomic_uint*)&data[w.target_offset];
    atomicSubFloat(addr, product);
}

// ============================================================================
// Two-phase deterministic sparse elimination
// Phase 1: compute products into scratch buffer (no atomics)
// Phase 2: accumulate per-target in fixed order (deterministic)
// ============================================================================

// Segment descriptor: groups work items that write to the same target
struct SegmentInfo {
  int32_t target_offset;  // data offset for target element
  int32_t scratch_start;  // start index in scratch buffer
  int32_t count;          // number of products to sum
};

// Phase 1 (LU): compute L*U products into scratch (no atomics, fully parallel)
kernel void lu_sparse_elim_phase1_float(
    device float* data [[buffer(0)]],
    constant LUSparseWorkItem* items [[buffer(1)]],
    device float* scratch [[buffer(2)]],
    constant int64_t& numItems [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= numItems) return;
    LUSparseWorkItem w = items[tid];
    scratch[tid] = data[w.L_offset] * data[w.U_offset];
}

// Phase 2: deterministic segmented sum — one thread per target element
kernel void sparse_elim_phase2_float(
    device float* data [[buffer(0)]],
    device float* scratch [[buffer(1)]],
    constant SegmentInfo* segments [[buffer(2)]],
    constant int64_t& numSegments [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= numSegments) return;
    SegmentInfo seg = segments[tid];
    float sum = 0.0f;
    for (int32_t i = 0; i < seg.count; i++) {
        sum += scratch[seg.scratch_start + i];
    }
    data[seg.target_offset] -= sum;
}

// Phase 1 (Cholesky): compute dot products into scratch (no atomics)
// Each work item computes a dot product between two source rows
struct CholSparseWorkItem {
  int32_t srcRow_offset;   // start offset of row in source B matrix (in data[])
  int32_t srcCol_offset;   // start offset of row in source C matrix (in data[])
  int16_t numK;            // dot product length (= lumpSize)
  int16_t padding;
  int32_t target_offset;   // target element in data[]
};  // 16 bytes per item

kernel void chol_sparse_elim_phase1_float(
    device float* data [[buffer(0)]],
    constant CholSparseWorkItem* items [[buffer(1)]],
    device float* scratch [[buffer(2)]],
    constant int64_t& numItems [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= numItems) return;
    CholSparseWorkItem w = items[tid];
    float val = 0.0f;
    for (int16_t k = 0; k < w.numK; k++) {
        val += data[w.srcRow_offset + k] * data[w.srcCol_offset + k];
    }
    scratch[tid] = val;
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

// Solves L * x = b in-place where L has non-unit diagonal (Cholesky)
template <typename T>
inline void solveLowerNonUnit_rm(device T* L, int ldl, int n, device T* v) {
    for (int i = 0; i < n; i++) {
        T x = v[i];
        for (int j = 0; j < i; j++) {
            x -= L[i * ldl + j] * v[j];
        }
        v[i] = x / L[i * ldl + i];  // Non-unit diagonal
    }
}

// Backward substitution for L^T * x = b (L is row-major lower triangular, Cholesky)
// L^T[i,j] = L[j,i] = L_rm[j*ldl+i], upper triangular
template <typename T>
inline void solveLtRM(device T* L, int ldl, int n, device T* v) {
    for (int i = n - 1; i >= 0; i--) {
        T x = v[i];
        for (int j = i + 1; j < n; j++) {
            x -= L[j * ldl + i] * v[j];  // L^T[i,j] = L[j*ldl+i]
        }
        v[i] = x / L[i * ldl + i];  // diagonal is same for L and L^T
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

// Parallel right-looking LU factorization kernel.
// Single threadgroup, all threads cooperate on each column k:
//   1. Parallel pivot search (max |A[i,k]|) via reduction in threadgroup memory
//   2. Parallel row swap
//   3. Parallel column scale (A[i,k] /= A[k,k])
//   4. Parallel trailing matrix update (A[i,j] -= A[i,k]*A[k,j])
// Same algorithm as sequential lu_factor() — identical pivots and numerical results.
// Ported from CUDA luFactorRowMajorKernel (MatOpsCuda.cu:773).
kernel void lu_getrf_kernel_float(
    device float* data [[buffer(0)]],
    constant int64_t& offA [[buffer(1)]],
    constant int64_t& m [[buffer(2)]],
    constant int64_t& n [[buffer(3)]],
    device int64_t* pivots [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]])
{
    // Threadgroup memory for parallel max-abs reduction
    threadgroup float svals[256];
    threadgroup int64_t sidx[256];

    device float* A = data + offA;
    int64_t minMN = min(m, n);

    for (int64_t k = 0; k < minMN; k++) {
        // Step 1: Find pivot row — max |A[i,k]| for i in [k, m)
        float myMaxAbs = 0.0f;
        int64_t myBestRow = k;
        for (int64_t i = k + tid; i < m; i += nt) {
            float val = A[i * n + k];
            float absVal = (val >= 0.0f) ? val : -val;
            if (absVal > myMaxAbs) {
                myMaxAbs = absVal;
                myBestRow = i;
            }
        }
        svals[tid] = myMaxAbs;
        sidx[tid] = myBestRow;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Parallel reduction (requires power-of-2 threadgroup size)
        for (uint s = nt / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < nt) {
                if (svals[tid + s] > svals[tid]) {
                    svals[tid] = svals[tid + s];
                    sidx[tid] = sidx[tid + s];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        int64_t pivotRow = sidx[0];
        if (tid == 0) {
            pivots[k] = pivotRow;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 2: Swap rows k and pivotRow (all columns)
        if (pivotRow != k) {
            for (int64_t j = tid; j < n; j += nt) {
                float tmp = A[k * n + j];
                A[k * n + j] = A[pivotRow * n + j];
                A[pivotRow * n + j] = tmp;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Step 3: Scale column k below diagonal
        float diag = A[k * n + k];
        if (diag != 0.0f) {
            float invDiag = 1.0f / diag;
            for (int64_t i = k + 1 + tid; i < m; i += nt) {
                A[i * n + k] *= invDiag;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Step 4: Rank-1 update of trailing matrix
        // A[i,j] -= A[i,k] * A[k,j] for i in [k+1,m), j in [k+1,n)
        for (int64_t i = k + 1 + tid; i < m; i += nt) {
            float lik = A[i * n + k];
            for (int64_t j = k + 1; j < n; j++) {
                A[i * n + j] -= lik * A[k * n + j];
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
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

// ============================================================================
// Recursive TRSM→GEMM for parallel triangular solve (multi-RHS)
// ============================================================================
//
// Generalizes TRSV→GEMV to multiple right-hand sides (n > 1).
// For L * X = B (unit lower triangular, splitting on m rows):
//   TRSM(m) = TRSM(m/2) + GEMM + TRSM(m/2)
//
// Base case (m ≤ THRESHOLD): column-parallel sequential row solve (same as
// original kernel, with THRESHOLD barriers).
//
// GEMM steps parallelize across rows × cols of the output block, giving
// better thread utilization than the row-sequential approach for large m.
// ============================================================================

constant constexpr int TRSM_THRESHOLD = 16;
constant constexpr int TRSM_MAX_DEPTH = 12;  // supports m up to 16 * 2^12 = 65536

// Recursive TRSM: L * X = B, where L is m×m unit lower triangular (row-major,
// stride ldl). B is m×n row-major with stride ldb. Solves in place.
// L is in constant address space (read-only diagonal block from factored data).
template <typename T>
inline void iterativeTrsmLowerUnit(constant T* L, int64_t ldl,
                                    device T* B, int64_t ldb,
                                    int64_t m, int64_t n,
                                    uint tid, uint nt)
{
    int depth = 0;
    int64_t s_off[TRSM_MAX_DEPTH];   // row offset into L diagonal / B rows
    int64_t s_size[TRSM_MAX_DEPTH];  // sub-problem row count
    int8_t  s_phase[TRSM_MAX_DEPTH];

    s_off[0] = 0;
    s_size[0] = m;
    s_phase[0] = 0;

    while (depth >= 0) {
        int64_t off = s_off[depth];
        int64_t sz = s_size[depth];
        int8_t phase = s_phase[depth];

        if (sz <= TRSM_THRESHOLD) {
            // Base case: parallel across columns, sequential across rows
            for (int64_t i = 0; i < sz; i++) {
                for (int64_t j = int64_t(tid); j < n; j += int64_t(nt)) {
                    T val = B[(off + i) * ldb + j];
                    for (int64_t k = 0; k < i; k++) {
                        val -= L[(off + i) * ldl + off + k] * B[(off + k) * ldb + j];
                    }
                    B[(off + i) * ldb + j] = val;
                }
                threadgroup_barrier(mem_flags::mem_device);
            }
            depth--;
            if (depth >= 0) s_phase[depth]++;
            continue;
        }

        int64_t mid = sz / 2;

        if (phase == 0) {
            // Descend to top half: solve L11 * X1 = B1 (rows 0..mid-1)
            depth++;
            s_off[depth] = off;
            s_size[depth] = mid;
            s_phase[depth] = 0;
            continue;
        }

        if (phase == 1) {
            // GEMM: B_bottom -= L21 * B_top
            // L21 at L[(off+mid)*ldl + off], size (sz-mid) × mid
            // B_top at B[off*ldb], size mid × n
            // B_bottom at B[(off+mid)*ldb], size (sz-mid) × n
            // Distribute across all (rows × cols) output elements
            int64_t rows = sz - mid;
            int64_t total = rows * n;
            for (int64_t idx = int64_t(tid); idx < total; idx += int64_t(nt)) {
                int64_t i = idx / n;   // row in bottom block
                int64_t j = idx % n;   // column
                T sum = T(0);
                for (int64_t k = 0; k < mid; k++) {
                    sum += L[(off + mid + i) * ldl + off + k] * B[(off + k) * ldb + j];
                }
                B[(off + mid + i) * ldb + j] -= sum;
            }
            threadgroup_barrier(mem_flags::mem_device);

            // Descend to bottom half: solve L22 * X2 = B2
            depth++;
            s_off[depth] = off + mid;
            s_size[depth] = sz - mid;
            s_phase[depth] = 0;
            continue;
        }

        // phase >= 2: pop
        depth--;
        if (depth >= 0) s_phase[depth]++;
    }
}

// TRSM: Solve L * X = B where L is m×m unit lower triangular (row-major)
// B is m×n row-major with stride ldb
// Uses recursive TRSM→GEMM decomposition for threadgroup-parallel solve.
kernel void lu_trsmLowerUnit_kernel_float(
    constant float* L [[buffer(0)]],
    constant int64_t& offL [[buffer(1)]],
    device float* B [[buffer(2)]],
    constant int64_t& offB [[buffer(3)]],
    constant int64_t& m [[buffer(4)]],
    constant int64_t& n [[buffer(5)]],
    constant int64_t& ldb [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]])
{
    iterativeTrsmLowerUnit(L + offL, m, B + offB, ldb, m, n, tid, nt);
}

// TRSM: Solve X * U = B where U is n×n upper triangular (row-major)
// B is m×n row-major with stride ldb
// Parallel across rows (threads divide i), barrier after each column
kernel void lu_trsmUpperRight_kernel_float(
    constant float* U [[buffer(0)]],
    constant int64_t& offU [[buffer(1)]],
    device float* B [[buffer(2)]],
    constant int64_t& offB [[buffer(3)]],
    constant int64_t& m [[buffer(4)]],
    constant int64_t& n [[buffer(5)]],
    constant int64_t& ldb [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]])
{
    device float* Bp = B + offB;
    constant float* Up = U + offU;

    for (int64_t j = 0; j < n; j++) {
        float inv_diag = 1.0f / Up[j * n + j];
        for (int64_t i = tid; i < m; i += nt) {
            float val = Bp[i * ldb + j];
            for (int64_t k = 0; k < j; k++) {
                val -= Bp[i * ldb + k] * Up[k * n + j];
            }
            Bp[i * ldb + j] = val * inv_diag;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// ============================================================================
// Fused post-getrf kernel for LU factorization dense loop.
// Combines perturbSmallDiagonals + below-diag (applyRowPerm + trsmUpperRight)
// + all upper spans (applyRowPerm + trsmLowerUnit) into a single dispatch.
//
// Threadgroup 0: perturbDiag → below-diag pivot swap → trsmUpperRight
// Threadgroups 1..N: upper span pivot swap → trsmLowerUnit (each independent)
// ============================================================================
kernel void lu_postGetrf_kernel_float(
    device float* data [[buffer(0)]],
    constant float* dataConst [[buffer(1)]],   // same buffer, constant for L reads
    device int64_t* pivots [[buffer(2)]],
    constant int64_t& pivotOffset [[buffer(3)]],
    constant int64_t& diagOffset [[buffer(4)]],
    constant int64_t& lumpSize [[buffer(5)]],
    // perturbDiag params
    constant float& threshold [[buffer(6)]],
    device atomic_uint* perturbCount [[buffer(7)]],
    constant int& enablePerturb [[buffer(8)]],
    // below-diag params
    constant int64_t& belowDiagOffset [[buffer(9)]],
    constant int64_t& numRowsBelowDiag [[buffer(10)]],
    // upper span params
    constant int64_t* upperChainColSpan [[buffer(11)]],
    constant int64_t* upperChainData [[buffer(12)]],
    constant int64_t* spanStartArr [[buffer(13)]],
    constant int64_t& upperDataBase [[buffer(14)]],
    constant int64_t& rangeStart [[buffer(15)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    device int64_t* piv = pivots + pivotOffset;

    if (tg_id == 0) {
        // === Threadgroup 0: perturbDiag + below-diag processing ===
        device float* diag = data + diagOffset;

        // Phase 0: Perturb small diagonals (thread-parallel scan)
        if (enablePerturb) {
            for (int64_t i = int64_t(tid); i < lumpSize; i += int64_t(tg_size)) {
                float val = diag[i * lumpSize + i];
                if (!isfinite(val) || abs(val) < threshold) {
                    diag[i * lumpSize + i] = (val >= 0.0f) ? threshold : -threshold;
                    atomic_fetch_add_explicit(perturbCount, 1u, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
        }

        // Phases 1-2: Below-diag processing (if any rows below diagonal)
        if (numRowsBelowDiag > 0) {
            device float* belowDiag = data + belowDiagOffset;

            // Phase 1: Apply row permutation (ld=lumpSize, numCols=numRowsBelowDiag)
            for (int64_t i = 0; i < lumpSize; i++) {
                int64_t swapRow = piv[i];
                if (swapRow != i) {
                    for (int64_t c = int64_t(tid); c < numRowsBelowDiag;
                         c += int64_t(tg_size)) {
                        float tmp = belowDiag[i + c * lumpSize];
                        belowDiag[i + c * lumpSize] = belowDiag[swapRow + c * lumpSize];
                        belowDiag[swapRow + c * lumpSize] = tmp;
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
            }

            // Phase 2: trsmUpperRight (X * U = B, parallel across rows)
            // U at diag (upper triangle), B at belowDiag
            // Read U through device pointer (not constant) since perturbDiag modified it
            for (int64_t j = 0; j < lumpSize; j++) {
                float inv_diag = 1.0f / diag[j * lumpSize + j];
                for (int64_t i = int64_t(tid); i < numRowsBelowDiag;
                     i += int64_t(tg_size)) {
                    float val = belowDiag[i * lumpSize + j];
                    for (int64_t k = 0; k < j; k++) {
                        val -= belowDiag[i * lumpSize + k] * diag[k * lumpSize + j];
                    }
                    belowDiag[i * lumpSize + j] = val * inv_diag;
                }
                threadgroup_barrier(mem_flags::mem_device);
            }
        }
    } else {
        // === Threadgroups 1..N: upper span processing ===
        int64_t idx = rangeStart + int64_t(tg_id) - 1;

        int64_t colSpan = upperChainColSpan[idx];
        int64_t colSize = spanStartArr[colSpan + 1] - spanStartArr[colSpan];
        int64_t blockOffset = upperDataBase + upperChainData[idx];

        device float* block = data + blockOffset;

        // Phase 1: Apply row permutation (ld=colSize, numCols=1)
        for (int64_t i = 0; i < lumpSize; i++) {
            int64_t swapRow = piv[i];
            if (swapRow != i) {
                for (int64_t c = int64_t(tid); c < colSize; c += int64_t(tg_size)) {
                    float tmp = block[i + c * colSize];
                    block[i + c * colSize] = block[swapRow + c * colSize];
                    block[swapRow + c * colSize] = tmp;
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
        }

        // Phase 2: TRSM Lower Unit (L * X = B, unit lower triangular)
        // L reads through constant address space (not modified by perturbDiag)
        iterativeTrsmLowerUnit(dataConst + diagOffset, lumpSize,
                               block, colSize, lumpSize, colSize, tid, tg_size);
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

// prepareAssemble: GPU kernel that replaces CPU loop + memcpy.
// Reads device-resident skeleton arrays (chainColPtr, chainRowSpan, chainData)
// and writes spanToChainOffset[chainRowSpan[i]] = chainData[i] for all chain
// entries of the target lump. One thread per chain entry.
kernel void prepareAssemble_kernel_float(
    constant int64_t* chainColPtr [[buffer(0)]],
    constant int64_t* chainRowSpan [[buffer(1)]],
    constant int64_t* chainData [[buffer(2)]],
    device int64_t* spanToChainOffset [[buffer(3)]],
    constant int64_t& targetLump [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t start = chainColPtr[targetLump];
    int64_t end = chainColPtr[targetLump + 1];
    if (int64_t(tid) >= end - start) return;
    int64_t i = start + int64_t(tid);
    spanToChainOffset[chainRowSpan[i]] = chainData[i];
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
// Recursive TRSV→GEMV for parallel triangular solve
// ============================================================================
//
// Converts O(n²) single-thread triangular solve into recursive decomposition:
//   TRSV(n) = TRSV(n/2) + GEMV(n/2) + TRSV(n/2)
// The GEMV steps are parallelized across threadgroup threads while base-case
// TRSV (n ≤ THRESHOLD) uses existing sequential solvers on thread 0.
//
// Uses iterative traversal with per-thread stack in registers. All threads
// compute identical control flow (stack transitions are deterministic), so
// no threadgroup memory is needed for the stack. Only the vector v (in device
// memory) is shared, synchronized via threadgroup_barrier(mem_flags::mem_device).
//
// Reference: arXiv 2504.13821 (recursive TRSM→GEMM on Apple Silicon Metal)
// ============================================================================

constant constexpr int TRSV_THRESHOLD = 16;
constant constexpr int TRSV_MAX_DEPTH = 12;  // supports n up to 16 * 2^12 = 65536

// Iterative recursive forward substitution: L * x = b (unit lower triangular)
// L is row-major with leading dimension ldl, sub-problem within [0, n).
// All threads in the threadgroup cooperate on GEMV steps.
template <typename T>
inline void iterativeSolveLowerUnit(device T* L, int64_t ldl, int64_t n,
                                     device T* v, uint tid, uint nt)
{
    // Per-thread stack (all threads compute identical values)
    int depth = 0;
    int64_t s_off[TRSV_MAX_DEPTH];
    int64_t s_size[TRSV_MAX_DEPTH];
    int8_t  s_phase[TRSV_MAX_DEPTH];

    s_off[0] = 0;
    s_size[0] = n;
    s_phase[0] = 0;

    while (depth >= 0) {
        int64_t off = s_off[depth];
        int64_t sz = s_size[depth];
        int8_t phase = s_phase[depth];

        if (sz <= TRSV_THRESHOLD) {
            // Base case: thread 0 runs sequential forward substitution
            if (tid == 0) {
                solveLowerUnit_rm(L + off * ldl + off, int(ldl), int(sz), v + off);
            }
            threadgroup_barrier(mem_flags::mem_device);
            depth--;
            if (depth >= 0) s_phase[depth]++;
            continue;
        }

        int64_t mid = sz / 2;

        if (phase == 0) {
            // Descend left: solve L11 * x1 = b1 (top-left block, size mid)
            depth++;
            s_off[depth] = off;
            s_size[depth] = mid;
            s_phase[depth] = 0;
            continue;
        }

        if (phase == 1) {
            // Left child done. GEMV: b2 -= L21 * x1
            // L21 is at L[(off+mid)*ldl + off], size (sz-mid) × mid
            int64_t rows = sz - mid;
            for (int64_t i = int64_t(tid); i < rows; i += int64_t(nt)) {
                T sum = T(0);
                for (int64_t k = 0; k < mid; k++) {
                    sum += L[(off + mid + i) * ldl + off + k] * v[off + k];
                }
                v[off + mid + i] -= sum;
            }
            threadgroup_barrier(mem_flags::mem_device);

            // Descend right: solve L22 * x2 = b2 (bottom-right block, size sz-mid)
            depth++;
            s_off[depth] = off + mid;
            s_size[depth] = sz - mid;
            s_phase[depth] = 0;
            continue;
        }

        // phase >= 2: right child done, pop this frame
        depth--;
        if (depth >= 0) s_phase[depth]++;
    }
}

// Iterative recursive backward substitution: U * x = b (upper triangular)
// U is row-major with leading dimension ldu, sub-problem within [0, n).
// Solves bottom-right first, then GEMV, then top-left.
template <typename T>
inline void iterativeSolveUpper(device T* U, int64_t ldu, int64_t n,
                                 device T* v, uint tid, uint nt)
{
    int depth = 0;
    int64_t s_off[TRSV_MAX_DEPTH];
    int64_t s_size[TRSV_MAX_DEPTH];
    int8_t  s_phase[TRSV_MAX_DEPTH];

    s_off[0] = 0;
    s_size[0] = n;
    s_phase[0] = 0;

    while (depth >= 0) {
        int64_t off = s_off[depth];
        int64_t sz = s_size[depth];
        int8_t phase = s_phase[depth];

        if (sz <= TRSV_THRESHOLD) {
            // Base case: thread 0 runs sequential backward substitution
            if (tid == 0) {
                solveUpperRM(U + off * ldu + off, int(ldu), int(sz), v + off);
            }
            threadgroup_barrier(mem_flags::mem_device);
            depth--;
            if (depth >= 0) s_phase[depth]++;
            continue;
        }

        int64_t mid = sz / 2;

        if (phase == 0) {
            // Descend right first: solve U22 * x2 = b2 (bottom-right block)
            depth++;
            s_off[depth] = off + mid;
            s_size[depth] = sz - mid;
            s_phase[depth] = 0;
            continue;
        }

        if (phase == 1) {
            // Right child done. GEMV: b1 -= U12 * x2
            // U12 is at U[off*ldu + (off+mid)], size mid × (sz-mid)
            int64_t cols = sz - mid;
            for (int64_t i = int64_t(tid); i < mid; i += int64_t(nt)) {
                T sum = T(0);
                for (int64_t k = 0; k < cols; k++) {
                    sum += U[(off + i) * ldu + off + mid + k] * v[off + mid + k];
                }
                v[off + i] -= sum;
            }
            threadgroup_barrier(mem_flags::mem_device);

            // Descend left: solve U11 * x1 = b1 (top-left block)
            depth++;
            s_off[depth] = off;
            s_size[depth] = mid;
            s_phase[depth] = 0;
            continue;
        }

        // phase >= 2: left child done, pop this frame
        depth--;
        if (depth >= 0) s_phase[depth]++;
    }
}

// ============================================================================
// Direct-offset LU solve kernels (for per-lump calls with explicit offsets)
// ============================================================================

// Solve L * x = b where L is unit lower triangular (row-major at data+offM, n×n)
// x is col-major at C+offC with stride ldc
// Uses recursive TRSV→GEMV decomposition for threadgroup-parallel solve.
kernel void lu_solveLUnit_direct_kernel_float(
    device float* data [[buffer(0)]],
    constant int64_t& offM [[buffer(1)]],
    constant int64_t& n [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant int64_t& offC [[buffer(4)]],
    constant int64_t& ldc [[buffer(5)]],
    constant int64_t& nRHS [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]])
{
    device float* L = data + offM;
    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        iterativeSolveLowerUnit(L, n, n, C + offC + rhs * ldc, tid, nt);
    }
}

// Solve U * x = b where U is upper triangular (row-major at data+offM, n×n)
// x is col-major at C+offC with stride ldc
// Uses recursive TRSV→GEMV decomposition for threadgroup-parallel solve.
kernel void lu_solveU_direct_kernel_float(
    device float* data [[buffer(0)]],
    constant int64_t& offM [[buffer(1)]],
    constant int64_t& n [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant int64_t& offC [[buffer(4)]],
    constant int64_t& ldc [[buffer(5)]],
    constant int64_t& nRHS [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]])
{
    device float* U = data + offM;
    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        iterativeSolveUpper(U, n, n, C + offC + rhs * ldc, tid, nt);
    }
}

// ============================================================================
// Device helper functions for fused dense solve kernels
// ============================================================================
// These are reusable inline functions for matvec, scatter, and gather operations.
// Used by the fused per-lump kernels below.

// Strided-loop GEMV: tempVec[row*nRHS+rhs] = alpha * sum(M[row,col] * x[col])
// M is row-major nRows×nCols at data+offset. x is col-major at A+offA with stride lda.
// Each thread handles rows in a strided pattern.
template <typename T>
inline void deviceGemv(device T* data, int64_t offset,
                       int64_t nRows, int64_t nCols,
                       device T* A, int64_t offA, int64_t lda,
                       T alpha, int64_t nRHS,
                       device T* tempVec,
                       uint tid, uint nt)
{
    for (int64_t row = int64_t(tid); row < nRows; row += int64_t(nt)) {
        for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            T sum = T(0);
            for (int64_t col = 0; col < nCols; col++) {
                sum += data[offset + row * nCols + col] * A[offA + col + rhs * lda];
            }
            tempVec[row * nRHS + rhs] = alpha * sum;
        }
    }
}

// Strided-loop assembleVec: scatter tempVec → C using chain structure.
// Each thread handles chain entries in a strided pattern.
template <typename T>
inline void deviceAssembleVec(constant int64_t* chainRowsTillEnd,
                              constant int64_t* toSpan,
                              constant int64_t* spanStarts,
                              device T* tempVec,
                              int64_t numColItems,
                              device T* C, int64_t ldc, int64_t nRHS,
                              int64_t startRow,
                              uint tid, uint nt)
{
    for (int64_t item = int64_t(tid); item < numColItems; item += int64_t(nt)) {
        int64_t rowsBefore = (item > 0) ? (chainRowsTillEnd[item - 1] - startRow) : 0;
        int64_t rowsAfter = chainRowsTillEnd[item] - startRow;
        int64_t blockRows = rowsAfter - rowsBefore;

        int64_t span = toSpan[item];
        int64_t spanStart = spanStarts[span];

        for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            for (int64_t i = 0; i < blockRows; i++) {
                C[spanStart + i + rhs * ldc] += tempVec[(rowsBefore + i) * nRHS + rhs];
            }
        }
    }
}

// Strided-loop gemvDirect: dst += alpha * M * src
// M is row-major nRows×nCols at data+offset.
// src is at vec+srcOff, dst is at vec+dstOff, stride ldVec.
template <typename T>
inline void deviceGemvDirect(device T* data, int64_t offset,
                             int64_t nRows, int64_t nCols,
                             device T* vec, int64_t srcOff, int64_t dstOff,
                             int64_t ldVec, T alpha, int64_t nRHS,
                             uint tid, uint nt)
{
    for (int64_t row = int64_t(tid); row < nRows; row += int64_t(nt)) {
        for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            T sum = T(0);
            for (int64_t col = 0; col < nCols; col++) {
                sum += data[offset + row * nCols + col] * vec[srcOff + col + rhs * ldVec];
            }
            vec[dstOff + row + rhs * ldVec] += alpha * sum;
        }
    }
}

// ============================================================================
// Fused per-lump dense solve kernels
// ============================================================================

// Fused forward L solve: solveLUnit + gemv + assembleVec in a single dispatch.
// Single threadgroup, power-of-2 threads. Phases separated by threadgroup barriers.
kernel void lu_fusedForwardLUnit_kernel_float(
    device float* data [[buffer(0)]],
    constant int64_t& diagOffset [[buffer(1)]],
    constant int64_t& n [[buffer(2)]],
    constant int64_t& belowDiagOffset [[buffer(3)]],
    constant int64_t& numRowsBelowDiag [[buffer(4)]],
    constant int64_t* chainRowsTillEnd [[buffer(5)]],
    constant int64_t* chainRowSpan [[buffer(6)]],
    constant int64_t* spanStarts [[buffer(7)]],
    constant int64_t& numColItems [[buffer(8)]],
    constant int64_t& startRow [[buffer(9)]],
    device float* vecData [[buffer(10)]],
    constant int64_t& lumpStart [[buffer(11)]],
    constant int64_t& stride [[buffer(12)]],
    constant int64_t& nRHS [[buffer(13)]],
    device float* tempVec [[buffer(14)]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]])
{
    // Phase 1: solve L * x = b with unit diagonal (recursive TRSV→GEMV)
    device float* L = data + diagOffset;
    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        iterativeSolveLowerUnit(L, n, n, vecData + lumpStart + rhs * stride, tid, nt);
    }

    if (numRowsBelowDiag <= 0) return;

    // Phase 2: tempVec = -1.0 * M_below * x_diag
    threadgroup_barrier(mem_flags::mem_device);
    deviceGemv(
        data, belowDiagOffset,
        numRowsBelowDiag, n,
        vecData, lumpStart, stride,
        -1.0f, nRHS, tempVec, tid, nt);

    // Phase 3: scatter tempVec → vecData using chain structure
    threadgroup_barrier(mem_flags::mem_device);
    deviceAssembleVec(
        chainRowsTillEnd, chainRowSpan, spanStarts,
        tempVec, numColItems,
        vecData, stride, nRHS, startRow, tid, nt);
}

// Fused backward U solve: iterate upper chain gemvDirect + solveU in a single dispatch.
// Single threadgroup, power-of-2 threads. Iterates upper chain entries on GPU.
kernel void lu_fusedBackwardU_kernel_float(
    device float* data [[buffer(0)]],
    constant int64_t& diagOffset [[buffer(1)]],
    constant int64_t& n [[buffer(2)]],
    constant int64_t& upperDataBase [[buffer(3)]],
    constant int64_t* upperChainRowPtr [[buffer(4)]],
    constant int64_t* upperChainColSpan [[buffer(5)]],
    constant int64_t* upperChainData [[buffer(6)]],
    constant int64_t* spanStarts [[buffer(7)]],
    constant int64_t& lump [[buffer(8)]],
    device float* vecData [[buffer(9)]],
    constant int64_t& lumpStart [[buffer(10)]],
    constant int64_t& stride [[buffer(11)]],
    constant int64_t& nRHS [[buffer(12)]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]])
{
    // Phase 1: for each upper chain entry, accumulate y(lump) -= U_{lump,k} * x(k)
    int64_t upperRowStart = upperChainRowPtr[lump];
    int64_t upperRowEnd = upperChainRowPtr[lump + 1];

    for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
        int64_t colSpan = upperChainColSpan[i];
        int64_t colStart = spanStarts[colSpan];
        int64_t colSize = spanStarts[colSpan + 1] - colStart;
        int64_t upperDataOffset = upperDataBase + upperChainData[i];

        deviceGemvDirect(
            data, upperDataOffset,
            n, colSize,
            vecData, colStart, lumpStart,
            stride, -1.0f, nRHS, tid, nt);

        // Barrier between upper chain entries for correctness
        threadgroup_barrier(mem_flags::mem_device);
    }

    // Phase 2: solve U * x = y for diagonal block (recursive TRSV→GEMV)
    device float* U = data + diagOffset;
    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        iterativeSolveUpper(U, n, n, vecData + lumpStart + rhs * stride, tid, nt);
    }
}

// ============================================================================
// Iterative refinement step kernel (fused unpermute + accumulate + SpMV + permute)
// ============================================================================

// ============================================================================
// Iterative refinement with compensated (double-float) arithmetic.
// Metal has no native float64; instead we use (hi, lo) float32 pairs
// that together represent a value with ~48 bits of mantissa.
// This enables convergence of iterative refinement for large matrices
// where float32 alone would diverge.
// ============================================================================

// Two-Sum: exact floating-point addition a + b = s + t
// where s = fl(a+b) and t captures the rounding error exactly
inline void twoSum(float a, float b, thread float& s, thread float& t) {
    s = a + b;
    float v = s - a;
    t = (a - (s - v)) + (b - v);
}

// Two-Prod via FMA: exact floating-point multiplication a * b = p + e
// where p = fl(a*b) and e captures the rounding error exactly
inline void twoProd(float a, float b, thread float& p, thread float& e) {
    p = a * b;
    e = fma(a, b, -p);  // Metal supports hardware FMA
}

// Double-float addition: (a_hi, a_lo) + b → (s_hi, s_lo)
inline void dfAdd(float a_hi, float a_lo, float b,
                  thread float& s_hi, thread float& s_lo) {
    float t1, t2;
    twoSum(a_hi, b, t1, t2);
    t2 += a_lo;
    twoSum(t1, t2, s_hi, s_lo);
}

// Double-float addition: (a_hi, a_lo) + (b_hi, b_lo) → (s_hi, s_lo)
inline void dfAdd2(float a_hi, float a_lo, float b_hi, float b_lo,
                   thread float& s_hi, thread float& s_lo) {
    float t1, t2;
    twoSum(a_hi, b_hi, t1, t2);
    t2 += a_lo + b_lo;
    twoSum(t1, t2, s_hi, s_lo);
}

// Step 1: Unpermute correction and accumulate into double-float solution.
// One thread per element.
// x_accum = (hi, lo) pair; correction = colScale[j] * xGpu[perm[j]]
kernel void refine_accumulate_kernel_float(
    device const int64_t* perm [[buffer(0)]],
    device const float* colScale [[buffer(1)]],
    device float* x_accum_hi [[buffer(2)]],
    device float* x_accum_lo [[buffer(3)]],
    device const float* xGpu [[buffer(4)]],
    constant int64_t& n [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uint(n)) return;
    int64_t j = int64_t(tid);
    float correction = colScale[j] * xGpu[perm[j]];
    float hi, lo;
    dfAdd(x_accum_hi[j], x_accum_lo[j], correction, hi, lo);
    x_accum_hi[j] = hi;
    x_accum_lo[j] = lo;
}

// Step 2: SpMV residual with double-float precision.
// One thread per row. Uses TwoProd (FMA) for exact A*x products and
// compensated summation for the dot product. Achieves ~float64 residual
// precision, enabling iterative refinement convergence for κ(A) up to ~1e14.
kernel void refine_spmv_kernel_float(
    device const int64_t* csrRowPtr [[buffer(0)]],
    device const int64_t* csrColInd [[buffer(1)]],
    device const float* csrValues [[buffer(2)]],
    device const int64_t* perm [[buffer(3)]],
    device const int64_t* rowPerm [[buffer(4)]],
    device const float* rowScale [[buffer(5)]],
    device const float* b_hi [[buffer(6)]],
    device const float* x_accum_hi [[buffer(7)]],
    device const float* x_accum_lo [[buffer(8)]],
    device float* xGpu [[buffer(9)]],
    constant int64_t& n [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= uint(n)) return;
    int64_t j = int64_t(tid);
    int64_t srcRow = rowPerm[j];

    // Double-float dot product: sum = A[srcRow,:] * x_accum
    // Each product uses TwoProd for exact error capture, then
    // accumulated via double-float addition.
    float sum_hi = 0.0f, sum_lo = 0.0f;
    for (int64_t k = csrRowPtr[srcRow]; k < csrRowPtr[srcRow + 1]; k++) {
        int64_t col = csrColInd[k];
        float val = csrValues[k];
        float x_hi = x_accum_hi[col];
        float x_lo = x_accum_lo[col];

        // TwoProd: val * x_hi = prod_hi + prod_err (exact via FMA)
        float prod_hi, prod_err;
        twoProd(val, x_hi, prod_hi, prod_err);
        // Full product ≈ prod_hi + (prod_err + val * x_lo)
        float prod_lo = prod_err + val * x_lo;

        // Accumulate (prod_hi, prod_lo) into (sum_hi, sum_lo)
        dfAdd2(sum_hi, sum_lo, prod_hi, prod_lo, sum_hi, sum_lo);
    }
    // Residual: b - sum, computed in double-float
    // (sum_hi + sum_lo) ≈ A[row,:] * x with ~float64 precision
    float residual = (b_hi[srcRow] - sum_hi) - sum_lo;
    xGpu[perm[j]] = rowScale[j] * residual;
}

// Legacy single-threaded version for small matrices (n <= ~100).
// Uses simple float32 — sufficient for well-conditioned small systems.
kernel void refine_step_kernel_float(
    device const int64_t* csrRowPtr [[buffer(0)]],
    device const int64_t* csrColInd [[buffer(1)]],
    device const float* csrValues [[buffer(2)]],
    device const int64_t* perm [[buffer(3)]],
    device const int64_t* rowPerm [[buffer(4)]],
    device const float* rowScale [[buffer(5)]],
    device const float* colScale [[buffer(6)]],
    device const float* b [[buffer(7)]],
    device float* x_accum [[buffer(8)]],
    device float* xGpu [[buffer(9)]],
    constant int64_t& n [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;

    for (int64_t j = 0; j < n; j++) {
        x_accum[j] += colScale[j] * xGpu[perm[j]];
    }

    for (int64_t j = 0; j < n; j++) {
        int64_t srcRow = rowPerm[j];
        float dot = 0.0f;
        for (int64_t k = csrRowPtr[srcRow]; k < csrRowPtr[srcRow + 1]; k++) {
            dot += csrValues[k] * x_accum[csrColInd[k]];
        }
        xGpu[perm[j]] = rowScale[j] * (b[srcRow] - dot);
    }
}

// ============================================================================
// Cholesky dense solve kernels
// ============================================================================

// Solve L * x = b where L is lower triangular with non-unit diagonal (row-major)
// x is col-major at C+offC with stride ldc. Single-thread sequential solve.
kernel void cholesky_solveL_kernel_float(
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
        solveLowerNonUnit_rm(L, int(n), int(n), C + offC + rhs * ldc);
    }
}

// Solve L^T * x = b (backward substitution with L transpose, Cholesky)
// L is row-major lower triangular at data+offset, x is col-major at C+offC with stride ldc.
kernel void cholesky_solveLt_kernel_float(
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
        solveLtRM(L, int(n), int(n), C + offC + rhs * ldc);
    }
}

// Cholesky gemv: tempVec = alpha * M * A (below-diagonal matvec, forward solve)
// M is row-major nRows×nCols at data+offset. A is col-major at A+offA with stride lda.
// Result written to tempVecBuffer (row-major nRows×nRHS). One thread per row.
kernel void cholesky_gemv_kernel_float(
    constant float* data [[buffer(0)]],
    constant int64_t& offset [[buffer(1)]],
    constant int64_t& nRows [[buffer(2)]],
    constant int64_t& nCols [[buffer(3)]],
    constant float* A [[buffer(4)]],
    constant int64_t& offA [[buffer(5)]],
    constant int64_t& lda [[buffer(6)]],
    constant float& alpha [[buffer(7)]],
    constant int64_t& nRHS [[buffer(8)]],
    device float* tempVec [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= nRows) return;

    int64_t row = tid;
    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        float sum = 0.0f;
        for (int64_t col = 0; col < nCols; col++) {
            sum += data[offset + row * nCols + col] * A[offA + col + rhs * lda];
        }
        tempVec[row * nRHS + rhs] = alpha * sum;  // assignment, not accumulation
    }
}

// Cholesky gemvT: A += alpha * M^T * tempVec (transpose matvec, backward solve)
// M is row-major nRows×nCols at data+offset. tempVec is row-major nRows×nRHS.
// A is col-major at A+offA with stride lda. One thread per output column.
kernel void cholesky_gemvT_kernel_float(
    constant float* data [[buffer(0)]],
    constant int64_t& offset [[buffer(1)]],
    constant int64_t& nRows [[buffer(2)]],
    constant int64_t& nCols [[buffer(3)]],
    device float* A [[buffer(4)]],
    constant int64_t& offA [[buffer(5)]],
    constant int64_t& lda [[buffer(6)]],
    constant float& alpha [[buffer(7)]],
    constant int64_t& nRHS [[buffer(8)]],
    constant float* tempVec [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= nCols) return;

    int64_t col = tid;
    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        float sum = 0.0f;
        for (int64_t row = 0; row < nRows; row++) {
            sum += data[offset + row * nCols + col] * tempVec[row * nRHS + rhs];
        }
        A[offA + col + rhs * lda] += alpha * sum;
    }
}

// Cholesky symm: D += alpha * selfadjoint(A) * C
// A is row-major n×n (lower triangle stored). C is col-major at C+offC with stride ldc.
// D is col-major with stride ldd. One thread per row.
kernel void cholesky_symm_kernel_float(
    constant float* data [[buffer(0)]],
    constant int64_t& offset [[buffer(1)]],
    constant int64_t& n [[buffer(2)]],
    constant float* C [[buffer(3)]],
    constant int64_t& offC [[buffer(4)]],
    constant int64_t& ldc [[buffer(5)]],
    device float* D [[buffer(6)]],
    constant int64_t& ldd [[buffer(7)]],
    constant float& alpha [[buffer(8)]],
    constant int64_t& nRHS [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (int64_t(tid) >= n) return;

    int64_t row = tid;
    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        float sum = 0.0f;
        for (int64_t col = 0; col < n; col++) {
            // selfadjointView<Lower>: use lower triangle element
            float val = (row >= col) ? data[offset + row * n + col]
                                     : data[offset + col * n + row];
            sum += val * C[offC + col + rhs * ldc];
        }
        D[row + rhs * ldd] += alpha * sum;
    }
}

// ============================================================================
// maxAbsDiag GPU reduction kernel
// ============================================================================

// Compute max|diag| across all lumps. One thread per lump, each thread finds its
// local max. Threadgroup reduction finds group max. Atomic max across groups.
// Result: single float in output[0].
kernel void maxAbsDiag_kernel_float(
    constant int64_t* lumpStarts [[buffer(0)]],
    constant int64_t* chainColPtr [[buffer(1)]],
    constant int64_t* chainData [[buffer(2)]],
    constant float* data [[buffer(3)]],
    constant int64_t& startLump [[buffer(4)]],
    constant int64_t& numLumps [[buffer(5)]],
    device atomic_uint* result [[buffer(6)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    float localMax = 0.0f;

    int64_t lump = startLump + tid;
    if (int64_t(tid) < numLumps) {
        int64_t lumpSize = lumpStarts[lump + 1] - lumpStarts[lump];
        int64_t diagOff = chainData[chainColPtr[lump]];
        for (int64_t i = 0; i < lumpSize; i++) {
            float absVal = abs(data[diagOff + i * lumpSize + i]);
            localMax = max(localMax, absVal);
        }
    }

    // Threadgroup reduction using shared memory
    threadgroup float shared[256];
    shared[lid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] = max(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Atomically update global maximum (using uint bit representation for float max)
    if (lid == 0) {
        uint val = as_type<uint>(shared[0]);
        // Atomic max for positive floats: IEEE 754 bit pattern preserves ordering
        // for non-negative floats, so atomic_max on uint gives correct float max.
        atomic_fetch_max_explicit(result, val, memory_order_relaxed);
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
// Forward L^T solve: transpose sub-diagonal gather
// For each lump, computes v[lump] -= block^T * v[rowSpan]
// One threadgroup per lump — parallelize inner GEMV across columns.
// Safe: each lump writes only to its own v[lumpStart..lumpStart+lumpSize].
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
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]])
{
    int64_t lump = lumpIndexStart + int64_t(gid);
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
        // Parallelize across columns (c) — each thread handles a subset
        for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            for (int64_t c = int64_t(tid); c < lumpSize; c += int64_t(nt)) {
                float sum = 0.0f;
                for (int64_t r = 0; r < rowSpanSize; r++) {
                    sum += data[blockPtr + r * lumpSize + c] * v[rowSpanStart + r + rhs * ldc];
                }
                v[lumpStart + c + rhs * ldc] -= sum;
            }
        }
        // Barrier between chain entries: next entry reads from v[rowSpan]
        // which may overlap with v[lump] written above
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// Size-1 lump specialization: flat dispatch (1 thread per lump).
// For circuits like c6288 where all sparse lumps are size-1, the threadgroup
// kernel wastes 255/256 threads. This kernel gives every thread useful work.
kernel void sparseElim_subDiagMultT_size1_float(
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
    int64_t lump = lumpIndexStart + int64_t(tid);
    if (lump >= lumpIndexEnd) return;

    int64_t lumpStart = lumpStarts[lump];
    // lumpSize is always 1: column loop disappears, leaving a dot product
    int64_t colStart = chainColPtr[lump];
    int64_t colEnd = chainColPtr[lump + 1];

    for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
        int64_t rowSpan = chainRowSpan[colPtr];
        int64_t rowSpanStart = spanStarts[rowSpan];
        int64_t rowSpanSize = spanStarts[rowSpan + 1] - rowSpanStart;
        int64_t blockPtr = chainData[colPtr];

        for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            float sum = 0.0f;
            for (int64_t r = 0; r < rowSpanSize; r++) {
                sum += data[blockPtr + r] * v[rowSpanStart + r + rhs * ldc];
            }
            v[lumpStart + rhs * ldc] -= sum;
        }
    }
}

// ============================================================================
// LU solve kernels: upper triangle gather and diagonal divide
// ============================================================================

// Backward U solve: gather from upper triangle entries
// For each lump, computes v[lump] -= U[lump, colSpan] * v[colSpan]
// One threadgroup per lump — parallelize inner GEMV across rows.
// Safe: each lump writes only to its own v[lumpStart..lumpStart+lumpSize].
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
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]])
{
    int64_t lump = lumpIndexStart + int64_t(gid);
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
        // Parallelize across rows (r) — each thread handles a subset
        for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            for (int64_t r = int64_t(tid); r < lumpSize; r += int64_t(nt)) {
                float sum = 0.0f;
                for (int64_t c = 0; c < colSize; c++) {
                    sum += data[uDataOffset + r * colSize + c] * v[colStart + c + rhs * ldc];
                }
                v[lumpStart + r + rhs * ldc] -= sum;
            }
        }
        // Barrier between upper chain entries: ensures all rows updated
        // before next entry's GEMV reads from potentially overlapping v
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// Backward U solve: divide by U diagonal
// For each lump, computes v[lump] /= U_diagonal
// One threadgroup per lump — uses recursive TRSV→GEMV for parallelism.
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
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]])
{
    int64_t lump = lumpIndexStart + int64_t(gid);
    if (lump >= lumpIndexEnd) {
        return;
    }

    int64_t lumpStart = lumpStarts[lump];
    int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
    int64_t colStart = chainColPtr[lump];
    int64_t diagDataPtr = chainData[colStart];

    device float* diagBlock = data + diagDataPtr;

    // For LU, diagonal block has U in upper triangle (row-major, from getrf)
    // Solve U * x = b: recursive TRSV→GEMV with threadgroup parallelism
    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        iterativeSolveUpper(diagBlock, lumpSize, lumpSize,
                            v + lumpStart + ldc * rhs, tid, nt);
    }
}

// Size-1 lump specialization for upper gather: flat dispatch (1 thread per lump).
kernel void sparseElim_upperGather_size1_float(
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
    int64_t lump = lumpIndexStart + int64_t(tid);
    if (lump >= lumpIndexEnd) return;

    int64_t lumpStart = lumpStarts[lump];
    // lumpSize is always 1: row loop disappears, leaving a dot product
    int64_t uRowStart = upperChainRowPtr[lump];
    int64_t uRowEnd = upperChainRowPtr[lump + 1];

    for (int64_t uIdx = uRowStart; uIdx < uRowEnd; uIdx++) {
        int64_t colSpan = upperChainColSpan[uIdx];
        int64_t colStart = spanStarts[colSpan];
        int64_t colSize = spanStarts[colSpan + 1] - colStart;
        int64_t uDataOffset = upperDataBase + upperChainData[uIdx];

        for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            float sum = 0.0f;
            for (int64_t c = 0; c < colSize; c++) {
                sum += data[uDataOffset + c] * v[colStart + c + rhs * ldc];
            }
            v[lumpStart + rhs * ldc] -= sum;
        }
    }
}

// Size-1 lump specialization for diagonal U divide: flat dispatch (1 thread per lump).
// The recursive TRSV collapses to a single scalar division.
kernel void sparseElim_diagDivU_size1_float(
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
    int64_t lump = lumpIndexStart + int64_t(tid);
    if (lump >= lumpIndexEnd) return;

    int64_t lumpStart = lumpStarts[lump];
    int64_t diagDataPtr = chainData[chainColPtr[lump]];

    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        v[lumpStart + rhs * ldc] /= data[diagDataPtr];
    }
}

// Size-1 lump specialization: fused upper gather + diagonal U divide in one dispatch.
// Combines sparseElim_upperGather_size1 + sparseElim_diagDivU_size1 to halve dispatch count.
kernel void sparseElim_fusedSolveU_size1_float(
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
    constant int64_t* chainColPtr [[buffer(12)]],
    constant int64_t* chainData [[buffer(13)]],
    uint tid [[thread_position_in_grid]])
{
    int64_t lump = lumpIndexStart + int64_t(tid);
    if (lump >= lumpIndexEnd) return;

    int64_t lumpStart = lumpStarts[lump];

    // Phase 1: upper gather — v[lump] -= U_row * v[colSpan]
    int64_t uRowStart = upperChainRowPtr[lump];
    int64_t uRowEnd = upperChainRowPtr[lump + 1];

    for (int64_t uIdx = uRowStart; uIdx < uRowEnd; uIdx++) {
        int64_t colSpan = upperChainColSpan[uIdx];
        int64_t colStart = spanStarts[colSpan];
        int64_t colSize = spanStarts[colSpan + 1] - colStart;
        int64_t uDataOffset = upperDataBase + upperChainData[uIdx];

        for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            float sum = 0.0f;
            for (int64_t c = 0; c < colSize; c++) {
                sum += data[uDataOffset + c] * v[colStart + c + rhs * ldc];
            }
            v[lumpStart + rhs * ldc] -= sum;
        }
    }

    // Phase 2: diagonal divide — v[lump] /= U_diagonal
    int64_t diagDataPtr = chainData[chainColPtr[lump]];
    for (int64_t rhs = 0; rhs < nRHS; rhs++) {
        v[lumpStart + rhs * ldc] /= data[diagDataPtr];
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

