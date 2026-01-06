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
    // The target is in the column corresponding to iSpan's lump
    int64_t targetLump = spanToLump[iSpan];
    int64_t targetSpanOffsetInLump = spanOffsetInLump[iSpan];
    int64_t targetStartPtr = chainColPtr[targetLump];  // includes diagonal
    int64_t targetEndPtr = chainColPtr[targetLump + 1];
    int64_t targetLumpSize = lumpStart[targetLump + 1] - lumpStart[targetLump];

    // Use bisect to find jSpan in the target column's chain
    // The target block (j,i) is where we write the elimination result
    int64_t targetPos = bisect(chainRowSpan + targetStartPtr, targetEndPtr - targetStartPtr, jSpan);
    int64_t jiDataPtr = chainData[targetStartPtr + targetPos];

    // Source blocks (row-major, stride = lumpSize)
    device float* srcI = data + iDataPtr;  // iSize rows x lumpSize cols
    device float* srcJ = data + jDataPtr;  // jSize rows x lumpSize cols

    // Target block with offset for span position within the lump
    // Target is jSize rows x iSize cols, stride = targetLumpSize
    device float* target = data + jiDataPtr + targetSpanOffsetInLump;

    // Perform elimination: target -= srcJ * srcI^T (with atomics)
    // srcJ is (jSize x lumpSize), srcI is (iSize x lumpSize)
    // Result is (jSize x iSize)
    locked_sub_product_float(target, int(targetLumpSize),
                             srcJ, int(jSize), int(lumpSize), int(lumpSize),
                             srcI, int(iSize), int(lumpSize));
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
// Solve kernels: sparseElim_updateL (below-diagonal updates for forward solve)
// One thread per elimination entry
// Computes: Q[rowSpan] -= block * C[lump]
// ============================================================================
kernel void sparseElim_updateL_float(
    constant int64_t* rowPtr [[buffer(0)]],        // CSR row pointers for elimination
    constant int64_t* colLump [[buffer(1)]],       // Column lump for each entry
    constant int64_t* chainColOrd [[buffer(2)]],   // Chain column order for each entry
    constant int64_t* spanStart [[buffer(3)]],     // Span start indices
    constant int64_t* lumpStart [[buffer(4)]],     // Lump start indices
    constant int64_t* chainColPtr [[buffer(5)]],   // Chain column pointers
    constant int64_t* chainData [[buffer(6)]],     // Chain data pointers
    constant float* data [[buffer(7)]],            // Factor data (read-only)
    device float* C [[buffer(8)]],                 // Solution vector (read-write)
    constant int64_t& ldc [[buffer(9)]],           // Leading dimension of C
    constant int64_t& nRHS [[buffer(10)]],         // Number of right-hand sides
    constant int64_t& spanRowBegin [[buffer(11)]], // First span row in this elimination
    constant int64_t& numElimRows [[buffer(12)]],  // Number of elimination rows
    uint tid [[thread_position_in_grid]])
{
    // Each thread processes one elimination row
    if (int64_t(tid) >= numElimRows) {
        return;
    }

    int64_t sRel = tid;
    int64_t rowSpan = sRel + spanRowBegin;
    int64_t rowStart = spanStart[rowSpan];
    int64_t rowSize = spanStart[rowSpan + 1] - rowStart;

    // Process all entries in this elimination row
    for (int64_t i = rowPtr[sRel]; i < rowPtr[sRel + 1]; i++) {
        int64_t lump = colLump[i];
        int64_t colOrd = chainColOrd[i];
        int64_t ptr = chainColPtr[lump] + colOrd;
        int64_t lumpStartIdx = lumpStart[lump];
        int64_t lumpSize = lumpStart[lump + 1] - lumpStartIdx;
        int64_t blockPtr = chainData[ptr];

        // block is rowSize x lumpSize (row-major)
        constant float* block = data + blockPtr;

        // matC is at C + lumpStartIdx, size lumpSize x nRHS (column-major, stride ldc)
        // matQ is at C + rowStart, size rowSize x nRHS (column-major, stride ldc)
        // Compute: matQ -= block * matC

        for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            for (int64_t r = 0; r < rowSize; r++) {
                float sum = 0.0f;
                for (int64_t k = 0; k < lumpSize; k++) {
                    // block[r, k] in row-major = block[r * lumpSize + k]
                    // C[lumpStartIdx + k, rhs] in col-major = C[lumpStartIdx + k + rhs * ldc]
                    sum += block[r * lumpSize + k] * C[lumpStartIdx + k + rhs * ldc];
                }
                // Q[rowStart + r, rhs] -= sum
                // Use atomic to handle potential races from different elimination entries
                device atomic_uint* addr = (device atomic_uint*)&C[rowStart + r + rhs * ldc];
                atomicSubFloat(addr, sum);
            }
        }
    }
}

// ============================================================================
// Solve kernels: sparseElim_updateLt (below-diagonal updates for backward solve)
// One thread per elimination row
// Computes: C[lump] -= block^T * Q[rowSpan]
// ============================================================================
kernel void sparseElim_updateLt_float(
    constant int64_t* rowPtr [[buffer(0)]],        // CSR row pointers for elimination
    constant int64_t* colLump [[buffer(1)]],       // Column lump for each entry
    constant int64_t* chainColOrd [[buffer(2)]],   // Chain column order for each entry
    constant int64_t* spanStart [[buffer(3)]],     // Span start indices
    constant int64_t* lumpStart [[buffer(4)]],     // Lump start indices
    constant int64_t* chainColPtr [[buffer(5)]],   // Chain column pointers
    constant int64_t* chainData [[buffer(6)]],     // Chain data pointers
    constant float* data [[buffer(7)]],            // Factor data (read-only)
    device float* C [[buffer(8)]],                 // Solution vector (read-write)
    constant int64_t& ldc [[buffer(9)]],           // Leading dimension of C
    constant int64_t& nRHS [[buffer(10)]],         // Number of right-hand sides
    constant int64_t& spanRowBegin [[buffer(11)]], // First span row in this elimination
    constant int64_t& numElimRows [[buffer(12)]],  // Number of elimination rows
    uint tid [[thread_position_in_grid]])
{
    // Each thread processes one elimination row
    if (int64_t(tid) >= numElimRows) {
        return;
    }

    int64_t sRel = tid;
    int64_t rowSpan = sRel + spanRowBegin;
    int64_t rowStart = spanStart[rowSpan];
    int64_t rowSize = spanStart[rowSpan + 1] - rowStart;

    // Process all entries in this elimination row
    for (int64_t i = rowPtr[sRel]; i < rowPtr[sRel + 1]; i++) {
        int64_t lump = colLump[i];
        int64_t colOrd = chainColOrd[i];
        int64_t ptr = chainColPtr[lump] + colOrd;
        int64_t lumpStartIdx = lumpStart[lump];
        int64_t lumpSize = lumpStart[lump + 1] - lumpStartIdx;
        int64_t blockPtr = chainData[ptr];

        // block is rowSize x lumpSize (row-major)
        constant float* block = data + blockPtr;

        // matQ is at C + rowStart, size rowSize x nRHS (column-major, stride ldc)
        // matC is at C + lumpStartIdx, size lumpSize x nRHS (column-major, stride ldc)
        // Compute: matC -= block^T * matQ

        for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            for (int64_t c = 0; c < lumpSize; c++) {
                float sum = 0.0f;
                for (int64_t r = 0; r < rowSize; r++) {
                    // block^T[c, r] = block[r, c] in row-major = block[r * lumpSize + c]
                    // Q[rowStart + r, rhs] in col-major = C[rowStart + r + rhs * ldc]
                    sum += block[r * lumpSize + c] * C[rowStart + r + rhs * ldc];
                }
                // C[lumpStartIdx + c, rhs] -= sum
                // Use atomic to handle potential races
                device atomic_uint* addr = (device atomic_uint*)&C[lumpStartIdx + c + rhs * ldc];
                atomicSubFloat(addr, sum);
            }
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
