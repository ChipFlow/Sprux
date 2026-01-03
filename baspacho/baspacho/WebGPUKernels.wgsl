/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ============================================================================
// BaSpaCho WebGPU Kernels (WGSL)
// Ported from Metal/OpenCL implementations
// ============================================================================

// ============================================================================
// Helper functions
// ============================================================================

// Convert linear index to ordered pair (x, y) where 0 <= x <= y < n
// p varies in 0 <= p < n*(n+1)/2
fn toOrderedPair(n: i64, p: i64) -> vec2<i32> {
    let odd: i64 = n & 1;
    let m: i64 = n + 1 - odd;
    var x: i64 = p % m;
    var y: i64 = n - 1 - (p / m);
    if (x > y) {
        x = x - y - 1;
        y = n - 1 - odd - y;
    }
    return vec2<i32>(i32(x), i32(y));
}

// Binary search: find largest i such that array[i] <= needle
fn bisect(array: ptr<storage, array<i64>, read>, size: i64, needle: i64) -> i64 {
    var a: i64 = 0;
    var b: i64 = size;
    while (b - a > 1) {
        let mid: i64 = (a + b) / 2;
        if (needle >= (*array)[mid]) {
            a = mid;
        } else {
            b = mid;
        }
    }
    return a;
}

// ============================================================================
// In-place Cholesky decomposition for small blocks
// A is row-major with stride lda
// Note: WGSL doesn't allow pointer arithmetic like Metal/OpenCL
// We use offset-based addressing instead
// ============================================================================
fn cholesky(data: ptr<storage, array<f32>, read_write>, offset: u32, lda: u32, n: u32) {
    var b_ii: u32 = offset;

    for (var i: u32 = 0u; i < n; i++) {
        let d: f32 = sqrt((*data)[b_ii]);
        (*data)[b_ii] = d;

        var b_ji: u32 = b_ii + lda;
        for (var j: u32 = i + 1u; j < n; j++) {
            let c: f32 = (*data)[b_ji] / d;
            (*data)[b_ji] = c;

            var b_ki: u32 = b_ii + lda;
            var b_jk: u32 = b_ji + 1u;
            for (var k: u32 = i + 1u; k <= j; k++) {
                (*data)[b_jk] -= c * (*data)[b_ki];
                b_ki += lda;
                b_jk += 1u;
            }

            b_ji += lda;
        }

        b_ii += lda + 1u;
    }
}

// ============================================================================
// In-place solver for A^T (A built upper-diagonal col-major)
// ============================================================================
fn solveUpperT(data: ptr<storage, array<f32>, read_write>, aOffset: u32, lda: u32, n: u32, vOffset: u32) {
    var b_ii: u32 = aOffset;
    for (var i: u32 = 0u; i < n; i++) {
        var x: f32 = (*data)[vOffset + i];

        for (var j: u32 = 0u; j < i; j++) {
            x -= (*data)[b_ii + j] * (*data)[vOffset + j];
        }

        (*data)[vOffset + i] = x / (*data)[b_ii + i];
        b_ii += lda;
    }
}

// ============================================================================
// In-place solver for A (A built upper-diagonal col-major)
// ============================================================================
fn solveUpper(data: ptr<storage, array<f32>, read_write>, aOffset: u32, lda: u32, n: u32, vOffset: u32) {
    var b_ii: u32 = aOffset + (lda + 1u) * (n - 1u);
    for (var i: i32 = i32(n) - 1; i >= 0; i--) {
        var x: f32 = (*data)[vOffset + u32(i)];

        var b_ij: u32 = b_ii;
        for (var j: u32 = u32(i) + 1u; j < n; j++) {
            b_ij += lda;
            x -= (*data)[b_ij] * (*data)[vOffset + j];
        }

        (*data)[vOffset + u32(i)] = x / (*data)[b_ii];
        b_ii -= lda + 1u;
    }
}

// ============================================================================
// Atomic operations for sparse elimination
// WGSL lacks native float atomics, so we use CAS-based emulation
// ============================================================================

// Atomic subtract for float using compare-and-swap
fn atomicSubFloat(atomicData: ptr<storage, array<atomic<u32>>, read_write>, index: u32, val: f32) {
    var expected: u32 = atomicLoad(&(*atomicData)[index]);
    loop {
        let current: f32 = bitcast<f32>(expected);
        let newVal: f32 = current - val;
        let desired: u32 = bitcast<u32>(newVal);
        let result = atomicCompareExchangeWeak(&(*atomicData)[index], expected, desired);
        if (result.exchanged) {
            break;
        }
        expected = result.old_value;
    }
}

// Atomic add for float using compare-and-swap
fn atomicAddFloat(atomicData: ptr<storage, array<atomic<u32>>, read_write>, index: u32, val: f32) {
    var expected: u32 = atomicLoad(&(*atomicData)[index]);
    loop {
        let current: f32 = bitcast<f32>(expected);
        let newVal: f32 = current + val;
        let desired: u32 = bitcast<u32>(newVal);
        let result = atomicCompareExchangeWeak(&(*atomicData)[index], expected, desired);
        if (result.exchanged) {
            break;
        }
        expected = result.old_value;
    }
}

// ============================================================================
// Kernel 1: factor_lumps_kernel (Cholesky on diagonal blocks)
// One thread per lump
// ============================================================================

struct FactorLumpsParams {
    lumpIndexStart: i64,
    lumpIndexEnd: i64,
}

@group(0) @binding(0) var<storage, read> lumpStart: array<i64>;
@group(0) @binding(1) var<storage, read> chainColPtr: array<i64>;
@group(0) @binding(2) var<storage, read> chainData: array<i64>;
@group(0) @binding(3) var<storage, read> boardColPtr: array<i64>;
@group(0) @binding(4) var<storage, read> boardChainColOrd: array<i64>;
@group(0) @binding(5) var<storage, read> chainRowsTillEnd: array<i64>;
@group(0) @binding(6) var<storage, read_write> data: array<f32>;
@group(0) @binding(7) var<uniform> factorParams: FactorLumpsParams;

@compute @workgroup_size(64)
fn factor_lumps_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid: i64 = i64(gid.x);
    let lump: i64 = factorParams.lumpIndexStart + tid;
    if (lump >= factorParams.lumpIndexEnd) {
        return;
    }

    let lumpSize: i64 = lumpStart[lump + 1] - lumpStart[lump];
    let colStart: i64 = chainColPtr[lump];
    let dataPtr: i64 = chainData[colStart];

    // In-place lower diag Cholesky on diagonal block
    cholesky(&data, u32(dataPtr), u32(lumpSize), u32(lumpSize));

    // Below-diagonal solve
    let gatheredStart: i64 = boardColPtr[lump];
    let gatheredEnd: i64 = boardColPtr[lump + 1];
    let rowDataStart: i64 = boardChainColOrd[gatheredStart + 1];
    let rowDataEnd: i64 = boardChainColOrd[gatheredEnd - 1];
    let belowDiagStart: i64 = chainData[colStart + rowDataStart];
    let numRows: i64 = chainRowsTillEnd[colStart + rowDataEnd - 1]
                     - chainRowsTillEnd[colStart + rowDataStart - 1];

    var belowDiagBlockPtr: i64 = belowDiagStart;
    for (var i: i64 = 0; i < numRows; i++) {
        solveUpperT(&data, u32(dataPtr), u32(lumpSize), u32(lumpSize), u32(belowDiagBlockPtr));
        belowDiagBlockPtr += lumpSize;
    }
}

// ============================================================================
// Kernel 2: assemble_kernel (Assemble rectangular sections)
// ============================================================================

struct AssembleParams {
    numBlockRows: i64,
    numBlockCols: i64,
    startRow: i64,
    srcRectWidth: i64,
    dstStride: i64,
}

@group(1) @binding(0) var<uniform> assembleParams: AssembleParams;
@group(1) @binding(1) var<storage, read> pChainRowsTillEnd: array<i64>;
@group(1) @binding(2) var<storage, read> pToSpan: array<i64>;
@group(1) @binding(3) var<storage, read> pSpanToChainOffset: array<i64>;
@group(1) @binding(4) var<storage, read> pSpanOffsetInLump: array<i64>;
@group(1) @binding(5) var<storage, read> matRectPtr: array<f32>;
@group(1) @binding(6) var<storage, read_write> assembleData: array<atomic<u32>>;

@compute @workgroup_size(64)
fn assemble_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid: i64 = i64(gid.x);
    if (tid >= assembleParams.numBlockRows * assembleParams.numBlockCols) {
        return;
    }

    let r: i64 = tid % assembleParams.numBlockRows;
    let c: i64 = tid / assembleParams.numBlockRows;

    // Only process lower triangle
    if (c > r) {
        return;
    }

    // Handle r=0 explicitly to avoid negative indexing
    var rBegin: i64;
    if (r > 0) {
        rBegin = pChainRowsTillEnd[r - 1] - assembleParams.startRow;
    } else {
        rBegin = 0;
    }
    let rEnd: i64 = pChainRowsTillEnd[r] - assembleParams.startRow;
    let rSize: i64 = rEnd - rBegin;
    let rParam: i64 = pToSpan[r];
    let rOffset: i64 = pSpanToChainOffset[rParam];

    // Handle c=0 explicitly to avoid negative indexing
    var cStart: i64;
    if (c > 0) {
        cStart = pChainRowsTillEnd[c - 1] - assembleParams.startRow;
    } else {
        cStart = 0;
    }
    let cEnd: i64 = pChainRowsTillEnd[c] - assembleParams.startRow;
    let cSize: i64 = cEnd - cStart;
    let offset: i64 = rOffset + pSpanOffsetInLump[pToSpan[c]];

    // Source pointer in temporary rectangle (row-major layout)
    let srcBase: i64 = rBegin * assembleParams.srcRectWidth + cStart;

    // Subtract source from destination (stridedMatSub)
    for (var i: i64 = 0; i < rSize; i++) {
        for (var j: i64 = 0; j < cSize; j++) {
            let srcIdx: i64 = srcBase + i * assembleParams.srcRectWidth + j;
            let dstIdx: i64 = offset + i * assembleParams.dstStride + j;
            atomicSubFloat(&assembleData, u32(dstIdx), matRectPtr[srcIdx]);
        }
    }
}

// ============================================================================
// Kernel 3: assembleVec_kernel (Vector assembly during solve)
// ============================================================================

struct AssembleVecParams {
    numColItems: i64,
    ldc: i64,
    nRHS: i64,
    startRow: i64,
}

@group(2) @binding(0) var<uniform> assembleVecParams: AssembleVecParams;
@group(2) @binding(1) var<storage, read> vecChainRowsTillEnd: array<i64>;
@group(2) @binding(2) var<storage, read> vecToSpan: array<i64>;
@group(2) @binding(3) var<storage, read> vecSpanStarts: array<i64>;
@group(2) @binding(4) var<storage, read> A_vec: array<f32>;
@group(2) @binding(5) var<storage, read_write> C_vec: array<f32>;

@compute @workgroup_size(64)
fn assembleVec_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid: i64 = i64(gid.x);
    if (tid >= assembleVecParams.numColItems) {
        return;
    }

    var rowsBefore: i64;
    if (tid > 0) {
        rowsBefore = vecChainRowsTillEnd[tid - 1] - assembleVecParams.startRow;
    } else {
        rowsBefore = 0;
    }
    let rowsAfter: i64 = vecChainRowsTillEnd[tid] - assembleVecParams.startRow;
    let blockRows: i64 = rowsAfter - rowsBefore;

    let span: i64 = vecToSpan[tid];
    let spanStart: i64 = vecSpanStarts[span];

    // A (temp buffer) is row-major with stride nRHS
    // C (output vector) is column-major with stride ldc
    let srcBase: i64 = rowsBefore * assembleVecParams.nRHS;
    let dstBase: i64 = spanStart;

    for (var rhs: i64 = 0; rhs < assembleVecParams.nRHS; rhs++) {
        for (var i: i64 = 0; i < blockRows; i++) {
            let srcIdx: i64 = srcBase + i * assembleVecParams.nRHS + rhs;
            let dstIdx: i64 = dstBase + i + rhs * assembleVecParams.ldc;
            C_vec[dstIdx] += A_vec[srcIdx];
        }
    }
}

// ============================================================================
// Kernel 4: assembleVecT_kernel (Transposed vector assembly)
// ============================================================================

@group(3) @binding(0) var<uniform> assembleVecTParams: AssembleVecParams;
@group(3) @binding(1) var<storage, read> vecTChainRowsTillEnd: array<i64>;
@group(3) @binding(2) var<storage, read> vecTToSpan: array<i64>;
@group(3) @binding(3) var<storage, read> vecTSpanStarts: array<i64>;
@group(3) @binding(4) var<storage, read> C_vecT: array<f32>;
@group(3) @binding(5) var<storage, read_write> A_vecT: array<f32>;

@compute @workgroup_size(64)
fn assembleVecT_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid: i64 = i64(gid.x);
    if (tid >= assembleVecTParams.numColItems) {
        return;
    }

    var rowsBefore: i64;
    if (tid > 0) {
        rowsBefore = vecTChainRowsTillEnd[tid - 1] - assembleVecTParams.startRow;
    } else {
        rowsBefore = 0;
    }
    let rowsAfter: i64 = vecTChainRowsTillEnd[tid] - assembleVecTParams.startRow;
    let blockRows: i64 = rowsAfter - rowsBefore;

    let span: i64 = vecTToSpan[tid];
    let spanStart: i64 = vecTSpanStarts[span];

    // A (temp buffer) is row-major with stride nRHS
    // C (input vector) is column-major with stride ldc
    let dstBase: i64 = rowsBefore * assembleVecTParams.nRHS;
    let srcBase: i64 = spanStart;

    for (var rhs: i64 = 0; rhs < assembleVecTParams.nRHS; rhs++) {
        for (var i: i64 = 0; i < blockRows; i++) {
            let dstIdx: i64 = dstBase + i * assembleVecTParams.nRHS + rhs;
            let srcIdx: i64 = srcBase + i + rhs * assembleVecTParams.ldc;
            A_vecT[dstIdx] = C_vecT[srcIdx];
        }
    }
}

// ============================================================================
// Kernel 5: sparseElim_diagSolveL_kernel
// ============================================================================

struct DiagSolveParams {
    ldc: i64,
    nRHS: i64,
    lumpIndexStart: i64,
    lumpIndexEnd: i64,
}

@group(4) @binding(0) var<uniform> diagSolveLParams: DiagSolveParams;
@group(4) @binding(1) var<storage, read> diagSolveLumpStarts: array<i64>;
@group(4) @binding(2) var<storage, read> diagSolveChainColPtr: array<i64>;
@group(4) @binding(3) var<storage, read> diagSolveChainData: array<i64>;
@group(4) @binding(4) var<storage, read_write> diagSolveData: array<f32>;
@group(4) @binding(5) var<storage, read_write> diagSolveV: array<f32>;

@compute @workgroup_size(64)
fn sparseElim_diagSolveL_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid: i64 = i64(gid.x);
    let lump: i64 = diagSolveLParams.lumpIndexStart + tid;
    if (lump >= diagSolveLParams.lumpIndexEnd) {
        return;
    }

    let lumpStartVal: i64 = diagSolveLumpStarts[lump];
    let lumpSize: i64 = diagSolveLumpStarts[lump + 1] - lumpStartVal;
    let colStart: i64 = diagSolveChainColPtr[lump];
    let diagDataPtr: i64 = diagSolveChainData[colStart];

    for (var rhs: i64 = 0; rhs < diagSolveLParams.nRHS; rhs++) {
        let vOffset: i64 = lumpStartVal + diagSolveLParams.ldc * rhs;
        solveUpperT(&diagSolveData, u32(diagDataPtr), u32(lumpSize), u32(lumpSize), u32(vOffset));
    }
}

// ============================================================================
// Kernel 6: sparseElim_diagSolveLt_kernel
// ============================================================================

@group(5) @binding(0) var<uniform> diagSolveLtParams: DiagSolveParams;
@group(5) @binding(1) var<storage, read> diagSolveLtLumpStarts: array<i64>;
@group(5) @binding(2) var<storage, read> diagSolveLtChainColPtr: array<i64>;
@group(5) @binding(3) var<storage, read> diagSolveLtChainData: array<i64>;
@group(5) @binding(4) var<storage, read_write> diagSolveLtData: array<f32>;
@group(5) @binding(5) var<storage, read_write> diagSolveLtV: array<f32>;

@compute @workgroup_size(64)
fn sparseElim_diagSolveLt_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid: i64 = i64(gid.x);
    let lump: i64 = diagSolveLtParams.lumpIndexStart + tid;
    if (lump >= diagSolveLtParams.lumpIndexEnd) {
        return;
    }

    let lumpStartVal: i64 = diagSolveLtLumpStarts[lump];
    let lumpSize: i64 = diagSolveLtLumpStarts[lump + 1] - lumpStartVal;
    let colStart: i64 = diagSolveLtChainColPtr[lump];
    let diagDataPtr: i64 = diagSolveLtChainData[colStart];

    for (var rhs: i64 = 0; rhs < diagSolveLtParams.nRHS; rhs++) {
        let vOffset: i64 = lumpStartVal + diagSolveLtParams.ldc * rhs;
        solveUpper(&diagSolveLtData, u32(diagDataPtr), u32(lumpSize), u32(lumpSize), u32(vOffset));
    }
}

// ============================================================================
// Kernel 7: sparse_elim_straight_kernel (Sparse elimination)
// One thread per block pair
// ============================================================================

struct SparseElimParams {
    lumpIndexStart: i64,
    lumpIndexEnd: i64,
    numBlockPairs: i64,
}

@group(6) @binding(0) var<uniform> sparseElimParams: SparseElimParams;
@group(6) @binding(1) var<storage, read> elimChainColPtr: array<i64>;
@group(6) @binding(2) var<storage, read> elimLumpStart: array<i64>;
@group(6) @binding(3) var<storage, read> elimChainRowSpan: array<i64>;
@group(6) @binding(4) var<storage, read> elimSpanStart: array<i64>;
@group(6) @binding(5) var<storage, read> elimChainData: array<i64>;
@group(6) @binding(6) var<storage, read> elimSpanToLump: array<i64>;
@group(6) @binding(7) var<storage, read> elimSpanOffsetInLump: array<i64>;
@group(6) @binding(8) var<storage, read_write> elimData: array<f32>;
@group(6) @binding(9) var<storage, read> makeBlockPairEnumStraight: array<i64>;

@compute @workgroup_size(64)
fn sparse_elim_straight_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid: i64 = i64(gid.x);
    if (tid >= sparseElimParams.numBlockPairs) {
        return;
    }

    // Find which lump this block pair belongs to
    let pos: i64 = bisect(&makeBlockPairEnumStraight, sparseElimParams.lumpIndexEnd - sparseElimParams.lumpIndexStart, tid);
    let l: i64 = sparseElimParams.lumpIndexStart + pos;

    // Get the number of below-diagonal blocks in this column
    let colStart: i64 = elimChainColPtr[l] + 1;  // skip diagonal
    let colEnd: i64 = elimChainColPtr[l + 1];
    let n: i64 = colEnd - colStart;

    // Convert linear index to block pair (di, dj)
    let di_dj: vec2<i32> = toOrderedPair(n, tid - makeBlockPairEnumStraight[pos]);
    let di: i64 = i64(di_dj.x);
    let dj: i64 = i64(di_dj.y);

    // Get block information
    let lumpSize: i64 = elimLumpStart[l + 1] - elimLumpStart[l];
    let iSpan: i64 = elimChainRowSpan[colStart + di];
    let jSpan: i64 = elimChainRowSpan[colStart + dj];
    let iSize: i64 = elimSpanStart[iSpan + 1] - elimSpanStart[iSpan];
    let jSize: i64 = elimSpanStart[jSpan + 1] - elimSpanStart[jSpan];
    let iDataPtr: i64 = elimChainData[colStart + di];
    let jDataPtr: i64 = elimChainData[colStart + dj];

    // Find target block in factored matrix
    let iLump: i64 = elimSpanToLump[iSpan];
    let iSpanOff: i64 = elimSpanOffsetInLump[iSpan];
    let jSpanOff: i64 = elimSpanOffsetInLump[jSpan];
    let targetLumpSize: i64 = elimLumpStart[iLump + 1] - elimLumpStart[iLump];

    // Note: Full implementation requires target chain lookup
    // This is a skeleton - actual implementation needs chain lookup for target pointer
    // For now, this kernel shows the structure but doesn't perform the actual elimination
}
