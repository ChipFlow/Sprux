/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma nv_diag_suppress 20236
#pragma nv_diag_suppress 20012

#include <chrono>
#include <iostream>
#include "baspacho/baspacho/CudaAtomic.cuh"
#include "baspacho/baspacho/CudaDefs.h"
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/MathUtils.h"
#include "baspacho/baspacho/Utils.h"

#ifdef BASPACHO_USE_BLAS
#include "baspacho/baspacho/BlasDefs.h"
#endif

namespace BaSpaCho {

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

// CPU transpose for BLAS fallback (row-major ↔ col-major conversion)
template <typename T>
static void transposeSquareInPlace(T* data, int64_t n) {
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = i + 1; j < n; j++) {
      std::swap(data[i * n + j], data[j * n + i]);
    }
  }
}

using OuterStride = Eigen::OuterStride<>;
template <typename T>
using OuterStridedMatM =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, OuterStride>;
template <typename T>
using OuterStridedCMajMatM =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, OuterStride>;
template <typename T>
using OuterStridedCMajMatK =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
               OuterStride>;

// synchronization ops, Cuda version
struct CudaSyncOps {
  static void sync() {
    // can call 'cudaDeviceSynchronize()', but not needed
    cuCHECK(cudaStreamSynchronize(0));
  }
};

struct CudaSymElimCtx : SymElimCtx {
  CudaSymElimCtx() {}
  virtual ~CudaSymElimCtx() override {}

  int64_t numColumns;
  int64_t numBlockPairs;
  DevMirror<int64_t> makeBlockPairEnumStraight;
};

struct CudaSymbolicCtx : SymbolicCtx {
  CudaSymbolicCtx(const CoalescedBlockMatrixSkel& skel, const std::vector<int64_t>& permutation)
      : skel(skel) {
    // TODO: support custom stream in the future
    cublasCHECK(cublasCreate(&cublasH));
    // cublasCHECK(cublasSetStream(cublasH, stream));
    cusolverCHECK(cusolverDnCreate(&cusolverDnH));
    // cusolverCHECK(cusolverDnSetStream(cusolverDnH, stream));

    devLumpToSpan.load(skel.lumpToSpan);
    devChainRowsTillEnd.load(skel.chainRowsTillEnd);
    devChainRowSpan.load(skel.chainRowSpan);
    devSpanOffsetInLump.load(skel.spanOffsetInLump);
    devLumpStart.load(skel.lumpStart);
    devChainColPtr.load(skel.chainColPtr);
    devChainData.load(skel.chainData);
    devBoardColPtr.load(skel.boardColPtr);
    devBoardChainColOrd.load(skel.boardChainColOrd);
    devSpanStart.load(skel.spanStart);
    devSpanToLump.load(skel.spanToLump);
    devPermutation.load(permutation);

    // Upload upper triangle structure for LU factorization
    if (skel.isGeneral()) {
      devUpperChainRowPtr.load(skel.upperChainRowPtr);
      devUpperChainColSpan.load(skel.upperChainColSpan);
      devUpperChainData.load(skel.upperChainData);
    }
  }

  virtual ~CudaSymbolicCtx() override {
    if (cublasH) {
      cublasCHECK(cublasDestroy(cublasH));
    }
    if (cusolverDnH) {
      cusolverCHECK(cusolverDnDestroy(cusolverDnH));
    }
  }

  virtual PermutedCoalescedAccessor deviceAccessor() override {
    PermutedCoalescedAccessor retv;
    retv.init(devSpanStart.ptr, devSpanToLump.ptr, devLumpStart.ptr, devSpanOffsetInLump.ptr,
              devChainColPtr.ptr, devChainRowSpan.ptr, devChainData.ptr, devPermutation.ptr);
    return retv;
  }

  virtual SymElimCtxPtr prepareElimination(int64_t lumpsBegin, int64_t lumpsEnd) override {
    CudaSymElimCtx* elim = new CudaSymElimCtx;

    vector<int64_t> makeStraight(lumpsEnd - lumpsBegin + 1);

    // for each lump, compute number of pairs contributing to elim
    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      int64_t startPtr = skel.chainColPtr[l] + 1;  // skip diag block
      int64_t endPtr = skel.chainColPtr[l + 1];
      int64_t n = endPtr - startPtr;
      makeStraight[l - lumpsBegin] = n * (n + 1) / 2;
    }
    cumSumVec(makeStraight);

    elim->numColumns = lumpsEnd - lumpsBegin;
    elim->numBlockPairs = makeStraight[makeStraight.size() - 1];
    elim->makeBlockPairEnumStraight.load(makeStraight);

    return SymElimCtxPtr(elim);
  }

  virtual SymElimCtxPtr prepareLUElimination(int64_t lumpsBegin, int64_t lumpsEnd) override {
    if (!skel.isGeneral()) return nullptr;

    CudaSymElimCtx* elim = new CudaSymElimCtx;

    vector<int64_t> pairEnum(lumpsEnd - lumpsBegin + 1);

    // For LU: n^2 pairs per lump (L_rows x U_cols) instead of n*(n+1)/2
    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      int64_t startPtr = skel.chainColPtr[l] + 1;  // skip diag block
      int64_t endPtr = skel.chainColPtr[l + 1];
      int64_t n = endPtr - startPtr;
      pairEnum[l - lumpsBegin] = n * n;
    }
    cumSumVec(pairEnum);

    elim->numColumns = lumpsEnd - lumpsBegin;
    elim->numBlockPairs = pairEnum[pairEnum.size() - 1];
    elim->makeBlockPairEnumStraight.load(pairEnum);

    return SymElimCtxPtr(elim);
  }

  virtual NumericCtxBase* createNumericCtxForType(type_index tIdx, int64_t tempBufSize,
                                                  int batchSize) override;

  virtual SolveCtxBase* createSolveCtxForType(type_index tIdx, int nRHS, int batchSize) override;

  const CoalescedBlockMatrixSkel& skel;

  cublasHandle_t cublasH = nullptr;
  cusolverDnHandle_t cusolverDnH = nullptr;

  DevMirror<int64_t> devLumpToSpan;
  DevMirror<int64_t> devChainRowsTillEnd;
  DevMirror<int64_t> devChainRowSpan;
  DevMirror<int64_t> devSpanOffsetInLump;
  DevMirror<int64_t> devLumpStart;
  DevMirror<int64_t> devChainColPtr;
  DevMirror<int64_t> devChainData;
  DevMirror<int64_t> devBoardColPtr;
  DevMirror<int64_t> devBoardChainColOrd;
  DevMirror<int64_t> devSpanStart;
  DevMirror<int64_t> devSpanToLump;
  DevMirror<int64_t> devPermutation;

  // Upper triangle buffers (for LU factorization)
  DevMirror<int64_t> devUpperChainRowPtr;
  DevMirror<int64_t> devUpperChainColSpan;
  DevMirror<int64_t> devUpperChainData;
};

// cuda ops implemented using CUBLAS and custom kernels
struct CudaOps : Ops {
  virtual SymbolicCtxPtr createSymbolicCtx(const CoalescedBlockMatrixSkel& skel,
                                           const std::vector<int64_t>& permutation) override {
    // cout << "create sym..." << endl;
    return SymbolicCtxPtr(new CudaSymbolicCtx(skel, permutation));
  }
};

template <typename TT, typename B>
__global__ static void factor_lumps_kernel(const int64_t* lumpStart, const int64_t* chainColPtr,
                                           const int64_t* chainData, const int64_t* boardColPtr,
                                           const int64_t* boardChainColOrd,
                                           const int64_t* chainRowsTillEnd, TT* dataB,
                                           int64_t lumpIndexStart, int64_t lumpIndexEnd, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  T* data = batch.get(dataB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t lump = lumpIndexStart + i;
  if (lump >= lumpIndexEnd) {
    return;
  }
  int64_t lumpSize = lumpStart[lump + 1] - lumpStart[lump];
  int64_t colStart = chainColPtr[lump];
  int64_t dataPtr = chainData[colStart];

  // in-place lower diag cholesky dec on diagonal block
  T* diagBlockPtr = data + dataPtr;
  cholesky(diagBlockPtr, lumpSize, lumpSize);

  int64_t gatheredStart = boardColPtr[lump];
  int64_t gatheredEnd = boardColPtr[lump + 1];
  int64_t rowDataStart = boardChainColOrd[gatheredStart + 1];
  int64_t rowDataEnd = boardChainColOrd[gatheredEnd - 1];
  int64_t belowDiagStart = chainData[colStart + rowDataStart];
  int64_t numRows =
      chainRowsTillEnd[colStart + rowDataEnd - 1] - chainRowsTillEnd[colStart + rowDataStart - 1];

  T* belowDiagBlockPtr = data + belowDiagStart;
  for (int i = 0; i < numRows; i++) {
    solveUpperT(diagBlockPtr, lumpSize, lumpSize, belowDiagBlockPtr);
    belowDiagBlockPtr += lumpSize;
  }
}

template <typename TT, typename B>
__global__ static void factor_spans_kernel(const int64_t* spanToLump,
                                           const int64_t* spanOffsetInLumpV,
                                           const int64_t* lumpToSpan, const int64_t* spanStart,

                                           const int64_t* lumpStartV, const int64_t* chainColPtr,
                                           const int64_t* chainData, const int64_t* boardColPtr,
                                           const int64_t* boardChainColOrd,
                                           const int64_t* chainRowsTillEnd, TT* dataB,
                                           int64_t spanIndexStart, int64_t spanIndexEnd, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  T* data = batch.get(dataB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t span = spanIndexStart + i;
  if (span >= spanIndexEnd) {
    return;
  }
  int64_t lump = spanToLump[span];
  int64_t spanOffsetInLump = spanOffsetInLumpV[span];
  int64_t spanIndexInLump = span - lumpToSpan[lump];
  int64_t spanSize = spanStart[span + 1] - spanStart[span];
  int64_t lumpStart = lumpStartV[lump];
  int64_t lumpSize = lumpStartV[lump + 1] - lumpStart;
  int64_t colStart = chainColPtr[lump];
  int64_t dataPtr = chainData[colStart + spanIndexInLump] + spanOffsetInLump;

  // in-place lower diag cholesky dec on diagonal block
  T* diagBlockPtr = data + dataPtr;
  cholesky(diagBlockPtr, lumpSize, spanSize);

  int64_t gatheredEnd = boardColPtr[lump + 1];
  int64_t rowDataEnd = boardChainColOrd[gatheredEnd - 1];
  int64_t belowDiagPtr = chainData[colStart + spanIndexInLump + 1] + spanOffsetInLump;
  int64_t numRows =
      chainRowsTillEnd[colStart + rowDataEnd - 1] - chainRowsTillEnd[colStart + spanIndexInLump];

  T* belowDiagBlockPtr = data + belowDiagPtr;
  for (int i = 0; i < numRows; i++) {
    solveUpperT(diagBlockPtr, lumpSize, spanSize, belowDiagBlockPtr);
    belowDiagBlockPtr += lumpSize;
  }
}

template <typename T>
__device__ static inline void do_sparse_elim(const int64_t* chainColPtr, const int64_t* lumpStart,
                                             const int64_t* chainRowSpan, const int64_t* spanStart,
                                             const int64_t* chainData, const int64_t* spanToLump,
                                             const int64_t* spanOffsetInLump, T* data, int64_t l,
                                             int64_t di, int64_t dj) {
  int64_t startPtr = chainColPtr[l] + 1;  // skip diag block
  int64_t lColSize = lumpStart[l + 1] - lumpStart[l];

  int64_t i = startPtr + di;
  int64_t si = chainRowSpan[i];
  int64_t siSize = spanStart[si + 1] - spanStart[si];
  int64_t siDataPtr = chainData[i];
  Eigen::Map<MatRMaj<T>> ilBlock(data + siDataPtr, siSize, lColSize);

  int64_t targetLump = spanToLump[si];
  int64_t targetSpanOffsetInLump = spanOffsetInLump[si];
  int64_t targetStartPtr = chainColPtr[targetLump];  // skip diag block
  int64_t targetEndPtr = chainColPtr[targetLump + 1];
  int64_t targetLumpSize = lumpStart[targetLump + 1] - lumpStart[targetLump];

  int64_t j = startPtr + dj;
  int64_t sj = chainRowSpan[j];
  int64_t sjSize = spanStart[sj + 1] - spanStart[sj];
  int64_t sjDataPtr = chainData[j];

  Eigen::Map<MatRMaj<T>> jlBlock(data + sjDataPtr, sjSize, lColSize);

  uint64_t pos = bisect(chainRowSpan + targetStartPtr, targetEndPtr - targetStartPtr, sj);
  int64_t jiDataPtr = chainData[targetStartPtr + pos];
  OuterStridedMatM<T> jiBlock(data + jiDataPtr + targetSpanOffsetInLump, sjSize, siSize,
                              OuterStride(targetLumpSize));
  // jiBlock -= jlBlock * ilBlock.transpose();
  locked_sub_product(jiBlock, jlBlock, ilBlock);
}

// "naive" elimination kernel, in the sense there is one kernel instance
// per column, and will internally iterate over pairs of blocks (two
// nested loops). Not meant for performance, but as a simpler testing
// version of the below "straigthened" kernel.
template <typename TT, typename B>
__global__ static void sparse_elim_2loops_kernel(
    const int64_t* chainColPtr, const int64_t* lumpStart, const int64_t* chainRowSpan,
    const int64_t* spanStart, const int64_t* chainData, const int64_t* spanToLump,
    const int64_t* spanOffsetInLump, TT* dataB, int64_t lumpIndexStart, int64_t lumpIndexEnd,
    B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  T* data = batch.get(dataB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t l = lumpIndexStart + i;
  if (l >= lumpIndexEnd) {
    return;
  }
  int64_t startPtr = chainColPtr[l] + 1;  // skip diag block
  int64_t endPtr = chainColPtr[l + 1];
  for (int64_t i = startPtr; i < endPtr; i++) {
    for (int64_t j = i; j < endPtr; j++) {
      do_sparse_elim(chainColPtr, lumpStart, chainRowSpan, spanStart, chainData, spanToLump,
                     spanOffsetInLump, data, l, i - startPtr, j - startPtr);
    }
  }
}

// makeBlockPairEnumStraight contains the cumulated sum of nb*(nb+1)/2
// over all columns, nb being the number of blocks in the columns.
// `i` is the index in the index in the list of all pairs of block in
// the same column. We bisect and get as position the column (relative to
// lumpIndexStart), and as offset to the found value the index in range
// 0..nb*(nb+1)/2-1 in the list of *pairs* of blocks in the column. Such
// index if converted to the ordered pair di/dj
template <typename TT, typename B>
__global__ static void sparse_elim_straight_kernel(
    const int64_t* chainColPtr, const int64_t* lumpStart, const int64_t* chainRowSpan,
    const int64_t* spanStart, const int64_t* chainData, const int64_t* spanToLump,
    const int64_t* spanOffsetInLump, TT* dataB, int64_t lumpIndexStart, int64_t lumpIndexEnd,
    const int64_t* makeBlockPairEnumStraight, int64_t numBlockPairs, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  T* data = batch.get(dataB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numBlockPairs) {
    return;
  }
  int64_t pos = bisect(makeBlockPairEnumStraight, lumpIndexEnd - lumpIndexStart, i);
  int64_t l = lumpIndexStart + pos;
  int64_t n = chainColPtr[l + 1] - (chainColPtr[l] + 1);
  auto di_dj = toOrderedPair(n, i - makeBlockPairEnumStraight[pos]);
  do_sparse_elim(chainColPtr, lumpStart, chainRowSpan, spanStart, chainData, spanToLump,
                 spanOffsetInLump, data, l, std::get<0>(di_dj), std::get<1>(di_dj));
}

template <typename T>
__device__ static inline void stridedMatSubDev(T* dst, int64_t dstStride, const T* src,
                                               int64_t srcStride, int64_t rSize, int64_t cSize) {
  for (uint j = 0; j < rSize; j++) {
    for (uint i = 0; i < cSize; i++) {
      dst[i] -= src[i];
    }
    dst += dstStride;
    src += srcStride;
  }
}

struct Plain {
  __device__ bool verify() { return true; }
  template <typename T>
  __device__ T* get(T* ptr) {
    return ptr;
  }
};

struct Batched {
  int batchSize;
  int batchIndex;
  __device__ bool verify() {
    batchIndex = blockIdx.y * blockDim.y + threadIdx.y;
    return batchIndex < batchSize;
  }
  template <typename T>
  __device__ T* get(T* const* ptr) {
    return ptr[batchIndex];
  }
  template <typename T>
  const __device__ T* get(const T* const* ptr) {
    return ptr[batchIndex];
  }
};

// ============================================================================
// LU-specific CUDA kernels
// ============================================================================

// Transpose square matrix in-place on GPU
template <typename T>
__global__ void transposeSquareInPlaceKernel(T* mat, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = n * (n - 1) / 2;
  if (idx >= total) return;

  // Map linear index to upper triangle (i, j) where j > i
  // Use a simple row-based mapping
  int64_t i = 0, j = 0;
  int64_t count = 0;
  for (i = 0; i < n - 1; i++) {
    int64_t rowElems = n - 1 - i;
    if (count + rowElems > idx) {
      j = i + 1 + (idx - count);
      break;
    }
    count += rowElems;
  }

  T tmp = mat[i * n + j];
  mat[i * n + j] = mat[j * n + i];
  mat[j * n + i] = tmp;
}

// Perturb small diagonal elements in a row-major n×n matrix on GPU.
// One thread per diagonal element. Atomically increments *count for each perturbation.
template <typename T>
__global__ void perturbSmallDiagonalsKernel(int64_t n, T* data, int64_t offset, int64_t stride,
                                            T threshold, int64_t* count) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  T& diag = data[offset + i * stride + i];
  T absVal = (diag >= T(0)) ? diag : -diag;
  if (isnan(diag) || isinf(diag) || absVal < threshold) {
    diag = (diag >= T(0) && !isnan(diag)) ? threshold : -threshold;
    atomicAdd(reinterpret_cast<unsigned long long*>(count), 1ULL);
  }
}

// Convert cuSolver pivots (int, 1-based) to BaSpaCho format (int64_t, 0-based) on GPU
__global__ void convertPivotsKernel(const int* src, int64_t* dst, int64_t count) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    dst[tid] = (int64_t)(src[tid] - 1);
  }
}

// Batched small GEMM: C -= L * U (row-major) for many small matrix multiplies.
// Each block handles one work item. Threads within a block parallelize over output elements.
// This avoids the ~10μs cuBLAS dispatch overhead per GEMM for thousands of tiny operations.
struct GemmWorkItem {
  int64_t offL, offU, offC;
  int32_t m, n, k;
  int32_t ldL, ldU, ldC;
};

// One thread per work item. Each thread computes all output elements of its GEMM sequentially.
// Uses atomicAdd because multiple work items (from different source lumps/boards) can target
// the same output element in the Schur complement update.
// For c6288: ~250K work items, mostly 1×1 scalar GEMMs → one multiply + atomicAdd per thread.
template <typename T>
__global__ void batchedSmallGemmKernel(T* data, const GemmWorkItem* work, int64_t numWork) {
  int64_t wid = blockIdx.x * blockDim.x + threadIdx.x;
  if (wid >= numWork) return;

  const GemmWorkItem& w = work[wid];
  int total = w.m * w.n;

  for (int idx = 0; idx < total; idx++) {
    int i = idx / w.n;
    int j = idx % w.n;
    T sum = T(0);
    for (int t = 0; t < w.k; t++) {
      sum += data[w.offL + i * w.ldL + t] * data[w.offU + t * w.ldU + j];
    }
    atomicAdd(&data[w.offC + i * w.ldC + j], -sum);
  }
}

// Threshold: GEMMs with m*n <= this value are batched; larger ones use cuBLAS directly
static const int kBatchGemmMaxMN = 64;

// Apply row permutation to matrix columns on GPU
// Sequential swaps (data dependency), parallel across columns within one block.
// IMPORTANT: Must launch with exactly 1 block since __syncthreads only syncs within a block.
template <typename T>
__global__ void applyRowPermKernel(const int64_t* pivots, int64_t n, T* data,
                                    int64_t offData, int64_t ld, int64_t numCols) {
  int64_t c = threadIdx.x;

  for (int64_t i = 0; i < n; i++) {
    int64_t swapRow = pivots[i];
    if (swapRow != i) {
      // Each thread handles a column stride
      for (int64_t col = c; col < numCols; col += blockDim.x) {
        T tmp = data[offData + i + col * ld];
        data[offData + i + col * ld] = data[offData + swapRow + col * ld];
        data[offData + swapRow + col * ld] = tmp;
      }
    }
    __syncthreads();
  }
}

// Apply row permutation to solve vector (forward direction)
// IMPORTANT: Must launch with exactly 1 block.
template <typename T>
__global__ void applyRowPermVecKernel(const int64_t* pivots, int64_t n, T* vec,
                                       int64_t ldVec, int64_t nRHS) {
  int64_t tid = threadIdx.x;

  for (int64_t i = 0; i < n; i++) {
    int64_t swapRow = pivots[i];
    if (swapRow != i) {
      for (int64_t rhs = tid; rhs < nRHS; rhs += blockDim.x) {
        T tmp = vec[i + rhs * ldVec];
        vec[i + rhs * ldVec] = vec[swapRow + rhs * ldVec];
        vec[swapRow + rhs * ldVec] = tmp;
      }
    }
    __syncthreads();
  }
}

// Apply inverse row permutation to solve vector (reverse direction)
// IMPORTANT: Must launch with exactly 1 block.
template <typename T>
__global__ void applyRowPermVecInvKernel(const int64_t* pivots, int64_t n, T* vec,
                                          int64_t ldVec, int64_t nRHS) {
  int64_t tid = threadIdx.x;

  for (int64_t i = n - 1; i >= 0; i--) {
    int64_t swapRow = pivots[i];
    if (swapRow != i) {
      for (int64_t rhs = tid; rhs < nRHS; rhs += blockDim.x) {
        T tmp = vec[i + rhs * ldVec];
        vec[i + rhs * ldVec] = vec[swapRow + rhs * ldVec];
        vec[swapRow + rhs * ldVec] = tmp;
      }
    }
    __syncthreads();
  }
}

// ============================================================================
// LU sparse elimination kernels (factor + Schur complement for scalar lumps)
// ============================================================================

// LU factor: divide below-diagonal by diagonal, with static pivoting.
// One thread per lump.
template <typename T>
__global__ void lu_factor_lumps_kernel(const int64_t* lumpStart, const int64_t* chainColPtr,
                                       const int64_t* chainData, const int64_t* boardColPtr,
                                       const int64_t* boardChainColOrd,
                                       const int64_t* chainRowsTillEnd, T* data,
                                       int64_t lumpIndexStart, int64_t lumpIndexEnd,
                                       T staticPivotThreshold, int64_t* perturbCount) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t lump = lumpIndexStart + i;
  if (lump >= lumpIndexEnd) return;

  int64_t lumpSize = lumpStart[lump + 1] - lumpStart[lump];
  int64_t colStart = chainColPtr[lump];
  int64_t dataPtr = chainData[colStart];

  // Read diagonal value
  T* diagPtr = data + dataPtr;
  T diag = *diagPtr;

  // Static pivoting: perturb near-zero or non-finite diagonals
  if (staticPivotThreshold >= T(0)) {
    T absVal = (diag >= T(0)) ? diag : -diag;
    if (isnan(diag) || isinf(diag) || absVal < staticPivotThreshold) {
      diag = (diag >= T(0) && !isnan(diag)) ? staticPivotThreshold : -staticPivotThreshold;
      *diagPtr = diag;
      atomicAdd(reinterpret_cast<unsigned long long*>(perturbCount), 1ULL);
    }
  }

  // Divide below-diagonal entries by diagonal to get L column
  int64_t gatheredStart = boardColPtr[lump];
  int64_t gatheredEnd = boardColPtr[lump + 1];
  int64_t rowDataStart = boardChainColOrd[gatheredStart + 1];
  int64_t rowDataEnd = boardChainColOrd[gatheredEnd - 1];
  int64_t belowDiagStart = chainData[colStart + rowDataStart];
  int64_t numRows =
      chainRowsTillEnd[colStart + rowDataEnd - 1] - chainRowsTillEnd[colStart + rowDataStart - 1];

  T* belowDiagPtr = data + belowDiagStart;
  T invDiag = T(1) / diag;
  for (int64_t r = 0; r < numRows * lumpSize; r++) {
    belowDiagPtr[r] *= invDiag;
  }
}

// LU Schur complement: target -= L[a,k] * U[k,b]
// One thread per (L_row, U_col) pair. n^2 pairs per lump.
template <typename T>
__global__ void lu_sparse_elim_kernel(
    const int64_t* chainColPtr, const int64_t* lumpStart, const int64_t* chainRowSpan,
    const int64_t* spanStart, const int64_t* chainData, const int64_t* spanToLump,
    const int64_t* spanOffsetInLump, T* data, int64_t lumpIndexStart, int64_t lumpIndexEnd,
    const int64_t* blockPairEnum, int64_t numBlockPairs, const int64_t* upperChainRowPtr,
    const int64_t* upperChainColSpan, const int64_t* upperChainData, int64_t upperDataBase) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numBlockPairs) return;

  // Find which lump this pair belongs to
  int64_t pos = bisect(blockPairEnum, lumpIndexEnd - lumpIndexStart, (int64_t)i);
  int64_t l = lumpIndexStart + pos;

  // Number of below-diagonal chain entries
  int64_t colStart = chainColPtr[l] + 1;  // skip diagonal
  int64_t colEnd = chainColPtr[l + 1];
  int64_t n = colEnd - colStart;

  // Map linear pair index to (row_idx, col_idx) in [0,n) x [0,n)
  int64_t localPair = (int64_t)i - blockPairEnum[pos];
  int64_t row_idx = localPair / n;
  int64_t col_idx = localPair % n;

  // Get spans for this pair
  int64_t aSpan = chainRowSpan[colStart + row_idx];  // row (L entry)
  int64_t bSpan = chainRowSpan[colStart + col_idx];  // col (U entry)

  // L value from lower chain
  T L_val = data[chainData[colStart + row_idx]];

  // U value from upper chain
  int64_t uRowStart = upperChainRowPtr[l];
  T U_val = data[upperDataBase + upperChainData[uRowStart + col_idx]];

  T product = L_val * U_val;

  if (aSpan >= bSpan) {
    // Target in lower triangle at (row=aSpan, col=bSpan)
    int64_t bLump = spanToLump[bSpan];
    int64_t bSpanOff = spanOffsetInLump[bSpan];
    int64_t targetStartPtr = chainColPtr[bLump];
    int64_t targetEndPtr = chainColPtr[bLump + 1];
    int64_t targetPos = bisect(chainRowSpan + targetStartPtr, targetEndPtr - targetStartPtr, aSpan);
    int64_t targetDataPtr = chainData[targetStartPtr + targetPos];

    atomicAdd(data + targetDataPtr + bSpanOff, -product);
  } else {
    // Target in upper triangle at (row=aSpan, col=bSpan)
    int64_t aLump = spanToLump[aSpan];
    int64_t targetURowStart = upperChainRowPtr[aLump];
    int64_t targetURowEnd = upperChainRowPtr[aLump + 1];
    int64_t targetPos = bisect(upperChainColSpan + targetURowStart,
                               targetURowEnd - targetURowStart, bSpan);
    int64_t targetDataPtr = upperDataBase + upperChainData[targetURowStart + targetPos];

    atomicAdd(data + targetDataPtr, -product);
  }
}

// ============================================================================
// LU sparse elimination solve kernels
// ============================================================================

// Backward U solve: gather from upper triangle entries
// For each lump, computes v[lump] -= U[lump, colSpan] * v[colSpan]
template <typename TT, typename B>
__global__ void sparseElim_upperGather(const int64_t* lumpStarts, const int64_t* spanStarts,
                                       const int64_t* upperChainRowPtr,
                                       const int64_t* upperChainColSpan,
                                       const int64_t* upperChainData, const TT* dataB, TT* vB,
                                       int64_t ldc, int64_t nRHS, int64_t lumpIndexStart,
                                       int64_t lumpIndexEnd, int64_t upperDataBase, B batch) {
  if (!batch.verify()) return;
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  const T* data = batch.get(dataB);
  T* v = batch.get(vB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t lump = lumpIndexStart + i;
  if (lump >= lumpIndexEnd) return;

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
        T sum = T(0);
        for (int64_t c = 0; c < colSize; c++) {
          sum += data[uDataOffset + r * colSize + c] * v[colStart + c + rhs * ldc];
        }
        v[lumpStart + r + rhs * ldc] -= sum;
      }
    }
  }
}

// Backward U solve: divide by U diagonal (row-major upper triangular)
template <typename TT, typename B>
__global__ void sparseElim_diagDivU(const int64_t* lumpStarts, const int64_t* chainColPtr,
                                    const int64_t* chainData, const TT* dataB, TT* vB,
                                    int64_t ldc, int64_t nRHS, int64_t lumpIndexStart,
                                    int64_t lumpIndexEnd, B batch) {
  if (!batch.verify()) return;
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  const T* data = batch.get(dataB);
  T* v = batch.get(vB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t lump = lumpIndexStart + i;
  if (lump >= lumpIndexEnd) return;

  int64_t lumpStart = lumpStarts[lump];
  int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
  int64_t colStart = chainColPtr[lump];
  int64_t diagDataPtr = chainData[colStart];

  const T* diagBlock = data + diagDataPtr;

  // For LU, diagonal block has U in upper triangle (row-major, from getrf)
  // Solve U * x = b: back-substitution with row-major U
  for (int64_t rhs = 0; rhs < nRHS; rhs++) {
    solveUpperRowMajor(diagBlock, (int)lumpSize, (int)lumpSize, v + lumpStart + ldc * rhs);
  }
}

template <typename TT, typename B>
__global__ void assemble_kernel(int64_t numBlockRows, int64_t numBlockCols, int64_t rectRowBegin,
                                int64_t srcRectWidth, int64_t dstStride,
                                const int64_t* pChainRowsTillEnd, const int64_t* pToSpan,
                                const int64_t* pSpanToChainOffset, const int64_t* pSpanOffsetInLump,
                                const TT* matRectPtrB, TT* dataB, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  const T* matRectPtr = batch.get(matRectPtrB);
  T* data = batch.get(dataB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numBlockRows * numBlockCols) {
    return;
  }
  int64_t r = i % numBlockRows;
  int64_t c = i / numBlockRows;
  if (c > r) {
    return;
  }

  int64_t rBegin = pChainRowsTillEnd[r - 1] - rectRowBegin;
  int64_t rSize = pChainRowsTillEnd[r] - rBegin - rectRowBegin;
  int64_t rParam = pToSpan[r];
  int64_t rOffset = pSpanToChainOffset[rParam];
  const T* matRowPtr = matRectPtr + rBegin * srcRectWidth;

  int64_t cStart = pChainRowsTillEnd[c - 1] - rectRowBegin;
  int64_t cSize = pChainRowsTillEnd[c] - cStart - rectRowBegin;
  int64_t offset = rOffset + pSpanOffsetInLump[pToSpan[c]];

  T* dst = data + offset;
  const T* src = matRowPtr + cStart;
  stridedMatSubDev(dst, dstStride, src, srcRectWidth, rSize, cSize);
}

template <typename T>
struct CudaNumericCtx : NumericCtx<T> {
  CudaNumericCtx(const CudaSymbolicCtx& sym, int64_t bufSize, int64_t numSpans)
      : spanToChainOffset(numSpans), sym(sym) {
    devTempBuffer.resizeToAtLeast(bufSize);
    devSpanToChainOffset.resizeToAtLeast(spanToChainOffset.size());
  }

  virtual ~CudaNumericCtx() override {
    if (pinnedBuf_) cudaFreeHost(pinnedBuf_);
  }

  virtual void pseudoFactorSpans(T* data, int64_t spanBegin, int64_t spanEnd) override {
    auto timer = sym.pseudoFactorStat.instance<CudaSyncOps>();

    int wgs = 32;
    int numGroups = (spanEnd - spanBegin + wgs - 1) / wgs;
    factor_spans_kernel<T>
        <<<numGroups, wgs>>>(sym.devSpanToLump.ptr, sym.devSpanOffsetInLump.ptr,
                             sym.devLumpToSpan.ptr, sym.devSpanStart.ptr,

                             sym.devLumpStart.ptr, sym.devChainColPtr.ptr, sym.devChainData.ptr,
                             sym.devBoardColPtr.ptr, sym.devBoardChainColOrd.ptr,
                             sym.devChainRowsTillEnd.ptr, data, spanBegin, spanEnd, Plain{});
  }

  virtual void doElimination(const SymElimCtx& elimData, T* data, int64_t lumpsBegin,
                             int64_t lumpsEnd) override {
    const CudaSymElimCtx* pElim = dynamic_cast<const CudaSymElimCtx*>(&elimData);
    BASPACHO_CHECK_NOTNULL(pElim);
    const CudaSymElimCtx& elim = *pElim;

    auto timer = elim.elimStat.instance<CudaSyncOps>();

    int wgs = 32;
    int numGroups = (lumpsEnd - lumpsBegin + wgs - 1) / wgs;
    factor_lumps_kernel<T>
        <<<numGroups, wgs>>>(sym.devLumpStart.ptr, sym.devChainColPtr.ptr, sym.devChainData.ptr,
                             sym.devBoardColPtr.ptr, sym.devBoardChainColOrd.ptr,
                             sym.devChainRowsTillEnd.ptr, data, lumpsBegin, lumpsEnd, Plain{});

#if 0
    // double inner loop
    sparse_elim_2loops_kernel<T><<<numGroups, wgs>>>(
        sym.devChainColPtr.ptr, sym.devLumpStart.ptr,
        sym.devChainRowSpan.ptr, sym.devSpanStart.ptr, sym.devChainData.ptr,
        sym.devSpanToLump.ptr, sym.devSpanOffsetInLump.ptr, data,
        lumpsBegin, lumpsEnd, Plain{});
#else
    int wgs2 = 32;
    int numGroups2 = (elim.numBlockPairs + wgs2 - 1) / wgs2;
    sparse_elim_straight_kernel<T><<<numGroups2, wgs2>>>(
        sym.devChainColPtr.ptr, sym.devLumpStart.ptr, sym.devChainRowSpan.ptr, sym.devSpanStart.ptr,
        sym.devChainData.ptr, sym.devSpanToLump.ptr, sym.devSpanOffsetInLump.ptr, data, lumpsBegin,
        lumpsEnd, elim.makeBlockPairEnumStraight.ptr, elim.numBlockPairs, Plain{});
#endif
  }

  virtual T readValue(const T* data, int64_t offset) override {
    if (cpuBlasMode_) return hostData_[offset];
    // Lazy cache: on first readValue, bulk-copy device data to host
    // to avoid per-element cudaMemcpy overhead (~10μs × N = 250ms for 25K lumps).
    // Cache is invalidated by beginDenseOps (which re-copies post-sparse-elim data).
    if (!readCacheValid_) {
      int64_t totalSize = sym.skel.totalDataSize();
      if ((int64_t)hostData_.size() < totalSize) {
        hostData_.resize(totalSize);
      }
      cuCHECK(cudaMemcpy(hostData_.data(), data, totalSize * sizeof(T), cudaMemcpyDeviceToHost));
      readCacheValid_ = true;
    }
    return hostData_[offset];
  }

  virtual void potrf(int64_t n, T* data, int64_t offA) override;

  virtual void trsm(int64_t n, int64_t k, T* data, int64_t offA, int64_t offB) override;

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const T* data,
                            int64_t offset) override;

  // LU factorization methods
  virtual int getrf(int64_t m, int64_t n, T* data, int64_t offA, int64_t* pivots) override;

  virtual void trsmLowerUnit(int64_t m, int64_t n, const T* L, int64_t offL, T* B, int64_t offB,
                              int64_t ldb) override;

  virtual void trsmUpperRight(int64_t m, int64_t n, const T* U, int64_t offU, T* B, int64_t offB,
                               int64_t ldb) override;

  virtual void saveGemm(int64_t m, int64_t n, int64_t k, const T* L, int64_t offL, int64_t ldL,
                         const T* U, int64_t offU, int64_t ldU, T* C, int64_t offC,
                         int64_t ldC) override;

  virtual void applyRowPerm(int64_t* pivots, int64_t n, T* data, int64_t offData, int64_t ld,
                             int64_t numCols) override;

  virtual void flush() override {
    if (cpuBlasMode_) {
      // Copy modified host data back to device
      cuCHECK(cudaMemcpy(devDataPtr_, hostData_.data(), totalDataSize_ * sizeof(T),
                          cudaMemcpyHostToDevice));
      cpuBlasMode_ = false;
      devDataPtr_ = nullptr;
      // Don't clear hostData_ (keep allocation for next factorization)
      return;
    }
    flushGemmBatch();
    // Flush deferred pivot copies: one bulk sync + sequential copies
    if (!deferredPivotCopies_.empty()) {
      for (auto& dc : deferredPivotCopies_) {
        cuCHECK(cudaMemcpy(dc.cpuDst, devDensePivots.ptr + dc.gpuSrcOffset,
                            dc.count * sizeof(int64_t), cudaMemcpyDeviceToHost));
      }
      deferredPivotCopies_.clear();
      densePivotWriteOffset_ = 0;
      lastGetrfPivotOff_ = -1;
    }
  }

  int64_t deferredPerturbCount() override {
    if (cpuPerturbCount_ > 0) {
      int64_t count = cpuPerturbCount_;
      cpuPerturbCount_ = 0;
      return count;
    }
    if (!perturbCountPending_) return 0;
    int64_t count = 0;
    cuCHECK(cudaMemcpy(&count, devPerturbCount.ptr, sizeof(int64_t), cudaMemcpyDeviceToHost));
    perturbCountPending_ = false;
    return count;
  }

  virtual int64_t perturbSmallDiagonals(int64_t n, T* data, int64_t offset, int64_t stride,
                                        T threshold) override {
    if (n <= 0) return 0;

    if (cpuBlasMode_) {
      int64_t count = 0;
      T* h = hostData_.data();
      for (int64_t i = 0; i < n; i++) {
        T& diag = h[offset + i * stride + i];
        if (!std::isfinite(diag) || std::abs(diag) < threshold) {
          diag = (diag >= T(0)) ? threshold : -threshold;
          count++;
        }
      }
      cpuPerturbCount_ += count;
      return 0;  // Actual count returned via deferredPerturbCount()
    }

    // Initialize persistent counter on first call (once per factorization)
    if (!perturbCountPending_) {
      devPerturbCount.resizeToAtLeast(1);
      cuCHECK(cudaMemsetAsync(devPerturbCount.ptr, 0, sizeof(int64_t), 0));
      perturbCountPending_ = true;
    }
    int wgs = 256;
    int numGroups = (n + wgs - 1) / wgs;
    perturbSmallDiagonalsKernel<<<numGroups, wgs>>>(n, data, offset, stride, threshold,
                                                    devPerturbCount.ptr);
    return 0;  // Actual count read in deferredPerturbCount() after flush
  }

  virtual void doEliminationLU(const SymElimCtx& elimData, T* data, int64_t lumpsBegin,
                               int64_t lumpsEnd, T staticPivotThreshold,
                               int64_t& perturbCount) override {
    const CudaSymElimCtx* pElim = dynamic_cast<const CudaSymElimCtx*>(&elimData);
    BASPACHO_CHECK_NOTNULL(pElim);
    const CudaSymElimCtx& elim = *pElim;

    int64_t numLumps = lumpsEnd - lumpsBegin;
    if (numLumps <= 0) return;

    // Allocate GPU counter for perturbations
    devPerturbCount.resizeToAtLeast(1);
    cuCHECK(cudaMemset(devPerturbCount.ptr, 0, sizeof(int64_t)));

    // Step 1: LU factor lumps (divide below-diag by diagonal)
    {
      int wgs = 32;
      int numGroups = (numLumps + wgs - 1) / wgs;
      lu_factor_lumps_kernel<T><<<numGroups, wgs>>>(
          sym.devLumpStart.ptr, sym.devChainColPtr.ptr, sym.devChainData.ptr, sym.devBoardColPtr.ptr,
          sym.devBoardChainColOrd.ptr, sym.devChainRowsTillEnd.ptr, data, lumpsBegin, lumpsEnd,
          staticPivotThreshold, devPerturbCount.ptr);
    }

    // Step 2: LU Schur complement (L*U updates to both triangles)
    if (elim.numBlockPairs > 0) {
      int64_t upperDataBase = sym.skel.dataSize();
      int wgs = 32;
      int numGroups = (elim.numBlockPairs + wgs - 1) / wgs;
      lu_sparse_elim_kernel<T><<<numGroups, wgs>>>(
          sym.devChainColPtr.ptr, sym.devLumpStart.ptr, sym.devChainRowSpan.ptr,
          sym.devSpanStart.ptr, sym.devChainData.ptr, sym.devSpanToLump.ptr,
          sym.devSpanOffsetInLump.ptr, data, lumpsBegin, lumpsEnd,
          elim.makeBlockPairEnumStraight.ptr, elim.numBlockPairs, sym.devUpperChainRowPtr.ptr,
          sym.devUpperChainColSpan.ptr, sym.devUpperChainData.ptr, upperDataBase);
    }

    // Read back perturb count
    int64_t count = 0;
    cuCHECK(cudaMemcpy(&count, devPerturbCount.ptr, sizeof(int64_t), cudaMemcpyDeviceToHost));
    perturbCount = count;
  }

  virtual void prepareAssemble(int64_t targetLump) override {
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    // Compute spanToChainOffset (needed for Cholesky assemble path, but also
    // called in LU dense loop where it's not used for assemble).
    for (int64_t i = skel.chainColPtr[targetLump], iEnd = skel.chainColPtr[targetLump + 1];
         i < iEnd; i++) {
      spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
    }

    // In CPU BLAS mode, skip GPU upload (no GPU operations in dense loop)
    if (cpuBlasMode_) return;

    // Upload to GPU using pinned memory for async H→D copy
    size_t bytes = spanToChainOffset.size() * sizeof(int64_t);
    ensurePinnedBuf(bytes);
    memcpy(pinnedBuf_, spanToChainOffset.data(), bytes);
    cuCHECK(cudaMemcpyAsync(devSpanToChainOffset.ptr, pinnedBuf_, bytes,
                             cudaMemcpyHostToDevice, 0));
  }

  virtual void assemble(T* data, int64_t rectRowBegin,
                        int64_t dstStride,  //
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    auto timer = sym.asmblStat.instance<CudaSyncOps>(sizeof(T), numBlockRows, numBlockCols);
    const int64_t* pChainRowsTillEnd = sym.devChainRowsTillEnd.ptr + srcColDataOffset;
    const int64_t* pToSpan = sym.devChainRowSpan.ptr + srcColDataOffset;
    const int64_t* pSpanToChainOffset = devSpanToChainOffset.ptr;
    const int64_t* pSpanOffsetInLump = sym.devSpanOffsetInLump.ptr;

    int wgs = 32;
    int numGroups = (numBlockRows * numBlockCols + wgs - 1) / wgs;
    assemble_kernel<T><<<numGroups, wgs>>>(
        numBlockRows, numBlockCols, rectRowBegin, srcRectWidth, dstStride, pChainRowsTillEnd,
        pToSpan, pSpanToChainOffset, pSpanOffsetInLump, devTempBuffer.ptr, data, Plain{});
  }

  void flushGemmBatch() {
    if (gemmBatch_.empty()) return;
    size_t bytes = gemmBatch_.size() * sizeof(GemmWorkItem);
    devGemmWork_.resizeToAtLeast(bytes / sizeof(T) + 1);
    // Use pinned memory for async H→D copy (avoids blocking CPU)
    ensurePinnedBuf(bytes);
    memcpy(pinnedBuf_, gemmBatch_.data(), bytes);
    cuCHECK(cudaMemcpyAsync(devGemmWork_.ptr, pinnedBuf_, bytes,
                             cudaMemcpyHostToDevice, 0));
    int wgs = 256;  // threads per block — one thread per work item
    int numBlocks = ((int)gemmBatch_.size() + wgs - 1) / wgs;
    batchedSmallGemmKernel<T>
        <<<numBlocks, wgs>>>(gemmDataPtr_, (GemmWorkItem*)devGemmWork_.ptr,
                             (int64_t)gemmBatch_.size());
    gemmBatch_.clear();
    gemmDataPtr_ = nullptr;
  }

  // Ensure pinned staging buffer is large enough for async H→D copies.
  // Pinned memory allows cudaMemcpyAsync to return immediately to the CPU.
  void ensurePinnedBuf(size_t bytes) {
    if (pinnedBufSize_ >= bytes) return;
    if (pinnedBuf_) cuCHECK(cudaFreeHost(pinnedBuf_));
    cuCHECK(cudaHostAlloc(&pinnedBuf_, bytes, cudaHostAllocDefault));
    pinnedBufSize_ = bytes;
  }

  // Grow devDensePivots with data preservation (DevMirror::resizeToAtLeast destroys data).
  // Uses doubling strategy to minimize reallocations across multiple getrf calls.
  void ensureDensePivotCapacity(int64_t needed) {
    if (devDensePivots.allocSize >= (size_t)needed) return;
    size_t newSize = std::max(devDensePivots.allocSize * 2, (size_t)needed);
    int64_t* newPtr = nullptr;
    cuCHECK(cudaMalloc((void**)&newPtr, newSize * sizeof(int64_t)));
    if (devDensePivots.ptr && densePivotWriteOffset_ > 0) {
      cuCHECK(cudaMemcpyAsync(newPtr, devDensePivots.ptr,
                                densePivotWriteOffset_ * sizeof(int64_t),
                                cudaMemcpyDeviceToDevice, 0));
    }
    devDensePivots.clear();
    devDensePivots.ptr = newPtr;
    devDensePivots.allocSize = newSize;
  }

  DevMirror<T> devTempBuffer;
  DevMirror<int> devPotrfSingIndex;
  DevMirror<int64_t> devSpanToChainOffset;
  vector<int64_t> spanToChainOffset;
  DevMirror<int> devGetrfPivots;       // cuSolver int pivots (1-based)
  DevMirror<int64_t> devPivotBuf;      // Our int64_t pivots (0-based, for CPU-uploaded pivots)
  DevMirror<int64_t> devPerturbCount;  // GPU atomic counter for perturbSmallDiagonals
  bool perturbCountPending_ = false;   // True when devPerturbCount has accumulated data
  int64_t lastGetrfPivotN_ = 0;       // Size of pivots from last getrf

  // Deferred pivot state: store all dense lumps' converted pivots on GPU,
  // defer D→H copies to flush() to eliminate per-lump sync barriers.
  DevMirror<int64_t> devDensePivots;   // Stores all dense lumps' converted pivots
  int64_t densePivotWriteOffset_ = 0;  // Current write offset in devDensePivots
  int64_t lastGetrfPivotOff_ = -1;     // Offset of last getrf pivots in devDensePivots
  struct DeferredPivotCopy {
    int64_t* cpuDst;
    int64_t gpuSrcOffset;
    int64_t count;
  };
  std::vector<DeferredPivotCopy> deferredPivotCopies_;

  // Pinned host memory for async H→D copies (reusable staging buffer)
  void* pinnedBuf_ = nullptr;
  size_t pinnedBufSize_ = 0;

  // Batched small GEMM buffer
  vector<GemmWorkItem> gemmBatch_;
  DevMirror<T> devGemmWork_;  // GPU storage for GemmWorkItem array (reinterpret_cast)
  T* gemmDataPtr_ = nullptr;  // data pointer for current batch (for validation)

  // CPU BLAS mode: after sparse elimination completes on GPU, copy entire data buffer
  // to host and run all dense LU operations (boards + factor) on CPU using BLAS.
  // This avoids cuSolver/cuBLAS dispatch overhead and the expensive CPU iteration
  // in eliminateBoardLU with GPU-side GEMMs. Modeled after Metal's CPU BLAS fallback
  // (which uses unified memory; CUDA needs explicit D→H + H→D copies).
  bool cpuBlasMode_ = false;
  bool readCacheValid_ = false;   // Lazy read cache for readValue (avoids per-element cudaMemcpy)
  T* devDataPtr_ = nullptr;       // Device data pointer (for H→D copy back in flush)
  int64_t totalDataSize_ = 0;     // Total data buffer size in elements
  std::vector<T> hostData_;       // Host copy of data buffer (shared by readCache + cpuBlasMode)
  int64_t cpuPerturbCount_ = 0;   // Accumulated perturb count during CPU mode

  virtual void beginDenseOps(T* data, int64_t totalDataSize) override {
#ifdef BASPACHO_USE_BLAS
    // Sync GPU to ensure all prior work (sparse elimination) is complete
    cuCHECK(cudaDeviceSynchronize());

    // Invalidate lazy read cache (data changed by GPU sparse elimination)
    readCacheValid_ = false;

    // Copy entire data buffer from device to host
    devDataPtr_ = data;
    totalDataSize_ = totalDataSize;
    hostData_.resize(totalDataSize);
    cuCHECK(cudaMemcpy(hostData_.data(), data, totalDataSize * sizeof(T),
                        cudaMemcpyDeviceToHost));
    cpuBlasMode_ = true;
    cpuPerturbCount_ = 0;
#else
    (void)data;
    (void)totalDataSize;
#endif
  }

  const CudaSymbolicCtx& sym;
};

template <>
void CudaNumericCtx<double>::potrf(int64_t n, double* data, int64_t offA) {
  auto timer = sym.potrfStat.instance<CudaSyncOps>(sizeof(double), n);
  sym.potrfBiggestN = max(sym.potrfBiggestN, n);

  int workspaceSize;
  cusolverCHECK(cusolverDnDpotrf_bufferSize(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, data + offA,
                                            n, &workspaceSize));

  devTempBuffer.resizeToAtLeast(workspaceSize);
  devPotrfSingIndex.resizeToAtLeast(1);

  cusolverCHECK(cusolverDnDpotrf(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, data + offA, n,
                                 devTempBuffer.ptr, workspaceSize, devPotrfSingIndex.ptr));

  // TODO: handle/report singularity
  // int info;
  // cuCHECK(cudaMemcpy(&info, devPotrfSingIndex.ptr, 1 * sizeof(int), cudaMemcpyDeviceToHost));
  // std::cout << info << std::endl;
}

template <>
void CudaNumericCtx<float>::potrf(int64_t n, float* data, int64_t offA) {
  auto timer = sym.potrfStat.instance<CudaSyncOps>(sizeof(float), n);
  sym.potrfBiggestN = max(sym.potrfBiggestN, n);

  int workspaceSize;
  cusolverCHECK(cusolverDnSpotrf_bufferSize(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, data + offA,
                                            n, &workspaceSize));

  devTempBuffer.resizeToAtLeast(workspaceSize);
  devPotrfSingIndex.resizeToAtLeast(1);

  cusolverCHECK(cusolverDnSpotrf(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, data + offA, n,
                                 devTempBuffer.ptr, workspaceSize, devPotrfSingIndex.ptr));

  // TODO: handle/report singularity
  // int info;
  // cuCHECK(cudaMemcpy(&info, devPotrfSingIndex.ptr, 1 * sizeof(int), cudaMemcpyDeviceToHost));
  // std::cout << info << std::endl;
}

template <>
void CudaNumericCtx<double>::trsm(int64_t n, int64_t k, double* data, int64_t offA, int64_t offB) {
  auto timer = sym.trsmStat.instance<CudaSyncOps>(sizeof(double), n, k);

  double alpha(1.0);
  cublasCHECK(cublasDtrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                          CUBLAS_DIAG_NON_UNIT, n, k, &alpha, data + offA, n, data + offB, n));
}

template <>
void CudaNumericCtx<float>::trsm(int64_t n, int64_t k, float* data, int64_t offA, int64_t offB) {
  auto timer = sym.trsmStat.instance<CudaSyncOps>(sizeof(float), n, k);

  float alpha(1.0);
  cublasCHECK(cublasStrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                          CUBLAS_DIAG_NON_UNIT, n, k, &alpha, data + offA, n, data + offB, n));
}

template <>
void CudaNumericCtx<double>::saveSyrkGemm(int64_t m, int64_t n, int64_t k, const double* data,
                                          int64_t offset) {
  auto timer = sym.sygeStat.instance<CudaSyncOps>(sizeof(double), m, n, k);

  double alpha(1.0), beta(0.0);
  cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k, &alpha, data + offset, k,
                          data + offset, k, &beta, devTempBuffer.ptr, m));

  sym.gemmCalls++;
}

template <>
void CudaNumericCtx<float>::saveSyrkGemm(int64_t m, int64_t n, int64_t k, const float* data,
                                         int64_t offset) {
  auto timer = sym.sygeStat.instance<CudaSyncOps>(sizeof(float), m, n, k);

  float alpha(1.0), beta(0.0);
  cublasCHECK(cublasSgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k, &alpha, data + offset, k,
                          data + offset, k, &beta, devTempBuffer.ptr, m));

  sym.gemmCalls++;
}

// ============ LU NumericCtx implementations ============

// getrf: GPU implementation using cuSolver + transpose kernel
// Deferred pivot readback: pivots are converted on GPU and stored in devDensePivots
// at per-lump offsets. D→H copy is deferred to flush() to eliminate per-lump sync.
// applyRowPerm reads directly from devDensePivots via lastGetrfPivotOff_.
template <>
int CudaNumericCtx<double>::getrf(int64_t m, int64_t n, double* data, int64_t offA,
                                   int64_t* pivots) {
  if (m <= 0 || n <= 0) return 0;

#ifdef BASPACHO_USE_BLAS
  if (cpuBlasMode_) {
    int64_t minMN = std::min(m, n);
    double* h = hostData_.data();
    if (m == n) transposeSquareInPlace(h + offA, n);
    std::vector<BLAS_INT> ipiv(minMN);
    int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, h + offA, m, ipiv.data());
    if (m == n) transposeSquareInPlace(h + offA, n);
    for (int64_t i = 0; i < minMN; i++) pivots[i] = ipiv[i] - 1;
    return info;
  }
#endif

  flushGemmBatch();  // Ensure all pending GEMM updates are complete before factoring
  int64_t minMN = std::min(m, n);

  // Step 1: Transpose row-major → col-major on GPU (in-place for square)
  if (m == n) {
    int64_t numPairs = n * (n - 1) / 2;
    if (numPairs > 0) {
      int wgs = 256;
      int numGroups = (numPairs + wgs - 1) / wgs;
      transposeSquareInPlaceKernel<<<numGroups, wgs>>>(data + offA, n);
    }
  }

  // Step 2: cuSolver getrf (col-major LU with partial pivoting)
  int workspaceSize;
  cusolverCHECK(cusolverDnDgetrf_bufferSize(sym.cusolverDnH, m, n, data + offA, m, &workspaceSize));

  devTempBuffer.resizeToAtLeast(workspaceSize);
  devGetrfPivots.resizeToAtLeast(minMN);
  devPotrfSingIndex.resizeToAtLeast(1);

  cusolverCHECK(cusolverDnDgetrf(sym.cusolverDnH, m, n, data + offA, m,
                                  devTempBuffer.ptr, devGetrfPivots.ptr, devPotrfSingIndex.ptr));

  // Step 3: Transpose col-major → row-major on GPU
  if (m == n) {
    int64_t numPairs = n * (n - 1) / 2;
    if (numPairs > 0) {
      int wgs = 256;
      int numGroups = (numPairs + wgs - 1) / wgs;
      transposeSquareInPlaceKernel<<<numGroups, wgs>>>(data + offA, n);
    }
  }

  // Step 4: Convert pivots on GPU into devDensePivots at current offset
  int64_t pivotOff = densePivotWriteOffset_;
  ensureDensePivotCapacity(pivotOff + minMN);
  {
    int wgs = 256;
    int numGroups = (minMN + wgs - 1) / wgs;
    convertPivotsKernel<<<numGroups, wgs>>>(devGetrfPivots.ptr,
                                             devDensePivots.ptr + pivotOff, minMN);
  }
  lastGetrfPivotOff_ = pivotOff;
  lastGetrfPivotN_ = minMN;
  densePivotWriteOffset_ = pivotOff + minMN;

  // Step 5: Defer D→H pivot copy to flush() — no sync barrier here
  deferredPivotCopies_.push_back({pivots, pivotOff, minMN});

  return 0;
}

template <>
int CudaNumericCtx<float>::getrf(int64_t m, int64_t n, float* data, int64_t offA,
                                  int64_t* pivots) {
  if (m <= 0 || n <= 0) return 0;

#ifdef BASPACHO_USE_BLAS
  if (cpuBlasMode_) {
    int64_t minMN = std::min(m, n);
    float* h = hostData_.data();
    if (m == n) transposeSquareInPlace(h + offA, n);
    std::vector<BLAS_INT> ipiv(minMN);
    int info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, m, n, h + offA, m, ipiv.data());
    if (m == n) transposeSquareInPlace(h + offA, n);
    for (int64_t i = 0; i < minMN; i++) pivots[i] = ipiv[i] - 1;
    return info;
  }
#endif

  flushGemmBatch();  // Ensure all pending GEMM updates are complete before factoring
  int64_t minMN = std::min(m, n);

  // Step 1: Transpose row-major → col-major on GPU (in-place for square)
  if (m == n) {
    int64_t numPairs = n * (n - 1) / 2;
    if (numPairs > 0) {
      int wgs = 256;
      int numGroups = (numPairs + wgs - 1) / wgs;
      transposeSquareInPlaceKernel<<<numGroups, wgs>>>(data + offA, n);
    }
  }

  // Step 2: cuSolver getrf (col-major LU with partial pivoting)
  int workspaceSize;
  cusolverCHECK(cusolverDnSgetrf_bufferSize(sym.cusolverDnH, m, n, data + offA, m, &workspaceSize));

  devTempBuffer.resizeToAtLeast(workspaceSize);
  devGetrfPivots.resizeToAtLeast(minMN);
  devPotrfSingIndex.resizeToAtLeast(1);

  cusolverCHECK(cusolverDnSgetrf(sym.cusolverDnH, m, n, data + offA, m,
                                  devTempBuffer.ptr, devGetrfPivots.ptr, devPotrfSingIndex.ptr));

  // Step 3: Transpose col-major → row-major on GPU
  if (m == n) {
    int64_t numPairs = n * (n - 1) / 2;
    if (numPairs > 0) {
      int wgs = 256;
      int numGroups = (numPairs + wgs - 1) / wgs;
      transposeSquareInPlaceKernel<<<numGroups, wgs>>>(data + offA, n);
    }
  }

  // Step 4: Convert pivots on GPU into devDensePivots at current offset
  int64_t pivotOff = densePivotWriteOffset_;
  ensureDensePivotCapacity(pivotOff + minMN);
  {
    int wgs = 256;
    int numGroups = (minMN + wgs - 1) / wgs;
    convertPivotsKernel<<<numGroups, wgs>>>(devGetrfPivots.ptr,
                                             devDensePivots.ptr + pivotOff, minMN);
  }
  lastGetrfPivotOff_ = pivotOff;
  lastGetrfPivotN_ = minMN;
  densePivotWriteOffset_ = pivotOff + minMN;

  // Step 5: Defer D→H pivot copy to flush() — no sync barrier here
  deferredPivotCopies_.push_back({pivots, pivotOff, minMN});

  return 0;
}

// trsmLowerUnit: solve L * X = B, L is m×m unit lower, B is m×n (row-major)
// Row-major → col-major: CblasRight, CblasUpper, CblasNoTrans, CblasUnit
template <>
void CudaNumericCtx<double>::trsmLowerUnit(int64_t m, int64_t n, const double* L, int64_t offL,
                                            double* B, int64_t offB, int64_t ldb) {
#ifdef BASPACHO_USE_BLAS
  if (cpuBlasMode_) {
    double* h = hostData_.data();
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasUnit,
                n, m, 1.0, h + offL, m, h + offB, ldb);
    return;
  }
#endif
  double alpha(1.0);
  cublasCHECK(cublasDtrsm(sym.cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                           CUBLAS_DIAG_UNIT, n, m, &alpha, L + offL, m, B + offB, ldb));
}

template <>
void CudaNumericCtx<float>::trsmLowerUnit(int64_t m, int64_t n, const float* L, int64_t offL,
                                           float* B, int64_t offB, int64_t ldb) {
#ifdef BASPACHO_USE_BLAS
  if (cpuBlasMode_) {
    float* h = hostData_.data();
    cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasUnit,
                n, m, 1.0f, h + offL, m, h + offB, ldb);
    return;
  }
#endif
  float alpha(1.0);
  cublasCHECK(cublasStrsm(sym.cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                           CUBLAS_DIAG_UNIT, n, m, &alpha, L + offL, m, B + offB, ldb));
}

// trsmUpperRight: solve X * U = B, U is n×n upper, B is m×n (row-major)
// Row-major → col-major: CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit
template <>
void CudaNumericCtx<double>::trsmUpperRight(int64_t m, int64_t n, const double* U, int64_t offU,
                                             double* B, int64_t offB, int64_t /*ldb*/) {
#ifdef BASPACHO_USE_BLAS
  if (cpuBlasMode_) {
    double* h = hostData_.data();
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                n, m, 1.0, h + offU, n, h + offB, n);
    return;
  }
#endif
  double alpha(1.0);
  cublasCHECK(cublasDtrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                           CUBLAS_DIAG_NON_UNIT, n, m, &alpha, U + offU, n, B + offB, n));
}

template <>
void CudaNumericCtx<float>::trsmUpperRight(int64_t m, int64_t n, const float* U, int64_t offU,
                                            float* B, int64_t offB, int64_t /*ldb*/) {
#ifdef BASPACHO_USE_BLAS
  if (cpuBlasMode_) {
    float* h = hostData_.data();
    cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                n, m, 1.0f, h + offU, n, h + offB, n);
    return;
  }
#endif
  float alpha(1.0);
  cublasCHECK(cublasStrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                           CUBLAS_DIAG_NON_UNIT, n, m, &alpha, U + offU, n, B + offB, n));
}

// saveGemm: C -= L * U (row-major)
// Col-major view: C_cm -= U_cm * L_cm (transposed data, NoTrans, NoTrans)
template <>
void CudaNumericCtx<double>::saveGemm(int64_t m, int64_t n, int64_t k, const double* L,
                                       int64_t offL, int64_t ldL, const double* U, int64_t offU,
                                       int64_t ldU, double* C, int64_t offC, int64_t ldC) {
  sym.gemmCalls++;

#ifdef BASPACHO_USE_BLAS
  if (cpuBlasMode_) {
    double* h = hostData_.data();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, k, -1.0,
                h + offU, ldU, h + offL, ldL, 1.0, h + offC, ldC);
    return;
  }
#endif

  // Batch small GEMMs into a single kernel dispatch
  if (m * n <= kBatchGemmMaxMN) {
    if (gemmDataPtr_ == nullptr) {
      gemmDataPtr_ = const_cast<double*>(L);
    }
    gemmBatch_.push_back({offL, offU, offC, (int32_t)m, (int32_t)n, (int32_t)k,
                          (int32_t)ldL, (int32_t)ldU, (int32_t)ldC});
    return;
  }

  flushGemmBatch();
  double alpha(-1.0), beta(1.0);
  cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, U + offU, ldU,
                           L + offL, ldL, &beta, C + offC, ldC));
}

template <>
void CudaNumericCtx<float>::saveGemm(int64_t m, int64_t n, int64_t k, const float* L,
                                      int64_t offL, int64_t ldL, const float* U, int64_t offU,
                                      int64_t ldU, float* C, int64_t offC, int64_t ldC) {
  sym.gemmCalls++;

#ifdef BASPACHO_USE_BLAS
  if (cpuBlasMode_) {
    float* h = hostData_.data();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, k, -1.0f,
                h + offU, ldU, h + offL, ldL, 1.0f, h + offC, ldC);
    return;
  }
#endif

  // Batch small GEMMs into a single kernel dispatch to avoid cuBLAS overhead
  if (m * n <= kBatchGemmMaxMN) {
    // All pointers must refer to the same data buffer (they do in BaSpaCho)
    if (gemmDataPtr_ == nullptr) {
      gemmDataPtr_ = const_cast<float*>(L);  // Track the base data pointer
    }
    gemmBatch_.push_back({offL, offU, offC, (int32_t)m, (int32_t)n, (int32_t)k,
                          (int32_t)ldL, (int32_t)ldU, (int32_t)ldC});
    return;
  }

  // Large GEMM: flush any pending batch first, then use cuBLAS
  flushGemmBatch();
  float alpha(-1.0), beta(1.0);
  cublasCHECK(cublasSgemm(sym.cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, U + offU, ldU,
                           L + offL, ldL, &beta, C + offC, ldC));
}

// applyRowPerm: GPU kernel - use GPU-resident pivots from deferred getrf when available,
// otherwise copy small pivots H→D. Run permutation kernel on device.
template <>
void CudaNumericCtx<double>::applyRowPerm(int64_t* pivots, int64_t n, double* data, int64_t offData,
                                           int64_t ld, int64_t numCols) {
  if (n <= 0 || numCols <= 0) return;

  if (cpuBlasMode_) {
    double* h = hostData_.data();
    for (int64_t i = 0; i < n; i++) {
      int64_t swapRow = pivots[i];
      if (swapRow != i) {
        for (int64_t c = 0; c < numCols; c++) {
          std::swap(h[offData + i + c * ld], h[offData + swapRow + c * ld]);
        }
      }
    }
    return;
  }

  int64_t* devPivPtr;
  if (n == lastGetrfPivotN_ && lastGetrfPivotOff_ >= 0) {
    // Use GPU-resident pivots from deferred getrf (no sync needed)
    devPivPtr = devDensePivots.ptr + lastGetrfPivotOff_;
  } else {
    // Pivots not from recent getrf — upload from CPU
    devPivotBuf.resizeToAtLeast(n);
    cuCHECK(cudaMemcpy(devPivotBuf.ptr, pivots, n * sizeof(int64_t), cudaMemcpyHostToDevice));
    devPivPtr = devPivotBuf.ptr;
  }

  // Single block launch (sequential pivot dependency requires __syncthreads)
  int wgs = std::min((int64_t)256, numCols);
  applyRowPermKernel<<<1, wgs>>>(devPivPtr, n, data, offData, ld, numCols);
}

template <>
void CudaNumericCtx<float>::applyRowPerm(int64_t* pivots, int64_t n, float* data, int64_t offData,
                                          int64_t ld, int64_t numCols) {
  if (n <= 0 || numCols <= 0) return;

  if (cpuBlasMode_) {
    float* h = hostData_.data();
    for (int64_t i = 0; i < n; i++) {
      int64_t swapRow = pivots[i];
      if (swapRow != i) {
        for (int64_t c = 0; c < numCols; c++) {
          std::swap(h[offData + i + c * ld], h[offData + swapRow + c * ld]);
        }
      }
    }
    return;
  }

  int64_t* devPivPtr;
  if (n == lastGetrfPivotN_ && lastGetrfPivotOff_ >= 0) {
    // Use GPU-resident pivots from deferred getrf (no sync needed)
    devPivPtr = devDensePivots.ptr + lastGetrfPivotOff_;
  } else {
    // Pivots not from recent getrf — upload from CPU
    devPivotBuf.resizeToAtLeast(n);
    cuCHECK(cudaMemcpy(devPivotBuf.ptr, pivots, n * sizeof(int64_t), cudaMemcpyHostToDevice));
    devPivPtr = devPivotBuf.ptr;
  }

  int wgs = std::min((int64_t)256, numCols);
  applyRowPermKernel<<<1, wgs>>>(devPivPtr, n, data, offData, ld, numCols);
}

template <typename T>
string printVec(const vector<T>& ints) {
  stringstream ss;
  ss << "[";
  bool first = true;
  for (auto c : ints) {
    ss << (first ? "" : ", ") << c;
    first = false;
  }
  ss << "]";
  return ss.str();
}

template <typename T>
struct CudaNumericCtx<vector<T*>> : NumericCtx<vector<T*>> {
  CudaNumericCtx(const CudaSymbolicCtx& sym, int64_t bufSize, int64_t numSpans, int batchSize)
      : spanToChainOffset(numSpans), devTempBufs(batchSize, nullptr), sym(sym) {
    devAllJoinedTempBufs.resizeToAtLeast(bufSize * batchSize);
    for (int i = 0; i < batchSize; i++) {
      devTempBufs[i] = devAllJoinedTempBufs.ptr + bufSize * i;
    }
    devTempBufsDev.load(devTempBufs);
    devSpanToChainOffset.resizeToAtLeast(spanToChainOffset.size());
  }

  virtual ~CudaNumericCtx() override {}

  virtual void pseudoFactorSpans(vector<T*>* data, int64_t spanBegin, int64_t spanEnd) override {
    BASPACHO_UNUSED(data, spanBegin, spanEnd);
    throw std::runtime_error("pseudo factor not implemented for batched ops");
  }

  virtual void doElimination(const SymElimCtx& elimData, vector<T*>* data, int64_t lumpsBegin,
                             int64_t lumpsEnd) override {
    const CudaSymElimCtx* pElim = dynamic_cast<const CudaSymElimCtx*>(&elimData);
    BASPACHO_CHECK_NOTNULL(pElim);
    const CudaSymElimCtx& elim = *pElim;

    auto timer = elim.elimStat.instance<CudaSyncOps>();
    devPtrsA.load(*data, 0);

    int batchWgs = 32;
    while (batchWgs / 2 >= (int)data->size()) {
      batchWgs /= 2;
    }
    int batchGroups = (data->size() + batchWgs - 1) / batchWgs;
    int wgs = 32 / batchWgs;
    int numGroups = (lumpsEnd - lumpsBegin + wgs - 1) / wgs;
    dim3 gridDim(numGroups, batchGroups);
    dim3 blockDim(wgs, batchWgs);

    factor_lumps_kernel<T*><<<gridDim, blockDim>>>(
        sym.devLumpStart.ptr, sym.devChainColPtr.ptr, sym.devChainData.ptr, sym.devBoardColPtr.ptr,
        sym.devBoardChainColOrd.ptr, sym.devChainRowsTillEnd.ptr, devPtrsA.ptr, lumpsBegin,
        lumpsEnd, Batched{.batchSize = (int)data->size(), .batchIndex = 0});

#if 0
    // double inner loop
    sparse_elim_2loops_kernel<T*><<<numGroups, wgs>>>(
        sym.devChainColPtr.ptr, sym.devLumpStart.ptr,
        sym.devChainRowSpan.ptr, sym.devSpanStart.ptr, sym.devChainData.ptr,
        sym.devSpanToLump.ptr, sym.devSpanOffsetInLump.ptr, devPtrsA.ptr,
        lumpsBegin, lumpsEnd,
        Batched{.batchSize = (int)data->size(), .batchIndex = 0});
#else
    int wgs2 = 32 / batchWgs;
    int numGroups2 = (elim.numBlockPairs + wgs2 - 1) / wgs2;
    dim3 gridDim2(numGroups2, batchGroups);
    dim3 blockDim2(wgs2, batchWgs);
    sparse_elim_straight_kernel<T*><<<gridDim2, blockDim2>>>(
        sym.devChainColPtr.ptr, sym.devLumpStart.ptr, sym.devChainRowSpan.ptr, sym.devSpanStart.ptr,
        sym.devChainData.ptr, sym.devSpanToLump.ptr, sym.devSpanOffsetInLump.ptr, devPtrsA.ptr,
        lumpsBegin, lumpsEnd, elim.makeBlockPairEnumStraight.ptr, elim.numBlockPairs,
        Batched{.batchSize = (int)data->size(), .batchIndex = 0});
#endif
  }

  virtual void potrf(int64_t n, vector<T*>* data, int64_t offA) override;

  virtual void trsm(int64_t n, int64_t k, vector<T*>* data, int64_t offA, int64_t offB) override;

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const vector<T*>* data,
                            int64_t offset) override;

  virtual void prepareAssemble(int64_t targetLump) override {
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    // FIXME: compute on CPU and copy, not ideal
    for (int64_t i = skel.chainColPtr[targetLump], iEnd = skel.chainColPtr[targetLump + 1];
         i < iEnd; i++) {
      spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
    }
    cuCHECK(cudaMemcpy(devSpanToChainOffset.ptr, spanToChainOffset.data(),
                       spanToChainOffset.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
  }

  virtual void assemble(vector<T*>* data, int64_t rectRowBegin,
                        int64_t dstStride,  //
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    BASPACHO_CHECK_LE(data->size(), devTempBufs.size());
    auto timer = sym.asmblStat.instance<CudaSyncOps>(sizeof(T) + data->size() * 100, numBlockRows,
                                                     numBlockCols);
    devPtrsA.load(*data, 0);
    const int64_t* pChainRowsTillEnd = sym.devChainRowsTillEnd.ptr + srcColDataOffset;
    const int64_t* pToSpan = sym.devChainRowSpan.ptr + srcColDataOffset;
    const int64_t* pSpanToChainOffset = devSpanToChainOffset.ptr;
    const int64_t* pSpanOffsetInLump = sym.devSpanOffsetInLump.ptr;

    int batchWgs = 32;
    while (batchWgs / 2 >= (int)data->size()) {
      batchWgs /= 2;
    }
    int batchGroups = (data->size() + batchWgs - 1) / batchWgs;
    int wgs = 32 / batchWgs;
    int numGroups = (numBlockRows * numBlockCols + wgs - 1) / wgs;
    dim3 gridDim(numGroups, batchGroups);
    dim3 blockDim(wgs, batchWgs);
    assemble_kernel<T*><<<gridDim, blockDim>>>(
        numBlockRows, numBlockCols, rectRowBegin, srcRectWidth, dstStride, pChainRowsTillEnd,
        pToSpan, pSpanToChainOffset, pSpanOffsetInLump, devTempBufsDev.ptr, devPtrsA.ptr,
        Batched{.batchSize = (int)data->size(), .batchIndex = 0});
  }

  DevMirror<T> devAllJoinedTempBufs;
  vector<T*> devTempBufs;
  DevPtrMirror<T> devTempBufsDev;
  DevMirror<int> devPotrfSingIndex;
  DevPtrMirror<T> devPtrsA, devPtrsB;
  DevMirror<int64_t> devSpanToChainOffset;
  vector<int64_t> spanToChainOffset;

  const CudaSymbolicCtx& sym;
};

template <>
void CudaNumericCtx<vector<double*>>::potrf(int64_t n, vector<double*>* data, int64_t offA) {
  auto timer = sym.potrfStat.instance<CudaSyncOps>(sizeof(double) + data->size() * 100, n);
  devPtrsA.load(*data, offA);
  devPotrfSingIndex.resizeToAtLeast(data->size());

  cusolverCHECK(cusolverDnDpotrfBatched(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, devPtrsA.ptr, n,
                                        devPotrfSingIndex.ptr, data->size()));

  // TODO: handle/report singularity
  // vector<int> info(data->size());
  // devPotrfSingIndex.get(info);
  // cout << "info: " << printVec(info) << endl;
}

template <>
void CudaNumericCtx<vector<float*>>::potrf(int64_t n, vector<float*>* data, int64_t offA) {
  auto timer = sym.potrfStat.instance<CudaSyncOps>(sizeof(float) + data->size() * 100, n);
  devPtrsA.load(*data, offA);
  devPotrfSingIndex.resizeToAtLeast(data->size());

  cusolverCHECK(cusolverDnSpotrfBatched(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, devPtrsA.ptr, n,
                                        devPotrfSingIndex.ptr, data->size()));

  // TODO: handle/report singularity
  // vector<int> info(data->size());
  // devPotrfSingIndex.get(info);
  // cout << "info: " << printVec(info) << endl;
}

template <>
void CudaNumericCtx<vector<double*>>::trsm(int64_t n, int64_t k, vector<double*>* data,
                                           int64_t offA, int64_t offB) {
  auto timer = sym.trsmStat.instance<CudaSyncOps>(sizeof(double) + data->size() * 100, n, k);
  devPtrsA.load(*data, offA);
  devPtrsB.load(*data, offB);

  double alpha(1.0);
  cublasCHECK(cublasDtrsmBatched(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                                 CUBLAS_DIAG_NON_UNIT, n, k, &alpha, devPtrsA.ptr, n, devPtrsB.ptr,
                                 n, data->size()));
}

template <>
void CudaNumericCtx<vector<float*>>::trsm(int64_t n, int64_t k, vector<float*>* data, int64_t offA,
                                          int64_t offB) {
  auto timer = sym.trsmStat.instance<CudaSyncOps>(sizeof(float) + data->size() * 100, n, k);
  devPtrsA.load(*data, offA);
  devPtrsB.load(*data, offB);

  float alpha(1.0);
  cublasCHECK(cublasStrsmBatched(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                                 CUBLAS_DIAG_NON_UNIT, n, k, &alpha, devPtrsA.ptr, n, devPtrsB.ptr,
                                 n, data->size()));
}

template <>
void CudaNumericCtx<vector<double*>>::saveSyrkGemm(int64_t m, int64_t n, int64_t k,
                                                   const vector<double*>* data, int64_t offset) {
  BASPACHO_CHECK_LE(data->size(), devTempBufs.size());
  auto timer = sym.sygeStat.instance<CudaSyncOps>(sizeof(double) + data->size() * 100, m, n, k);
  devPtrsA.load(*data, offset);
  double alpha(1.0), beta(0.0);
  cublasCHECK(cublasDgemmBatched(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k, &alpha,
                                 devPtrsA.ptr, k, devPtrsA.ptr, k, &beta, devTempBufsDev.ptr, m,
                                 data->size()));
  sym.gemmCalls++;
}

template <>
void CudaNumericCtx<vector<float*>>::saveSyrkGemm(int64_t m, int64_t n, int64_t k,
                                                  const vector<float*>* data, int64_t offset) {
  BASPACHO_CHECK_LE(data->size(), devTempBufs.size());
  auto timer = sym.sygeStat.instance<CudaSyncOps>(sizeof(float) + data->size() * 100, m, n, k);
  devPtrsA.load(*data, offset);
  float alpha(1.0), beta(0.0);
  cublasCHECK(cublasSgemmBatched(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k, &alpha,
                                 devPtrsA.ptr, k, devPtrsA.ptr, k, &beta, devTempBufsDev.ptr, m,
                                 data->size()));
  sym.gemmCalls++;
}

// SolveCtx helpers
template <typename T>
__device__ static inline void stridedTransAdd(T* dst, int64_t dstStride, const T* src,
                                              int64_t srcStride, int64_t rSize, int64_t cSize) {
  for (uint j = 0; j < rSize; j++) {
    T* pDst = dst + j;
    for (uint i = 0; i < cSize; i++) {
      *pDst += src[i];
      pDst += dstStride;
    }
    src += srcStride;
  }
}

template <typename T>
__device__ static inline void stridedTransSet(T* dst, int64_t dstStride, const T* src,
                                              int64_t srcStride, int64_t rSize, int64_t cSize) {
  for (uint j = 0; j < rSize; j++) {
    T* pDst = dst + j;
    for (uint i = 0; i < cSize; i++) {
      *pDst = src[i];
      pDst += dstStride;
    }
    src += srcStride;
  }
}

template <typename TT, typename B>
__global__ void assembleVec_kernel(const int64_t* chainRowsTillEnd, const int64_t* toSpan,
                                   const int64_t* spanStarts, const TT* AB, int64_t numColItems,
                                   TT* CB, int64_t ldc, int64_t nRHS, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(CB)[0])>>;
  T* C = batch.get(CB);
  const T* A = batch.get(AB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numColItems) {
    return;
  }
  int64_t rowOffset = chainRowsTillEnd[i - 1] - chainRowsTillEnd[-1];
  int64_t span = toSpan[i];
  int64_t spanStart = spanStarts[span];
  int64_t spanSize = spanStarts[span + 1] - spanStart;

  stridedTransAdd(C + spanStart, ldc, A + rowOffset * nRHS, nRHS, spanSize, nRHS);
}

template <typename TT, typename B>
__global__ void assembleVecT_kernel(const int64_t* chainRowsTillEnd, const int64_t* toSpan,
                                    const int64_t* spanStarts, const TT* CB, int64_t ldc,
                                    int64_t nRHS, TT* AB, int64_t numColItems, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(AB)[0])>>;
  const T* C = batch.get(CB);
  T* A = batch.get(AB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numColItems) {
    return;
  }
  int64_t rowOffset = chainRowsTillEnd[i - 1] - chainRowsTillEnd[-1];
  int64_t span = toSpan[i];
  int64_t spanStart = spanStarts[span];
  int64_t spanSize = spanStarts[span + 1] - spanStart;

  stridedTransSet(A + rowOffset * nRHS, nRHS, C + spanStart, ldc, nRHS, spanSize);
}

// kernels for sparse-elim solve
template <typename TT, typename B>
__global__ void sparseElim_diagSolveL(const int64_t* lumpStarts, const int64_t* chainColPtr,
                                      const int64_t* chainData, const TT* dataB, TT* vB,
                                      int64_t ldc, int64_t nRHS, int64_t lumpIndexStart,
                                      int64_t lumpIndexEnd, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  const T* data = batch.get(dataB);
  T* v = batch.get(vB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t lump = lumpIndexStart + i;
  if (lump >= lumpIndexEnd) {
    return;
  }

  int64_t lumpStart = lumpStarts[lump];
  int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
  int64_t colStart = chainColPtr[lump];
  int64_t diagDataPtr = chainData[colStart];

  for (int i = 0; i < nRHS; i++) {
    solveUpperT(data + diagDataPtr, lumpSize, lumpSize, v + lumpStart + ldc * i);
  }
}

template <typename TT, typename B>
__global__ void sparseElim_subDiagMult(const int64_t* lumpStarts, const int64_t* spanStarts,
                                       const int64_t* chainColPtr, const int64_t* chainRowSpan,
                                       const int64_t* chainData, const TT* dataB, TT* vB,
                                       int64_t ldc, int64_t nRHS, int64_t lumpIndexStart,
                                       int64_t lumpIndexEnd, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  const T* data = batch.get(dataB);
  T* v = batch.get(vB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t lump = lumpIndexStart + i;
  if (lump >= lumpIndexEnd) {
    return;
  }

  int64_t lumpStart = lumpStarts[lump];
  int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
  int64_t colStart = chainColPtr[lump];
  int64_t colEnd = chainColPtr[lump + 1];
  OuterStridedCMajMatM<T> matC(v + lumpStart, lumpSize, nRHS, OuterStride(ldc));

  for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
    int64_t rowSpan = chainRowSpan[colPtr];
    int64_t rowSpanStart = spanStarts[rowSpan];
    int64_t rowSpanSize = spanStarts[rowSpan + 1] - rowSpanStart;
    int64_t blockPtr = chainData[colPtr];
    Eigen::Map<const MatRMaj<T>> block(data + blockPtr, rowSpanSize, lumpSize);
    OuterStridedCMajMatM<T> matQ(v + rowSpanStart, rowSpanSize, nRHS, OuterStride(ldc));
    // matQ -= block * matC;
    locked_sub_AxB(matQ, block, matC);
  }
}

// kernels for sparse-elim solve
template <typename TT, typename B>
__global__ void sparseElim_diagSolveLt(const int64_t* lumpStarts, const int64_t* chainColPtr,
                                       const int64_t* chainData, const TT* dataB, TT* vB,
                                       int64_t ldc, int64_t nRHS, int64_t lumpIndexStart,
                                       int64_t lumpIndexEnd, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  const T* data = batch.get(dataB);
  T* v = batch.get(vB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t lump = lumpIndexStart + i;
  if (lump >= lumpIndexEnd) {
    return;
  }

  int64_t lumpStart = lumpStarts[lump];
  int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
  int64_t colStart = chainColPtr[lump];
  int64_t diagDataPtr = chainData[colStart];

  for (int i = 0; i < nRHS; i++) {
    solveUpper(data + diagDataPtr, lumpSize, lumpSize, v + lumpStart + ldc * i);
  }
}

template <typename TT, typename B>
__global__ void sparseElim_subDiagMultT(const int64_t* lumpStarts, const int64_t* spanStarts,
                                        const int64_t* chainColPtr, const int64_t* chainRowSpan,
                                        const int64_t* chainData, const TT* dataB, TT* vB,
                                        int64_t ldc, int64_t nRHS, int64_t lumpIndexStart,
                                        int64_t lumpIndexEnd, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  const T* data = batch.get(dataB);
  T* v = batch.get(vB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t lump = lumpIndexStart + i;
  if (lump >= lumpIndexEnd) {
    return;
  }

  int64_t lumpStart = lumpStarts[lump];
  int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
  int64_t colStart = chainColPtr[lump];
  int64_t colEnd = chainColPtr[lump + 1];
  OuterStridedCMajMatM<T> matC(v + lumpStart, lumpSize, nRHS, OuterStride(ldc));

  for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
    int64_t rowSpan = chainRowSpan[colPtr];
    int64_t rowSpanStart = spanStarts[rowSpan];
    int64_t rowSpanSize = spanStarts[rowSpan + 1] - rowSpanStart;
    int64_t blockPtr = chainData[colPtr];
    Eigen::Map<const MatRMaj<T>> block(data + blockPtr, rowSpanSize, lumpSize);
    OuterStridedCMajMatM<T> matQ(v + rowSpanStart, rowSpanSize, nRHS, OuterStride(ldc));
    // matC -= block * matQ;
    locked_sub_ATxB(matC, block, matQ);
  }
}

template <typename T>
struct CudaSolveCtx : SolveCtx<T> {
  CudaSolveCtx(const CudaSymbolicCtx& sym, int64_t nRHS) : sym(sym), nRHS(nRHS) {
    devSolveBuf.resizeToAtLeast(sym.skel.order() * nRHS);
  }
  virtual ~CudaSolveCtx() override {}

  virtual void sparseElimSolveL(const SymElimCtx& /*elimData*/, const T* data, int64_t lumpsBegin,
                                int64_t lumpsEnd, T* C, int64_t ldc) override {
    auto timer = sym.solveSparseLStat.instance<CudaSyncOps>();

    int wgs = 32;
    int numGroups = (lumpsEnd - lumpsBegin + wgs - 1) / wgs;
    sparseElim_diagSolveL<T><<<numGroups, wgs>>>(sym.devLumpStart.ptr, sym.devChainColPtr.ptr,
                                                 sym.devChainData.ptr, data, C, ldc, nRHS,
                                                 lumpsBegin, lumpsEnd, Plain{});

    // TODO: consider "straightening" inner loop
    sparseElim_subDiagMult<T><<<numGroups, wgs>>>(
        sym.devLumpStart.ptr, sym.devSpanStart.ptr, sym.devChainColPtr.ptr, sym.devChainRowSpan.ptr,
        sym.devChainData.ptr, data, C, ldc, nRHS, lumpsBegin, lumpsEnd, Plain{});
  }

  virtual void sparseElimSolveLt(const SymElimCtx& /*elimData*/, const T* data, int64_t lumpsBegin,
                                 int64_t lumpsEnd, T* C, int64_t ldc) override {
    auto timer = sym.solveSparseLtStat.instance<CudaSyncOps>();

    int wgs = 32;
    int numGroups = (lumpsEnd - lumpsBegin + wgs - 1) / wgs;

    // TODO: consider "straightening" inner loop
    sparseElim_subDiagMultT<T><<<numGroups, wgs>>>(
        sym.devLumpStart.ptr, sym.devSpanStart.ptr, sym.devChainColPtr.ptr, sym.devChainRowSpan.ptr,
        sym.devChainData.ptr, data, C, ldc, nRHS, lumpsBegin, lumpsEnd, Plain{});

    sparseElim_diagSolveLt<T><<<numGroups, wgs>>>(sym.devLumpStart.ptr, sym.devChainColPtr.ptr,
                                                  sym.devChainData.ptr, data, C, ldc, nRHS,
                                                  lumpsBegin, lumpsEnd, Plain{});
  }

  // LU sparse elimination forward solve: unit L (skip diagonal, just scatter)
  virtual void sparseElimSolveLUnit(const SymElimCtx& /*elimData*/, const T* data,
                                    int64_t lumpsBegin, int64_t lumpsEnd, T* C,
                                    int64_t ldc) override {
    int64_t numLumps = lumpsEnd - lumpsBegin;
    if (numLumps <= 0) return;

    int wgs = 32;
    int numGroups = (numLumps + wgs - 1) / wgs;

    // No diagonal solve for unit L (diagonal is implicitly 1)
    // Only dispatch below-diagonal scatter: v[rowSpan] -= L_below * v[lump]
    sparseElim_subDiagMult<T><<<numGroups, wgs>>>(
        sym.devLumpStart.ptr, sym.devSpanStart.ptr, sym.devChainColPtr.ptr, sym.devChainRowSpan.ptr,
        sym.devChainData.ptr, data, C, ldc, nRHS, lumpsBegin, lumpsEnd, Plain{});
  }

  // LU sparse elimination backward solve: gather from upper triangle then U diagonal solve
  virtual void sparseElimSolveU(const SymElimCtx& /*elimData*/, const T* data, int64_t lumpsBegin,
                                int64_t lumpsEnd, T* C, int64_t ldc) override {
    int64_t numLumps = lumpsEnd - lumpsBegin;
    if (numLumps <= 0) return;

    int wgs = 32;
    int numGroups = (numLumps + wgs - 1) / wgs;

    int64_t upperDataBase = sym.skel.dataSize();

    // First: gather from upper triangle entries: v[lump] -= U_row * v[colSpan]
    sparseElim_upperGather<T><<<numGroups, wgs>>>(
        sym.devLumpStart.ptr, sym.devSpanStart.ptr, sym.devUpperChainRowPtr.ptr,
        sym.devUpperChainColSpan.ptr, sym.devUpperChainData.ptr, data, C, ldc, nRHS, lumpsBegin,
        lumpsEnd, upperDataBase, Plain{});

    // Then: diagonal U solve: v[lump] /= U_diagonal (row-major)
    sparseElim_diagDivU<T><<<numGroups, wgs>>>(sym.devLumpStart.ptr, sym.devChainColPtr.ptr,
                                               sym.devChainData.ptr, data, C, ldc, nRHS, lumpsBegin,
                                               lumpsEnd, Plain{});
  }

  virtual void symm(const T* data, int64_t offset, int64_t n, const T* C, int64_t offC, int64_t ldc,
                    T* D, int64_t ldd, T alpha) override;

  virtual void solveL(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                      int64_t ldc) override;

  virtual void gemv(const T* data, int64_t offM, int64_t nRows, int64_t nCols, const T* A,
                    int64_t offA, int64_t lda, T alpha) override;

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, T* C, int64_t ldc) override {
    auto timer = sym.solveAssVStat.instance<CudaSyncOps>();
    int wgs = 32;
    int numGroups = (numColItems + wgs - 1) / wgs;
    assembleVec_kernel<T><<<numGroups, wgs>>>(
        sym.devChainRowsTillEnd.ptr + chainColPtr, sym.devChainRowSpan.ptr + chainColPtr,
        sym.devSpanStart.ptr, devSolveBuf.ptr, numColItems, C, ldc, nRHS, Plain{});
  }

  virtual void solveLt(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                       int64_t ldc) override;

  virtual void gemvT(const T* data, int64_t offM, int64_t nRows, int64_t nCols, T* A, int64_t offA,
                     int64_t lda, T alpha) override;

  virtual void assembleVecT(const T* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    auto timer = sym.solveAssVTStat.instance<CudaSyncOps>();
    int wgs = 32;
    int numGroups = (numColItems + wgs - 1) / wgs;
    assembleVecT_kernel<T><<<numGroups, wgs>>>(
        sym.devChainRowsTillEnd.ptr + chainColPtr, sym.devChainRowSpan.ptr + chainColPtr,
        sym.devSpanStart.ptr, C, ldc, nRHS, devSolveBuf.ptr, numColItems, Plain{});
  }

  // LU solve methods
  virtual void solveLUnit(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                           int64_t ldc) override;

  virtual void solveU(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                      int64_t ldc) override;

  virtual void applyRowPermVec(const int64_t* pivots, int64_t n, T* vec,
                                int64_t ldVec) override;

  virtual void applyRowPermVecInv(const int64_t* pivots, int64_t n, T* vec,
                                   int64_t ldVec) override;

  virtual void gemvDirect(const T* data, int64_t offset, int64_t nRows, int64_t nCols, T* vec,
                           int64_t srcOff, int64_t dstOff, int64_t ldVec, T alpha) override;

  const CudaSymbolicCtx& sym;
  int64_t nRHS;
  DevMirror<T> devSolveBuf;
  DevMirror<int64_t> devPivotBuf;  // GPU buffer for LU pivots
};

template <>
void CudaSolveCtx<double>::symm(const double* data, int64_t offM, int64_t n, const double* C,
                                int64_t offC, int64_t ldc, double* D, int64_t ldd, double alpha) {
  auto timer = sym.symmStat.instance<CudaSyncOps>();
  double beta(1.0);
  cublasCHECK(cublasDsymm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, n, nRHS, &alpha,
                          data + offM, n, C + offC, ldc, &beta, D + offC, ldd));
}

template <>
void CudaSolveCtx<float>::symm(const float* data, int64_t offM, int64_t n, const float* C,
                               int64_t offC, int64_t ldc, float* D, int64_t ldd, float alpha) {
  auto timer = sym.symmStat.instance<CudaSyncOps>();
  float beta(1.0);
  cublasCHECK(cublasSsymm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, n, nRHS, &alpha,
                          data + offM, n, C + offC, ldc, &beta, D + offC, ldd));
}

template <>
void CudaSolveCtx<double>::solveL(const double* data, int64_t offM, int64_t n, double* C,
                                  int64_t offC, int64_t ldc) {
  auto timer = sym.solveLStat.instance<CudaSyncOps>();
  double alpha(1.0);
  cublasCHECK(cublasDtrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                          CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

template <>
void CudaSolveCtx<float>::solveL(const float* data, int64_t offM, int64_t n, float* C, int64_t offC,
                                 int64_t ldc) {
  auto timer = sym.solveLStat.instance<CudaSyncOps>();
  float alpha(1.0);
  cublasCHECK(cublasStrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                          CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

template <>
void CudaSolveCtx<double>::gemv(const double* data, int64_t offM, int64_t nRows, int64_t nCols,
                                const double* A, int64_t offA, int64_t lda, double alpha) {
  auto timer = sym.solveGemvStat.instance<CudaSyncOps>();
  double beta(0.0);
  cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, nRHS, nRows, nCols, &alpha,
                          A + offA, lda, data + offM, nCols, &beta, devSolveBuf.ptr, nRHS));
}

template <>
void CudaSolveCtx<float>::gemv(const float* data, int64_t offM, int64_t nRows, int64_t nCols,
                               const float* A, int64_t offA, int64_t lda, float alpha) {
  auto timer = sym.solveGemvStat.instance<CudaSyncOps>();
  float beta(0.0);
  cublasCHECK(cublasSgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, nRHS, nRows, nCols, &alpha,
                          A + offA, lda, data + offM, nCols, &beta, devSolveBuf.ptr, nRHS));
}

template <>
void CudaSolveCtx<double>::solveLt(const double* data, int64_t offM, int64_t n, double* C,
                                   int64_t offC, int64_t ldc) {
  auto timer = sym.solveLtStat.instance<CudaSyncOps>();
  double alpha(1.0);
  cublasCHECK(cublasDtrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                          CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

template <>
void CudaSolveCtx<float>::solveLt(const float* data, int64_t offM, int64_t n, float* C,
                                  int64_t offC, int64_t ldc) {
  auto timer = sym.solveLtStat.instance<CudaSyncOps>();
  float alpha(1.0);
  cublasCHECK(cublasStrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                          CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

template <>
void CudaSolveCtx<double>::gemvT(const double* data, int64_t offM, int64_t nRows, int64_t nCols,
                                 double* A, int64_t offA, int64_t lda, double alpha) {
  auto timer = sym.solveGemvTStat.instance<CudaSyncOps>();
  double beta(1.0);
  cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_N, CUBLAS_OP_C, nCols, nRHS, nRows, &alpha,
                          data + offM, nCols, devSolveBuf.ptr, nRHS, &beta, A + offA, lda));
}

template <>
void CudaSolveCtx<float>::gemvT(const float* data, int64_t offM, int64_t nRows, int64_t nCols,
                                float* A, int64_t offA, int64_t lda, float alpha) {
  auto timer = sym.solveGemvTStat.instance<CudaSyncOps>();
  float beta(1.0);
  cublasCHECK(cublasSgemm(sym.cublasH, CUBLAS_OP_N, CUBLAS_OP_C, nCols, nRHS, nRows, &alpha,
                          data + offM, nCols, devSolveBuf.ptr, nRHS, &beta, A + offA, lda));
}

// ============ LU SolveCtx implementations ============

// solveLUnit: solve L * x = b where L is unit lower triangular
// Row-major lower → col-major upper. Use OP_C to get lower from upper.
template <>
void CudaSolveCtx<double>::solveLUnit(const double* data, int64_t offM, int64_t n, double* C,
                                       int64_t offC, int64_t ldc) {
  double alpha(1.0);
  cublasCHECK(cublasDtrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                           CUBLAS_DIAG_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

template <>
void CudaSolveCtx<float>::solveLUnit(const float* data, int64_t offM, int64_t n, float* C,
                                      int64_t offC, int64_t ldc) {
  float alpha(1.0);
  cublasCHECK(cublasStrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                           CUBLAS_DIAG_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

// solveU: solve U * x = b where U is upper triangular (non-unit)
// Row-major upper → col-major lower. Use OP_C to get upper from lower.
template <>
void CudaSolveCtx<double>::solveU(const double* data, int64_t offM, int64_t n, double* C,
                                   int64_t offC, int64_t ldc) {
  double alpha(1.0);
  cublasCHECK(cublasDtrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C,
                           CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

template <>
void CudaSolveCtx<float>::solveU(const float* data, int64_t offM, int64_t n, float* C,
                                  int64_t offC, int64_t ldc) {
  float alpha(1.0);
  cublasCHECK(cublasStrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C,
                           CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

// applyRowPermVec: GPU kernel - copy small pivots H→D, run kernel on device
template <>
void CudaSolveCtx<double>::applyRowPermVec(const int64_t* pivots, int64_t n, double* vec,
                                            int64_t ldVec) {
  if (n <= 0) return;

  devPivotBuf.resizeToAtLeast(n);
  cuCHECK(cudaMemcpy(devPivotBuf.ptr, pivots, n * sizeof(int64_t), cudaMemcpyHostToDevice));

  // Single block (sequential pivot dependency)
  int wgs = std::min((int64_t)256, nRHS);
  applyRowPermVecKernel<<<1, wgs>>>(devPivotBuf.ptr, n, vec, ldVec, nRHS);
}

template <>
void CudaSolveCtx<float>::applyRowPermVec(const int64_t* pivots, int64_t n, float* vec,
                                           int64_t ldVec) {
  if (n <= 0) return;

  devPivotBuf.resizeToAtLeast(n);
  cuCHECK(cudaMemcpy(devPivotBuf.ptr, pivots, n * sizeof(int64_t), cudaMemcpyHostToDevice));

  int wgs = std::min((int64_t)256, nRHS);
  applyRowPermVecKernel<<<1, wgs>>>(devPivotBuf.ptr, n, vec, ldVec, nRHS);
}

// applyRowPermVecInv: GPU kernel - copy small pivots H→D, run kernel on device (reverse)
template <>
void CudaSolveCtx<double>::applyRowPermVecInv(const int64_t* pivots, int64_t n, double* vec,
                                               int64_t ldVec) {
  if (n <= 0) return;

  devPivotBuf.resizeToAtLeast(n);
  cuCHECK(cudaMemcpy(devPivotBuf.ptr, pivots, n * sizeof(int64_t), cudaMemcpyHostToDevice));

  int wgs = std::min((int64_t)256, nRHS);
  applyRowPermVecInvKernel<<<1, wgs>>>(devPivotBuf.ptr, n, vec, ldVec, nRHS);
}

template <>
void CudaSolveCtx<float>::applyRowPermVecInv(const int64_t* pivots, int64_t n, float* vec,
                                              int64_t ldVec) {
  if (n <= 0) return;

  devPivotBuf.resizeToAtLeast(n);
  cuCHECK(cudaMemcpy(devPivotBuf.ptr, pivots, n * sizeof(int64_t), cudaMemcpyHostToDevice));

  int wgs = std::min((int64_t)256, nRHS);
  applyRowPermVecInvKernel<<<1, wgs>>>(devPivotBuf.ptr, n, vec, ldVec, nRHS);
}

// gemvDirect: result += alpha * M * x, M is row-major (nRows × nCols)
// Same pattern as existing gemv but with different source/dest offsets
template <>
void CudaSolveCtx<double>::gemvDirect(const double* data, int64_t offset, int64_t nRows,
                                       int64_t nCols, double* vec, int64_t srcOff, int64_t dstOff,
                                       int64_t ldVec, double alpha) {
  double beta(1.0);
  // Row-major M(nRows×nCols) in col-major is M^T(nCols×nRows)
  // y += alpha * M * x = alpha * M_cm^T * x → use OP_C
  cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, nRows, nRHS, nCols, &alpha,
                           data + offset, nCols, vec + srcOff, ldVec, &beta, vec + dstOff, ldVec));
}

template <>
void CudaSolveCtx<float>::gemvDirect(const float* data, int64_t offset, int64_t nRows,
                                      int64_t nCols, float* vec, int64_t srcOff, int64_t dstOff,
                                      int64_t ldVec, float alpha) {
  float beta(1.0);
  cublasCHECK(cublasSgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, nRows, nRHS, nCols, &alpha,
                           data + offset, nCols, vec + srcOff, ldVec, &beta, vec + dstOff, ldVec));
}

// solve context, batched version
template <typename T>
struct CudaSolveCtx<vector<T*>> : SolveCtx<vector<T*>> {
  CudaSolveCtx(const CudaSymbolicCtx& sym, int64_t nRHS, int batchSize)
      : sym(sym), nRHS(nRHS), devSolveBufs(batchSize, nullptr) {
    int64_t solveBufSize = sym.skel.order() * nRHS;
    devAllJoinedSolveBufs.resizeToAtLeast(batchSize * solveBufSize);
    for (int i = 0; i < batchSize; i++) {
      devSolveBufs[i] = devAllJoinedSolveBufs.ptr + i * solveBufSize;
    }
    devSolveBufsDev.load(devSolveBufs);
  }
  virtual ~CudaSolveCtx() override {}

  virtual void sparseElimSolveL(const SymElimCtx& /*elimData*/, const vector<T*>* data,
                                int64_t lumpsBegin, int64_t lumpsEnd, vector<T*>* C,
                                int64_t ldc) override {
    auto timer = sym.solveSparseLStat.instance<CudaSyncOps>();

    devPtrsX.load(*data, 0);
    devPtrsY.load(*C, 0);

    int batchWgs = 32;
    while (batchWgs / 2 >= (int)C->size()) {
      batchWgs /= 2;
    }
    int batchGroups = (C->size() + batchWgs - 1) / batchWgs;
    int wgs = 32 / batchWgs;
    int numGroups = (lumpsEnd - lumpsBegin + wgs - 1) / wgs;
    dim3 gridDim(numGroups, batchGroups);
    dim3 blockDim(wgs, batchWgs);

    sparseElim_diagSolveL<T*>
        <<<gridDim, blockDim>>>(sym.devLumpStart.ptr, sym.devChainColPtr.ptr, sym.devChainData.ptr,
                                devPtrsX.ptr, devPtrsY.ptr, ldc, nRHS, lumpsBegin, lumpsEnd,
                                Batched{.batchSize = (int)C->size(), .batchIndex = 0});

    // TODO: consider "straightening" inner loop
    sparseElim_subDiagMult<T*><<<gridDim, blockDim>>>(
        sym.devLumpStart.ptr, sym.devSpanStart.ptr, sym.devChainColPtr.ptr, sym.devChainRowSpan.ptr,
        sym.devChainData.ptr, devPtrsX.ptr, devPtrsY.ptr, ldc, nRHS, lumpsBegin, lumpsEnd,
        Batched{.batchSize = (int)C->size(), .batchIndex = 0});
  }

  virtual void sparseElimSolveLt(const SymElimCtx& /*elimData*/, const vector<T*>* data,
                                 int64_t lumpsBegin, int64_t lumpsEnd, vector<T*>* C,
                                 int64_t ldc) override {
    auto timer = sym.solveSparseLtStat.instance<CudaSyncOps>();

    devPtrsX.load(*data, 0);
    devPtrsY.load(*C, 0);

    int batchWgs = 32;
    while (batchWgs / 2 >= (int)C->size()) {
      batchWgs /= 2;
    }
    int batchGroups = (C->size() + batchWgs - 1) / batchWgs;
    int wgs = 32 / batchWgs;
    int numGroups = (lumpsEnd - lumpsBegin + wgs - 1) / wgs;
    dim3 gridDim(numGroups, batchGroups);
    dim3 blockDim(wgs, batchWgs);

    // TODO: consider "straightening" inner loop
    sparseElim_subDiagMultT<T*><<<gridDim, blockDim>>>(
        sym.devLumpStart.ptr, sym.devSpanStart.ptr, sym.devChainColPtr.ptr, sym.devChainRowSpan.ptr,
        sym.devChainData.ptr, devPtrsX.ptr, devPtrsY.ptr, ldc, nRHS, lumpsBegin, lumpsEnd,
        Batched{.batchSize = (int)C->size(), .batchIndex = 0});

    sparseElim_diagSolveLt<T*>
        <<<gridDim, blockDim>>>(sym.devLumpStart.ptr, sym.devChainColPtr.ptr, sym.devChainData.ptr,
                                devPtrsX.ptr, devPtrsY.ptr, ldc, nRHS, lumpsBegin, lumpsEnd,
                                Batched{.batchSize = (int)C->size(), .batchIndex = 0});
  }

  virtual void symm(const vector<T*>* data, int64_t offset, int64_t n, const vector<T*>* C,
                    int64_t offC, int64_t ldc, vector<T*>* D, int64_t ldd, T alpha) override;

  virtual void solveL(const vector<T*>* data, int64_t offM, int64_t n, vector<T*>* C, int64_t offC,
                      int64_t ldc) override;

  virtual void gemv(const vector<T*>* data, int64_t offM, int64_t nRows, int64_t nCols,
                    const vector<T*>* A, int64_t offA, int64_t lda, T alpha) override;

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, vector<T*>* C,
                           int64_t ldc) override {
    auto timer = sym.solveAssVStat.instance<CudaSyncOps>();
    devPtrsX.load(*C, 0);
    int batchWgs = 32;
    while (batchWgs / 2 >= (int)C->size()) {
      batchWgs /= 2;
    }
    int batchGroups = (C->size() + batchWgs - 1) / batchWgs;
    int wgs = 32 / batchWgs;
    int numGroups = (numColItems + wgs - 1) / wgs;
    dim3 gridDim(numGroups, batchGroups);
    dim3 blockDim(wgs, batchWgs);
    assembleVec_kernel<T*><<<gridDim, blockDim>>>(
        sym.devChainRowsTillEnd.ptr + chainColPtr, sym.devChainRowSpan.ptr + chainColPtr,
        sym.devSpanStart.ptr, devSolveBufsDev.ptr, numColItems, devPtrsX.ptr, ldc, nRHS,
        Batched{.batchSize = (int)C->size(), .batchIndex = 0});
  }

  virtual void solveLt(const vector<T*>* data, int64_t offM, int64_t n, vector<T*>* C, int64_t offC,
                       int64_t ldc) override;

  virtual void gemvT(const vector<T*>* data, int64_t offM, int64_t nRows, int64_t nCols,
                     vector<T*>* A, int64_t offA, int64_t lda, T alpha) override;

  virtual void assembleVecT(const vector<T*>* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    auto timer = sym.solveAssVTStat.instance<CudaSyncOps>();
    devPtrsX.load(*C, 0);
    int batchWgs = 32;
    while (batchWgs / 2 >= (int)C->size()) {
      batchWgs /= 2;
    }
    int batchGroups = (C->size() + batchWgs - 1) / batchWgs;
    int wgs = 32 / batchWgs;
    int numGroups = (numColItems + wgs - 1) / wgs;
    dim3 gridDim(numGroups, batchGroups);
    dim3 blockDim(wgs, batchWgs);
    assembleVecT_kernel<T*><<<gridDim, blockDim>>>(
        sym.devChainRowsTillEnd.ptr + chainColPtr, sym.devChainRowSpan.ptr + chainColPtr,
        sym.devSpanStart.ptr, devPtrsX.ptr, ldc, nRHS, devSolveBufsDev.ptr, numColItems,
        Batched{.batchSize = (int)C->size(), .batchIndex = 0});
  }

  const CudaSymbolicCtx& sym;
  int64_t nRHS;
  DevMirror<T> devAllJoinedSolveBufs;
  vector<T*> devSolveBufs;
  DevPtrMirror<T> devSolveBufsDev;
  DevPtrMirror<T> devPtrsX, devPtrsY;
};

template <>
void CudaSolveCtx<vector<double*>>::symm(const vector<double*>* data, int64_t offset, int64_t n,
                                         const vector<double*>* C, int64_t offC, int64_t ldc,
                                         vector<double*>* D, int64_t ldd, double alpha) {
  auto timer = sym.symmStat.instance<CudaSyncOps>();
  BASPACHO_UNUSED(data, offset, n, C, offC, ldc, D, ldd, alpha);
  throw std::runtime_error("symm not implemented for batched ops");
}

template <>
void CudaSolveCtx<vector<float*>>::symm(const vector<float*>* data, int64_t offset, int64_t n,
                                        const vector<float*>* C, int64_t offC, int64_t ldc,
                                        vector<float*>* D, int64_t ldd, float alpha) {
  auto timer = sym.symmStat.instance<CudaSyncOps>();
  BASPACHO_UNUSED(data, offset, n, C, offC, ldc, D, ldd, alpha);
  throw std::runtime_error("symm not implemented for batched ops");
}

template <>
void CudaSolveCtx<vector<double*>>::solveL(const vector<double*>* data, int64_t offM, int64_t n,
                                           vector<double*>* C, int64_t offC, int64_t ldc) {
  auto timer = sym.solveLStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*C, offC);
  double alpha(1.0);
  cublasCHECK(cublasDtrsmBatched(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                                 CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, devPtrsX.ptr, n,
                                 devPtrsY.ptr, ldc, data->size()));
}

template <>
void CudaSolveCtx<vector<float*>>::solveL(const vector<float*>* data, int64_t offM, int64_t n,
                                          vector<float*>* C, int64_t offC, int64_t ldc) {
  auto timer = sym.solveLStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*C, offC);
  float alpha(1.0);
  cublasCHECK(cublasStrsmBatched(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                                 CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, devPtrsX.ptr, n,
                                 devPtrsY.ptr, ldc, data->size()));
}

template <>
void CudaSolveCtx<vector<double*>>::gemv(const vector<double*>* data, int64_t offM, int64_t nRows,
                                         int64_t nCols, const vector<double*>* A, int64_t offA,
                                         int64_t lda, double alpha) {
  auto timer = sym.solveGemvStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*A, offA);
  double beta(0.0);
  cublasCHECK(cublasDgemmBatched(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, nRHS, nRows, nCols, &alpha,
                                 devPtrsY.ptr, lda, devPtrsX.ptr, nCols, &beta, devSolveBufsDev.ptr,
                                 nRHS, data->size()));
}

template <>
void CudaSolveCtx<vector<float*>>::gemv(const vector<float*>* data, int64_t offM, int64_t nRows,
                                        int64_t nCols, const vector<float*>* A, int64_t offA,
                                        int64_t lda, float alpha) {
  auto timer = sym.solveGemvStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*A, offA);
  float beta(0.0);
  cublasCHECK(cublasSgemmBatched(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, nRHS, nRows, nCols, &alpha,
                                 devPtrsY.ptr, lda, devPtrsX.ptr, nCols, &beta, devSolveBufsDev.ptr,
                                 nRHS, data->size()));
}

template <>
void CudaSolveCtx<vector<double*>>::solveLt(const vector<double*>* data, int64_t offM, int64_t n,
                                            vector<double*>* C, int64_t offC, int64_t ldc) {
  auto timer = sym.solveLtStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*C, offC);
  double alpha(1.0);
  cublasCHECK(cublasDtrsmBatched(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, devPtrsX.ptr, n,
                                 devPtrsY.ptr, ldc, data->size()));
}

template <>
void CudaSolveCtx<vector<float*>>::solveLt(const vector<float*>* data, int64_t offM, int64_t n,
                                           vector<float*>* C, int64_t offC, int64_t ldc) {
  auto timer = sym.solveLtStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*C, offC);
  float alpha(1.0);
  cublasCHECK(cublasStrsmBatched(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, devPtrsX.ptr, n,
                                 devPtrsY.ptr, ldc, data->size()));
}

template <>
void CudaSolveCtx<vector<double*>>::gemvT(const vector<double*>* data, int64_t offM, int64_t nRows,
                                          int64_t nCols, vector<double*>* A, int64_t offA,
                                          int64_t lda, double alpha) {
  auto timer = sym.solveGemvTStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*A, offA);
  double beta(1.0);
  cublasCHECK(cublasDgemmBatched(sym.cublasH, CUBLAS_OP_N, CUBLAS_OP_C, nCols, nRHS, nRows, &alpha,
                                 devPtrsX.ptr, nCols, devSolveBufsDev.ptr, nRHS, &beta,
                                 devPtrsY.ptr, lda, data->size()));
}

template <>
void CudaSolveCtx<vector<float*>>::gemvT(const vector<float*>* data, int64_t offM, int64_t nRows,
                                         int64_t nCols, vector<float*>* A, int64_t offA,
                                         int64_t lda, float alpha) {
  auto timer = sym.solveGemvTStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*A, offA);
  float beta(1.0);
  cublasCHECK(cublasSgemmBatched(sym.cublasH, CUBLAS_OP_N, CUBLAS_OP_C, nCols, nRHS, nRows, &alpha,
                                 devPtrsX.ptr, nCols, devSolveBufsDev.ptr, nRHS, &beta,
                                 devPtrsY.ptr, lda, data->size()));
}

NumericCtxBase* CudaSymbolicCtx::createNumericCtxForType(type_index tIdx, int64_t tempBufSize,
                                                         int batchSize) {
  if (tIdx == type_index(typeid(double))) {
    BASPACHO_CHECK_EQ(batchSize, 1);
    return new CudaNumericCtx<double>(*this, tempBufSize, skel.spanStart.size() - 1);
  } else if (tIdx == type_index(typeid(float))) {
    BASPACHO_CHECK_EQ(batchSize, 1);
    return new CudaNumericCtx<float>(*this, tempBufSize, skel.spanStart.size() - 1);
  } else if (tIdx == type_index(typeid(vector<double*>))) {
    return new CudaNumericCtx<vector<double*>>(*this, tempBufSize, skel.spanStart.size() - 1,
                                               batchSize);
  } else if (tIdx == type_index(typeid(vector<float*>))) {
    return new CudaNumericCtx<vector<float*>>(*this, tempBufSize, skel.spanStart.size() - 1,
                                              batchSize);
  } else {
    return nullptr;
  }
}

SolveCtxBase* CudaSymbolicCtx::createSolveCtxForType(type_index tIdx, int nRHS, int batchSize) {
  if (tIdx == type_index(typeid(double))) {
    BASPACHO_CHECK_EQ(batchSize, 1);
    return new CudaSolveCtx<double>(*this, nRHS);
  } else if (tIdx == type_index(typeid(float))) {
    BASPACHO_CHECK_EQ(batchSize, 1);
    return new CudaSolveCtx<float>(*this, nRHS);
  } else if (tIdx == type_index(typeid(vector<double*>))) {
    return new CudaSolveCtx<vector<double*>>(*this, nRHS, batchSize);
  } else if (tIdx == type_index(typeid(vector<float*>))) {
    return new CudaSolveCtx<vector<float*>>(*this, nRHS, batchSize);
  } else {
    return nullptr;
  }
}

OpsPtr cudaOps() { return OpsPtr(new CudaOps); }

}  // end namespace BaSpaCho
