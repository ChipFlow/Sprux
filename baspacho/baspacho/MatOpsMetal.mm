/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <typeindex>

#include <Eigen/Dense>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/MetalDefs.h"
#include "baspacho/baspacho/Utils.h"
#ifdef BASPACHO_USE_BLAS
#include "baspacho/baspacho/BlasDefs.h"
#endif

namespace BaSpaCho {

using namespace std;

// Read an MPS operation threshold from an environment variable, returning
// defaultVal if the variable is unset or not a valid non-negative integer.
static int64_t getMpsThreshold(const char* envVar, int64_t defaultVal) {
  const char* val = std::getenv(envVar);
  if (val) {
    char* end;
    errno = 0;
    long long parsed = std::strtoll(val, &end, 10);
    if (end != val && *end == '\0' && parsed >= 0 && errno == 0) {
      return static_cast<int64_t>(parsed);
    }
  }
  return defaultVal;
}

// Work item struct matching the Metal shader definition
struct LUGemmWorkItem {
  int64_t offL, ldL;
  int64_t offU, ldU;
  int64_t offC, ldC;
  int64_t m, n, k;
};


// Binary search on CPU: find largest i such that array[i] <= needle
// Mirrors GPU bisect() in MetalKernels.metal exactly.
static int64_t cpuBisect(const int64_t* array, int64_t size, int64_t needle) {
  int64_t a = 0, b = size;
  while (b - a > 1) {
    int64_t mid = (a + b) / 2;
    if (needle >= array[mid])
      a = mid;
    else
      b = mid;
  }
  return a;
}

// Synchronization ops for Metal
struct MetalSyncOps {
  static void sync() { MetalContext::instance().synchronize(); }
};

// Pre-computed work item for LU sparse elimination (matches Metal shader struct)
struct LUWorkItem {
  int32_t L_offset;       // data offset for L value
  int32_t U_offset;       // data offset for U value
  int32_t target_offset;  // data offset for target
};

// Segment descriptor for two-phase deterministic sparse elimination
struct SegmentInfo {
  int32_t target_offset;  // data offset for target element
  int32_t scratch_start;  // start index in scratch buffer
  int32_t count;          // number of products to sum
};

// Cholesky element-level work item for two-phase elimination
struct CholWorkItem {
  int32_t srcRow_offset;   // start offset of row in source B matrix (in data[])
  int32_t srcCol_offset;   // start offset of row in source C matrix (in data[])
  int16_t numK;            // dot product length (= lumpSize)
  int16_t padding;
  int32_t target_offset;   // target element in data[]
};

// Symbolic elimination context for Metal
struct MetalSymElimCtx : SymElimCtx {
  MetalSymElimCtx() {}
  virtual ~MetalSymElimCtx() override {}

  int64_t numColumns;
  int64_t numBlockPairs;
  MetalMirror<int64_t> makeBlockPairEnumStraight;

  // Pre-computed work list for LU sparse elimination
  int64_t numWorkItems = 0;
  MetalMirror<int32_t> devWorkItems;  // packed: 3 int32 per LUWorkItem

  // Two-phase deterministic elimination: segments (sorted by target)
  int64_t numSegments = 0;
  MetalMirror<int32_t> devSegments;  // packed: 3 int32 per SegmentInfo

  // Two-phase Cholesky element-level work items
  int64_t numCholWorkItems = 0;
  MetalMirror<int32_t> devCholWorkItems;  // packed: 4 int32 per CholWorkItem (16 bytes)
  int64_t numCholSegments = 0;
  MetalMirror<int32_t> devCholSegments;  // packed: 3 int32 per SegmentInfo
};

// Forward declarations
struct MetalSymbolicCtx;

template <typename T>
struct MetalNumericCtx;

template <typename T>
struct MetalSolveCtx;

// Symbolic context for Metal operations
struct MetalSymbolicCtx : SymbolicCtx {
  MetalSymbolicCtx(const CoalescedBlockMatrixSkel& skel_, const std::vector<int64_t>& permutation)
      : skel(skel_) {
    @autoreleasepool {
      device = (__bridge id<MTLDevice>)MetalContext::instance().device();
      commandQueue = (__bridge id<MTLCommandQueue>)MetalContext::instance().commandQueue();
      asyncCommandQueue = (__bridge id<MTLCommandQueue>)MetalContext::instance().asyncQueue();

      // Load all skeleton data to GPU buffers
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
        NSLog(@"MetalSymbolicCtx: uploaded upper triangle data for LU");
      }

      NSLog(@"MetalSymbolicCtx initialized");
    }
  }

  virtual ~MetalSymbolicCtx() override {}

  virtual PermutedCoalescedAccessor deviceAccessor() override {
    PermutedCoalescedAccessor retv;
    retv.init(devSpanStart.ptr(), devSpanToLump.ptr(), devLumpStart.ptr(), devSpanOffsetInLump.ptr(),
              devChainColPtr.ptr(), devChainRowSpan.ptr(), devChainData.ptr(), devPermutation.ptr());
    return retv;
  }

  virtual SymElimCtxPtr prepareElimination(int64_t lumpsBegin, int64_t lumpsEnd) override {
    MetalSymElimCtx* elim = new MetalSymElimCtx;

    vector<int64_t> makeStraight(lumpsEnd - lumpsBegin + 1);

    // For each lump, compute number of pairs contributing to elimination
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

    // Build element-level work items for two-phase deterministic elimination.
    // Expand each block pair into individual element dot products.
    BASPACHO_CHECK(skel.totalDataSize() < INT32_MAX);

    vector<CholWorkItem> cholItems;
    cholItems.reserve(elim->numBlockPairs * 4);  // heuristic: ~4 elements per block pair

    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      int64_t colStart = skel.chainColPtr[l] + 1;  // skip diagonal
      int64_t colEnd = skel.chainColPtr[l + 1];
      int64_t n = colEnd - colStart;
      int64_t lumpSize = skel.lumpStart[l + 1] - skel.lumpStart[l];

      for (int64_t p = 0; p < n * (n + 1) / 2; p++) {
        // Convert linear index to ordered pair (di, dj) where di <= dj
        int64_t odd = n & 1;
        int64_t m = n + 1 - odd;
        int64_t di = p % m;
        int64_t dj = n - 1 - (p / m);
        if (di > dj) {
          di = di - dj - 1;
          dj = n - 1 - odd - dj;
        }

        // Block info (mirrors GPU kernel logic)
        int64_t iSpan = skel.chainRowSpan[colStart + di];
        int64_t jSpan = skel.chainRowSpan[colStart + dj];
        int64_t iSize = skel.spanStart[iSpan + 1] - skel.spanStart[iSpan];
        int64_t jSize = skel.spanStart[jSpan + 1] - skel.spanStart[jSpan];
        int64_t iDataPtr = skel.chainData[colStart + di];
        int64_t jDataPtr = skel.chainData[colStart + dj];

        // Find target block via bisect
        int64_t iLump = skel.spanToLump[iSpan];
        int64_t iSpanOff = skel.spanOffsetInLump[iSpan];
        int64_t targetLumpSize = skel.lumpStart[iLump + 1] - skel.lumpStart[iLump];
        int64_t targetStartPtr = skel.chainColPtr[iLump];
        int64_t targetEndPtr = skel.chainColPtr[iLump + 1];
        int64_t targetPos = cpuBisect(skel.chainRowSpan.data() + targetStartPtr,
                                      targetEndPtr - targetStartPtr, jSpan);
        int64_t jiDataPtr = skel.chainData[targetStartPtr + targetPos];

        // target = data + jiDataPtr + iSpanOff
        // locked_sub_product_float args: (target, targetLumpSize,
        //   srcJ=data+jDataPtr, jSize, lumpSize, lumpSize,
        //   srcI=data+iDataPtr, iSize, lumpSize)
        // Element (i,j): val = dot(srcJ[i*lumpSize:], srcI[j*lumpSize:], lumpSize)
        //   target_elem = target + i * targetLumpSize + j
        for (int64_t i = 0; i < jSize; i++) {
          for (int64_t j = 0; j < iSize; j++) {
            CholWorkItem item;
            item.srcRow_offset = (int32_t)(jDataPtr + i * lumpSize);
            item.srcCol_offset = (int32_t)(iDataPtr + j * lumpSize);
            item.numK = (int16_t)lumpSize;
            item.padding = 0;
            item.target_offset = (int32_t)(jiDataPtr + iSpanOff + i * targetLumpSize + j);
            cholItems.push_back(item);
          }
        }
      }
    }

    // Sort by target_offset for deterministic accumulation
    stable_sort(cholItems.begin(), cholItems.end(),
                [](const CholWorkItem& a, const CholWorkItem& b) {
                  return a.target_offset < b.target_offset;
                });

    elim->numCholWorkItems = (int64_t)cholItems.size();
    if (elim->numCholWorkItems > 0) {
      // Upload as packed int32 array (4 int32 per CholWorkItem = 16 bytes)
      vector<int32_t> packed(4 * elim->numCholWorkItems);
      for (int64_t i = 0; i < elim->numCholWorkItems; i++) {
        packed[4 * i + 0] = cholItems[i].srcRow_offset;
        packed[4 * i + 1] = cholItems[i].srcCol_offset;
        // Pack numK and padding into one int32
        int32_t numK_packed;
        memcpy(&numK_packed, &cholItems[i].numK, sizeof(int32_t));
        packed[4 * i + 2] = numK_packed;
        packed[4 * i + 3] = cholItems[i].target_offset;
      }
      elim->devCholWorkItems.load(packed);

      // Build segment table
      vector<SegmentInfo> segments;
      segments.reserve(elim->numCholWorkItems);
      int32_t segStart = 0;
      for (int64_t i = 1; i <= elim->numCholWorkItems; i++) {
        if (i == elim->numCholWorkItems ||
            cholItems[i].target_offset != cholItems[segStart].target_offset) {
          SegmentInfo seg;
          seg.target_offset = cholItems[segStart].target_offset;
          seg.scratch_start = segStart;
          seg.count = (int32_t)(i - segStart);
          segments.push_back(seg);
          segStart = (int32_t)i;
        }
      }

      elim->numCholSegments = (int64_t)segments.size();
      vector<int32_t> segPacked(3 * elim->numCholSegments);
      for (int64_t i = 0; i < elim->numCholSegments; i++) {
        segPacked[3 * i + 0] = segments[i].target_offset;
        segPacked[3 * i + 1] = segments[i].scratch_start;
        segPacked[3 * i + 2] = segments[i].count;
      }
      elim->devCholSegments.load(segPacked);
    }

    return SymElimCtxPtr(elim);
  }

  virtual SymElimCtxPtr prepareLUElimination(int64_t lumpsBegin, int64_t lumpsEnd) override {
    if (!skel.isGeneral()) return nullptr;

    MetalSymElimCtx* elim = new MetalSymElimCtx;

    vector<int64_t> pairEnum(lumpsEnd - lumpsBegin + 1);

    // For LU: n^2 pairs per lump (L_rows × U_cols) instead of n*(n+1)/2
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

    // Pre-compute work list: resolve all binary searches on CPU
    int64_t upperDataBase = skel.dataSize();
    BASPACHO_CHECK(skel.totalDataSize() < INT32_MAX);

    vector<LUWorkItem> workItems;
    workItems.reserve(elim->numBlockPairs);

    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      int64_t colStart = skel.chainColPtr[l] + 1;  // skip diagonal
      int64_t n = skel.chainColPtr[l + 1] - colStart;
      int64_t uRowStart = skel.upperChainRowPtr[l];

      for (int64_t row_idx = 0; row_idx < n; row_idx++) {
        for (int64_t col_idx = 0; col_idx < n; col_idx++) {
          LUWorkItem item;
          item.L_offset = (int32_t)skel.chainData[colStart + row_idx];
          item.U_offset =
              (int32_t)(upperDataBase + skel.upperChainData[uRowStart + col_idx]);

          // Target: CPU bisect (mirrors GPU bisect in MetalKernels.metal)
          int64_t aSpan = skel.chainRowSpan[colStart + row_idx];
          int64_t bSpan = skel.chainRowSpan[colStart + col_idx];

          if (aSpan >= bSpan) {
            // Lower triangle target
            int64_t bLump = skel.spanToLump[bSpan];
            int64_t bSpanOff = skel.spanOffsetInLump[bSpan];
            int64_t tStart = skel.chainColPtr[bLump];
            int64_t tEnd = skel.chainColPtr[bLump + 1];
            int64_t tPos =
                cpuBisect(skel.chainRowSpan.data() + tStart, tEnd - tStart, aSpan);
            item.target_offset = (int32_t)(skel.chainData[tStart + tPos] + bSpanOff);
          } else {
            // Upper triangle target
            int64_t aLump = skel.spanToLump[aSpan];
            int64_t uStart = skel.upperChainRowPtr[aLump];
            int64_t uEnd = skel.upperChainRowPtr[aLump + 1];
            int64_t tPos = cpuBisect(skel.upperChainColSpan.data() + uStart,
                                     uEnd - uStart, bSpan);
            item.target_offset =
                (int32_t)(upperDataBase + skel.upperChainData[uStart + tPos]);
          }
          workItems.push_back(item);
        }
      }
    }

    // Sort work items by target_offset for deterministic accumulation.
    // Phase 1 reads L/U values (indexed by work item order) — sorting by
    // target disrupts L/U locality but Phase 1 is embarrassingly parallel.
    // Phase 2 does the accumulation in fixed order per target.
    std::stable_sort(workItems.begin(), workItems.end(),
                     [](const LUWorkItem& a, const LUWorkItem& b) {
                       return a.target_offset < b.target_offset;
                     });

    elim->numWorkItems = (int64_t)workItems.size();
    if (elim->numWorkItems > 0) {
      // Upload sorted work items as packed int32 array (3 int32 per item)
      vector<int32_t> packed(3 * elim->numWorkItems);
      for (int64_t i = 0; i < elim->numWorkItems; i++) {
        packed[3 * i + 0] = workItems[i].L_offset;
        packed[3 * i + 1] = workItems[i].U_offset;
        packed[3 * i + 2] = workItems[i].target_offset;
      }
      elim->devWorkItems.load(packed);

      // Build segment table: group consecutive items with same target
      vector<SegmentInfo> segments;
      segments.reserve(elim->numWorkItems);  // upper bound
      int32_t segStart = 0;
      for (int64_t i = 1; i <= elim->numWorkItems; i++) {
        if (i == elim->numWorkItems ||
            workItems[i].target_offset != workItems[segStart].target_offset) {
          SegmentInfo seg;
          seg.target_offset = workItems[segStart].target_offset;
          seg.scratch_start = segStart;
          seg.count = (int32_t)(i - segStart);
          segments.push_back(seg);
          segStart = (int32_t)i;
        }
      }

      elim->numSegments = (int64_t)segments.size();
      vector<int32_t> segPacked(3 * elim->numSegments);
      for (int64_t i = 0; i < elim->numSegments; i++) {
        segPacked[3 * i + 0] = segments[i].target_offset;
        segPacked[3 * i + 1] = segments[i].scratch_start;
        segPacked[3 * i + 2] = segments[i].count;
      }
      elim->devSegments.load(segPacked);
    }

    return SymElimCtxPtr(elim);
  }

  virtual NumericCtxBase* createNumericCtxForType(type_index tIdx, int64_t tempBufSize,
                                                  int batchSize) override;

  virtual SolveCtxBase* createSolveCtxForType(type_index tIdx, int nRHS, int batchSize) override;

  void setExternalEncoder(void* cmd_buffer, void* encoder) override {
    externalCmdBuf = (__bridge id<MTLCommandBuffer>)cmd_buffer;
    externalEncoder = (__bridge id<MTLComputeCommandEncoder>)encoder;
    usingExternalEncoder = true;
  }

  void clearExternalEncoder() override {
    if (externalEncoder) {
      [externalEncoder endEncoding];
      externalEncoder = nil;
    }
    if (externalCmdBuf) {
      [externalCmdBuf commit];
      [externalCmdBuf waitUntilCompleted];
      externalCmdBuf = nil;
    }
    usingExternalEncoder = false;
  }

  void* getExternalEncoder() override {
    return (__bridge void*)externalEncoder;
  }

  const CoalescedBlockMatrixSkel& skel;

  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;       // Primary queue (solve, dense factor, MPS)
  id<MTLCommandQueue> asyncCommandQueue;  // Async queue (pipelined sparse elimination)

  // Device buffers (mirrors of skeleton data)
  MetalMirror<int64_t> devLumpToSpan;
  MetalMirror<int64_t> devChainRowsTillEnd;
  MetalMirror<int64_t> devChainRowSpan;
  MetalMirror<int64_t> devSpanOffsetInLump;
  MetalMirror<int64_t> devLumpStart;
  MetalMirror<int64_t> devChainColPtr;
  MetalMirror<int64_t> devChainData;
  MetalMirror<int64_t> devBoardColPtr;
  MetalMirror<int64_t> devBoardChainColOrd;
  MetalMirror<int64_t> devSpanStart;
  MetalMirror<int64_t> devSpanToLump;
  MetalMirror<int64_t> devPermutation;

  // Upper triangle buffers (for LU factorization)
  MetalMirror<int64_t> devUpperChainRowPtr;
  MetalMirror<int64_t> devUpperChainColSpan;
  MetalMirror<int64_t> devUpperChainData;

  // External encoder state: when set, MetalNumericCtx and MetalSolveCtx
  // record dispatches into this encoder instead of creating their own.
  // Set via setExternalEncoder().
  id<MTLCommandBuffer> externalCmdBuf = nil;
  id<MTLComputeCommandEncoder> externalEncoder = nil;
  bool usingExternalEncoder = false;
};

// Metal operations factory
struct MetalOps : Ops {
  virtual SymbolicCtxPtr createSymbolicCtx(const CoalescedBlockMatrixSkel& skel,
                                           const std::vector<int64_t>& permutation) override {
    return SymbolicCtxPtr(new MetalSymbolicCtx(skel, permutation));
  }
};

// Helper to get a Metal compute pipeline by kernel name
static id<MTLComputePipelineState> getPipeline(const char* name) {
  return (__bridge id<MTLComputePipelineState>)
      MetalContext::instance().getPipelineState(name);
}

static void dispatchKernel(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                           void (^encodeBlock)(id<MTLComputeCommandEncoder>),
                           NSUInteger numThreads, bool sync = true) {
  @autoreleasepool {
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    encodeBlock(encoder);

    // Calculate thread group size
    NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    threadGroupSize = MIN(threadGroupSize, numThreads);

    MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
    MTLSize numGroups = MTLSizeMake((numThreads + threadGroupSize - 1) / threadGroupSize, 1, 1);

    [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [cmdBuf commit];

    if (sync) {
      [cmdBuf waitUntilCompleted];
    }
  }
}

// Numeric context for float - Metal implementation
template <>
struct MetalNumericCtx<float> : NumericCtx<float> {
  MetalNumericCtx(MetalSymbolicCtx& sym_, int64_t tempBufSize, int64_t numSpans)
      : sym(sym_), numSpans_(numSpans), spanToChainOffset(numSpans) {
    tempBuffer.resizeToAtLeast(tempBufSize);
    devSpanToChainOffset.resizeToAtLeast(numSpans);
    // Pre-allocate GPU pivot buffer for all rows (avoids reallocation during factorization)
    if (sym.skel.isGeneral()) {
      devAllPivots.resizeToAtLeast(sym.skel.order());
    }
  }

  virtual ~MetalNumericCtx() override {
    // Flush any pending work before destruction
    commitAndWait();
  }

  // Encode a kernel dispatch onto a persistent compute encoder within the
  // pending command buffer. Uses a single encoder for all dispatches with
  // memory barriers between them to ensure correct data ordering.
  // This avoids the ~4μs overhead of creating/ending a new encoder per dispatch.
  // In external encoder mode, dispatches are recorded into the caller-provided
  // encoder instead of a self-managed command buffer.
  void encodeKernel(id<MTLComputePipelineState> pipeline,
                    void (^encodeBlock)(id<MTLComputeCommandEncoder>),
                    NSUInteger numThreads) {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder;
      if (sym.usingExternalEncoder) {
        // External encoder mode: use caller-provided encoder.
        encoder = sym.externalEncoder;
      } else {
        // Normal mode: manage our own command buffer and encoder.
        if (!pendingCmdBuf_) {
          pendingCmdBuf_ = [sym.commandQueue commandBuffer];
        }
        if (!pendingEncoder_) {
          pendingEncoder_ = [pendingCmdBuf_ computeCommandEncoder];
          pendingDispatchCount_ = 0;
        }
        encoder = pendingEncoder_;
      }

      // Insert memory barrier so previous dispatches' buffer writes are visible
      if (pendingDispatchCount_ > 0) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }

      [encoder setComputePipelineState:pipeline];
      encodeBlock(encoder);

      NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
      threadGroupSize = MIN(threadGroupSize, numThreads);

      MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
      MTLSize numGroups =
          MTLSizeMake((numThreads + threadGroupSize - 1) / threadGroupSize, 1, 1);

      [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
      pendingDispatchCount_++;
    }
  }

  // Encode a kernel dispatch with explicit threadgroup counts.
  // Used when each threadgroup must cooperate internally (e.g., per-span
  // batched TRSM with threadgroup_barrier).
  void encodeKernelWithGroups(id<MTLComputePipelineState> pipeline,
                              void (^encodeBlock)(id<MTLComputeCommandEncoder>),
                              NSUInteger numGroupCount, NSUInteger threadsPerGroup) {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder;
      if (sym.usingExternalEncoder) {
        encoder = sym.externalEncoder;
      } else {
        if (!pendingCmdBuf_) {
          pendingCmdBuf_ = [sym.commandQueue commandBuffer];
        }
        if (!pendingEncoder_) {
          pendingEncoder_ = [pendingCmdBuf_ computeCommandEncoder];
          pendingDispatchCount_ = 0;
        }
        encoder = pendingEncoder_;
      }

      if (pendingDispatchCount_ > 0) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }

      [encoder setComputePipelineState:pipeline];
      encodeBlock(encoder);

      MTLSize tpg = MTLSizeMake(threadsPerGroup, 1, 1);
      MTLSize ng = MTLSizeMake(numGroupCount, 1, 1);
      [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tpg];
      pendingDispatchCount_++;
    }
  }

  // Commit the pending command buffer WITHOUT waiting for GPU completion.
  // Metal command queue ordering guarantees that command buffers execute in
  // submission order, so subsequent work on the same queue will see the results.
  // Tracks the last committed buffer so waitForGpu() can wait on it later.
  // Signals the shared event for lower-overhead CPU waiting.
  // In external encoder mode, this is a no-op — the caller manages the command buffer.
  void commitPending() {
    if (sym.usingExternalEncoder) return;
    if (pendingCmdBuf_) {
      if (pendingEncoder_) {
        [pendingEncoder_ endEncoding];
        pendingEncoder_ = nil;
      }
      // Signal shared event for lower-overhead waiting in waitForGpu()
      if (!sharedEvent_) {
        sharedEvent_ = [sym.device newSharedEvent];
      }
      sharedEventValue_++;
      [pendingCmdBuf_ encodeSignalEvent:sharedEvent_ value:sharedEventValue_];
      [pendingCmdBuf_ commit];
      lastCommittedCmdBuf_ = pendingCmdBuf_;
      pendingCmdBuf_ = nil;
      pendingDispatchCount_ = 0;
    }
  }

  // Wait for the most recently committed command buffer to complete.
  // Prefers MTLSharedEvent polling (lower overhead) when available,
  // falls back to waitUntilCompleted on the command buffer.
  // In external encoder mode, this is a no-op — the caller manages synchronization.
  void waitForGpu() {
    if (sym.usingExternalEncoder) return;
    if (sharedEvent_ && sharedEventValue_ > 0) {
      // Lower-overhead wait: polls a shared memory value instead of
      // full command buffer lifecycle tracking.
      [sharedEvent_ waitUntilSignaledValue:sharedEventValue_ timeoutMS:5000];
      lastCommittedCmdBuf_ = nil;
    } else if (lastCommittedCmdBuf_) {
      [lastCommittedCmdBuf_ waitUntilCompleted];
      lastCommittedCmdBuf_ = nil;
    }
    checkDeferredPotrfStatus();
    flushDeferredState();
    collectDeferredElimPerturb();
  }

  // Check deferred potrf status after GPU work has completed.
  void checkDeferredPotrfStatus() {
    if (potrfStatusPending_ && potrfStatusBuf_) {
      auto status = *reinterpret_cast<MPSMatrixDecompositionStatus*>([potrfStatusBuf_ contents]);
      if (status != MPSMatrixDecompositionStatusSuccess) {
        fprintf(stderr, "Metal potrf: MPS Cholesky failed (status=%d)\n", (int)status);
      }
      potrfStatusPending_ = false;
    }
  }

  // Flush deferred GPU state back to CPU after GPU work has completed.
  // Copies all GPU-resident pivots back to the CPU pivot array.
  void flushDeferredState() {
    if (pivotsOnGpu_ && allPivotsCpuBase_ && allPivotsCount_ > 0) {
      memcpy(allPivotsCpuBase_, devAllPivots.ptr(), allPivotsCount_ * sizeof(int64_t));
      pivotsOnGpu_ = false;
    }
  }

  // Collect deferred perturb count from sparse elimination GPU buffer.
  void collectDeferredElimPerturb() {
    if (deferredElimPerturbBuf_) {
      deferredElimPerturbCount_ += *(uint32_t*)[deferredElimPerturbBuf_ contents];
      deferredElimPerturbBuf_ = nil;
    }
  }

  // Signal start of dense LU operations — wait for deferred sparse elim GPU work.
  void beginDenseOps(float* data, int64_t totalDataSize) override {
    if (explicitRecording_) return;  // no-op during explicit recording
    (void)data;
    (void)totalDataSize;
    waitForGpu();
  }

  // Commit the pending command buffer and wait for all GPU work to complete.
  // This is needed when the CPU must access data that the GPU may have written
  // (e.g., before CPU fallback paths or memcpy to shared buffers).
  void commitAndWait() {
    flushPendingGemms();
    commitPending();
    waitForGpu();
  }

  // Flush buffered saveGemm work items as a single batched kernel dispatch.
  // Uses append-only strategy: each flush writes at increasing offsets in
  // devGemmWorkBuf_, avoiding the need to wait for GPU to finish reading
  // previous data. Buffer reset happens at flush() time between factorizations.
  void flushPendingGemms() {
    if (explicitRecording_) {
      // Explicit recording mode (for graph capture): record flush point, skip GPU work
      if (recordingBatchCount_ > 0) {
        size_t startIdx = recordedItems_.size() - recordingBatchCount_;
        recordedFlushPoints_.push_back({startIdx, recordingBatchCount_});
        recordingBatchCount_ = 0;
      }
      pendingGemms_.clear();
      return;
    }

    if (recordState_ == RecordState::Recording) {
      // Auto-recording: record flush point, then fall through to normal execution
      if (recordingBatchCount_ > 0) {
        size_t startIdx = recordedItems_.size() - recordingBatchCount_;
        recordedFlushPoints_.push_back({startIdx, recordingBatchCount_});
        recordingBatchCount_ = 0;
      }
      // Fall through to normal dispatch below
    }

    if (recordState_ == RecordState::Ready) {
      // Pre-computed mode: dispatch from device-resident items
      if (precomputedFlushIdx_ >= recordedFlushPoints_.size()) return;
      auto [startIdx, count] = recordedFlushPoints_[precomputedFlushIdx_];
      precomputedFlushIdx_++;
      if (count == 0) return;

      // Byte offset into pre-computed buffer (MetalMirror backing is already aligned)
      size_t byteOffset = startIdx * sizeof(LUGemmWorkItem);

      int64_t countI64 = (int64_t)count;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_batchedSaveGemm_kernel_float");

      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:cachedDataBuffer_ offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)devPrecomputedItems_.buffer()
                        offset:byteOffset
                       atIndex:1];
            [encoder setBytes:&countI64 length:sizeof(int64_t) atIndex:2];
          },
          (NSUInteger)count);
      pendingGemms_.clear();
      return;
    }

    if (pendingGemms_.empty()) return;

    int64_t count = (int64_t)pendingGemms_.size();
    size_t bytesNeeded = count * sizeof(LUGemmWorkItem);
    // Align to 16 bytes for Metal buffer offset requirements
    size_t alignedBytes = (bytesNeeded + 15) & ~size_t(15);

    size_t newUsedBytes = gemmWorkBufUsedBytes_ + alignedBytes;
    size_t int64sNeeded = (newUsedBytes + sizeof(int64_t) - 1) / sizeof(int64_t);

    if (int64sNeeded > devGemmWorkBuf_.allocSize()) {
      // Buffer too small: commit and wait before reallocating to avoid
      // invalidating in-flight buffer references.
      if (gemmWorkBufInFlight_) {
        commitPending();
        waitForGpu();
        gemmWorkBufInFlight_ = false;
      }
      // Reset position since we waited — old data is consumed, buffer will be replaced
      gemmWorkBufUsedBytes_ = 0;
      newUsedBytes = alignedBytes;
      int64sNeeded = (newUsedBytes + sizeof(int64_t) - 1) / sizeof(int64_t);
      // Grow with 4x headroom to minimize future reallocations
      devGemmWorkBuf_.resizeToAtLeast(int64sNeeded * 4);
    }

    // Append work items at current offset (safe: GPU reads from earlier offsets)
    size_t writeOffset = gemmWorkBufUsedBytes_;
    memcpy((char*)devGemmWorkBuf_.ptr() + writeOffset, pendingGemms_.data(), bytesNeeded);

    id<MTLComputePipelineState> pipeline = getPipeline(
            "lu_batchedSaveGemm_kernel_float");

    encodeKernel(
        pipeline,
        ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:cachedDataBuffer_ offset:0 atIndex:0];
          [encoder setBuffer:(__bridge id<MTLBuffer>)devGemmWorkBuf_.buffer()
                      offset:writeOffset
                     atIndex:1];
          [encoder setBytes:&count length:sizeof(int64_t) atIndex:2];
        },
        (NSUInteger)count);

    gemmWorkBufUsedBytes_ = newUsedBytes;
    gemmWorkBufInFlight_ = true;
    pendingGemms_.clear();
  }

  virtual void pseudoFactorSpans(float* data, int64_t spanBegin, int64_t spanEnd) override {
    if (explicitRecording_) return;
    @autoreleasepool {
      // Find the MTLBuffer for data
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::pseudoFactorSpans: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      // Get pipeline state for factor_spans_kernel_float
      id<MTLComputePipelineState> pipeline = getPipeline(
              "factor_spans_kernel_float");

      int64_t numSpans = spanEnd - spanBegin;
      if (numSpans <= 0) return;

      dispatchKernel(
          sym.commandQueue, pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanToLump.buffer() offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanOffsetInLump.buffer()
                        offset:0
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpToSpan.buffer() offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer() offset:0 atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer() offset:0 atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                        offset:0
                       atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer() offset:0 atIndex:6];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardColPtr.buffer()
                        offset:0
                       atIndex:7];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardChainColOrd.buffer()
                        offset:0
                       atIndex:8];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:0
                       atIndex:9];
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:10];
            [encoder setBytes:&spanBegin length:sizeof(int64_t) atIndex:11];
            [encoder setBytes:&spanEnd length:sizeof(int64_t) atIndex:12];
          },
          (NSUInteger)numSpans);
    }
  }

  virtual void doElimination(const SymElimCtx& elimData, float* data, int64_t lumpsBegin,
                             int64_t lumpsEnd) override {
    if (explicitRecording_) return;
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);
      const MetalSymElimCtx& elim = *pElim;

      // Find the MTLBuffer for data
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::doElimination: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      // Step 1: Factor lumps (Cholesky on diagonal blocks + below-diagonal solve)
      {
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
                "factor_lumps_kernel_float");

        dispatchKernel(
            sym.commandQueue, pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                          offset:0
                         atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                          offset:0
                         atIndex:1];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                          offset:0
                         atIndex:2];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardColPtr.buffer()
                          offset:0
                         atIndex:3];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardChainColOrd.buffer()
                          offset:0
                         atIndex:4];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                          offset:0
                         atIndex:5];
              [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:6];
              [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
              [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];
            },
            (NSUInteger)numLumps);
      }

      // Step 2: Sparse elimination — two-phase deterministic
      if (elim.numCholWorkItems > 0 && elim.numCholSegments > 0) {
        elimScratchBuffer.resizeToAtLeast(elim.numCholWorkItems);

        // Phase 1: compute dot products into scratch
        id<MTLComputePipelineState> cp1Pipeline =
            (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
                "chol_sparse_elim_phase1_float");

        dispatchKernel(
            sym.commandQueue, cp1Pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)elim.devCholWorkItems.buffer()
                          offset:0
                         atIndex:1];
              [encoder setBuffer:(__bridge id<MTLBuffer>)elimScratchBuffer.buffer()
                          offset:0
                         atIndex:2];
              [encoder setBytes:&elim.numCholWorkItems length:sizeof(int64_t) atIndex:3];
            },
            (NSUInteger)elim.numCholWorkItems);

        // Phase 2: deterministic segmented sum
        id<MTLComputePipelineState> p2Pipeline =
            (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
                "sparse_elim_phase2_float");

        dispatchKernel(
            sym.commandQueue, p2Pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)elimScratchBuffer.buffer()
                          offset:0
                         atIndex:1];
              [encoder setBuffer:(__bridge id<MTLBuffer>)elim.devCholSegments.buffer()
                          offset:0
                         atIndex:2];
              [encoder setBytes:&elim.numCholSegments length:sizeof(int64_t) atIndex:3];
            },
            (NSUInteger)elim.numCholSegments);
      }
    }
  }

  virtual void doEliminationLU(const SymElimCtx& elimData, float* data, int64_t lumpsBegin,
                               int64_t lumpsEnd, float staticPivotThreshold,
                               int64_t& perturbCount) override {
    if (explicitRecording_) return;
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);
      const MetalSymElimCtx& elim = *pElim;

      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error(
            "MetalNumericCtx<float>::doEliminationLU: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      // Allocate GPU buffer for perturb count (atomic counter)
      id<MTLBuffer> perturbBuf = [sym.device newBufferWithLength:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
      *(uint32_t*)[perturbBuf contents] = 0;

      // Step 1: LU factor lumps (divide below-diag by diagonal)
      {
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
                "lu_factor_lumps_kernel_float");

        float threshold = staticPivotThreshold;
        dispatchKernel(
            sym.commandQueue, pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                          offset:0
                         atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                          offset:0
                         atIndex:1];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                          offset:0
                         atIndex:2];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardColPtr.buffer()
                          offset:0
                         atIndex:3];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardChainColOrd.buffer()
                          offset:0
                         atIndex:4];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                          offset:0
                         atIndex:5];
              [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:6];
              [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
              [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];
              [encoder setBytes:&threshold length:sizeof(float) atIndex:9];
              [encoder setBuffer:perturbBuf offset:0 atIndex:10];
            },
            (NSUInteger)numLumps);
      }

      // Step 2: LU Schur complement — atomic elimination (single kernel)
      // Uses atomicSub directly; iterative refinement compensates for float precision.
      if (elim.numWorkItems > 0) {
        id<MTLComputePipelineState> elimPipeline =
            (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
                "lu_sparse_elim_precomputed_float");

        dispatchKernel(
            sym.commandQueue, elimPipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)elim.devWorkItems.buffer()
                          offset:0
                         atIndex:1];
              [encoder setBytes:&elim.numWorkItems length:sizeof(int64_t) atIndex:2];
            },
            (NSUInteger)elim.numWorkItems);
      }

      // Read back perturb count
      perturbCount = *(uint32_t*)[perturbBuf contents];
    }
  }

  // Batch all LU sparse elimination levels into a single command buffer
  // on the ASYNC queue. Uses atomic elimination (single kernel per level):
  //   Phase 2: segmented sum per target in fixed order (deterministic)
  // Signals the shared event so beginDenseOps/waitForGpu can synchronize.
  void doAllEliminationsLU(const std::vector<SymElimCtxPtr>& elimCtxs,
                           const std::vector<int64_t>& ranges, float* data,
                           float staticPivotThreshold,
                           int64_t& totalPerturbCount) override {
    if (explicitRecording_) return;
    @autoreleasepool {
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error(
            "MetalNumericCtx<float>::doAllEliminationsLU: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      // Single perturb counter for all levels
      id<MTLBuffer> perturbBuf = [sym.device newBufferWithLength:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
      *(uint32_t*)[perturbBuf contents] = 0;

      id<MTLComputePipelineState> factorPipeline =
          getPipeline("lu_factor_lumps_kernel_float");
      id<MTLComputePipelineState> elimPipeline =
          getPipeline("lu_sparse_elim_precomputed_float");

      // Create a dedicated command buffer on the ASYNC queue.
      id<MTLCommandBuffer> asyncCmdBuf = [sym.asyncCommandQueue commandBuffer];
      id<MTLComputeCommandEncoder> asyncEncoder = [asyncCmdBuf computeCommandEncoder];
      int dispatchCount = 0;

      for (size_t l = 0; l + 1 < ranges.size(); l++) {
        if (!elimCtxs[l]) continue;
        const MetalSymElimCtx& elim =
            *dynamic_cast<const MetalSymElimCtx*>(elimCtxs[l].get());

        int64_t lumpsBegin = ranges[l];
        int64_t lumpsEnd = ranges[l + 1];
        int64_t numLumps = lumpsEnd - lumpsBegin;
        if (numLumps <= 0) continue;

        float threshold = staticPivotThreshold;

        // Memory barrier between dispatches
        if (dispatchCount > 0) {
          [asyncEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // LU factor lumps (divide below-diag by diagonal)
        [asyncEncoder setComputePipelineState:factorPipeline];
        [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                         offset:0 atIndex:0];
        [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                         offset:0 atIndex:1];
        [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                         offset:0 atIndex:2];
        [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardColPtr.buffer()
                         offset:0 atIndex:3];
        [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardChainColOrd.buffer()
                         offset:0 atIndex:4];
        [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                         offset:0 atIndex:5];
        [asyncEncoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:6];
        [asyncEncoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
        [asyncEncoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];
        [asyncEncoder setBytes:&threshold length:sizeof(float) atIndex:9];
        [asyncEncoder setBuffer:perturbBuf offset:0 atIndex:10];

        NSUInteger tgs = MIN(factorPipeline.maxTotalThreadsPerThreadgroup, 256);
        tgs = MIN(tgs, (NSUInteger)numLumps);
        [asyncEncoder dispatchThreadgroups:MTLSizeMake(((NSUInteger)numLumps + tgs - 1) / tgs, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
        dispatchCount++;

        // LU Schur complement: atomic elimination (single kernel)
        // Uses atomicSub directly; iterative refinement compensates for float precision.
        if (elim.numWorkItems > 0) {
          [asyncEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
          [asyncEncoder setComputePipelineState:elimPipeline];
          [asyncEncoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
          [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)elim.devWorkItems.buffer()
                           offset:0 atIndex:1];
          [asyncEncoder setBytes:&elim.numWorkItems length:sizeof(int64_t) atIndex:2];

          NSUInteger etgs = MIN(elimPipeline.maxTotalThreadsPerThreadgroup, 256);
          etgs = MIN(etgs, (NSUInteger)elim.numWorkItems);
          [asyncEncoder dispatchThreadgroups:
              MTLSizeMake(((NSUInteger)elim.numWorkItems + etgs - 1) / etgs, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(etgs, 1, 1)];
          dispatchCount++;
        }
      }

      [asyncEncoder endEncoding];

      // Signal shared event on the async queue so waitForGpu() can detect completion.
      if (dispatchCount > 0) {
        if (!sharedEvent_) {
          sharedEvent_ = [sym.device newSharedEvent];
        }
        sharedEventValue_++;
        [asyncCmdBuf encodeSignalEvent:sharedEvent_ value:sharedEventValue_];
        [asyncCmdBuf commit];
        lastCommittedCmdBuf_ = asyncCmdBuf;
      }
      // Defer perturb count readback to beginDenseOps/waitForGpu
      deferredElimPerturbBuf_ = perturbBuf;
    }
  }

  // Batch all Cholesky sparse elimination levels into a single command buffer
  // on the ASYNC queue. Uses two-phase deterministic accumulation:
  //   Phase 1: compute dot products into scratch buffer (no atomics)
  //   Phase 2: segmented sum per target in fixed order (deterministic)
  void doAllEliminations(const std::vector<SymElimCtxPtr>& elimCtxs,
                         const std::vector<int64_t>& ranges, float* data) override {
    if (explicitRecording_) return;
    @autoreleasepool {
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error(
            "MetalNumericCtx<float>::doAllEliminations: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      id<MTLComputePipelineState> factorPipeline =
          getPipeline("factor_lumps_kernel_float");
      id<MTLComputePipelineState> cholPhase1Pipeline =
          getPipeline("chol_sparse_elim_phase1_float");
      id<MTLComputePipelineState> phase2Pipeline =
          getPipeline("sparse_elim_phase2_float");

      // Compute max scratch buffer size across all levels
      int64_t maxScratchSize = 0;
      for (size_t l = 0; l + 1 < ranges.size(); l++) {
        if (!elimCtxs[l]) continue;
        const MetalSymElimCtx& elim =
            *dynamic_cast<const MetalSymElimCtx*>(elimCtxs[l].get());
        maxScratchSize = max(maxScratchSize, elim.numCholWorkItems);
      }
      if (maxScratchSize > 0) {
        elimScratchBuffer.resizeToAtLeast(maxScratchSize);
      }

      // Create a dedicated command buffer on the ASYNC queue.
      id<MTLCommandBuffer> asyncCmdBuf = [sym.asyncCommandQueue commandBuffer];
      id<MTLComputeCommandEncoder> asyncEncoder = [asyncCmdBuf computeCommandEncoder];
      int dispatchCount = 0;

      for (size_t l = 0; l + 1 < ranges.size(); l++) {
        if (!elimCtxs[l]) continue;
        const MetalSymElimCtx& elim =
            *dynamic_cast<const MetalSymElimCtx*>(elimCtxs[l].get());

        int64_t lumpsBegin = ranges[l];
        int64_t lumpsEnd = ranges[l + 1];
        int64_t numLumps = lumpsEnd - lumpsBegin;
        if (numLumps <= 0) continue;

        if (dispatchCount > 0) {
          [asyncEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // Factor lumps (Cholesky on diagonal blocks + below-diagonal solve)
        [asyncEncoder setComputePipelineState:factorPipeline];
        [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                         offset:0 atIndex:0];
        [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                         offset:0 atIndex:1];
        [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                         offset:0 atIndex:2];
        [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardColPtr.buffer()
                         offset:0 atIndex:3];
        [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardChainColOrd.buffer()
                         offset:0 atIndex:4];
        [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                         offset:0 atIndex:5];
        [asyncEncoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:6];
        [asyncEncoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
        [asyncEncoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];

        NSUInteger tgs = MIN(factorPipeline.maxTotalThreadsPerThreadgroup, 256);
        tgs = MIN(tgs, (NSUInteger)numLumps);
        [asyncEncoder dispatchThreadgroups:MTLSizeMake(((NSUInteger)numLumps + tgs - 1) / tgs, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
        dispatchCount++;

        // Sparse elimination: two-phase deterministic
        if (elim.numCholWorkItems > 0 && elim.numCholSegments > 0) {
          // Phase 1: compute dot products into scratch buffer
          [asyncEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
          [asyncEncoder setComputePipelineState:cholPhase1Pipeline];
          [asyncEncoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
          [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)elim.devCholWorkItems.buffer()
                           offset:0 atIndex:1];
          [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)elimScratchBuffer.buffer()
                           offset:0 atIndex:2];
          [asyncEncoder setBytes:&elim.numCholWorkItems length:sizeof(int64_t) atIndex:3];

          NSUInteger etgs = MIN(cholPhase1Pipeline.maxTotalThreadsPerThreadgroup, 256);
          etgs = MIN(etgs, (NSUInteger)elim.numCholWorkItems);
          [asyncEncoder dispatchThreadgroups:
              MTLSizeMake(((NSUInteger)elim.numCholWorkItems + etgs - 1) / etgs, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(etgs, 1, 1)];
          dispatchCount++;

          // Phase 2: deterministic segmented sum
          [asyncEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
          [asyncEncoder setComputePipelineState:phase2Pipeline];
          [asyncEncoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
          [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)elimScratchBuffer.buffer()
                           offset:0 atIndex:1];
          [asyncEncoder setBuffer:(__bridge id<MTLBuffer>)elim.devCholSegments.buffer()
                           offset:0 atIndex:2];
          [asyncEncoder setBytes:&elim.numCholSegments length:sizeof(int64_t) atIndex:3];

          NSUInteger stgs = MIN(phase2Pipeline.maxTotalThreadsPerThreadgroup, 256);
          stgs = MIN(stgs, (NSUInteger)elim.numCholSegments);
          [asyncEncoder dispatchThreadgroups:
              MTLSizeMake(((NSUInteger)elim.numCholSegments + stgs - 1) / stgs, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(stgs, 1, 1)];
          dispatchCount++;
        }
      }

      [asyncEncoder endEncoding];

      // Signal shared event and commit on async queue — same pattern as LU.
      if (dispatchCount > 0) {
        if (!sharedEvent_) {
          sharedEvent_ = [sym.device newSharedEvent];
        }
        sharedEventValue_++;
        [asyncCmdBuf encodeSignalEvent:sharedEvent_ value:sharedEventValue_];
        [asyncCmdBuf commit];
        lastCommittedCmdBuf_ = asyncCmdBuf;
      }
    }
  }

  virtual double maxAbsDiag(const float* data, const int64_t* lumpStart, const int64_t* chainColPtr,
                            const int64_t* chainData, int64_t startLump, int64_t upToLump) override {
    if (explicitRecording_) return 0.0;
    @autoreleasepool {
      int64_t numLumps = upToLump - startLump;
      if (numLumps <= 0) return 0.0;

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!dataBufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::maxAbsDiag: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;

      // Result buffer: single uint32 for atomic max
      MetalMirror<uint32_t> resultBuf;
      resultBuf.resizeToAtLeast(1);
      resultBuf.ptr()[0] = 0;

      id<MTLComputePipelineState> pipeline = getPipeline("maxAbsDiag_kernel_float");

      int wgs = 256;
      int numGroups = (int)((numLumps + wgs - 1) / wgs);

      id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer() offset:0 atIndex:0];
      [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer() offset:0 atIndex:1];
      [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer() offset:0 atIndex:2];
      [encoder setBuffer:dataBuffer offset:dataOffset atIndex:3];
      [encoder setBytes:&startLump length:sizeof(int64_t) atIndex:4];
      [encoder setBytes:&numLumps length:sizeof(int64_t) atIndex:5];
      [encoder setBuffer:(__bridge id<MTLBuffer>)resultBuf.buffer() offset:0 atIndex:6];
      [encoder dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(wgs, 1, 1)];
      [encoder endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];

      // Convert uint bit pattern back to float
      uint32_t resultBits = resultBuf.ptr()[0];
      float result;
      memcpy(&result, &resultBits, sizeof(float));
      return result;
    }
  }

  virtual void potrf(int64_t n, float* data, int64_t offA) override {
    if (explicitRecording_) return;
    @autoreleasepool {
      if (n <= 0) return;

      // MPS Cholesky on GPU for all sizes
      flushPendingGemms();
      // If a previous potrf status hasn't been checked yet, wait and check now
      // before reusing the status buffer.
      if (potrfStatusPending_) {
        commitPending();
        waitForGpu();  // calls checkDeferredPotrfStatus()
      }
      if (pendingEncoder_) {
        [pendingEncoder_ endEncoding];
        pendingEncoder_ = nil;
      }
      if (!pendingCmdBuf_) {
        pendingCmdBuf_ = [sym.commandQueue commandBuffer];
      }

      // Look up MTLBuffer for data pointer
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      BASPACHO_CHECK_WHAT1(bufferInfo.first, "potrf: data buffer not registered");
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t baseOffset = bufferInfo.second;

      // Row-major n×n matrix descriptor
      MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
          matrixDescriptorWithRows:n columns:n
          rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32];
      MPSMatrix* mpsA = [[MPSMatrix alloc]
          initWithBuffer:dataBuffer
          offset:baseOffset + offA * sizeof(float)
          descriptor:descA];

      // MPS Cholesky: lower:YES — data is lower-triangular in row-major layout
      MPSMatrixDecompositionCholesky* mpsChol = [[MPSMatrixDecompositionCholesky alloc]
          initWithDevice:sym.device lower:YES order:n];

      // Reuse cached status buffer for detecting non-SPD matrices
      if (!potrfStatusBuf_) {
        potrfStatusBuf_ = [sym.device
            newBufferWithLength:sizeof(MPSMatrixDecompositionStatus)
                       options:MTLResourceStorageModeShared];
      }
      [mpsChol encodeToCommandBuffer:pendingCmdBuf_
          sourceMatrix:mpsA resultMatrix:mpsA status:potrfStatusBuf_];

      // Don't commit+wait here — leave pendingCmdBuf_ open so trsm() can be
      // encoded into the same command buffer (saving one GPU sync round-trip).
      // The status check is deferred to the next waitForGpu() call.
      potrfStatusPending_ = true;
    }
  }

  virtual void trsm(int64_t n, int64_t k, float* data, int64_t offA, int64_t offB) override {
    if (explicitRecording_) return;
    @autoreleasepool {
      if (n <= 0 || k <= 0) return;

      // MPS triangular solve on GPU for all sizes.
      // Flush pending work, end compute encoder (MPS needs its own encoding)
      flushPendingGemms();
      if (pendingEncoder_) {
        [pendingEncoder_ endEncoding];
        pendingEncoder_ = nil;
      }
      if (!pendingCmdBuf_) {
        pendingCmdBuf_ = [sym.commandQueue commandBuffer];
      }

      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      BASPACHO_CHECK_WHAT1(bufferInfo.first, "trsm: data buffer not registered");
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t baseOffset = bufferInfo.second;

      // L: n×n lower-triangular, row-major
      MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
          matrixDescriptorWithRows:n columns:n
          rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32];
      MPSMatrix* mpsA = [[MPSMatrix alloc]
          initWithBuffer:dataBuffer
          offset:baseOffset + offA * sizeof(float)
          descriptor:descA];

      // B: k×n, row-major
      MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
          matrixDescriptorWithRows:k columns:n
          rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32];
      MPSMatrix* mpsB = [[MPSMatrix alloc]
          initWithBuffer:dataBuffer
          offset:baseOffset + offB * sizeof(float)
          descriptor:descB];

      // Solve X * L^T = B in-place on B
      // right:YES (triangular matrix on the right), upper:NO (L is lower),
      // transpose:YES (using L^T), unit:NO (diagonal is not unit)
      MPSMatrixSolveTriangular* solve = [[MPSMatrixSolveTriangular alloc]
          initWithDevice:sym.device
                   right:YES
                   upper:NO
               transpose:YES
                    unit:NO
                   order:n
  numberOfRightHandSides:k
                   alpha:1.0];
      [solve encodeToCommandBuffer:pendingCmdBuf_
                sourceMatrix:mpsA
         rightHandSideMatrix:mpsB
              solutionMatrix:mpsB];

      // Submit potrf+trsm together without waiting. GPU executes asynchronously;
      // the next lump's prepareAssemble() will sync before CPU-visible writes.
      commitPending();
    }
  }

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const float* data,
                            int64_t offset) override {
    if (explicitRecording_) return;
    @autoreleasepool {
      if (m <= 0 || n <= 0 || k <= 0) return;

      // End compute encoder if active (MPS needs its own encoding pass)
      // but keep the same command buffer — Metal guarantees sequential execution
      // within a buffer, so assembly + GEMM can share one submission.
      if (pendingEncoder_) {
        [pendingEncoder_ endEncoding];
        pendingEncoder_ = nil;
      }

      // Ensure temp buffer is large enough
      tempBuffer.resizeToAtLeast(m * n);

      // MPS GEMM on GPU for all sizes.
      // C(n,m) = B(n,k) * A^T(k,m), where A and B are row-major at data+offset
      {
        auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
        if (!dataBufferInfo.first) {
          throw std::runtime_error("MetalNumericCtx<float>::saveSyrkGemm: data buffer not found");
        }
        id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
        size_t dataBaseOffset = dataBufferInfo.second;
        id<MTLBuffer> tempBuf = (__bridge id<MTLBuffer>)tempBuffer.buffer();

        // Matrix A: row-major (m rows, k cols), stride = k
        MPSMatrixDescriptor* descA =
            [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                  columns:k
                                                 rowBytes:k * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];

        // Matrix B: row-major (n rows, k cols), stride = k
        MPSMatrixDescriptor* descB =
            [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                  columns:k
                                                 rowBytes:k * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];

        // Matrix C: row-major (n rows, m cols), stride = m
        MPSMatrixDescriptor* descC =
            [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                  columns:m
                                                 rowBytes:m * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];

        MPSMatrix* mpsA = [[MPSMatrix alloc] initWithBuffer:dataBuffer
                                                     offset:dataBaseOffset + offset * sizeof(float)
                                                 descriptor:descA];
        MPSMatrix* mpsB = [[MPSMatrix alloc] initWithBuffer:dataBuffer
                                                     offset:dataBaseOffset + offset * sizeof(float)
                                                 descriptor:descB];
        MPSMatrix* mpsC = [[MPSMatrix alloc] initWithBuffer:tempBuf offset:0 descriptor:descC];

        // Compute: C(n,m) = B(n,k) * A^T(k,m)
        // left=B, right=A, transposeRight=YES
        MPSMatrixMultiplication* gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:sym.device
                                              transposeLeft:NO    // B as-is
                                             transposeRight:YES   // A^T
                                                 resultRows:n
                                              resultColumns:m
                                            interiorColumns:k
                                                      alpha:1.0
                                                       beta:0.0];

        // Encode MPS GEMM into pendingCmdBuf_ — don't commit yet.
        // The subsequent assemble() will encode into the same command buffer,
        // and Metal guarantees sequential execution within a command buffer.
        if (!pendingCmdBuf_) {
          pendingCmdBuf_ = [sym.commandQueue commandBuffer];
        }
        [gemm encodeToCommandBuffer:pendingCmdBuf_ leftMatrix:mpsB rightMatrix:mpsA
                       resultMatrix:mpsC];
      }
    }
  }

  virtual void prepareAssemble(int64_t targetLump) override {
    if (explicitRecording_) return;  // no-op during recording

    // Only flush if assemble() was actually called since last prepareAssemble.
    // For LU factorization (isGeneral()==true), eliminateBoardLU only calls
    // saveGemm — never assemble — so flushing is unnecessary and avoiding it
    // eliminates a per-lump CPU sync point.
    if (assembleWasCalled_) {
      commitAndWait();
      assembleWasCalled_ = false;
    }

    // GPU kernel: reads device-resident skeleton arrays, writes devSpanToChainOffset.
    // All inputs (devChainColPtr, devChainRowSpan, devChainData) are already on device
    // in MetalSymbolicCtx. No CPU loop, no memcpy — eliminates CPU→GPU sync point.
    const CoalescedBlockMatrixSkel& skel = sym.skel;
    int64_t numEntries = skel.chainColPtr[targetLump + 1] - skel.chainColPtr[targetLump];
    if (numEntries <= 0) return;

    id<MTLComputePipelineState> pipeline = getPipeline(
            "prepareAssemble_kernel_float");

    encodeKernel(
        pipeline,
        ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                      offset:0 atIndex:0];
          [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                      offset:0 atIndex:1];
          [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                      offset:0 atIndex:2];
          [encoder setBuffer:(__bridge id<MTLBuffer>)devSpanToChainOffset.buffer()
                      offset:0 atIndex:3];
          [encoder setBytes:&targetLump length:sizeof(int64_t) atIndex:4];
        },
        (NSUInteger)numEntries);
  }

  virtual void assemble(float* data, int64_t rectRowBegin, int64_t dstStride,
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    if (explicitRecording_) return;
    @autoreleasepool {
      if (numBlockRows <= 0 || numBlockCols <= 0) return;
      assembleWasCalled_ = true;

      // GPU assembly kernel for all sizes
      // Find the MTLBuffer for data
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::assemble: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      // Get pipeline state
      id<MTLComputePipelineState> pipeline = getPipeline(
              "assemble_kernel_float");

      int64_t numThreads = numBlockRows * numBlockCols;

      // Compute startRow = chainRowsTillEnd[srcColDataOffset - 1] (element before chain start)
      // This matches CPU ref where startRow = pChainRowsTillEnd[-1] after offsetting the pointer
      int64_t startRow = (srcColDataOffset > 0) ? sym.skel.chainRowsTillEnd[srcColDataOffset - 1] : 0;

      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBytes:&numBlockRows length:sizeof(int64_t) atIndex:0];
            [encoder setBytes:&numBlockCols length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&startRow length:sizeof(int64_t) atIndex:2];
            [encoder setBytes:&srcRectWidth length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&dstStride length:sizeof(int64_t) atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:srcColDataOffset * sizeof(int64_t)
                       atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:srcColDataOffset * sizeof(int64_t)
                       atIndex:6];
            [encoder setBuffer:(__bridge id<MTLBuffer>)devSpanToChainOffset.buffer()
                        offset:0
                       atIndex:7];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanOffsetInLump.buffer()
                        offset:0
                       atIndex:8];
            [encoder setBuffer:(__bridge id<MTLBuffer>)tempBuffer.buffer() offset:0 atIndex:9];
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:10];
          },
          (NSUInteger)numThreads);
    }
  }

  // Static pivoting: scan and perturb small diagonals.
  // CPU path for dense ops, GPU path for deferred execution.
  virtual int64_t perturbSmallDiagonals(int64_t n, float* data, int64_t offset, int64_t stride,
                                        float threshold) override {
    if (explicitRecording_) return 0;
    @autoreleasepool {
      if (n <= 0) return 0;

      // CPU path: no pending GPU work AND not using external encoder.
      // In external encoder mode, GPU work is encoded but not yet executed,
      // so CPU reads would see stale pre-factorization data.
      if (!pendingEncoder_ && !pendingCmdBuf_ && !sym.usingExternalEncoder) {
        int64_t count = 0;
        for (int64_t i = 0; i < n; i++) {
          int64_t idx = offset + i * stride + i;
          float diag = data[idx];
          if (!std::isfinite(diag) || std::abs(diag) < threshold) {
            data[idx] = (diag >= 0.0f) ? threshold : -threshold;
            count++;
          }
        }
        return count;
      }

      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error(
            "MetalNumericCtx<float>::perturbSmallDiagonals: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      // Allocate perturb count buffer on first call, zero it once.
      // All subsequent perturbDiag kernels atomically add to it.
      if (!perturbCountBuf_) {
        perturbCountBuf_ = [sym.device newBufferWithLength:sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
        *(uint32_t*)[perturbCountBuf_ contents] = 0;
        perturbCountPending_ = true;
      }

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_perturbDiag_kernel_float");

      // Adjust offset to be relative to the MTLBuffer start
      int64_t absOffset = (int64_t)(dataBaseOffset / sizeof(float)) + offset;

      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:0 atIndex:0];
            [encoder setBytes:&absOffset length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&stride length:sizeof(int64_t) atIndex:2];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&threshold length:sizeof(float) atIndex:4];
            [encoder setBuffer:perturbCountBuf_ offset:0 atIndex:5];
          },
          (NSUInteger)n);

      return 0;  // Actual count read back in deferredPerturbCount() after flush
    }
  }

  // ============ LU factorization methods ============

  virtual int getrf(int64_t m, int64_t n, float* data, int64_t offA, int64_t* pivots) override {
    if (explicitRecording_) { flushPendingGemms(); return 0; }
    @autoreleasepool {
      if (m <= 0 || n <= 0) return 0;

      int64_t minMN = std::min(m, n);

      // Flush pending saveGemm work items first — ensures all Schur
      // complement updates are dispatched before factorization of this lump.
      flushPendingGemms();

#ifdef BASPACHO_USE_BLAS
      // CPU BLAS path for larger lumps: multi-threaded LAPACK sgetrf (~10-50μs)
      // is much faster than single-threadgroup GPU kernel (~1ms for n≥64).
      // In external encoder mode, we cycle the encoder (end→commit→wait→CPU→
      // re-create) since unified memory means CPU BLAS works on the same data.
      if (minMN >= 64) {
        if (sym.usingExternalEncoder) {
          // End current encoder and commit the command buffer
          [sym.externalEncoder endEncoding];
          sym.externalEncoder = nil;
          [sym.externalCmdBuf commit];
          [sym.externalCmdBuf waitUntilCompleted];
          sym.externalCmdBuf = nil;
        } else {
          commitPending();
          waitForGpu();
        }
        float* A = data + offA;

        // BaSpaCho stores row-major; LAPACK expects col-major.
        // Transpose before + after gives correct row-major L*U result.
        if (m == n) {
          for (int64_t i = 0; i < n; i++)
            for (int64_t j = i + 1; j < n; j++)
              std::swap(A[i * n + j], A[j * n + i]);
        }

        std::vector<BLAS_INT> ipiv(minMN);
        int info = LAPACKE_sgetrf(0 /*col-major*/, (BLAS_INT)m, (BLAS_INT)n, A,
                                  (BLAS_INT)m, ipiv.data());

        if (m == n) {
          for (int64_t i = 0; i < n; i++)
            for (int64_t j = i + 1; j < n; j++)
              std::swap(A[i * n + j], A[j * n + i]);
        }

        // Write pivots to devAllPivots (shared memory, GPU-accessible) so that
        // downstream GPU kernels (applyRowPerm) can read them.
        if (devAllPivots.buffer()) {
          if (!allPivotsCpuBase_) allPivotsCpuBase_ = pivots;
          int64_t pivotOffset = pivots - allPivotsCpuBase_;
          allPivotsCount_ = std::max(allPivotsCount_, pivotOffset + minMN);
          int64_t* gpuPivots = devAllPivots.ptr() + pivotOffset;
          for (int64_t i = 0; i < minMN; i++) {
            gpuPivots[i] = ipiv[i] - 1;  // LAPACK is 1-based
          }
          pivotsOnGpu_ = true;
        }

        // Also write to host pivots array
        for (int64_t i = 0; i < minMN; i++) {
          pivots[i] = ipiv[i] - 1;
        }

        // Re-create external encoder for subsequent GPU dispatches
        if (sym.usingExternalEncoder) {
          sym.externalCmdBuf = [sym.commandQueue commandBuffer];
          sym.externalEncoder = [sym.externalCmdBuf computeCommandEncoder];
        }

        return info;
      }
#endif

      // GPU path: single-threadgroup Metal kernel (efficient for small lumps)
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::getrf: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      return getrfCustom(m, n, minMN, dataBuffer, dataBaseOffset, offA, pivots);
    }
  }

  // Custom Metal kernel LU factorization. Dispatched via encodeKernel() so it
  // stays within the same compute encoder — no encoder transitions per lump.
  // Outputs int64_t pivots directly (no uint32 conversion needed).
  int getrfCustom(int64_t m, int64_t n, int64_t minMN,
                  id<MTLBuffer> dataBuffer, size_t dataBaseOffset,
                  int64_t offA, int64_t* pivots) {

    // Determine pivot destination buffer
    id<MTLBuffer> pivotBuffer;
    NSUInteger pivotByteOffset;

    if (devAllPivots.buffer()) {
      if (!allPivotsCpuBase_) {
        allPivotsCpuBase_ = pivots;
      }
      int64_t pivotOffset = pivots - allPivotsCpuBase_;
      allPivotsCount_ = std::max(allPivotsCount_, pivotOffset + minMN);

      if (sym.usingExternalEncoder) {
        // External encoder: write directly to caller's pivot buffer
        auto bufInfo = MetalBufferRegistry::instance().findBuffer(allPivotsCpuBase_);
        if (bufInfo.first) {
          pivotBuffer = (__bridge id<MTLBuffer>)bufInfo.first;
          pivotByteOffset = (NSUInteger)bufInfo.second
              + pivotOffset * sizeof(int64_t);
        } else {
          pivotBuffer = (__bridge id<MTLBuffer>)devAllPivots.buffer();
          pivotByteOffset = pivotOffset * sizeof(int64_t);
        }
      } else {
        pivotBuffer = (__bridge id<MTLBuffer>)devAllPivots.buffer();
        pivotByteOffset = pivotOffset * sizeof(int64_t);
      }
      pivotsOnGpu_ = true;
    } else {
      // No pre-allocated pivot buffer (e.g. simple factorLU without
      // preAllocateForLU). Use devPivots as a temporary pivot buffer —
      // kernel writes pivots there, then we copy back to CPU after dispatch.
      devPivots.resizeToAtLeast(minMN);
      pivotBuffer = (__bridge id<MTLBuffer>)devPivots.buffer();
      pivotByteOffset = 0;
      pivotsOnGpu_ = false;  // Will copy back to CPU below
    }

    // Absolute element offset into the data buffer
    int64_t absOffA = (int64_t)(dataBaseOffset / sizeof(float)) + offA;

    id<MTLComputePipelineState> pipeline = getPipeline(
            "lu_getrf_kernel_float");

    // Compute power-of-2 threadgroup size (matches CUDA dispatch pattern).
    // Single threadgroup — all threads cooperate via threadgroup_barrier.
    int threads = std::min((int)std::max(m, n), (int)256);
    int t = 1;
    while (t < threads) t <<= 1;
    NSUInteger numThreads = (NSUInteger)t;

    // Dispatch via encodeKernel (stays in same encoder)
    bool needCopyBack = !pivotsOnGpu_;
    encodeKernel(
        pipeline,
        ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:dataBuffer offset:0 atIndex:0];
          [encoder setBytes:&absOffA length:sizeof(int64_t) atIndex:1];
          [encoder setBytes:&m length:sizeof(int64_t) atIndex:2];
          [encoder setBytes:&n length:sizeof(int64_t) atIndex:3];
          [encoder setBuffer:pivotBuffer offset:pivotByteOffset atIndex:4];
        }, numThreads);

    // Non-pre-allocated path: commit GPU work and copy pivots back to CPU
    if (needCopyBack) {
      commitPending();
      waitForGpu();
      memcpy(pivots, devPivots.ptr(), minMN * sizeof(int64_t));
    }

    return 0;
  }

  virtual void trsmLowerUnit(int64_t m, int64_t n, const float* L, int64_t offL, float* B,
                              int64_t offB, int64_t ldb) override {
    if (explicitRecording_) return;
    @autoreleasepool {
      if (m <= 0 || n <= 0) return;

      // GPU kernel for all sizes
      auto lBufferInfo = MetalBufferRegistry::instance().findBuffer(L);
      auto bBufferInfo = MetalBufferRegistry::instance().findBuffer(B);
      if (!lBufferInfo.first || !bBufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::trsmLowerUnit: buffer not found");
      }
      id<MTLBuffer> lBuffer = (__bridge id<MTLBuffer>)lBufferInfo.first;
      size_t lBaseOffset = lBufferInfo.second;
      id<MTLBuffer> bBuffer = (__bridge id<MTLBuffer>)bBufferInfo.first;
      size_t bBaseOffset = bBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_trsmLowerUnit_kernel_float");

      // Power-of-2 threadgroup size: use max(m, n) since recursive TRSM→GEMM
      // distributes GEMM work across both rows and columns
      int thr = std::min(std::max((int)m, (int)n), 256);
      int t = 1;
      while (t < thr) t <<= 1;
      NSUInteger numThreads = (NSUInteger)t;

      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:lBuffer offset:lBaseOffset atIndex:0];
            [encoder setBytes:&offL length:sizeof(int64_t) atIndex:1];
            [encoder setBuffer:bBuffer offset:bBaseOffset atIndex:2];
            [encoder setBytes:&offB length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&m length:sizeof(int64_t) atIndex:4];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&ldb length:sizeof(int64_t) atIndex:6];
          },
          numThreads);
    }
  }

  virtual void trsmUpperRight(int64_t m, int64_t n, const float* U, int64_t offU, float* B,
                               int64_t offB, int64_t ldb) override {
    if (explicitRecording_) return;
    @autoreleasepool {
      if (m <= 0 || n <= 0) return;

      // GPU kernel for all sizes
      auto uBufferInfo = MetalBufferRegistry::instance().findBuffer(U);
      auto bBufferInfo = MetalBufferRegistry::instance().findBuffer(B);
      if (!uBufferInfo.first || !bBufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::trsmUpperRight: buffer not found");
      }
      id<MTLBuffer> uBuffer = (__bridge id<MTLBuffer>)uBufferInfo.first;
      size_t uBaseOffset = uBufferInfo.second;
      id<MTLBuffer> bBuffer = (__bridge id<MTLBuffer>)bBufferInfo.first;
      size_t bBaseOffset = bBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_trsmUpperRight_kernel_float");

      // Power-of-2 threadgroup size for parallel row processing
      int thr = std::min((int)m, 256);
      int t = 1;
      while (t < thr) t <<= 1;
      NSUInteger numThreads = (NSUInteger)t;

      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:uBuffer offset:uBaseOffset atIndex:0];
            [encoder setBytes:&offU length:sizeof(int64_t) atIndex:1];
            [encoder setBuffer:bBuffer offset:bBaseOffset atIndex:2];
            [encoder setBytes:&offB length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&m length:sizeof(int64_t) atIndex:4];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&ldb length:sizeof(int64_t) atIndex:6];
          },
          numThreads);
    }
  }

  virtual void saveGemm(int64_t m, int64_t n, int64_t k, const float* L, int64_t offL,
                         int64_t ldL, const float* U, int64_t offU, int64_t ldU, float* C,
                         int64_t offC, int64_t ldC) override {
    @autoreleasepool {
      if (m <= 0 || n <= 0 || k <= 0) return;

      // Batched GPU path: buffer work items, flush later in flushPendingGemms()
      // On first call, cache the data buffer info (L, U, C all share the same buffer)
      if (!cachedDataBuffer_) {
        auto bufferInfo = MetalBufferRegistry::instance().findBuffer(C);
        if (!bufferInfo.first) {
          throw std::runtime_error("MetalNumericCtx<float>::saveGemm: buffer not found");
        }
        cachedDataBuffer_ = (__bridge id<MTLBuffer>)bufferInfo.first;
        cachedDataBaseOffset_ = bufferInfo.second;
      }

      // L, U, C all point into the same MTLBuffer. Compute absolute element
      // offsets from the buffer start using pointer arithmetic relative to C.
      int64_t cElemFromBufStart = (int64_t)(cachedDataBaseOffset_ / sizeof(float));

      LUGemmWorkItem item;
      item.offL = cElemFromBufStart + (int64_t)(L - C) + offL;
      item.ldL = ldL;
      item.offU = cElemFromBufStart + (int64_t)(U - (const float*)C) + offU;
      item.ldU = ldU;
      item.offC = cElemFromBufStart + offC;
      item.ldC = ldC;
      item.m = m;
      item.n = n;
      item.k = k;

      if (explicitRecording_) {
        // Explicit recording: capture item, skip GPU work
        recordedItems_.push_back(item);
        recordingBatchCount_++;
        sym.luGemmCalls++;
        return;
      }
      if (recordState_ == RecordState::Recording) {
        // Auto-recording: capture item AND fall through to normal execution
        recordedItems_.push_back(item);
        recordingBatchCount_++;
      }
      if (recordState_ == RecordState::Ready) {
        // Items already on device — dispatched from pre-computed buffer in flushPendingGemms.
        // Still need to cache the current data buffer so flushPendingGemms binds the
        // correct MTLBuffer (data pointer may differ between factorLU calls).
        if (!cachedDataBuffer_) {
          auto bufferInfo = MetalBufferRegistry::instance().findBuffer(C);
          if (bufferInfo.first) {
            cachedDataBuffer_ = (__bridge id<MTLBuffer>)bufferInfo.first;
            cachedDataBaseOffset_ = bufferInfo.second;
          }
        }
        sym.luGemmCalls++;
        return;
      }

      pendingGemms_.push_back(item);
      sym.luGemmCalls++;
    }
  }

  virtual void applyRowPerm(int64_t* pivots, int64_t n, float* data, int64_t offData, int64_t ld,
                             int64_t numCols) override {
    if (explicitRecording_) return;
    @autoreleasepool {
      if (n <= 0 || numCols <= 0) return;

      // GPU kernel for all sizes
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::applyRowPerm: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_applyRowPerm_kernel_float");

      // Determine pivot buffer and offset.
      // If pivots are GPU-resident (from getrf deferred path), use devAllPivots
      // at the correct offset. Otherwise, copy from CPU to devPivots.
      id<MTLBuffer> pivotBuffer;
      size_t pivotByteOffset = 0;
      if (pivotsOnGpu_ && allPivotsCpuBase_) {
        int64_t pivotElemOffset = pivots - allPivotsCpuBase_;
        pivotBuffer = (__bridge id<MTLBuffer>)devAllPivots.buffer();
        pivotByteOffset = pivotElemOffset * sizeof(int64_t);
      } else {
        devPivots.resizeToAtLeast(n);
        memcpy(devPivots.ptr(), pivots, n * sizeof(int64_t));
        pivotBuffer = (__bridge id<MTLBuffer>)devPivots.buffer();
        pivotByteOffset = 0;
      }

      // Dispatch as single threadgroup with min(256, numCols) threads
      // (threadgroup_barrier in kernel requires single threadgroup)
      NSUInteger numThreads = (NSUInteger)std::min((int64_t)256, numCols);

      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:pivotBuffer offset:pivotByteOffset atIndex:0];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:1];
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:2];
            [encoder setBytes:&offData length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&ld length:sizeof(int64_t) atIndex:4];
            [encoder setBytes:&numCols length:sizeof(int64_t) atIndex:5];
          },
          numThreads);
    }
  }

  bool hasPostGetrfFused() const override { return true; }

  void postGetrfFused(float* data, int64_t diagOffset, int64_t lumpSize,
                      int64_t* pivots, int64_t pivotOffset,
                      float threshold, bool enablePerturb,
                      int64_t belowDiagOffset, int64_t numRowsBelowDiag,
                      int64_t lump, int64_t upperDataBase) override {
    if (explicitRecording_) return;
    @autoreleasepool {
      // Count upper spans for threadgroup dispatch
      int64_t rangeStart = sym.skel.isGeneral() ? sym.skel.upperChainRowPtr[lump] : 0;
      int64_t rangeEnd = sym.skel.isGeneral() ? sym.skel.upperChainRowPtr[lump + 1] : 0;
      int64_t numUpperSpans = rangeEnd - rangeStart;

      // Total threadgroups: 1 (below-diag + perturb) + numUpperSpans
      int64_t numThreadgroups = 1 + numUpperSpans;

      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error(
            "MetalNumericCtx<float>::postGetrfFused: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      // Allocate perturb count buffer on first call (same as perturbSmallDiagonals)
      if (!perturbCountBuf_) {
        perturbCountBuf_ = [sym.device newBufferWithLength:sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
        *(uint32_t*)[perturbCountBuf_ contents] = 0;
        perturbCountPending_ = true;
      }
      int enablePerturbInt = enablePerturb ? 1 : 0;

      // Threadgroup size based on lumpSize (for TRSM parallelism within each threadgroup)
      int thr = std::min((int)lumpSize, 256);
      int t = 1;
      while (t < thr) t <<= 1;
      NSUInteger tgSize = (NSUInteger)t;

      id<MTLComputePipelineState> pipeline = getPipeline("lu_postGetrf_kernel_float");

      // Resolve pivot buffer (same logic as applyRowPerm)
      id<MTLBuffer> pivotBuffer;
      size_t pivotByteOffset = 0;
      int64_t zeroPivotOffset = 0;
      if (pivotsOnGpu_ && allPivotsCpuBase_) {
        int64_t pivotElemOffset = (pivots + pivotOffset) - allPivotsCpuBase_;
        pivotBuffer = (__bridge id<MTLBuffer>)devAllPivots.buffer();
        pivotByteOffset = pivotElemOffset * sizeof(int64_t);
      } else {
        // Pivots on CPU — copy to device first
        devPivots.resizeToAtLeast(lumpSize);
        memcpy(devPivots.ptr(), pivots + pivotOffset, lumpSize * sizeof(int64_t));
        pivotBuffer = (__bridge id<MTLBuffer>)devPivots.buffer();
      }

      encodeKernelWithGroups(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:1];   // constant alias
            [encoder setBuffer:pivotBuffer offset:pivotByteOffset atIndex:2];
            [encoder setBytes:&zeroPivotOffset length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&diagOffset length:sizeof(int64_t) atIndex:4];
            [encoder setBytes:&lumpSize length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&threshold length:sizeof(float) atIndex:6];
            [encoder setBuffer:perturbCountBuf_ offset:0 atIndex:7];
            [encoder setBytes:&enablePerturbInt length:sizeof(int) atIndex:8];
            [encoder setBytes:&belowDiagOffset length:sizeof(int64_t) atIndex:9];
            [encoder setBytes:&numRowsBelowDiag length:sizeof(int64_t) atIndex:10];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainColSpan.buffer()
                        offset:0
                       atIndex:11];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainData.buffer()
                        offset:0
                       atIndex:12];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                        offset:0
                       atIndex:13];
            [encoder setBytes:&upperDataBase length:sizeof(int64_t) atIndex:14];
            [encoder setBytes:&rangeStart length:sizeof(int64_t) atIndex:15];
          },
          (NSUInteger)numThreadgroups, tgSize);
    }
  }

  void flush() override {
    flushPendingGemms();
    if (explicitRecording_) return;  // skip pivot copies during recording

    if (!sym.usingExternalEncoder) {
      // Normal mode: commit pending work and wait for GPU to finish.
      // waitForGpu() calls flushDeferredState() to copy pivots to host.
      commitAndWait();
    } else {
      // External encoder mode: GPU work not committed yet. Skip pivot copy.
      // Caller must call flush() again AFTER committing the command buffer
      // and clearing the external encoder, to trigger the pivot copy below.
    }

    // Copy deferred GPU pivots to host. Safe to call in both modes:
    // - Normal mode: GPU work completed above, data is valid
    // - External encoder mode: this is a no-op because pivotsOnGpu_ is true
    //   but the GPU hasn't run yet, UNLESS external encoder was already cleared
    //   (post-commit call) in which case data IS valid
    if (!sym.usingExternalEncoder) {
      flushDeferredState();
    }

    // Reset batched state for next factorization
    cachedDataBuffer_ = nil;
    cachedDataBaseOffset_ = 0;
    gemmWorkBufUsedBytes_ = 0;
    gemmWorkBufInFlight_ = false;

    if (!sym.usingExternalEncoder) {
      // Only reset pivot state if NOT in external encoder mode — the caller
      // will need to flush pivots after committing the external command buffer.
      allPivotsCpuBase_ = nullptr;
      allPivotsCount_ = 0;
      MetalContext::instance().synchronize();
    }
  }

  // Reset per-factorization mutable state without deallocating any buffers.
  // Allows reusing this context across multiple factorLU calls.
  void reset() override {
    pendingGemms_.clear();
    gemmWorkBufUsedBytes_ = 0;
    gemmWorkBufInFlight_ = false;
    cachedDataBuffer_ = nil;
    cachedDataBaseOffset_ = 0;
    allPivotsCpuBase_ = nullptr;
    allPivotsCount_ = 0;
    pivotsOnGpu_ = false;
    perturbCountPending_ = false;
    deferredElimPerturbCount_ = 0;
    deferredElimPerturbBuf_ = nil;
    assembleWasCalled_ = false;
    potrfStatusPending_ = false;

    // Auto-recording state machine
    if (recordState_ == RecordState::Recording && !explicitRecording_) {
      // First factorLU completed — finalize recording and upload to device
      if (recordingBatchCount_ > 0) {
        size_t startIdx = recordedItems_.size() - recordingBatchCount_;
        recordedFlushPoints_.push_back({startIdx, recordingBatchCount_});
        recordingBatchCount_ = 0;
      }
      totalPrecomputedItems_ = recordedItems_.size();
      if (totalPrecomputedItems_ > 0) {
        size_t bytes = totalPrecomputedItems_ * sizeof(LUGemmWorkItem);
        size_t int64sNeeded = (bytes + sizeof(int64_t) - 1) / sizeof(int64_t);
        devPrecomputedItems_.resizeToAtLeast(int64sNeeded);
        memcpy(devPrecomputedItems_.ptr(), recordedItems_.data(), bytes);
      }
      recordedItems_.clear();
      recordedItems_.shrink_to_fit();
      recordState_ = RecordState::Ready;
      precomputedFlushIdx_ = 0;
    } else if (recordState_ == RecordState::Ready) {
      precomputedFlushIdx_ = 0;
    } else if (recordState_ == RecordState::Idle && !explicitRecording_) {
      // Start auto-recording on next factorLU
      recordState_ = RecordState::Recording;
      recordedItems_.clear();
      recordedFlushPoints_.clear();
      recordingBatchCount_ = 0;
    }
    // Buffers (tempBuffer, devSpanToChainOffset, devPivots, devAllPivots,
    // devGemmWorkBuf_, perturbCountBuf_, devPrecomputedItems_) are NOT freed — reused across calls.
  }

  // Pre-allocate all Metal buffers to max needed sizes so no allocation occurs
  // during the hot factorization path. Required for streamable/external encoder mode.
  void preAllocateForLU(int64_t maxDenseBlockSize, int64_t totalDensePivots) override {
    if (maxDenseBlockSize <= 0) return;

    // Pre-allocate pivot buffer for applyRowPerm
    devPivots.resizeToAtLeast(maxDenseBlockSize);

    // Pre-allocate all-pivots buffer if needed
    if (totalDensePivots > 0) {
      devAllPivots.resizeToAtLeast(totalDensePivots);
    }

    // Pre-allocate perturbation counter buffer
    if (!perturbCountBuf_) {
      perturbCountBuf_ = [sym.device
          newBufferWithLength:sizeof(uint32_t)
                     options:MTLResourceStorageModeShared];
      *(uint32_t*)[perturbCountBuf_ contents] = 0;
    }
  }

  // Flush deferred pivot copies as device-to-device (keeps pivots on GPU).
  // On Metal with unified memory, this is a simple memcpy between Metal buffer
  // backing stores — both are CPU-accessible, so no D->H->D round-trip.
  void flushDevicePivots(int64_t* devDstPivots) override {
    flushPendingGemms();
    if (explicitRecording_) return;
    if (pivotsOnGpu_ && allPivotsCount_ > 0) {
      if (sym.usingExternalEncoder) {
        // External encoder mode: getrf kernel already wrote directly to
        // devDstPivots buffer (see getrfCustom). No memcpy needed — data will
        // be valid when the command buffer executes.
      } else {
        // Normal mode: commit GPU work and copy from devAllPivots to host.
        commitAndWait();
        memcpy(devDstPivots, devAllPivots.ptr(), allPivotsCount_ * sizeof(int64_t));
      }
      pivotsOnGpu_ = false;
      allPivotsCpuBase_ = nullptr;
      allPivotsCount_ = 0;
    }
  }

  int64_t deferredPerturbCount() override {
    if (explicitRecording_) return 0;
    // Read the accumulated GPU atomic counter (valid after flush/commitAndWait)
    int64_t count = 0;
    if (perturbCountPending_ && perturbCountBuf_) {
      count = *(uint32_t*)[perturbCountBuf_ contents];
      perturbCountPending_ = false;
      // Reset buffer for next factorization
      *(uint32_t*)[perturbCountBuf_ contents] = 0;
    }
    // Include deferred sparse elimination perturb count
    count += deferredElimPerturbCount_;
    deferredElimPerturbCount_ = 0;
    return count;
  }

  // ============ Recording mode API ============

  void beginRecording() override {
    explicitRecording_ = true;
    recordState_ = RecordState::Recording;
    recordedItems_.clear();
    recordedFlushPoints_.clear();
    recordingBatchCount_ = 0;
  }

  void endRecording() override {
    // Flush any remaining batch
    if (recordingBatchCount_ > 0) {
      size_t startIdx = recordedItems_.size() - recordingBatchCount_;
      recordedFlushPoints_.push_back({startIdx, recordingBatchCount_});
      recordingBatchCount_ = 0;
    }

    explicitRecording_ = false;
    totalPrecomputedItems_ = recordedItems_.size();

    if (totalPrecomputedItems_ > 0) {
      // Upload all recorded items to device (single memcpy at init time).
      // On Metal unified memory this is fast — just a CPU write to shared buffer.
      size_t bytes = totalPrecomputedItems_ * sizeof(LUGemmWorkItem);
      size_t int64sNeeded = (bytes + sizeof(int64_t) - 1) / sizeof(int64_t);
      devPrecomputedItems_.resizeToAtLeast(int64sNeeded);
      memcpy(devPrecomputedItems_.ptr(), recordedItems_.data(), bytes);
    }

    recordState_ = RecordState::Ready;
    precomputedFlushIdx_ = 0;

    // Free host recording buffers (data is now on device)
    recordedItems_.clear();
    recordedItems_.shrink_to_fit();
  }

  MetalSymbolicCtx& sym;
  int64_t numSpans_;
  MetalMirror<float> tempBuffer;
  MetalMirror<int64_t> devSpanToChainOffset;
  std::vector<int64_t> spanToChainOffset;
  MetalMirror<int64_t> devPivots;        // GPU buffer for LU pivots
  id<MTLBuffer> potrfStatusBuf_ = nil;  // Cached status buffer for MPS Cholesky
  bool assembleWasCalled_ = false;      // Track whether assemble() was called

  // GPU-resident pivot state: after getrf, pivots live in devAllPivots on GPU
  // at the correct offset for each lump. applyRowPerm reads from devPivots
  // (a view into devAllPivots). CPU copy deferred to flush().
  bool pivotsOnGpu_ = false;           // True if devPivots has valid GPU-side pivots
  MetalMirror<int64_t> devAllPivots;   // Full pivot buffer for all lumps
  int64_t* allPivotsCpuBase_ = nullptr;  // CPU pivot array base (for deferred copy)
  int64_t allPivotsCount_ = 0;        // Total pivot count (for deferred copy)

  // GPU-resident perturbSmallDiagonals state
  id<MTLBuffer> perturbCountBuf_ = nil;  // Atomic counter for perturbed diagonals (accumulates)
  bool perturbCountPending_ = false;     // True if perturbCountBuf_ has unread count

  // Batched saveGemm state (append-only: each flush writes at increasing offsets)
  std::vector<LUGemmWorkItem> pendingGemms_;   // CPU-side work item accumulator
  MetalMirror<int64_t> devGemmWorkBuf_;        // GPU buffer for work items (append-only)
  size_t gemmWorkBufUsedBytes_ = 0;            // Current write position (bytes, 16-aligned)
  bool gemmWorkBufInFlight_ = false;           // True if GPU may be reading devGemmWorkBuf_
  id<MTLBuffer> cachedDataBuffer_ = nil;       // Cached MTLBuffer for data
  size_t cachedDataBaseOffset_ = 0;            // Cached byte offset into MTLBuffer

  // Command buffer batching state
  id<MTLCommandBuffer> pendingCmdBuf_ = nil;
  id<MTLComputeCommandEncoder> pendingEncoder_ = nil;
  int pendingDispatchCount_ = 0;
  id<MTLCommandBuffer> lastCommittedCmdBuf_ = nil;  // For deferred GPU sync
  bool potrfStatusPending_ = false;                 // Deferred potrf status check

  // MTLSharedEvent for lower-overhead CPU-GPU synchronization
  id<MTLSharedEvent> sharedEvent_ = nil;
  uint64_t sharedEventValue_ = 0;

  // Deferred sparse elimination perturb count (read after GPU completion)
  id<MTLBuffer> deferredElimPerturbBuf_ = nil;
  int64_t deferredElimPerturbCount_ = 0;

  // Scratch buffer for two-phase deterministic sparse elimination
  MetalMirror<float> elimScratchBuffer;

  // ============ Auto-recording for pre-computed GemmWorkItems ============
  // Transparent optimization: the first factorLU call records the GemmWorkItem
  // schedule (structure-dependent, never changes) while executing normally.
  // Subsequent calls dispatch from a pre-computed device buffer, eliminating
  // per-lump CPU memcpy in flushPendingGemms.
  //
  // State machine (transitions happen in reset()):
  //   Idle -> Recording  (first reset() call)
  //   Recording -> Ready (second reset(): finalize + upload items to device)
  //   Ready -> Ready     (subsequent reset(): just reset flush index)
  //
  // beginRecording()/endRecording() override auto-recording for explicit
  // no-op recording (needed for CUDA graph capture via FFI).
  enum class RecordState { Idle, Recording, Ready };
  RecordState recordState_ = RecordState::Idle;
  bool explicitRecording_ = false;                     // true when beginRecording() was called explicitly
  std::vector<LUGemmWorkItem> recordedItems_;          // all items across all flushes
  std::vector<std::pair<size_t, size_t>> recordedFlushPoints_;  // (startIdx, count) per flush
  size_t recordingBatchCount_ = 0;                     // items in current batch

  MetalMirror<int64_t> devPrecomputedItems_;           // LUGemmWorkItems on device (as int64_t for MetalMirror)
  size_t precomputedFlushIdx_ = 0;                     // current flush point index during dispatch
  size_t totalPrecomputedItems_ = 0;                   // total items for bounds checking

};

// Solve context for float - Metal implementation
template <>
struct MetalSolveCtx<float> : SolveCtx<float> {
  MetalSolveCtx(MetalSymbolicCtx& sym_, int nRHS_) : sym(sym_), nRHS(nRHS_) {
    // Allocate temp buffer for vector assembly
    tempVecBuffer.resizeToAtLeast(sym.skel.order() * nRHS);
  }

  virtual ~MetalSolveCtx() override {
    // Flush any pending GPU work before destruction
    commitAndWait();
  }

  // Check if all lumps in [lumpsBegin, lumpsEnd) are size-1.
  // Uses host-side skel.lumpStart — no GPU readback needed.
  bool allLumpsSize1(int64_t lumpsBegin, int64_t lumpsEnd) const {
    return sym.skel.lumpStart[lumpsEnd] - sym.skel.lumpStart[lumpsBegin]
           == (lumpsEnd - lumpsBegin);
  }

  // Encode a kernel dispatch onto a persistent compute encoder within the
  // pending command buffer. Uses a single encoder for all dispatches with
  // memory barriers between them to ensure correct data ordering.
  // In external encoder mode, dispatches are recorded into the caller-provided
  // encoder instead of a self-managed command buffer.
  void encodeKernel(id<MTLComputePipelineState> pipeline,
                    void (^encodeBlock)(id<MTLComputeCommandEncoder>),
                    NSUInteger numThreads) {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder;
      if (sym.usingExternalEncoder) {
        encoder = sym.externalEncoder;
      } else {
        if (!pendingCmdBuf_) {
          pendingCmdBuf_ = [sym.commandQueue commandBuffer];
        }
        if (!pendingEncoder_) {
          pendingEncoder_ = [pendingCmdBuf_ computeCommandEncoder];
          pendingDispatchCount_ = 0;
        }
        encoder = pendingEncoder_;
      }

      // Insert memory barrier so previous dispatches' buffer writes are visible.
      // skipNextBarrier_ allows skipping when consecutive dispatches operate on
      // disjoint memory ranges (e.g., perm on dense lumps, sparse elim on sparse lumps).
      if (pendingDispatchCount_ > 0 && !skipNextBarrier_) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      skipNextBarrier_ = false;

      [encoder setComputePipelineState:pipeline];
      encodeBlock(encoder);

      NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
      threadGroupSize = MIN(threadGroupSize, numThreads);

      MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
      MTLSize numGroups =
          MTLSizeMake((numThreads + threadGroupSize - 1) / threadGroupSize, 1, 1);

      [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
      pendingDispatchCount_++;
    }
  }

  // Encode a kernel dispatch with explicit threadgroup counts.
  // Used when each threadgroup must cooperate internally (e.g., per-lump
  // recursive TRSV with threadgroup_barrier).
  void encodeKernelWithGroups(id<MTLComputePipelineState> pipeline,
                              void (^encodeBlock)(id<MTLComputeCommandEncoder>),
                              NSUInteger numGroupCount, NSUInteger threadsPerGroup) {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder;
      if (sym.usingExternalEncoder) {
        encoder = sym.externalEncoder;
      } else {
        if (!pendingCmdBuf_) {
          pendingCmdBuf_ = [sym.commandQueue commandBuffer];
        }
        if (!pendingEncoder_) {
          pendingEncoder_ = [pendingCmdBuf_ computeCommandEncoder];
          pendingDispatchCount_ = 0;
        }
        encoder = pendingEncoder_;
      }

      if (pendingDispatchCount_ > 0 && !skipNextBarrier_) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      skipNextBarrier_ = false;

      [encoder setComputePipelineState:pipeline];
      encodeBlock(encoder);

      MTLSize tpg = MTLSizeMake(threadsPerGroup, 1, 1);
      MTLSize ng = MTLSizeMake(numGroupCount, 1, 1);
      [encoder dispatchThreadgroups:ng threadsPerThreadgroup:tpg];
      pendingDispatchCount_++;
    }
  }

  // Commit the pending command buffer WITHOUT waiting for GPU completion.
  // In external encoder mode, this is a no-op — the caller manages the command buffer.
  void commitPending() {
    if (sym.usingExternalEncoder) return;
    if (pendingCmdBuf_) {
      if (pendingEncoder_) {
        [pendingEncoder_ endEncoding];
        pendingEncoder_ = nil;
      }
      [pendingCmdBuf_ commit];
      lastCommittedCmdBuf_ = pendingCmdBuf_;
      pendingCmdBuf_ = nil;
      pendingDispatchCount_ = 0;
    }
  }

  // Wait for the most recently committed command buffer to complete.
  // In external encoder mode, this is a no-op — the caller manages synchronization.
  void waitForGpu() {
    if (sym.usingExternalEncoder) return;
    if (lastCommittedCmdBuf_) {
      [lastCommittedCmdBuf_ waitUntilCompleted];
      lastCommittedCmdBuf_ = nil;
    }
  }

  // Commit the pending command buffer and wait for all GPU work to complete.
  void commitAndWait() {
    commitPending();
    waitForGpu();
  }

  virtual void sparseElimSolveL(const SymElimCtx& elimData, const float* data, int64_t lumpsBegin,
                                int64_t lumpsEnd, float* C, int64_t ldc) override {
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);

      // Find buffers
      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!dataBufferInfo.first || !cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::sparseElimSolveL: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;
      size_t cOffset = cBufferInfo.second;

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      // Encode diagonal solve kernel
      id<MTLComputePipelineState> pipeline = getPipeline(
              "sparseElim_diagSolveL_float");

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer() offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                        offset:0
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer() offset:0 atIndex:2];
            [encoder setBuffer:dataBuffer offset:dataOffset atIndex:3];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:4];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];
          },
          (NSUInteger)numLumps);

      // Encode below-diagonal multiply: matQ -= block * matC
      id<MTLComputePipelineState> subDiagPipeline = getPipeline(
              "sparseElim_subDiagMult_float");

      encodeKernel(
          subDiagPipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer() offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer() offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                        offset:0
                       atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:0
                       atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer() offset:0 atIndex:4];
            [encoder setBuffer:dataBuffer offset:dataOffset atIndex:5];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:6];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:8];
            [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:9];
            [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:10];
          },
          (NSUInteger)numLumps);
    }
  }

  virtual void sparseElimSolveLt(const SymElimCtx& elimData, const float* data, int64_t lumpsBegin,
                                 int64_t lumpsEnd, float* C, int64_t ldc) override {
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);

      // Find buffers
      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!dataBufferInfo.first || !cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::sparseElimSolveLt: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;
      size_t cOffset = cBufferInfo.second;

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      int64_t nRHS64 = nRHS;

      // Encode below-diagonal transpose multiply first: matC -= block^T * matQ
      // Use size-1 flat kernel when all lumps are size-1 (e.g. c6288)
      auto subDiagEncodeBlock = ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer() offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                    offset:0
                   atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                    offset:0
                   atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer() offset:0 atIndex:4];
        [encoder setBuffer:dataBuffer offset:dataOffset atIndex:5];
        [encoder setBuffer:cBuffer offset:cOffset atIndex:6];
        [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:7];
        [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:8];
        [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:9];
        [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:10];
      };

      if (allLumpsSize1(lumpsBegin, lumpsEnd)) {
        id<MTLComputePipelineState> subDiagPipeline = getPipeline(
                "sparseElim_subDiagMultT_size1_float");
        encodeKernel(subDiagPipeline, subDiagEncodeBlock, (NSUInteger)numLumps);
      } else {
        id<MTLComputePipelineState> subDiagPipeline = getPipeline(
                "sparseElim_subDiagMultT_float");
        NSUInteger subDiagTgSize = MIN(subDiagPipeline.maxTotalThreadsPerThreadgroup, 256);
        encodeKernelWithGroups(subDiagPipeline, subDiagEncodeBlock,
                               (NSUInteger)numLumps, subDiagTgSize);
      }

      // Then encode diagonal solve kernel
      id<MTLComputePipelineState> pipeline = getPipeline(
              "sparseElim_diagSolveLt_float");

      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer() offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                        offset:0
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer() offset:0 atIndex:2];
            [encoder setBuffer:dataBuffer offset:dataOffset atIndex:3];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:4];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];
          },
          (NSUInteger)numLumps);
    }
  }

  // LU sparse elimination forward solve: unit L (skip diagonal, just scatter)
  virtual void sparseElimSolveLUnit(const SymElimCtx& elimData, const float* data,
                                    int64_t lumpsBegin, int64_t lumpsEnd, float* C,
                                    int64_t ldc) override {
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!dataBufferInfo.first || !cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::sparseElimSolveLUnit: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;
      size_t cOffset = cBufferInfo.second;

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      int64_t nRHS64 = nRHS;

      // No diagonal solve for unit L (diagonal is implicitly 1)
      // Only encode below-diagonal scatter: v[rowSpan] -= L_below * v[lump]
      id<MTLComputePipelineState> subDiagPipeline =
          getPipeline("sparseElim_subDiagMult_float");

      encodeKernel(
          subDiagPipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer() offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer() offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                        offset:0
                       atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:0
                       atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer() offset:0 atIndex:4];
            [encoder setBuffer:dataBuffer offset:dataOffset atIndex:5];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:6];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:8];
            [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:9];
            [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:10];
          },
          (NSUInteger)numLumps);
    }
  }

  // LU sparse elimination backward solve: gather from upper triangle then U diagonal solve
  virtual void sparseElimSolveU(const SymElimCtx& elimData, const float* data, int64_t lumpsBegin,
                                int64_t lumpsEnd, float* C, int64_t ldc) override {
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!dataBufferInfo.first || !cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::sparseElimSolveU: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;
      size_t cOffset = cBufferInfo.second;

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      int64_t nRHS64 = nRHS;
      int64_t upperDataBase = sym.skel.dataSize();

      // First: gather from upper triangle entries: v[lump] -= U_row * v[colSpan]
      // Use size-1 flat kernel when all lumps are size-1
      auto gatherEncodeBlock = ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer() offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainRowPtr.buffer()
                    offset:0
                   atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainColSpan.buffer()
                    offset:0
                   atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainData.buffer()
                    offset:0
                   atIndex:4];
        [encoder setBuffer:dataBuffer offset:dataOffset atIndex:5];
        [encoder setBuffer:cBuffer offset:cOffset atIndex:6];
        [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:7];
        [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:8];
        [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:9];
        [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:10];
        [encoder setBytes:&upperDataBase length:sizeof(int64_t) atIndex:11];
      };

      bool size1 = allLumpsSize1(lumpsBegin, lumpsEnd);

      if (size1) {
        // Fused kernel: upper gather + diagonal divide in one dispatch
        id<MTLComputePipelineState> fusedPipeline =
            getPipeline("sparseElim_fusedSolveU_size1_float");
        encodeKernel(
            fusedPipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                          offset:0
                         atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                          offset:0
                         atIndex:1];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainRowPtr.buffer()
                          offset:0
                         atIndex:2];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainColSpan.buffer()
                          offset:0
                         atIndex:3];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainData.buffer()
                          offset:0
                         atIndex:4];
              [encoder setBuffer:dataBuffer offset:dataOffset atIndex:5];
              [encoder setBuffer:cBuffer offset:cOffset atIndex:6];
              [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:7];
              [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:8];
              [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:9];
              [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:10];
              [encoder setBytes:&upperDataBase length:sizeof(int64_t) atIndex:11];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                          offset:0
                         atIndex:12];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                          offset:0
                         atIndex:13];
            },
            (NSUInteger)numLumps);
      } else {
        id<MTLComputePipelineState> gatherPipeline =
            getPipeline("sparseElim_upperGather_float");
        NSUInteger gatherTgSize = MIN(gatherPipeline.maxTotalThreadsPerThreadgroup, 256);
        encodeKernelWithGroups(gatherPipeline, gatherEncodeBlock,
                               (NSUInteger)numLumps, gatherTgSize);

        // Then: diagonal U solve: v[lump] /= U_diagonal
        auto diagEncodeBlock = ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                      offset:0
                     atIndex:0];
          [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                      offset:0
                     atIndex:1];
          [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                      offset:0
                     atIndex:2];
          [encoder setBuffer:dataBuffer offset:dataOffset atIndex:3];
          [encoder setBuffer:cBuffer offset:cOffset atIndex:4];
          [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
          [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:6];
          [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
          [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];
        };

        id<MTLComputePipelineState> diagPipeline =
            getPipeline("sparseElim_diagDivU_float");
        NSUInteger tgSize = MIN(diagPipeline.maxTotalThreadsPerThreadgroup, 256);
        encodeKernelWithGroups(diagPipeline, diagEncodeBlock,
                               (NSUInteger)numLumps, tgSize);
      }
    }
  }

  virtual void symm(const float* data, int64_t offset, int64_t n, const float* C, int64_t offC,
                    int64_t ldc, float* D, int64_t ldd, float alpha) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      auto dBufferInfo = MetalBufferRegistry::instance().findBuffer(D);
      if (!dataBufferInfo.first || !cBufferInfo.first || !dBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::symm: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      id<MTLBuffer> dBuffer = (__bridge id<MTLBuffer>)dBufferInfo.first;
      size_t dataBaseOffset = dataBufferInfo.second;
      size_t cBaseOffset = cBufferInfo.second;
      size_t dBaseOffset = dBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "cholesky_symm_kernel_float");

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
            [encoder setBytes:&offset length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:2];
            [encoder setBuffer:cBuffer offset:cBaseOffset atIndex:3];
            [encoder setBytes:&offC length:sizeof(int64_t) atIndex:4];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
            [encoder setBuffer:dBuffer offset:dBaseOffset atIndex:6];
            [encoder setBytes:&ldd length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&alpha length:sizeof(float) atIndex:8];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:9];
          },
          (NSUInteger)n);
    }
  }

  virtual void solveL(const float* data, int64_t offset, int64_t n, float* C, int64_t offC,
                      int64_t ldc) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!dataBufferInfo.first || !cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::solveL: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;
      size_t cOffset = cBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "cholesky_solveL_kernel_float");

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:dataOffset atIndex:0];
            [encoder setBytes:&offset length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:2];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:3];
            [encoder setBytes:&offC length:sizeof(int64_t) atIndex:4];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:6];
          },
          1);
    }
  }

  virtual void gemv(const float* data, int64_t offset, int64_t nRows, int64_t nCols, const float* A,
                    int64_t offA, int64_t lda, float alpha) override {
    @autoreleasepool {
      if (nRows <= 0 || nCols <= 0 || nRHS <= 0) return;

      tempVecBuffer.resizeToAtLeast(nRows * nRHS);

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto aBufferInfo = MetalBufferRegistry::instance().findBuffer(A);
      if (!dataBufferInfo.first || !aBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::gemv: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> aBuffer = (__bridge id<MTLBuffer>)aBufferInfo.first;
      size_t dataBaseOffset = dataBufferInfo.second;
      size_t aBaseOffset = aBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "cholesky_gemv_kernel_float");

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
            [encoder setBytes:&offset length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&nRows length:sizeof(int64_t) atIndex:2];
            [encoder setBytes:&nCols length:sizeof(int64_t) atIndex:3];
            [encoder setBuffer:aBuffer offset:aBaseOffset atIndex:4];
            [encoder setBytes:&offA length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&lda length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&alpha length:sizeof(float) atIndex:7];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:8];
            [encoder setBuffer:(__bridge id<MTLBuffer>)tempVecBuffer.buffer() offset:0 atIndex:9];
          },
          (NSUInteger)nRows);
    }
  }

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, float* C, int64_t ldc) override {
    @autoreleasepool {
      if (numColItems <= 0) return;

      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::assembleVec: buffer not found");
      }
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t cOffset = cBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "assembleVec_kernel_float");

      // startRow = chainRowsTillEnd[chainColPtr - 1] (element before chain start)
      int64_t startRow =
          (chainColPtr > 0) ? sym.devChainRowsTillEnd.ptr()[chainColPtr - 1] : 0;

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:chainColPtr * sizeof(int64_t)
                       atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:chainColPtr * sizeof(int64_t)
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer() offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)tempVecBuffer.buffer() offset:0 atIndex:3];
            [encoder setBytes:&numColItems length:sizeof(int64_t) atIndex:4];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:5];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&startRow length:sizeof(int64_t) atIndex:8];
          },
          (NSUInteger)numColItems);
    }
  }

  virtual void solveLt(const float* data, int64_t offset, int64_t n, float* C, int64_t offC,
                       int64_t ldc) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!dataBufferInfo.first || !cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::solveLt: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;
      size_t cOffset = cBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "cholesky_solveLt_kernel_float");

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:dataOffset atIndex:0];
            [encoder setBytes:&offset length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:2];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:3];
            [encoder setBytes:&offC length:sizeof(int64_t) atIndex:4];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:6];
          },
          1);
    }
  }

  virtual void gemvT(const float* data, int64_t offset, int64_t nRows, int64_t nCols, float* A,
                     int64_t offA, int64_t lda, float alpha) override {
    @autoreleasepool {
      if (nRows <= 0 || nCols <= 0 || nRHS <= 0) return;

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto aBufferInfo = MetalBufferRegistry::instance().findBuffer(A);
      if (!dataBufferInfo.first || !aBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::gemvT: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> aBuffer = (__bridge id<MTLBuffer>)aBufferInfo.first;
      size_t dataBaseOffset = dataBufferInfo.second;
      size_t aBaseOffset = aBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "cholesky_gemvT_kernel_float");

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
            [encoder setBytes:&offset length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&nRows length:sizeof(int64_t) atIndex:2];
            [encoder setBytes:&nCols length:sizeof(int64_t) atIndex:3];
            [encoder setBuffer:aBuffer offset:aBaseOffset atIndex:4];
            [encoder setBytes:&offA length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&lda length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&alpha length:sizeof(float) atIndex:7];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:8];
            [encoder setBuffer:(__bridge id<MTLBuffer>)tempVecBuffer.buffer() offset:0 atIndex:9];
          },
          (NSUInteger)nCols);
    }
  }

  virtual void assembleVecT(const float* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    @autoreleasepool {
      if (numColItems <= 0) return;

      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::assembleVecT: buffer not found");
      }
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t cOffset = cBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "assembleVecT_kernel_float");

      // startRow = chainRowsTillEnd[chainColPtr - 1] (element before chain start)
      int64_t startRow =
          (chainColPtr > 0) ? sym.devChainRowsTillEnd.ptr()[chainColPtr - 1] : 0;

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:chainColPtr * sizeof(int64_t)
                       atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:chainColPtr * sizeof(int64_t)
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer() offset:0 atIndex:2];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:3];
            [encoder setBytes:&numColItems length:sizeof(int64_t) atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)tempVecBuffer.buffer() offset:0 atIndex:5];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&startRow length:sizeof(int64_t) atIndex:8];
          },
          (NSUInteger)numColItems);
    }
  }

  // ============ LU solve methods ============

  // Solve L * x = b where L is unit lower triangular (forward substitution)
  virtual void solveLUnit(const float* data, int64_t offM, int64_t n, float* C, int64_t offC,
                          int64_t ldc) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!dataBufferInfo.first || !cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::solveLUnit: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;
      size_t cOffset = cBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_solveLUnit_direct_kernel_float");

      // Power-of-2 threadgroup size for recursive TRSV→GEMV parallelism
      int thr = std::min((int)n, 256);
      int numThreads = 1;
      while (numThreads < thr) numThreads <<= 1;

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:dataOffset atIndex:0];
            [encoder setBytes:&offM length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:2];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:3];
            [encoder setBytes:&offC length:sizeof(int64_t) atIndex:4];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:6];
          },
          (NSUInteger)numThreads);
    }
  }

  // Solve U * x = b where U is upper triangular (backward substitution)
  virtual void solveU(const float* data, int64_t offM, int64_t n, float* C, int64_t offC,
                      int64_t ldc) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!dataBufferInfo.first || !cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::solveU: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;
      size_t cOffset = cBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_solveU_direct_kernel_float");

      // Power-of-2 threadgroup size for recursive TRSV→GEMV parallelism
      int thr = std::min((int)n, 256);
      int numThreads = 1;
      while (numThreads < thr) numThreads <<= 1;

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:dataOffset atIndex:0];
            [encoder setBytes:&offM length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:2];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:3];
            [encoder setBytes:&offC length:sizeof(int64_t) atIndex:4];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:6];
          },
          (NSUInteger)numThreads);
    }
  }

  // Pre-upload all pivots to GPU in a single memcpy.
  // Subsequent applyRowPermVec/Inv calls use offsets into this buffer.
  virtual void uploadPivots(const int64_t* pivots, int64_t totalSize) override {
    // Flush any pending GPU work that might be reading devPivots
    if (pendingEncoder_ || pendingCmdBuf_) {
      commitAndWait();
    }

    devPivots.resizeToAtLeast(totalSize);
    memcpy(devPivots.ptr(), pivots, totalSize * sizeof(int64_t));
    pivotsBase_ = pivots;
    pivotsSize_ = totalSize;
  }

  // Apply row permutation P to vector: for each i, swap row i with row pivots[i]
  virtual void applyRowPermVec(const int64_t* pivots, int64_t n, float* vec,
                                int64_t ldVec) override {
    @autoreleasepool {
      if (n <= 0) return;

      // Determine pivot buffer and offset
      id<MTLBuffer> pivotBuffer = nil;
      size_t pivotByteOffset = 0;

      if (externalDevPivots_ && pivotsBase_ &&
          pivots >= pivotsBase_ && pivots < pivotsBase_ + pivotsSize_) {
        // External device pivots: find the Metal buffer via registry
        auto pivBufInfo = MetalBufferRegistry::instance().findBuffer(externalDevPivots_);
        if (pivBufInfo.first) {
          pivotBuffer = (__bridge id<MTLBuffer>)pivBufInfo.first;
          pivotByteOffset = pivBufInfo.second + (pivots - pivotsBase_) * sizeof(int64_t);
        }
      }
      if (!pivotBuffer && pivotsBase_ &&
          pivots >= pivotsBase_ && pivots < pivotsBase_ + pivotsSize_) {
        // Pre-uploaded pivots in devPivots
        pivotBuffer = (__bridge id<MTLBuffer>)devPivots.buffer();
        pivotByteOffset = (pivots - pivotsBase_) * sizeof(int64_t);
      }
      if (!pivotBuffer) {
        // Fallback: no pre-upload, sync and copy per-call
        commitAndWait();
        devPivots.resizeToAtLeast(n);
        memcpy(devPivots.ptr(), pivots, n * sizeof(int64_t));
        pivotBuffer = (__bridge id<MTLBuffer>)devPivots.buffer();
        pivotByteOffset = 0;
      }

      auto vecBufferInfo = MetalBufferRegistry::instance().findBuffer(vec);
      if (!vecBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::applyRowPermVec: buffer not found");
      }
      id<MTLBuffer> vecBuffer = (__bridge id<MTLBuffer>)vecBufferInfo.first;
      size_t vecBaseOffset = vecBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_applyRowPermVec_kernel_float");

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:pivotBuffer
                        offset:pivotByteOffset
                       atIndex:0];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:1];
            [encoder setBuffer:vecBuffer offset:vecBaseOffset atIndex:2];
            [encoder setBytes:&ldVec length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:4];
          },
          1);
    }
  }

  // Apply inverse row permutation P^T to vector (reverse order)
  virtual void applyRowPermVecInv(const int64_t* pivots, int64_t n, float* vec,
                                   int64_t ldVec) override {
    @autoreleasepool {
      if (n <= 0) return;

      // Determine pivot buffer and offset
      id<MTLBuffer> pivotBuffer = nil;
      size_t pivotByteOffset = 0;

      if (externalDevPivots_ && pivotsBase_ &&
          pivots >= pivotsBase_ && pivots < pivotsBase_ + pivotsSize_) {
        // External device pivots: find the Metal buffer via registry
        auto pivBufInfo = MetalBufferRegistry::instance().findBuffer(externalDevPivots_);
        if (pivBufInfo.first) {
          pivotBuffer = (__bridge id<MTLBuffer>)pivBufInfo.first;
          pivotByteOffset = pivBufInfo.second + (pivots - pivotsBase_) * sizeof(int64_t);
        }
      }
      if (!pivotBuffer && pivotsBase_ &&
          pivots >= pivotsBase_ && pivots < pivotsBase_ + pivotsSize_) {
        // Pre-uploaded pivots in devPivots
        pivotBuffer = (__bridge id<MTLBuffer>)devPivots.buffer();
        pivotByteOffset = (pivots - pivotsBase_) * sizeof(int64_t);
      }
      if (!pivotBuffer) {
        // Fallback: no pre-upload, sync and copy per-call
        commitAndWait();
        devPivots.resizeToAtLeast(n);
        memcpy(devPivots.ptr(), pivots, n * sizeof(int64_t));
        pivotBuffer = (__bridge id<MTLBuffer>)devPivots.buffer();
        pivotByteOffset = 0;
      }

      auto vecBufferInfo = MetalBufferRegistry::instance().findBuffer(vec);
      if (!vecBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::applyRowPermVecInv: buffer not found");
      }
      id<MTLBuffer> vecBuffer = (__bridge id<MTLBuffer>)vecBufferInfo.first;
      size_t vecBaseOffset = vecBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_applyRowPermVecInv_kernel_float");

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:pivotBuffer
                        offset:pivotByteOffset
                       atIndex:0];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:1];
            [encoder setBuffer:vecBuffer offset:vecBaseOffset atIndex:2];
            [encoder setBytes:&ldVec length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:4];
          },
          1);
    }
  }

  // Direct gemv for U backward solve: result += alpha * M * x
  virtual void gemvDirect(const float* data, int64_t offset, int64_t nRows, int64_t nCols,
                           float* vec, int64_t srcOff, int64_t dstOff, int64_t ldVec,
                           float alpha) override {
    @autoreleasepool {
      if (nRows <= 0 || nCols <= 0 || nRHS <= 0) return;

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto vecBufferInfo = MetalBufferRegistry::instance().findBuffer(vec);
      if (!dataBufferInfo.first || !vecBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::gemvDirect: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> vecBuffer = (__bridge id<MTLBuffer>)vecBufferInfo.first;
      size_t dataBaseOffset = dataBufferInfo.second;
      size_t vecBaseOffset = vecBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_gemvDirect_kernel_float");

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
            [encoder setBytes:&offset length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&nRows length:sizeof(int64_t) atIndex:2];
            [encoder setBytes:&nCols length:sizeof(int64_t) atIndex:3];
            [encoder setBuffer:vecBuffer offset:vecBaseOffset atIndex:4];
            [encoder setBytes:&srcOff length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&dstOff length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&ldVec length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&alpha length:sizeof(float) atIndex:8];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:9];
          },
          (NSUInteger)nRows);
    }
  }

  // ============ Fused dense solve methods ============

  // Fused forward L: solveLUnit + gemv + assembleVec in one GPU dispatch.
  virtual void fusedForwardLUnit(const float* data, int64_t diagOffset, int64_t n,
                                 int64_t belowDiagOffset, int64_t numRowsBelowDiag,
                                 int64_t chainColPtr, int64_t numColItems, int64_t startRow,
                                 float* vecData, int64_t lumpStart, int64_t stride) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto vecBufferInfo = MetalBufferRegistry::instance().findBuffer(vecData);
      if (!dataBufferInfo.first || !vecBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::fusedForwardLUnit: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> vecBuffer = (__bridge id<MTLBuffer>)vecBufferInfo.first;
      size_t dataBaseOffset = dataBufferInfo.second;
      size_t vecBaseOffset = vecBufferInfo.second;

      tempVecBuffer.resizeToAtLeast(numRowsBelowDiag * nRHS);

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_fusedForwardLUnit_kernel_float");

      // Power-of-2 threadgroup size, single threadgroup
      int thr = std::max((int)n, (int)numRowsBelowDiag);
      thr = std::min(thr, 256);
      int numThreads = 1;
      while (numThreads < thr) numThreads <<= 1;

      int64_t nRHS64 = nRHS;
      encodeKernelWithGroups(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
            [encoder setBytes:&diagOffset length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:2];
            [encoder setBytes:&belowDiagOffset length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&numRowsBelowDiag length:sizeof(int64_t) atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:chainColPtr * sizeof(int64_t)
                       atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:chainColPtr * sizeof(int64_t)
                       atIndex:6];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                        offset:0
                       atIndex:7];
            [encoder setBytes:&numColItems length:sizeof(int64_t) atIndex:8];
            [encoder setBytes:&startRow length:sizeof(int64_t) atIndex:9];
            [encoder setBuffer:vecBuffer offset:vecBaseOffset atIndex:10];
            [encoder setBytes:&lumpStart length:sizeof(int64_t) atIndex:11];
            [encoder setBytes:&stride length:sizeof(int64_t) atIndex:12];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:13];
            [encoder setBuffer:(__bridge id<MTLBuffer>)tempVecBuffer.buffer()
                        offset:0
                       atIndex:14];
          },
          1,  // single threadgroup
          (NSUInteger)numThreads);
    }
  }

  // Fused backward U: iterate upper chain gemvDirect + solveU in one GPU dispatch.
  virtual void fusedBackwardU(const float* data, int64_t diagOffset, int64_t n,
                              int64_t lump, int64_t upperDataBase,
                              float* vecData, int64_t lumpStart, int64_t stride) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto vecBufferInfo = MetalBufferRegistry::instance().findBuffer(vecData);
      if (!dataBufferInfo.first || !vecBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::fusedBackwardU: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> vecBuffer = (__bridge id<MTLBuffer>)vecBufferInfo.first;
      size_t dataBaseOffset = dataBufferInfo.second;
      size_t vecBaseOffset = vecBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_fusedBackwardU_kernel_float");

      // Power-of-2 threadgroup size, single threadgroup
      int thr = std::min((int)n, 256);
      int numThreads = 1;
      while (numThreads < thr) numThreads <<= 1;

      int64_t nRHS64 = nRHS;
      encodeKernelWithGroups(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
            [encoder setBytes:&diagOffset length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:2];
            [encoder setBytes:&upperDataBase length:sizeof(int64_t) atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainRowPtr.buffer()
                        offset:0
                       atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainColSpan.buffer()
                        offset:0
                       atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainData.buffer()
                        offset:0
                       atIndex:6];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                        offset:0
                       atIndex:7];
            [encoder setBytes:&lump length:sizeof(int64_t) atIndex:8];
            [encoder setBuffer:vecBuffer offset:vecBaseOffset atIndex:9];
            [encoder setBytes:&lumpStart length:sizeof(int64_t) atIndex:10];
            [encoder setBytes:&stride length:sizeof(int64_t) atIndex:11];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:12];
          },
          1,  // single threadgroup
          (NSUInteger)numThreads);
    }
  }

  virtual bool hasFusedBackwardU() const override { return true; }

  // ============ Batched all-lumps dense solve methods ============

  virtual bool hasBatchedDenseSolve() const override { return true; }

  virtual void batchedApplyRowPermVec(float* vecData, int64_t stride,
      int64_t numLumps, const PermLumpInfo* lumpInfos) override {
    @autoreleasepool {
      if (numLumps <= 0) return;

      // Resolve pivot buffer
      id<MTLBuffer> pivotBuffer = nil;
      if (externalDevPivots_) {
        auto pivBufInfo = MetalBufferRegistry::instance().findBuffer(externalDevPivots_);
        if (pivBufInfo.first) {
          pivotBuffer = (__bridge id<MTLBuffer>)pivBufInfo.first;
        }
      }
      if (!pivotBuffer) {
        pivotBuffer = (__bridge id<MTLBuffer>)devPivots.buffer();
      }
      if (!pivotBuffer) {
        throw std::runtime_error("MetalSolveCtx::batchedApplyRowPermVec: no pivot buffer");
      }

      auto vecBufferInfo = MetalBufferRegistry::instance().findBuffer(vecData);
      if (!vecBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx::batchedApplyRowPermVec: vec buffer not found");
      }
      id<MTLBuffer> vecBuffer = (__bridge id<MTLBuffer>)vecBufferInfo.first;
      size_t vecBaseOffset = vecBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_batchedApplyRowPerm_kernel_float");

      int64_t nRHS64 = nRHS;
      int32_t numLumps32 = (int32_t)numLumps;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:pivotBuffer offset:0 atIndex:0];
            [encoder setBuffer:vecBuffer offset:vecBaseOffset atIndex:1];
            [encoder setBytes:&stride length:sizeof(int64_t) atIndex:2];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:lumpInfos length:numLumps * sizeof(PermLumpInfo) atIndex:4];
            [encoder setBytes:&numLumps32 length:sizeof(int32_t) atIndex:5];
          },
          (NSUInteger)numLumps);
    }
  }

  virtual void batchedForwardLUnit(const float* data, float* vecData, int64_t stride,
      int64_t numLumps, const ForwardLLumpInfo* lumpInfos) override {
    @autoreleasepool {
      if (numLumps <= 0) return;

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto vecBufferInfo = MetalBufferRegistry::instance().findBuffer(vecData);
      if (!dataBufferInfo.first || !vecBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx::batchedForwardLUnit: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> vecBuffer = (__bridge id<MTLBuffer>)vecBufferInfo.first;
      size_t dataBaseOffset = dataBufferInfo.second;
      size_t vecBaseOffset = vecBufferInfo.second;

      // Compute max numRowsBelowDiag for tempVec sizing
      int32_t maxRows = 0;
      for (int64_t i = 0; i < numLumps; i++) {
        if (lumpInfos[i].numRowsBelowDiag > maxRows)
          maxRows = lumpInfos[i].numRowsBelowDiag;
      }
      tempVecBuffer.resizeToAtLeast(maxRows * nRHS);

      // Compute threadgroup size: max of all lump sizes and numRowsBelowDiag
      int maxThr = 0;
      for (int64_t i = 0; i < numLumps; i++) {
        maxThr = std::max(maxThr, (int)lumpInfos[i].lumpSize);
        maxThr = std::max(maxThr, (int)lumpInfos[i].numRowsBelowDiag);
      }
      maxThr = std::min(maxThr, 256);
      int numThreads = 1;
      while (numThreads < maxThr) numThreads <<= 1;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_allLumpsForwardL_kernel_float");

      int64_t nRHS64 = nRHS;
      int32_t numLumps32 = (int32_t)numLumps;
      encodeKernelWithGroups(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
            [encoder setBuffer:vecBuffer offset:vecBaseOffset atIndex:1];
            [encoder setBytes:&stride length:sizeof(int64_t) atIndex:2];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:0 atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:0 atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                        offset:0 atIndex:6];
            [encoder setBuffer:(__bridge id<MTLBuffer>)tempVecBuffer.buffer()
                        offset:0 atIndex:7];
            [encoder setBytes:lumpInfos length:numLumps * sizeof(ForwardLLumpInfo) atIndex:8];
            [encoder setBytes:&numLumps32 length:sizeof(int32_t) atIndex:9];
          },
          1,  // single threadgroup
          (NSUInteger)numThreads);
    }
  }

  virtual void batchedBackwardU(const float* data, float* vecData, int64_t stride,
      int64_t numLumps, const BackwardULumpInfo* lumpInfos) override {
    @autoreleasepool {
      if (numLumps <= 0) return;

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto vecBufferInfo = MetalBufferRegistry::instance().findBuffer(vecData);
      if (!dataBufferInfo.first || !vecBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx::batchedBackwardU: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> vecBuffer = (__bridge id<MTLBuffer>)vecBufferInfo.first;
      size_t dataBaseOffset = dataBufferInfo.second;
      size_t vecBaseOffset = vecBufferInfo.second;

      // Compute threadgroup size: max of all lump sizes
      int maxThr = 0;
      for (int64_t i = 0; i < numLumps; i++) {
        maxThr = std::max(maxThr, (int)lumpInfos[i].lumpSize);
      }
      maxThr = std::min(maxThr, 256);
      int numThreads = 1;
      while (numThreads < maxThr) numThreads <<= 1;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_allLumpsBackwardU_kernel_float");

      int64_t nRHS64 = nRHS;
      int64_t upperDataBase = sym.skel.dataSize();
      int32_t numLumps32 = (int32_t)numLumps;
      encodeKernelWithGroups(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
            [encoder setBuffer:vecBuffer offset:vecBaseOffset atIndex:1];
            [encoder setBytes:&stride length:sizeof(int64_t) atIndex:2];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainRowPtr.buffer()
                        offset:0 atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainColSpan.buffer()
                        offset:0 atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainData.buffer()
                        offset:0 atIndex:6];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                        offset:0 atIndex:7];
            [encoder setBytes:&upperDataBase length:sizeof(int64_t) atIndex:8];
            [encoder setBytes:lumpInfos length:numLumps * sizeof(BackwardULumpInfo) atIndex:9];
            [encoder setBytes:&numLumps32 length:sizeof(int32_t) atIndex:10];
          },
          1,  // single threadgroup
          (NSUInteger)numThreads);
    }
  }

  void skipNextBarrier() override { skipNextBarrier_ = true; }

  virtual void fusedDenseSolveLU(const float* data, float* vecData, int64_t stride,
      int64_t numLumps, const ForwardLLumpInfo* fwdInfos,
      const BackwardULumpInfo* bwdInfos) override {
    @autoreleasepool {
      if (numLumps <= 0) return;

      // CPU path: cycle external encoder, run CPU triangular solves + gemv.
      // ~10-50us on CPU vs ~1.12ms on GPU (single threadgroup).
      {
        if (sym.usingExternalEncoder) {
          [sym.externalEncoder endEncoding];
          sym.externalEncoder = nil;
          [sym.externalCmdBuf commit];
          [sym.externalCmdBuf waitUntilCompleted];
          sym.externalCmdBuf = nil;
        } else {
          commitPending();
          waitForGpu();
        }

        const int64_t* chainRowsTillEnd = sym.skel.chainRowsTillEnd.data();
        const int64_t* chainRowSpan = sym.skel.chainRowSpan.data();
        const int64_t* spanStarts = sym.skel.spanStart.data();
        const int64_t* upperChainRowPtr = sym.skel.upperChainRowPtr.data();
        const int64_t* upperChainColSpan = sym.skel.upperChainColSpan.data();
        const int64_t* upperChainData = sym.skel.upperChainData.data();
        int64_t upperDataBase = sym.skel.dataSize();

        // Temp buffer for below-diagonal gemv results
        int32_t maxRows = 0;
        for (int64_t i = 0; i < numLumps; i++)
          maxRows = std::max(maxRows, fwdInfos[i].numRowsBelowDiag);
        std::vector<float> tempVec(maxRows * nRHS);

        // Cast away const for unified memory pointer arithmetic
        float* dataW = const_cast<float*>(data);
        float* vecW = vecData;

        // === Forward L phase ===
        for (int32_t li = 0; li < numLumps; li++) {
          const auto& info = fwdInfos[li];
          int64_t n = info.lumpSize;
          const float* L = data + info.diagOffset;
          float* x = vecW + info.lumpStart;

          // Unit lower triangular solve (row-major L)
          for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            float* xr = x + rhs * stride;
            for (int64_t i = 0; i < n; i++)
              for (int64_t j = i + 1; j < n; j++)
                xr[j] -= L[j * n + i] * xr[i];
          }

          if (info.numRowsBelowDiag <= 0) continue;

          // Gemv: tempVec = -M_below * x_diag
          const float* M = data + info.belowDiagOffset;
          int64_t rows = info.numRowsBelowDiag;
          for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            const float* xr = x + rhs * stride;
            float* tv = tempVec.data() + rhs * rows;
            for (int64_t r = 0; r < rows; r++) {
              float sum = 0.0f;
              for (int64_t c = 0; c < n; c++)
                sum += M[r * n + c] * xr[c];
              tv[r] = -sum;
            }
          }

          // Scatter tempVec → vecData using chain structure
          int64_t chainBase = info.chainColPtr;
          int64_t row = info.startRow;
          for (int32_t ci = 0; ci < info.numColItems; ci++) {
            int64_t rowEnd = chainRowsTillEnd[chainBase + ci];
            int64_t span = chainRowSpan[chainBase + ci];
            int64_t spanOff = spanStarts[span];
            for (; row < rowEnd; row++) {
              for (int64_t rhs = 0; rhs < nRHS; rhs++)
                vecW[spanOff + rhs * stride] += tempVec[rhs * rows + (row - info.startRow)];
              spanOff++;
            }
          }
        }

        // === Backward U phase ===
        for (int32_t li = numLumps - 1; li >= 0; li--) {
          const auto& bInfo = bwdInfos[li];
          int64_t n = bInfo.lumpSize;
          int32_t lump = bInfo.lumpIndex;

          // Gather: y -= U * x for each upper chain entry
          int64_t upperRowStart = upperChainRowPtr[lump];
          int64_t upperRowEnd = upperChainRowPtr[lump + 1];
          for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
            int64_t colSpan = upperChainColSpan[i];
            int64_t colStart = spanStarts[colSpan];
            int64_t colSize = spanStarts[colSpan + 1] - colStart;
            const float* U = data + upperDataBase + upperChainData[i];
            for (int64_t rhs = 0; rhs < nRHS; rhs++) {
              float* yr = vecW + bInfo.lumpStart + rhs * stride;
              const float* xr = vecW + colStart + rhs * stride;
              for (int64_t r = 0; r < n; r++) {
                float sum = 0.0f;
                for (int64_t c = 0; c < colSize; c++)
                  sum += U[r * colSize + c] * xr[c];
                yr[r] -= sum;
              }
            }
          }

          // Upper triangular solve (row-major U)
          const float* Udiag = data + bInfo.diagOffset;
          for (int64_t rhs = 0; rhs < nRHS; rhs++) {
            float* xr = vecW + bInfo.lumpStart + rhs * stride;
            for (int64_t i = n - 1; i >= 0; i--) {
              for (int64_t j = i + 1; j < n; j++)
                xr[i] -= Udiag[i * n + j] * xr[j];
              xr[i] /= Udiag[i * n + i];
            }
          }
        }

        // Re-create external encoder for subsequent GPU dispatches
        if (sym.usingExternalEncoder) {
          sym.externalCmdBuf = [sym.commandQueue commandBuffer];
          sym.externalEncoder = [sym.externalCmdBuf computeCommandEncoder];
        }
        return;
      }

      // GPU path (unreachable with CPU path above, kept for reference)
      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto vecBufferInfo = MetalBufferRegistry::instance().findBuffer(vecData);
      if (!dataBufferInfo.first || !vecBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx::fusedDenseSolveLU: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> vecBuffer = (__bridge id<MTLBuffer>)vecBufferInfo.first;
      size_t dataBaseOffset = dataBufferInfo.second;
      size_t vecBaseOffset = vecBufferInfo.second;

      // Compute max numRowsBelowDiag for tempVec sizing
      int32_t maxRows = 0;
      for (int64_t i = 0; i < numLumps; i++) {
        if (fwdInfos[i].numRowsBelowDiag > maxRows)
          maxRows = fwdInfos[i].numRowsBelowDiag;
      }
      tempVecBuffer.resizeToAtLeast(maxRows * nRHS);

      // Threadgroup size: max of all lump sizes and numRowsBelowDiag (from forward L)
      int maxThr = 0;
      for (int64_t i = 0; i < numLumps; i++) {
        maxThr = std::max(maxThr, (int)fwdInfos[i].lumpSize);
        maxThr = std::max(maxThr, (int)fwdInfos[i].numRowsBelowDiag);
      }
      maxThr = std::min(maxThr, 256);
      int numThreads = 1;
      while (numThreads < maxThr) numThreads <<= 1;

      id<MTLComputePipelineState> pipeline = getPipeline(
              "lu_fusedDenseSolve_kernel_float");

      int64_t nRHS64 = nRHS;
      int64_t upperDataBase = sym.skel.dataSize();
      int32_t numLumps32 = (int32_t)numLumps;
      encodeKernelWithGroups(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
            [encoder setBuffer:vecBuffer offset:vecBaseOffset atIndex:1];
            [encoder setBytes:&stride length:sizeof(int64_t) atIndex:2];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:3];
            // Forward L buffers
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:0 atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:0 atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                        offset:0 atIndex:6];
            [encoder setBuffer:(__bridge id<MTLBuffer>)tempVecBuffer.buffer()
                        offset:0 atIndex:7];
            [encoder setBytes:fwdInfos length:numLumps * sizeof(ForwardLLumpInfo) atIndex:8];
            // Backward U buffers
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainRowPtr.buffer()
                        offset:0 atIndex:9];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainColSpan.buffer()
                        offset:0 atIndex:10];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devUpperChainData.buffer()
                        offset:0 atIndex:11];
            [encoder setBytes:&upperDataBase length:sizeof(int64_t) atIndex:12];
            [encoder setBytes:bwdInfos length:numLumps * sizeof(BackwardULumpInfo) atIndex:13];
            // Shared
            [encoder setBytes:&numLumps32 length:sizeof(int32_t) atIndex:14];
          },
          1,  // single threadgroup
          (NSUInteger)numThreads);
    }
  }

  void flush() override { commitAndWait(); }

  // Reset per-solve mutable state without deallocating any buffers.
  // Allows reusing this context across multiple solveLU calls.
  void reset() override {
    pivotsBase_ = nullptr;
    pivotsSize_ = 0;
    externalDevPivots_ = nullptr;
    // devPivots, tempVecBuffer are NOT freed — reused across calls.
  }

  // Point solve context at device-resident pivots (no H2D upload needed).
  // Used when factorLU wrote pivots directly to device via flushDevicePivots.
  // pivotsBase_ is set to devPivots so that offset computation works:
  //   solveLU passes (devPivots + lumpStart[l]) to applyRowPermVec,
  //   which computes offset = (devPivots + lumpStart[l]) - pivotsBase_ = lumpStart[l].
  void useDevicePivots(const int64_t* devPivots_ext, int64_t totalSize) override {
    (void)totalSize;
    externalDevPivots_ = devPivots_ext;
    pivotsBase_ = devPivots_ext;  // used for offset computation only
    pivotsSize_ = totalSize;
  }

  MetalSymbolicCtx& sym;
  int nRHS;
  MetalMirror<float> tempVecBuffer;
  MetalMirror<int64_t> devPivots;  // GPU buffer for LU pivots
  const int64_t* pivotsBase_ = nullptr;  // Base pointer of pre-uploaded pivots
  int64_t pivotsSize_ = 0;               // Size of pre-uploaded pivot buffer
  const int64_t* externalDevPivots_ = nullptr;  // External device pivots (not owned)

  // Deferred sync state — batch multiple GPU dispatches into shared command buffers
  id<MTLCommandBuffer> pendingCmdBuf_ = nil;
  id<MTLComputeCommandEncoder> pendingEncoder_ = nil;
  int pendingDispatchCount_ = 0;
  bool skipNextBarrier_ = false;
  id<MTLCommandBuffer> lastCommittedCmdBuf_ = nil;
};

// Batched numeric context for float
template <>
struct MetalNumericCtx<std::vector<float*>> : NumericCtx<std::vector<float*>> {
  MetalNumericCtx(MetalSymbolicCtx& sym_, int64_t tempBufSize, int64_t numSpans, int batchSize_)
      : sym(sym_),
        numSpans_(numSpans),
        batchSize(batchSize_),
        tempBufSizePerBatch(tempBufSize),
        spanToChainOffset(numSpans),
        tempBufPtrs(batchSize_) {
    tempBuffer.resizeToAtLeast(tempBufSize * batchSize);
    for (int i = 0; i < batchSize; i++) {
      tempBufPtrs[i] = tempBuffer.ptr() + tempBufSize * i;
    }
    devSpanToChainOffset.resizeToAtLeast(numSpans);
  }

  virtual ~MetalNumericCtx() override {}

  virtual void pseudoFactorSpans(std::vector<float*>* data, int64_t spanBegin,
                                 int64_t spanEnd) override {
    throw std::runtime_error(
        "MetalNumericCtx<std::vector<float*>>::pseudoFactorSpans not supported in batched mode");
  }

  virtual void doElimination(const SymElimCtx& elimData, std::vector<float*>* data,
                             int64_t lumpsBegin, int64_t lumpsEnd) override {
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);
      const MetalSymElimCtx& elim = *pElim;

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      NSUInteger threadGroupSize;
      MTLSize threadsPerGroup, numGroups;

      // Step 1: Factor lumps - loop dispatch of non-batched kernel
      {
        id<MTLComputePipelineState> pipeline =
            getPipeline("factor_lumps_kernel_float");
        id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];

        // Set structural buffers (same for all batch items)
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                    offset:0
                   atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                    offset:0
                   atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                    offset:0
                   atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardColPtr.buffer()
                    offset:0
                   atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardChainColOrd.buffer()
                    offset:0
                   atIndex:4];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                    offset:0
                   atIndex:5];
        [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
        [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];

        threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        threadGroupSize = MIN(threadGroupSize, (NSUInteger)numLumps);
        threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        numGroups = MTLSizeMake(
            ((NSUInteger)numLumps + threadGroupSize - 1) / threadGroupSize, 1, 1);

        // Dispatch once per batch item, changing only the data buffer
        for (int b = 0; b < batchSize; b++) {
          auto bufferInfo = MetalBufferRegistry::instance().findBuffer((*data)[b]);
          BASPACHO_CHECK_WHAT1(bufferInfo.first,
                               "batched doElimination: data buffer not found");
          [encoder setBuffer:(__bridge id<MTLBuffer>)bufferInfo.first
                      offset:bufferInfo.second
                     atIndex:6];
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        }

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
      }

      // Step 2: Sparse elimination — two-phase deterministic
      if (elim.numCholWorkItems > 0 && elim.numCholSegments > 0) {
        elimScratchBuffer_.resizeToAtLeast(elim.numCholWorkItems);

        id<MTLComputePipelineState> p1Pipeline =
            getPipeline("chol_sparse_elim_phase1_float");
        id<MTLComputePipelineState> p2Pipeline =
            getPipeline("sparse_elim_phase2_float");

        id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        // Phase 1 + Phase 2 per batch item (scratch buffer reused between items)
        for (int b = 0; b < batchSize; b++) {
          auto bufferInfo = MetalBufferRegistry::instance().findBuffer((*data)[b]);
          BASPACHO_CHECK_WHAT1(bufferInfo.first,
                               "batched doElimination: data buffer not found");

          if (b > 0) {
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
          }

          // Phase 1: compute dot products into scratch
          [encoder setComputePipelineState:p1Pipeline];
          [encoder setBuffer:(__bridge id<MTLBuffer>)bufferInfo.first
                      offset:bufferInfo.second
                     atIndex:0];
          [encoder setBuffer:(__bridge id<MTLBuffer>)elim.devCholWorkItems.buffer()
                      offset:0
                     atIndex:1];
          [encoder setBuffer:(__bridge id<MTLBuffer>)elimScratchBuffer_.buffer()
                      offset:0
                     atIndex:2];
          [encoder setBytes:&elim.numCholWorkItems length:sizeof(int64_t) atIndex:3];

          threadGroupSize = MIN(p1Pipeline.maxTotalThreadsPerThreadgroup, 256);
          threadGroupSize = MIN(threadGroupSize, (NSUInteger)elim.numCholWorkItems);
          threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
          numGroups = MTLSizeMake(
              ((NSUInteger)elim.numCholWorkItems + threadGroupSize - 1) / threadGroupSize, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];

          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

          // Phase 2: deterministic segmented sum
          [encoder setComputePipelineState:p2Pipeline];
          [encoder setBuffer:(__bridge id<MTLBuffer>)bufferInfo.first
                      offset:bufferInfo.second
                     atIndex:0];
          [encoder setBuffer:(__bridge id<MTLBuffer>)elimScratchBuffer_.buffer()
                      offset:0
                     atIndex:1];
          [encoder setBuffer:(__bridge id<MTLBuffer>)elim.devCholSegments.buffer()
                      offset:0
                     atIndex:2];
          [encoder setBytes:&elim.numCholSegments length:sizeof(int64_t) atIndex:3];

          threadGroupSize = MIN(p2Pipeline.maxTotalThreadsPerThreadgroup, 256);
          threadGroupSize = MIN(threadGroupSize, (NSUInteger)elim.numCholSegments);
          threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
          numGroups = MTLSizeMake(
              ((NSUInteger)elim.numCholSegments + threadGroupSize - 1) / threadGroupSize, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        }

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
      }
    }
  }

  virtual void potrf(int64_t n, std::vector<float*>* data, int64_t offA) override {
    @autoreleasepool {
      if (n <= 0) return;

      // MPS Cholesky on GPU for all sizes — encode all batch items into one command buffer
      id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];
      NSMutableArray<id<MTLBuffer>>* statusBufs =
          [NSMutableArray arrayWithCapacity:batchSize];

      for (int b = 0; b < batchSize; b++) {
        auto bufferInfo = MetalBufferRegistry::instance().findBuffer((*data)[b]);
        BASPACHO_CHECK_WHAT1(bufferInfo.first, "batched potrf: data buffer not registered");
        id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
        size_t baseOffset = bufferInfo.second;

        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                                           columns:n
                                                                          rowBytes:n * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrix* mpsA =
            [[MPSMatrix alloc] initWithBuffer:dataBuffer
                                       offset:baseOffset + offA * sizeof(float)
                                   descriptor:descA];

        id<MTLBuffer> statusBuf = [sym.device
            newBufferWithLength:sizeof(MPSMatrixDecompositionStatus)
                       options:MTLResourceStorageModeShared];
        [statusBufs addObject:statusBuf];

        MPSMatrixDecompositionCholesky* mpsChol =
            [[MPSMatrixDecompositionCholesky alloc] initWithDevice:sym.device lower:YES order:n];
        [mpsChol encodeToCommandBuffer:cmdBuf
                          sourceMatrix:mpsA
                          resultMatrix:mpsA
                                status:statusBuf];
      }

      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];

      for (int b = 0; b < batchSize; b++) {
        auto status = *reinterpret_cast<MPSMatrixDecompositionStatus*>(
            [statusBufs[b] contents]);
        if (status != MPSMatrixDecompositionStatusSuccess) {
          fprintf(stderr,
                  "Metal batched potrf: MPS Cholesky failed (batch %d, status=%d, n=%lld)\n",
                  b, (int)status, (long long)n);
        }
      }
    }
  }

  virtual void trsm(int64_t n, int64_t k, std::vector<float*>* data, int64_t offA,
                    int64_t offB) override {
    @autoreleasepool {
      if (n <= 0 || k <= 0) return;

      // MPS triangular solve on GPU for all sizes — encode all batch items
      id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];

      for (int b = 0; b < batchSize; b++) {
        auto bufferInfo = MetalBufferRegistry::instance().findBuffer((*data)[b]);
        BASPACHO_CHECK_WHAT1(bufferInfo.first, "batched trsm: data buffer not registered");
        id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
        size_t baseOffset = bufferInfo.second;

        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                                           columns:n
                                                                          rowBytes:n * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrix* mpsA =
            [[MPSMatrix alloc] initWithBuffer:dataBuffer
                                       offset:baseOffset + offA * sizeof(float)
                                   descriptor:descA];

        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:k
                                                                           columns:n
                                                                          rowBytes:n * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrix* mpsB =
            [[MPSMatrix alloc] initWithBuffer:dataBuffer
                                       offset:baseOffset + offB * sizeof(float)
                                   descriptor:descB];

        MPSMatrixSolveTriangular* solve =
            [[MPSMatrixSolveTriangular alloc] initWithDevice:sym.device
                                                       right:YES
                                                       upper:NO
                                                   transpose:YES
                                                        unit:NO
                                                       order:n
                                      numberOfRightHandSides:k
                                                       alpha:1.0];
        [solve encodeToCommandBuffer:cmdBuf
                      sourceMatrix:mpsA
               rightHandSideMatrix:mpsB
                    solutionMatrix:mpsB];
      }

      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }
  }

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const std::vector<float*>* data,
                            int64_t offset) override {
    @autoreleasepool {
      if (m <= 0 || n <= 0 || k <= 0) return;

      // MPS GEMM on GPU for all sizes
      {
        id<MTLBuffer> tempBuf = (__bridge id<MTLBuffer>)tempBuffer.buffer();
        id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];

        for (int b = 0; b < batchSize; b++) {
          auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer((*data)[b]);
          BASPACHO_CHECK_WHAT1(dataBufferInfo.first,
                               "batched saveSyrkGemm: data buffer not found");
          id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
          size_t dataBaseOffset = dataBufferInfo.second;

          MPSMatrixDescriptor* descA =
              [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                   columns:k
                                                  rowBytes:k * sizeof(float)
                                                  dataType:MPSDataTypeFloat32];
          MPSMatrixDescriptor* descB =
              [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                   columns:k
                                                  rowBytes:k * sizeof(float)
                                                  dataType:MPSDataTypeFloat32];
          MPSMatrixDescriptor* descC =
              [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                   columns:m
                                                  rowBytes:m * sizeof(float)
                                                  dataType:MPSDataTypeFloat32];

          MPSMatrix* mpsA =
              [[MPSMatrix alloc] initWithBuffer:dataBuffer
                                         offset:dataBaseOffset + offset * sizeof(float)
                                     descriptor:descA];
          MPSMatrix* mpsB =
              [[MPSMatrix alloc] initWithBuffer:dataBuffer
                                         offset:dataBaseOffset + offset * sizeof(float)
                                     descriptor:descB];

          size_t tempBufOffset = b * tempBufSizePerBatch * sizeof(float);
          MPSMatrix* mpsC = [[MPSMatrix alloc] initWithBuffer:tempBuf
                                                       offset:tempBufOffset
                                                   descriptor:descC];

          MPSMatrixMultiplication* gemm =
              [[MPSMatrixMultiplication alloc] initWithDevice:sym.device
                                               transposeLeft:NO
                                              transposeRight:YES
                                                  resultRows:n
                                               resultColumns:m
                                             interiorColumns:k
                                                       alpha:1.0
                                                        beta:0.0];
          [gemm encodeToCommandBuffer:cmdBuf leftMatrix:mpsB rightMatrix:mpsA resultMatrix:mpsC];
        }

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
      }
    }
  }

  virtual void prepareAssemble(int64_t targetLump) override {
    if (assembleWasCalled_ && gpuAssemblyUsed_) {
      MetalContext::instance().synchronize();
    }
    assembleWasCalled_ = false;
    gpuAssemblyUsed_ = false;

    const CoalescedBlockMatrixSkel& skel = sym.skel;
    for (int64_t i = skel.chainColPtr[targetLump], iEnd = skel.chainColPtr[targetLump + 1];
         i < iEnd; i++) {
      spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
    }
    // Only copy to GPU buffer if GPU assembly might be used
    // (deferred: done in assemble() GPU path if needed)
  }

  virtual void assemble(std::vector<float*>* data, int64_t rectRowBegin, int64_t dstStride,
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    @autoreleasepool {
      if (numBlockRows <= 0 || numBlockCols <= 0) return;
      assembleWasCalled_ = true;

      // GPU assembly kernel for all sizes.
      // Upload spanToChainOffset if not yet done
      if (!gpuAssemblyUsed_) {
        memcpy(devSpanToChainOffset.ptr(), spanToChainOffset.data(),
               spanToChainOffset.size() * sizeof(int64_t));
        gpuAssemblyUsed_ = true;
      }

      id<MTLComputePipelineState> pipeline = getPipeline("assemble_kernel_float");

      int64_t numThreads = numBlockRows * numBlockCols;
      int64_t startRow =
          (srcColDataOffset > 0) ? sym.skel.chainRowsTillEnd[srcColDataOffset - 1] : 0;

      id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];

      // Set parameters that don't change per batch item
      [encoder setBytes:&numBlockRows length:sizeof(int64_t) atIndex:0];
      [encoder setBytes:&numBlockCols length:sizeof(int64_t) atIndex:1];
      [encoder setBytes:&startRow length:sizeof(int64_t) atIndex:2];
      [encoder setBytes:&srcRectWidth length:sizeof(int64_t) atIndex:3];
      [encoder setBytes:&dstStride length:sizeof(int64_t) atIndex:4];
      [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                  offset:srcColDataOffset * sizeof(int64_t)
                 atIndex:5];
      [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                  offset:srcColDataOffset * sizeof(int64_t)
                 atIndex:6];
      [encoder setBuffer:(__bridge id<MTLBuffer>)devSpanToChainOffset.buffer()
                  offset:0
                 atIndex:7];
      [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanOffsetInLump.buffer()
                  offset:0
                 atIndex:8];

      NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
      threadGroupSize = MIN(threadGroupSize, (NSUInteger)numThreads);
      MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
      MTLSize numGroups = MTLSizeMake(
          ((NSUInteger)numThreads + threadGroupSize - 1) / threadGroupSize, 1, 1);

      id<MTLBuffer> tempBuf = (__bridge id<MTLBuffer>)tempBuffer.buffer();

      // Dispatch once per batch item, changing temp buffer offset and data buffer
      for (int b = 0; b < batchSize; b++) {
        auto bufferInfo = MetalBufferRegistry::instance().findBuffer((*data)[b]);
        BASPACHO_CHECK_WHAT1(bufferInfo.first, "batched assemble: data buffer not found");

        size_t tempOffset = b * tempBufSizePerBatch * sizeof(float);
        [encoder setBuffer:tempBuf offset:tempOffset atIndex:9];
        [encoder setBuffer:(__bridge id<MTLBuffer>)bufferInfo.first
                    offset:bufferInfo.second
                   atIndex:10];
        [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
      }

      [encoder endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }
  }

  void flush() override { MetalContext::instance().synchronize(); }

  MetalSymbolicCtx& sym;
  int64_t numSpans_;
  int batchSize;
  int64_t tempBufSizePerBatch;
  MetalMirror<float> tempBuffer;
  std::vector<float*> tempBufPtrs;
  MetalMirror<int64_t> devSpanToChainOffset;
  std::vector<int64_t> spanToChainOffset;
  bool assembleWasCalled_ = false;
  bool gpuAssemblyUsed_ = false;

  // Scratch buffer for two-phase deterministic sparse elimination
  MetalMirror<float> elimScratchBuffer_;
};

// Batched solve context for float
template <>
struct MetalSolveCtx<std::vector<float*>> : SolveCtx<std::vector<float*>> {
  MetalSolveCtx(MetalSymbolicCtx& sym_, int nRHS_, int batchSize_)
      : sym(sym_), nRHS(nRHS_), batchSize(batchSize_), solveBufPtrs(batchSize_) {
    int64_t solveBufSize = sym_.skel.order() * nRHS_;
    allJoinedSolveBufs.resizeToAtLeast(batchSize_ * solveBufSize);
    for (int i = 0; i < batchSize_; i++) {
      solveBufPtrs[i] = allJoinedSolveBufs.ptr() + i * solveBufSize;
    }
  }

  virtual ~MetalSolveCtx() override {}

  virtual void sparseElimSolveL(const SymElimCtx& elimData, const std::vector<float*>* data,
                                int64_t lumpsBegin, int64_t lumpsEnd, std::vector<float*>* C,
                                int64_t ldc) override {
    @autoreleasepool {
      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      int64_t nRHS64 = nRHS;
      NSUInteger threadGroupSize;
      MTLSize threadsPerGroup, numGroups;

      // Step 1: Diagonal solve L - loop dispatch of non-batched kernel
      {
        id<MTLComputePipelineState> pipeline =
            getPipeline("sparseElim_diagSolveL_float");
        id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                    offset:0
                   atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                    offset:0
                   atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                    offset:0
                   atIndex:2];
        [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
        [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:6];
        [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
        [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];

        threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        threadGroupSize = MIN(threadGroupSize, (NSUInteger)numLumps);
        threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        numGroups = MTLSizeMake(
            ((NSUInteger)numLumps + threadGroupSize - 1) / threadGroupSize, 1, 1);

        for (int b = 0; b < batchSize; b++) {
          auto dataInfo = MetalBufferRegistry::instance().findBuffer((*data)[b]);
          auto cInfo = MetalBufferRegistry::instance().findBuffer((*C)[b]);
          BASPACHO_CHECK_WHAT1(dataInfo.first,
                               "batched sparseElimSolveL: data buffer not found");
          BASPACHO_CHECK_WHAT1(cInfo.first,
                               "batched sparseElimSolveL: C buffer not found");

          [encoder setBuffer:(__bridge id<MTLBuffer>)dataInfo.first
                      offset:dataInfo.second
                     atIndex:3];
          [encoder setBuffer:(__bridge id<MTLBuffer>)cInfo.first
                      offset:cInfo.second
                     atIndex:4];
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        }

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
      }

      // Step 2: Sub-diagonal multiply - loop dispatch of non-batched kernel
      {
        id<MTLComputePipelineState> pipeline =
            getPipeline("sparseElim_subDiagMult_float");
        id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                    offset:0
                   atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                    offset:0
                   atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                    offset:0
                   atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                    offset:0
                   atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                    offset:0
                   atIndex:4];
        [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:7];
        [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:8];
        [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:9];
        [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:10];

        threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        threadGroupSize = MIN(threadGroupSize, (NSUInteger)numLumps);
        threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        numGroups = MTLSizeMake(
            ((NSUInteger)numLumps + threadGroupSize - 1) / threadGroupSize, 1, 1);

        for (int b = 0; b < batchSize; b++) {
          auto dataInfo = MetalBufferRegistry::instance().findBuffer((*data)[b]);
          auto cInfo = MetalBufferRegistry::instance().findBuffer((*C)[b]);
          BASPACHO_CHECK_WHAT1(dataInfo.first,
                               "batched sparseElimSolveL: data buffer not found");
          BASPACHO_CHECK_WHAT1(cInfo.first,
                               "batched sparseElimSolveL: C buffer not found");

          [encoder setBuffer:(__bridge id<MTLBuffer>)dataInfo.first
                      offset:dataInfo.second
                     atIndex:5];
          [encoder setBuffer:(__bridge id<MTLBuffer>)cInfo.first
                      offset:cInfo.second
                     atIndex:6];
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        }

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
      }
    }
  }

  virtual void sparseElimSolveLt(const SymElimCtx& elimData, const std::vector<float*>* data,
                                 int64_t lumpsBegin, int64_t lumpsEnd, std::vector<float*>* C,
                                 int64_t ldc) override {
    @autoreleasepool {
      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      int64_t nRHS64 = nRHS;
      NSUInteger threadGroupSize;
      MTLSize threadsPerGroup, numGroups;

      // Step 1: Sub-diagonal transpose multiply - loop dispatch of non-batched kernel
      // Use size-1 flat kernel when all lumps are size-1
      bool size1 = sym.skel.lumpStart[lumpsEnd] - sym.skel.lumpStart[lumpsBegin]
                   == (lumpsEnd - lumpsBegin);
      {
        id<MTLComputePipelineState> pipeline = size1
            ? getPipeline("sparseElim_subDiagMultT_size1_float")
            : getPipeline("sparseElim_subDiagMultT_float");
        id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                    offset:0
                   atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                    offset:0
                   atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                    offset:0
                   atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                    offset:0
                   atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                    offset:0
                   atIndex:4];
        [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:7];
        [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:8];
        [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:9];
        [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:10];

        if (size1) {
          // Flat dispatch: 1 thread per lump
          threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
          threadGroupSize = MIN(threadGroupSize, (NSUInteger)numLumps);
          threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
          numGroups = MTLSizeMake(
              ((NSUInteger)numLumps + threadGroupSize - 1) / threadGroupSize, 1, 1);
        } else {
          // Threadgroup dispatch: 1 threadgroup per lump
          threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
          threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
          numGroups = MTLSizeMake((NSUInteger)numLumps, 1, 1);
        }

        for (int b = 0; b < batchSize; b++) {
          auto dataInfo = MetalBufferRegistry::instance().findBuffer((*data)[b]);
          auto cInfo = MetalBufferRegistry::instance().findBuffer((*C)[b]);
          BASPACHO_CHECK_WHAT1(dataInfo.first,
                               "batched sparseElimSolveLt: data buffer not found");
          BASPACHO_CHECK_WHAT1(cInfo.first,
                               "batched sparseElimSolveLt: C buffer not found");

          [encoder setBuffer:(__bridge id<MTLBuffer>)dataInfo.first
                      offset:dataInfo.second
                     atIndex:5];
          [encoder setBuffer:(__bridge id<MTLBuffer>)cInfo.first
                      offset:cInfo.second
                     atIndex:6];
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        }

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
      }

      // Step 2: Diagonal solve Lt - loop dispatch of non-batched kernel
      {
        id<MTLComputePipelineState> pipeline =
            getPipeline("sparseElim_diagSolveLt_float");
        id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                    offset:0
                   atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                    offset:0
                   atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                    offset:0
                   atIndex:2];
        [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
        [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:6];
        [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
        [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];

        threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        threadGroupSize = MIN(threadGroupSize, (NSUInteger)numLumps);
        threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        numGroups = MTLSizeMake(
            ((NSUInteger)numLumps + threadGroupSize - 1) / threadGroupSize, 1, 1);

        for (int b = 0; b < batchSize; b++) {
          auto dataInfo = MetalBufferRegistry::instance().findBuffer((*data)[b]);
          auto cInfo = MetalBufferRegistry::instance().findBuffer((*C)[b]);
          BASPACHO_CHECK_WHAT1(dataInfo.first,
                               "batched sparseElimSolveLt: data buffer not found");
          BASPACHO_CHECK_WHAT1(cInfo.first,
                               "batched sparseElimSolveLt: C buffer not found");

          [encoder setBuffer:(__bridge id<MTLBuffer>)dataInfo.first
                      offset:dataInfo.second
                     atIndex:3];
          [encoder setBuffer:(__bridge id<MTLBuffer>)cInfo.first
                      offset:cInfo.second
                     atIndex:4];
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        }

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
      }
    }
  }

  virtual void symm(const std::vector<float*>* data, int64_t offset, int64_t n,
                    const std::vector<float*>* C, int64_t offC, int64_t ldc, std::vector<float*>* D,
                    int64_t ldd, float alpha) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;

      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatK =
          Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;
      using OuterStridedCMajMatM =
          Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      for (int b = 0; b < batchSize; b++) {
        Eigen::Map<const MatRMaj> matA((*data)[b] + offset, n, n);
        OuterStridedCMajMatK matC((*C)[b] + offC, n, nRHS, OuterStride(ldc));
        OuterStridedCMajMatM matD((*D)[b], n, nRHS, OuterStride(ldd));
        matD.noalias() += alpha * (MatRMaj(matA.template selfadjointView<Eigen::Lower>()) * matC);
      }
    }
  }

  virtual void solveL(const std::vector<float*>* data, int64_t offset, int64_t n,
                      std::vector<float*>* C, int64_t offC, int64_t ldc) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;

      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatM =
          Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      for (int b = 0; b < batchSize; b++) {
        Eigen::Map<const MatRMaj> matA((*data)[b] + offset, n, n);
        OuterStridedCMajMatM matC((*C)[b] + offC, n, nRHS, OuterStride(ldc));
        matA.template triangularView<Eigen::Lower>().solveInPlace(matC);
      }
    }
  }

  virtual void gemv(const std::vector<float*>* data, int64_t offset, int64_t nRows, int64_t nCols,
                    const std::vector<float*>* A, int64_t offA, int64_t lda, float alpha) override {
    @autoreleasepool {
      if (nRows <= 0 || nCols <= 0 || nRHS <= 0) return;

      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatK =
          Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      for (int b = 0; b < batchSize; b++) {
        Eigen::Map<const MatRMaj> matM((*data)[b] + offset, nRows, nCols);
        OuterStridedCMajMatK matA((*A)[b] + offA, nCols, nRHS, OuterStride(lda));
        Eigen::Map<MatRMaj> matC(solveBufPtrs[b], nRows, nRHS);
        matC.noalias() = alpha * (matM * matA);
      }
    }
  }

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, std::vector<float*>* C,
                           int64_t ldc) override {
    @autoreleasepool {
      if (numColItems <= 0) return;

      id<MTLComputePipelineState> pipeline = getPipeline("assembleVec_kernel_float");

      int64_t startRow =
          (chainColPtr > 0) ? sym.devChainRowsTillEnd.ptr()[chainColPtr - 1] : 0;
      int64_t nRHS64 = nRHS;

      id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];

      // Set structural/constant buffers
      [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                  offset:chainColPtr * sizeof(int64_t)
                 atIndex:0];
      [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                  offset:chainColPtr * sizeof(int64_t)
                 atIndex:1];
      [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                  offset:0
                 atIndex:2];
      [encoder setBytes:&numColItems length:sizeof(int64_t) atIndex:4];
      [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:6];
      [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:7];
      [encoder setBytes:&startRow length:sizeof(int64_t) atIndex:8];

      NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
      threadGroupSize = MIN(threadGroupSize, (NSUInteger)numColItems);
      MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
      MTLSize numGroups = MTLSizeMake(
          ((NSUInteger)numColItems + threadGroupSize - 1) / threadGroupSize, 1, 1);

      id<MTLBuffer> solveBuf = (__bridge id<MTLBuffer>)allJoinedSolveBufs.buffer();
      int64_t solveBufSize = sym.skel.order() * nRHS;

      // Dispatch once per batch item, changing solve buffer offset and C buffer
      for (int b = 0; b < batchSize; b++) {
        auto cInfo = MetalBufferRegistry::instance().findBuffer((*C)[b]);
        BASPACHO_CHECK_WHAT1(cInfo.first, "batched assembleVec: C buffer not found");

        size_t solveBufOffset = b * solveBufSize * sizeof(float);
        [encoder setBuffer:solveBuf offset:solveBufOffset atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)cInfo.first
                    offset:cInfo.second
                   atIndex:5];
        [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
      }

      [encoder endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }
  }

  virtual void solveLt(const std::vector<float*>* data, int64_t offset, int64_t n,
                       std::vector<float*>* C, int64_t offC, int64_t ldc) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;

      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatM =
          Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      for (int b = 0; b < batchSize; b++) {
        Eigen::Map<const MatRMaj> matA((*data)[b] + offset, n, n);
        OuterStridedCMajMatM matC((*C)[b] + offC, n, nRHS, OuterStride(ldc));
        matA.template triangularView<Eigen::Lower>().adjoint().solveInPlace(matC);
      }
    }
  }

  virtual void gemvT(const std::vector<float*>* data, int64_t offset, int64_t nRows, int64_t nCols,
                     std::vector<float*>* A, int64_t offA, int64_t lda, float alpha) override {
    @autoreleasepool {
      if (nRows <= 0 || nCols <= 0 || nRHS <= 0) return;

      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatM =
          Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      for (int b = 0; b < batchSize; b++) {
        Eigen::Map<const MatRMaj> matM((*data)[b] + offset, nRows, nCols);
        OuterStridedCMajMatM matA((*A)[b] + offA, nCols, nRHS, OuterStride(lda));
        Eigen::Map<const MatRMaj> matC(solveBufPtrs[b], nRows, nRHS);
        matA.noalias() += alpha * (matM.transpose() * matC);
      }
    }
  }

  virtual void assembleVecT(const std::vector<float*>* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    @autoreleasepool {
      if (numColItems <= 0) return;

      id<MTLComputePipelineState> pipeline = getPipeline("assembleVecT_kernel_float");

      int64_t startRow =
          (chainColPtr > 0) ? sym.devChainRowsTillEnd.ptr()[chainColPtr - 1] : 0;
      int64_t nRHS64 = nRHS;

      id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];

      // Set structural/constant buffers
      [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                  offset:chainColPtr * sizeof(int64_t)
                 atIndex:0];
      [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                  offset:chainColPtr * sizeof(int64_t)
                 atIndex:1];
      [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                  offset:0
                 atIndex:2];
      [encoder setBytes:&numColItems length:sizeof(int64_t) atIndex:4];
      [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:6];
      [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:7];
      [encoder setBytes:&startRow length:sizeof(int64_t) atIndex:8];

      NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
      threadGroupSize = MIN(threadGroupSize, (NSUInteger)numColItems);
      MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
      MTLSize numGroups = MTLSizeMake(
          ((NSUInteger)numColItems + threadGroupSize - 1) / threadGroupSize, 1, 1);

      id<MTLBuffer> solveBuf = (__bridge id<MTLBuffer>)allJoinedSolveBufs.buffer();
      int64_t solveBufSize = sym.skel.order() * nRHS;

      // Dispatch once per batch item, changing C buffer and solve buffer offset
      for (int b = 0; b < batchSize; b++) {
        auto cInfo = MetalBufferRegistry::instance().findBuffer((*C)[b]);
        BASPACHO_CHECK_WHAT1(cInfo.first, "batched assembleVecT: C buffer not found");

        [encoder setBuffer:(__bridge id<MTLBuffer>)cInfo.first
                    offset:cInfo.second
                   atIndex:3];
        size_t solveBufOffset = b * solveBufSize * sizeof(float);
        [encoder setBuffer:solveBuf offset:solveBufOffset atIndex:5];
        [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
      }

      [encoder endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }
  }

  void flush() override { MetalContext::instance().synchronize(); }

  MetalSymbolicCtx& sym;
  int nRHS;
  int batchSize;
  MetalMirror<float> allJoinedSolveBufs;
  std::vector<float*> solveBufPtrs;
};

// Factory method implementations
NumericCtxBase* MetalSymbolicCtx::createNumericCtxForType(type_index tIdx, int64_t tempBufSize,
                                                          int batchSize) {
  int64_t numSpans = skel.spanStart.size() - 1;

  // Check for double precision - not supported on Metal
  if (tIdx == type_index(typeid(double)) || tIdx == type_index(typeid(std::vector<double*>))) {
    throw std::runtime_error(
        "Metal backend does not support double precision. "
        "Apple Silicon GPUs lack double-precision floating point support. "
        "Please use float or select a different backend (BackendFast or BackendCuda).");
  }

  if (tIdx == type_index(typeid(float))) {
    return new MetalNumericCtx<float>(*this, tempBufSize, numSpans);
  }
  if (tIdx == type_index(typeid(std::vector<float*>))) {
    return new MetalNumericCtx<std::vector<float*>>(*this, tempBufSize, numSpans, batchSize);
  }

  throw std::runtime_error("MetalSymbolicCtx: unsupported numeric type");
}

SolveCtxBase* MetalSymbolicCtx::createSolveCtxForType(type_index tIdx, int nRHS, int batchSize) {
  // Check for double precision - not supported on Metal
  if (tIdx == type_index(typeid(double)) || tIdx == type_index(typeid(std::vector<double*>))) {
    throw std::runtime_error(
        "Metal backend does not support double precision. "
        "Apple Silicon GPUs lack double-precision floating point support. "
        "Please use float or select a different backend (BackendFast or BackendCuda).");
  }

  if (tIdx == type_index(typeid(float))) {
    return new MetalSolveCtx<float>(*this, nRHS);
  }
  if (tIdx == type_index(typeid(std::vector<float*>))) {
    return new MetalSolveCtx<std::vector<float*>>(*this, nRHS, batchSize);
  }

  throw std::runtime_error("MetalSymbolicCtx: unsupported solve type");
}

// Public factory function
OpsPtr metalOps() { return OpsPtr(new MetalOps); }

}  // end namespace BaSpaCho
