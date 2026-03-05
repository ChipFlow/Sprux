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
#include <unordered_map>

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

// Threshold for CPU BLAS fallback: when lump size is below this, dense
// operations (getrf, trsm, gemm, applyRowPerm) use Accelerate BLAS on CPU
// instead of MPS/GPU kernels. This avoids GPU dispatch overhead for small
// matrices where CPU BLAS is faster (e.g., c6288 has max lump n=111).
static int64_t getCpuBlasThreshold() {
  static int64_t val = getMpsThreshold("BASPACHO_METAL_CPU_BLAS_THRESHOLD", 256);
  return val;
}

#ifdef BASPACHO_USE_BLAS
// Transpose square matrix in-place (for row-major ↔ col-major conversion)
static void transposeSquareInPlaceFloat(float* data, int64_t n) {
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = i + 1; j < n; j++) {
      std::swap(data[i * n + j], data[j * n + i]);
    }
  }
}
#endif

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

// Symbolic elimination context for Metal
struct MetalSymElimCtx : SymElimCtx {
  MetalSymElimCtx() {}
  virtual ~MetalSymElimCtx() override {}

  int64_t numColumns;
  int64_t numBlockPairs;
  MetalMirror<int64_t> makeBlockPairEnumStraight;

  // Pre-computed work list for LU sparse elimination (Phase 1)
  int64_t numWorkItems = 0;
  MetalMirror<int32_t> devWorkItems;  // packed: 3 int32 per LUWorkItem
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

    // Note: sorting by target_offset was benchmarked but hurts performance
    // by disrupting L/U read locality (consecutive threads from the same
    // lump share L/U cache lines in natural order).

    // Upload as packed int32 array (3 int32 per work item)
    elim->numWorkItems = (int64_t)workItems.size();
    if (elim->numWorkItems > 0) {
      vector<int32_t> packed(3 * elim->numWorkItems);
      for (int64_t i = 0; i < elim->numWorkItems; i++) {
        packed[3 * i + 0] = workItems[i].L_offset;
        packed[3 * i + 1] = workItems[i].U_offset;
        packed[3 * i + 2] = workItems[i].target_offset;
      }
      elim->devWorkItems.load(packed);
    }

    return SymElimCtxPtr(elim);
  }

  virtual NumericCtxBase* createNumericCtxForType(type_index tIdx, int64_t tempBufSize,
                                                  int batchSize) override;

  virtual SolveCtxBase* createSolveCtxForType(type_index tIdx, int nRHS, int batchSize) override;

  const CoalescedBlockMatrixSkel& skel;

  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;

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
};

// Metal operations factory
struct MetalOps : Ops {
  virtual SymbolicCtxPtr createSymbolicCtx(const CoalescedBlockMatrixSkel& skel,
                                           const std::vector<int64_t>& permutation) override {
    return SymbolicCtxPtr(new MetalSymbolicCtx(skel, permutation));
  }
};

// Helper to dispatch a Metal compute kernel
// GPU profiling: set BASPACHO_METAL_PROFILE=1 to log kernel names + GPU times
static bool metalProfilingEnabled() {
  static bool enabled = [] {
    const char* val = getenv("BASPACHO_METAL_PROFILE");
    return val && std::string(val) == "1";
  }();
  return enabled;
}

// Map pipeline states to their kernel names for profiling output
static std::unordered_map<const void*, std::string>& pipelineNameMap() {
  static std::unordered_map<const void*, std::string> map;
  return map;
}

static id<MTLComputePipelineState> getProfiledPipeline(const char* name) {
  auto pipeline =
      (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(name);
  if (metalProfilingEnabled()) {
    pipelineNameMap()[(__bridge const void*)pipeline] = name;
  }
  return pipeline;
}

static void dispatchKernel(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                           void (^encodeBlock)(id<MTLComputeCommandEncoder>),
                           NSUInteger numThreads, bool sync = true,
                           const char* kernelName = nullptr) {
  // When profiling, force sync to read GPU timestamps
  if (metalProfilingEnabled()) sync = true;

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

    if (sync && metalProfilingEnabled()) {
      double gpuTimeMs = ([cmdBuf GPUEndTime] - [cmdBuf GPUStartTime]) * 1000.0;
      const char* name = kernelName;
      std::string lookupName;
      if (!name) {
        auto it = pipelineNameMap().find((__bridge const void*)pipeline);
        if (it != pipelineNameMap().end()) {
          lookupName = it->second;
          name = lookupName.c_str();
        }
      }
      if (name) {
        NSLog(@"[GPU] %-45s  threads=%-6lu  gpu=%.3fms",
              name, (unsigned long)numThreads, gpuTimeMs);
      }
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
  void encodeKernel(id<MTLComputePipelineState> pipeline,
                    void (^encodeBlock)(id<MTLComputeCommandEncoder>),
                    NSUInteger numThreads) {
    @autoreleasepool {
      if (!pendingCmdBuf_) {
        pendingCmdBuf_ = [sym.commandQueue commandBuffer];
      }
      if (!pendingEncoder_) {
        pendingEncoder_ = [pendingCmdBuf_ computeCommandEncoder];
        pendingDispatchCount_ = 0;
      }

      // Insert memory barrier so previous dispatches' buffer writes are visible
      if (pendingDispatchCount_ > 0) {
        [pendingEncoder_ memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }

      [pendingEncoder_ setComputePipelineState:pipeline];
      encodeBlock(pendingEncoder_);

      NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
      threadGroupSize = MIN(threadGroupSize, numThreads);

      MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
      MTLSize numGroups =
          MTLSizeMake((numThreads + threadGroupSize - 1) / threadGroupSize, 1, 1);

      [pendingEncoder_ dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
      pendingDispatchCount_++;
    }
  }

  // Commit the pending command buffer WITHOUT waiting for GPU completion.
  // Metal command queue ordering guarantees that command buffers execute in
  // submission order, so subsequent work on the same queue will see the results.
  // Tracks the last committed buffer so waitForGpu() can wait on it later.
  void commitPending() {
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
  // Since command buffers execute in submission order on the same queue,
  // waiting for the last one implicitly waits for all prior work.
  void waitForGpu() {
    if (lastCommittedCmdBuf_) {
      [lastCommittedCmdBuf_ waitUntilCompleted];
      lastCommittedCmdBuf_ = nil;
    }
    checkDeferredPotrfStatus();
    flushDeferredState();
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

    id<MTLComputePipelineState> pipeline = getProfiledPipeline(
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
    @autoreleasepool {
      // Find the MTLBuffer for data
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::pseudoFactorSpans: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      // Get pipeline state for factor_spans_kernel_float
      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
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

      // Step 2: Sparse elimination
      if (elim.numBlockPairs > 0) {
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
                "sparse_elim_straight_kernel_float");

        dispatchKernel(
            sym.commandQueue, pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                          offset:0
                         atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                          offset:0
                         atIndex:1];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                          offset:0
                         atIndex:2];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                          offset:0
                         atIndex:3];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                          offset:0
                         atIndex:4];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanToLump.buffer()
                          offset:0
                         atIndex:5];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanOffsetInLump.buffer()
                          offset:0
                         atIndex:6];
              [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:7];
              [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:8];
              [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:9];
              [encoder setBuffer:(__bridge id<MTLBuffer>)elim.makeBlockPairEnumStraight.buffer()
                          offset:0
                         atIndex:10];
              [encoder setBytes:&elim.numBlockPairs length:sizeof(int64_t) atIndex:11];
            },
            (NSUInteger)elim.numBlockPairs);
      }
    }
  }

  virtual void doEliminationLU(const SymElimCtx& elimData, float* data, int64_t lumpsBegin,
                               int64_t lumpsEnd, float staticPivotThreshold,
                               int64_t& perturbCount) override {
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

      // Step 2: LU Schur complement (L*U updates to both triangles)
      if (elim.numWorkItems > 0) {
        // Pre-computed work list path: no binary searches, 3 buffer bindings
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
                "lu_sparse_elim_precomputed_float");

        dispatchKernel(
            sym.commandQueue, pipeline,
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

  // Batch all LU sparse elimination levels into a single command buffer.
  // Encodes all factor+elim kernels with memory barriers, single commit+wait.
  void doAllEliminationsLU(const std::vector<SymElimCtxPtr>& elimCtxs,
                           const std::vector<int64_t>& ranges, float* data,
                           float staticPivotThreshold,
                           int64_t& totalPerturbCount) override {
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
          getProfiledPipeline("lu_factor_lumps_kernel_float");
      id<MTLComputePipelineState> elimPipeline =
          getProfiledPipeline("lu_sparse_elim_precomputed_float");

      for (size_t l = 0; l + 1 < ranges.size(); l++) {
        if (!elimCtxs[l]) continue;
        const MetalSymElimCtx& elim =
            *dynamic_cast<const MetalSymElimCtx*>(elimCtxs[l].get());

        int64_t lumpsBegin = ranges[l];
        int64_t lumpsEnd = ranges[l + 1];
        int64_t numLumps = lumpsEnd - lumpsBegin;
        if (numLumps <= 0) continue;

        float threshold = staticPivotThreshold;

        // LU factor lumps (divide below-diag by diagonal)
        encodeKernel(
            factorPipeline,
            ^(id<MTLComputeCommandEncoder> enc) {
              [enc setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                      offset:0
                     atIndex:0];
              [enc setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                      offset:0
                     atIndex:1];
              [enc setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                      offset:0
                     atIndex:2];
              [enc setBuffer:(__bridge id<MTLBuffer>)sym.devBoardColPtr.buffer()
                      offset:0
                     atIndex:3];
              [enc setBuffer:(__bridge id<MTLBuffer>)sym.devBoardChainColOrd.buffer()
                      offset:0
                     atIndex:4];
              [enc setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                      offset:0
                     atIndex:5];
              [enc setBuffer:dataBuffer offset:dataBaseOffset atIndex:6];
              [enc setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
              [enc setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];
              [enc setBytes:&threshold length:sizeof(float) atIndex:9];
              [enc setBuffer:perturbBuf offset:0 atIndex:10];
            },
            (NSUInteger)numLumps);

        // LU Schur complement
        if (elim.numWorkItems > 0) {
          encodeKernel(
              elimPipeline,
              ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)elim.devWorkItems.buffer()
                        offset:0
                       atIndex:1];
                [enc setBytes:&elim.numWorkItems length:sizeof(int64_t) atIndex:2];
              },
              (NSUInteger)elim.numWorkItems);
        }
      }

      // Single commit+wait for all levels
      commitAndWait();
      totalPerturbCount += *(uint32_t*)[perturbBuf contents];
    }
  }

  // Batch all Cholesky sparse elimination levels into a single command buffer.
  void doAllEliminations(const std::vector<SymElimCtxPtr>& elimCtxs,
                         const std::vector<int64_t>& ranges, float* data) override {
    @autoreleasepool {
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error(
            "MetalNumericCtx<float>::doAllEliminations: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      id<MTLComputePipelineState> factorPipeline =
          getProfiledPipeline("factor_lumps_kernel_float");
      id<MTLComputePipelineState> elimStraightPipeline =
          getProfiledPipeline("sparse_elim_straight_kernel_float");

      for (size_t l = 0; l + 1 < ranges.size(); l++) {
        if (!elimCtxs[l]) continue;
        const MetalSymElimCtx& elim =
            *dynamic_cast<const MetalSymElimCtx*>(elimCtxs[l].get());

        int64_t lumpsBegin = ranges[l];
        int64_t lumpsEnd = ranges[l + 1];
        int64_t numLumps = lumpsEnd - lumpsBegin;
        if (numLumps <= 0) continue;

        // Factor lumps (Cholesky on diagonal blocks + below-diagonal solve)
        encodeKernel(
            factorPipeline,
            ^(id<MTLComputeCommandEncoder> enc) {
              [enc setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                      offset:0
                     atIndex:0];
              [enc setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                      offset:0
                     atIndex:1];
              [enc setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                      offset:0
                     atIndex:2];
              [enc setBuffer:(__bridge id<MTLBuffer>)sym.devBoardColPtr.buffer()
                      offset:0
                     atIndex:3];
              [enc setBuffer:(__bridge id<MTLBuffer>)sym.devBoardChainColOrd.buffer()
                      offset:0
                     atIndex:4];
              [enc setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                      offset:0
                     atIndex:5];
              [enc setBuffer:dataBuffer offset:dataBaseOffset atIndex:6];
              [enc setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
              [enc setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];
            },
            (NSUInteger)numLumps);

        // Sparse elimination
        if (elim.numBlockPairs > 0) {
          encodeKernel(
              elimStraightPipeline,
              ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                        offset:0
                       atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                        offset:0
                       atIndex:1];
                [enc setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:0
                       atIndex:2];
                [enc setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                        offset:0
                       atIndex:3];
                [enc setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                        offset:0
                       atIndex:4];
                [enc setBuffer:(__bridge id<MTLBuffer>)sym.devSpanToLump.buffer()
                        offset:0
                       atIndex:5];
                [enc setBuffer:(__bridge id<MTLBuffer>)sym.devSpanOffsetInLump.buffer()
                        offset:0
                       atIndex:6];
                [enc setBuffer:dataBuffer offset:dataBaseOffset atIndex:7];
                [enc setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:8];
                [enc setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:9];
                [enc setBuffer:(__bridge id<MTLBuffer>)elim.makeBlockPairEnumStraight.buffer()
                        offset:0
                       atIndex:10];
                [enc setBytes:&elim.numBlockPairs length:sizeof(int64_t) atIndex:11];
              },
              (NSUInteger)elim.numBlockPairs);
        }
      }

      // Single commit+wait for all levels
      commitAndWait();
    }
  }

  virtual void potrf(int64_t n, float* data, int64_t offA) override {
    @autoreleasepool {
      if (n <= 0) return;

#ifdef BASPACHO_USE_BLAS
      // CPU BLAS fallback when no pending GPU work (dense loop after sparse elim)
      if (!pendingEncoder_ && !pendingCmdBuf_ && n <= getCpuBlasThreshold()) {
        flushPendingGemms();
        LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', n, data + offA, n);
        return;
      }
#endif

      // Three-tier threshold for Cholesky factorization:
      // - n < kMinN: CPU Eigen (tiny matrices, MPS dispatch overhead dominates)
      // - n > kMaxN: CPU BLAS (large matrices, multi-threaded BLAS >> MPS)
      // - otherwise: MPS Cholesky (GPU acceleration for medium matrices)
      static const int64_t kMpsPotrfMinN =
          getMpsThreshold("BASPACHO_MPS_POTRF_MIN_N", 4);
      static const int64_t kMpsPotrfMaxN =
          getMpsThreshold("BASPACHO_MPS_POTRF_MAX_N", 128);

      if (n < kMpsPotrfMinN) {
        // Very small matrix (1-3) — MPS dispatch overhead exceeds computation
        commitAndWait();
        using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<MatRMaj> matA(data + offA, n, n);
        Eigen::LLT<Eigen::Ref<MatRMaj>> llt(matA);
        if (llt.info() != Eigen::Success) {
          fprintf(stderr, "Metal potrf: Cholesky failed\n");
        }
        return;
      }

#ifdef BASPACHO_USE_BLAS
      if (n > kMpsPotrfMaxN) {
        // Large matrix — multi-threaded CPU BLAS is faster than MPS.
        // col-major upper = row-major lower, matching BaSpaCho's storage.
        commitAndWait();
        LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', n, data + offA, n);
        return;
      }
#endif

      // Medium matrix — use MPS Cholesky on GPU
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
    @autoreleasepool {
      if (n <= 0 || k <= 0) return;

#ifdef BASPACHO_USE_BLAS
      // CPU BLAS fallback when no pending GPU work (dense loop after sparse elim)
      if (!pendingEncoder_ && !pendingCmdBuf_ && n <= getCpuBlasThreshold()) {
        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, n, k,
                    1.0f, data + offA, n, data + offB, n);
        return;
      }
#endif

      // Two-tier threshold for triangular solve:
      // - n*n*k > kMaxThreshold: CPU BLAS (multi-threaded, better for large ops)
      // - otherwise: MPS triangular solve (GPU acceleration)
      // Set BASPACHO_MPS_TRSM_THRESHOLD > 0 to add a minimum for MPS (Eigen below).
      // Set BASPACHO_MPS_TRSM_MAX_THRESHOLD to control the BLAS upper threshold.
      static const int64_t kMpsTrsmThreshold =
          getMpsThreshold("BASPACHO_MPS_TRSM_THRESHOLD", 0);
      static const int64_t kMpsTrsmMaxThreshold =
          getMpsThreshold("BASPACHO_MPS_TRSM_MAX_THRESHOLD", 128LL * 128 * 128);

      if ((int64_t)n * n * k < kMpsTrsmThreshold) {
        // CPU Eigen fallback — only used when env override sets threshold > 0
        commitAndWait();
        using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using MatCMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
        // col-major's upper = (row-major's lower).transpose()
        Eigen::Map<const MatCMaj> matA(data + offA, n, n);
        Eigen::Map<MatRMaj> matB(data + offB, k, n);
        matA.template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(matB);
        return;
      }

#ifdef BASPACHO_USE_BLAS
      if ((int64_t)n * n * k > kMpsTrsmMaxThreshold) {
        // Large triangular solve — multi-threaded CPU BLAS is faster than MPS.
        // col-major upper = row-major lower, matching BaSpaCho's storage.
        commitAndWait();
        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, n, k,
                    1.0f, data + offA, n, data + offB, n);
        return;
      }
#endif

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
    @autoreleasepool {
      if (m <= 0 || n <= 0 || k <= 0) return;

#ifdef BASPACHO_USE_BLAS
      // CPU BLAS fallback when no pending GPU work (dense loop after sparse elim)
      if (!pendingEncoder_ && !pendingCmdBuf_ && k <= getCpuBlasThreshold()) {
        tempBuffer.resizeToAtLeast(m * n);
        // C(n,m) = B(n,k) * A^T(k,m), where A and B are row-major at data+offset
        // Row-major m×k = col-major k×m (lda=k). Result n×m row-major = m×n col-major (ldc=m).
        // Fortran: C(m,n) = A^T(m,k) * B(k,n), transA='C', transB='N'
        cblas_sgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, (BLAS_INT)m, (BLAS_INT)n,
                    (BLAS_INT)k, 1.0f, data + offset, (BLAS_INT)k, data + offset, (BLAS_INT)k,
                    0.0f, tempBuffer.ptr(), (BLAS_INT)m);
        return;
      }
#endif

      // End compute encoder if active (MPS needs its own encoding pass)
      // but keep the same command buffer — Metal guarantees sequential execution
      // within a buffer, so assembly + GEMM can share one submission.
      if (pendingEncoder_) {
        [pendingEncoder_ endEncoding];
        pendingEncoder_ = nil;
      }

      // Ensure temp buffer is large enough
      tempBuffer.resizeToAtLeast(m * n);

      // Always use MPS GEMM by default — single-threaded Eigen CPU fallback is
      // far slower than MPS dispatch overhead for supernodal Cholesky workloads.
      // Set BASPACHO_MPS_GEMM_THRESHOLD > 0 to revert to size-based routing.
      static const int64_t kMpsThreshold =
          getMpsThreshold("BASPACHO_MPS_GEMM_THRESHOLD", 0);
      bool useMps = (m * n * k >= kMpsThreshold);

      if (useMps) {
        // Use MPS matrix multiplication: C = B * A^T
        // Input: data[offset] is row-major with dims (m, k) for A and (n, k) for B (same memory)
        // Output: tempBuffer is row-major with dims (n, m)
        //
        // MPS uses row-major storage. MPSMatrixMultiplication computes:
        //   result = alpha * op(left) * op(right) + beta * result
        //
        // We want: C(n,m) = B(n,k) * A^T(k,m)
        // So: left=B, right=A with transposeRight=YES

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
      } else {
        // CPU fallback — commit pending GPU work and wait before CPU accesses
        commitAndWait();

        using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        const float* AB = data + offset;
        Eigen::Map<const MatRMaj> matA(AB, m, k);
        Eigen::Map<const MatRMaj> matB(AB, n, k);
        Eigen::Map<MatRMaj> matC(tempBuffer.ptr(), n, m);
        matC.noalias() = matB * matA.transpose();
      }
    }
  }

  virtual void prepareAssemble(int64_t targetLump) override {
    // Only flush if assemble() was actually called since last prepareAssemble.
    // For LU factorization (isGeneral()==true), eliminateBoardLU only calls
    // saveGemm — never assemble — so flushing is unnecessary and avoiding it
    // eliminates a per-lump CPU sync point.
    if (assembleWasCalled_) {
      commitAndWait();
      assembleWasCalled_ = false;
    }

    // Prepare chain offset mapping for assembly (same as CUDA version)
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    for (int64_t i = skel.chainColPtr[targetLump], iEnd = skel.chainColPtr[targetLump + 1]; i < iEnd;
         i++) {
      spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
    }

    // Copy to device
    memcpy(devSpanToChainOffset.ptr(), spanToChainOffset.data(),
           spanToChainOffset.size() * sizeof(int64_t));
  }

  virtual void assemble(float* data, int64_t rectRowBegin, int64_t dstStride,
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    @autoreleasepool {
      if (numBlockRows <= 0 || numBlockCols <= 0) return;
      assembleWasCalled_ = true;

      // CPU fallback when no pending GPU work (avoids GPU dispatch overhead)
      if (!pendingEncoder_ && !pendingCmdBuf_) {
        const int64_t* chainRowsTillEnd =
            sym.skel.chainRowsTillEnd.data() + srcColDataOffset;
        const int64_t* pToSpan = sym.skel.chainRowSpan.data() + srcColDataOffset;
        const int64_t* pSpanToChainOffset = spanToChainOffset.data();
        const int64_t* pSpanOffsetInLump = sym.skel.spanOffsetInLump.data();
        const float* matRectPtr = tempBuffer.ptr();
        for (int64_t r = 0; r < numBlockRows; r++) {
          int64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
          int64_t rSize = chainRowsTillEnd[r] - rBegin - rectRowBegin;
          int64_t rParam = pToSpan[r];
          int64_t rOffset = pSpanToChainOffset[rParam];
          const float* matRowPtr = matRectPtr + rBegin * srcRectWidth;
          int64_t cEnd = std::min(numBlockCols, r + 1);
          for (int64_t c = 0; c < cEnd; c++) {
            int64_t cStart = chainRowsTillEnd[c - 1] - rectRowBegin;
            int64_t cSize = chainRowsTillEnd[c] - cStart - rectRowBegin;
            int64_t offset = rOffset + pSpanOffsetInLump[pToSpan[c]];
            float* dst = data + offset;
            const float* src = matRowPtr + cStart;
            for (int64_t i = 0; i < rSize; i++) {
              for (int64_t j = 0; j < cSize; j++) {
                dst[i * dstStride + j] -= src[i * srcRectWidth + j];
              }
            }
          }
        }
        return;
      }

      // Find the MTLBuffer for data
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::assemble: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      // Get pipeline state
      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
              "assemble_kernel_float");

      int64_t numThreads = numBlockRows * numBlockCols;

      // Compute startRow = chainRowsTillEnd[srcColDataOffset - 1] (element before chain start)
      // This matches CPU ref where startRow = pChainRowsTillEnd[-1] after offsetting the pointer
      int64_t startRow = (srcColDataOffset > 0) ? sym.skel.chainRowsTillEnd[srcColDataOffset - 1] : 0;

      // Profiling fallback: use dispatchKernel for per-kernel GPU timestamps
      if (metalProfilingEnabled()) {
        dispatchKernel(
            sym.commandQueue, pipeline,
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
        return;
      }

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
    @autoreleasepool {
      if (n <= 0) return 0;

      // CPU path: no pending GPU work, operate directly
      if (!pendingEncoder_ && !pendingCmdBuf_) {
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

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
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
    @autoreleasepool {
      if (m <= 0 || n <= 0) return 0;

      int64_t minMN = std::min(m, n);

      // Find the MTLBuffer for data
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::getrf: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      // Ensure devPivots is large enough for this lump's pivots
      devPivots.resizeToAtLeast(minMN);

      // Profiling fallback: use sync dispatch for per-kernel GPU timestamps
      if (metalProfilingEnabled()) {
        id<MTLComputePipelineState> pipeline = getProfiledPipeline(
                "lu_getrf_kernel_float");
        dispatchKernel(
            sym.commandQueue, pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:0];
              [encoder setBytes:&offA length:sizeof(int64_t) atIndex:1];
              [encoder setBytes:&m length:sizeof(int64_t) atIndex:2];
              [encoder setBytes:&n length:sizeof(int64_t) atIndex:3];
              [encoder setBuffer:(__bridge id<MTLBuffer>)devPivots.buffer() offset:0 atIndex:4];
            },
            1);
        memcpy(pivots, devPivots.ptr(), minMN * sizeof(int64_t));
        return 0;
      }

      // CPU BLAS path: for small matrices, use Accelerate sgetrf directly.
      // On Apple Silicon shared memory, CPU can access the Metal buffer without copy.
      // This avoids MPS dispatch overhead which dominates for small blocks.
#ifdef BASPACHO_USE_BLAS
      if (minMN <= getCpuBlasThreshold()) {
        // Flush any pending GPU work to ensure data is visible to CPU
        flushPendingGemms();
        if (pendingEncoder_ || pendingCmdBuf_) {
          commitPending();
          waitForGpu();
        }

        // Transpose row-major → col-major for LAPACK
        if (m == n) {
          transposeSquareInPlaceFloat(data + offA, n);
        }

        std::vector<BLAS_INT> ipiv(minMN);
        int info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, (BLAS_INT)m, (BLAS_INT)n,
                                  data + offA, (BLAS_INT)m, ipiv.data());

        // Transpose col-major → row-major
        if (m == n) {
          transposeSquareInPlaceFloat(data + offA, n);
        }

        // Convert pivots: LAPACK 1-based → 0-based
        for (int64_t i = 0; i < minMN; i++) {
          pivots[i] = ipiv[i] - 1;
        }

        // Pivots are on CPU — mark as not GPU-resident
        pivotsOnGpu_ = false;

        return info;
      }
#endif

      // MPS path: use MPSMatrixDecompositionLU for parallel GPU factorization.
      // Flush pending saveGemm work items first — ensures all Schur
      // complement updates are dispatched before factorization of this lump.
      flushPendingGemms();

      // End pending compute encoder (MPS needs its own encoding)
      if (pendingEncoder_) {
        [pendingEncoder_ endEncoding];
        pendingEncoder_ = nil;
      }

      // Ensure we have a command buffer
      if (!pendingCmdBuf_) {
        pendingCmdBuf_ = [sym.commandQueue commandBuffer];
      }

      // Create MPSMatrix view for the block at data+offA (row-major, m×n, stride=n)
      MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
          matrixDescriptorWithRows:m columns:n
          rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32];
      MPSMatrix* mpsA = [[MPSMatrix alloc]
          initWithBuffer:dataBuffer
          offset:dataBaseOffset + offA * sizeof(float)
          descriptor:descA];

      // Pivot buffer (UInt32 format required by MPS)
      devPivotBuf32.resizeToAtLeast(minMN);
      MPSMatrixDescriptor* descPiv = [MPSMatrixDescriptor
          matrixDescriptorWithRows:1 columns:minMN
          rowBytes:minMN * sizeof(uint32_t) dataType:MPSDataTypeUInt32];
      MPSMatrix* mpsPiv = [[MPSMatrix alloc]
          initWithBuffer:(__bridge id<MTLBuffer>)devPivotBuf32.buffer()
          offset:0 descriptor:descPiv];

      // Encode MPS LU factorization (in-place: resultMatrix = sourceMatrix)
      MPSMatrixDecompositionLU* mpsLU = [[MPSMatrixDecompositionLU alloc]
          initWithDevice:sym.device rows:m columns:n];
      [mpsLU encodeToCommandBuffer:pendingCmdBuf_
          sourceMatrix:mpsA resultMatrix:mpsA
          pivotIndices:mpsPiv status:nil];

      // GPU-resident pivot path: for general (LU) matrices with pre-allocated
      // devAllPivots, encode a GPU-side uint32→int64 conversion kernel and
      // keep pivots on GPU. For non-general matrices (e.g. simple LU tests),
      // fall back to CPU conversion with commitAndWait.
      if (devAllPivots.buffer()) {
        // Compute offset into the persistent all-pivots buffer.
        if (!allPivotsCpuBase_) {
          allPivotsCpuBase_ = pivots;  // First getrf call — record base
        }
        int64_t pivotOffset = pivots - allPivotsCpuBase_;
        allPivotsCount_ = std::max(allPivotsCount_, pivotOffset + minMN);

        // Encode GPU-side pivot conversion (uint32→int64) into the same cmd buffer.
        int64_t pivotByteOffset = pivotOffset * sizeof(int64_t);
        id<MTLComputePipelineState> convertPipeline = getProfiledPipeline(
                "lu_convertPivots_kernel_float");
        encodeKernel(
            convertPipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:(__bridge id<MTLBuffer>)devPivotBuf32.buffer()
                          offset:0 atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)devAllPivots.buffer()
                          offset:pivotByteOffset atIndex:1];
              [encoder setBytes:&minMN length:sizeof(int64_t) atIndex:2];
            },
            (NSUInteger)minMN);

        // Mark pivots as GPU-resident — applyRowPerm will skip memcpy.
        pivotsOnGpu_ = true;
      } else {
        // Fallback: non-general matrix, commit and read pivots on CPU
        commitPending();
        waitForGpu();
        uint32_t* mpsPivots = devPivotBuf32.ptr();
        for (int64_t i = 0; i < minMN; i++) {
          pivots[i] = static_cast<int64_t>(mpsPivots[i]);
        }
      }

      return 0;
    }
  }

  virtual void trsmLowerUnit(int64_t m, int64_t n, const float* L, int64_t offL, float* B,
                              int64_t offB, int64_t ldb) override {
    @autoreleasepool {
      if (m <= 0 || n <= 0) return;

      // CPU BLAS path for small operations
#ifdef BASPACHO_USE_BLAS
      if (m <= getCpuBlasThreshold() && !pendingEncoder_ && !pendingCmdBuf_) {
        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasUnit,
                    (BLAS_INT)n, (BLAS_INT)m, 1.0f, L + offL, (BLAS_INT)m, B + offB, (BLAS_INT)ldb);
        return;
      }
#endif

      auto lBufferInfo = MetalBufferRegistry::instance().findBuffer(L);
      auto bBufferInfo = MetalBufferRegistry::instance().findBuffer(B);
      if (!lBufferInfo.first || !bBufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::trsmLowerUnit: buffer not found");
      }
      id<MTLBuffer> lBuffer = (__bridge id<MTLBuffer>)lBufferInfo.first;
      size_t lBaseOffset = lBufferInfo.second;
      id<MTLBuffer> bBuffer = (__bridge id<MTLBuffer>)bBufferInfo.first;
      size_t bBaseOffset = bBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
              "lu_trsmLowerUnit_kernel_float");

      if (metalProfilingEnabled()) {
        dispatchKernel(
            sym.commandQueue, pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:lBuffer offset:lBaseOffset atIndex:0];
              [encoder setBytes:&offL length:sizeof(int64_t) atIndex:1];
              [encoder setBuffer:bBuffer offset:bBaseOffset atIndex:2];
              [encoder setBytes:&offB length:sizeof(int64_t) atIndex:3];
              [encoder setBytes:&m length:sizeof(int64_t) atIndex:4];
              [encoder setBytes:&n length:sizeof(int64_t) atIndex:5];
              [encoder setBytes:&ldb length:sizeof(int64_t) atIndex:6];
            },
            1);
        return;
      }

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
          1);
    }
  }

  virtual void trsmUpperRight(int64_t m, int64_t n, const float* U, int64_t offU, float* B,
                               int64_t offB, int64_t ldb) override {
    @autoreleasepool {
      if (m <= 0 || n <= 0) return;

      // CPU BLAS path for small operations
#ifdef BASPACHO_USE_BLAS
      if (n <= getCpuBlasThreshold() && !pendingEncoder_ && !pendingCmdBuf_) {
        (void)ldb;
        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                    (BLAS_INT)n, (BLAS_INT)m, 1.0f, U + offU, (BLAS_INT)n, B + offB, (BLAS_INT)n);
        return;
      }
#endif

      auto uBufferInfo = MetalBufferRegistry::instance().findBuffer(U);
      auto bBufferInfo = MetalBufferRegistry::instance().findBuffer(B);
      if (!uBufferInfo.first || !bBufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::trsmUpperRight: buffer not found");
      }
      id<MTLBuffer> uBuffer = (__bridge id<MTLBuffer>)uBufferInfo.first;
      size_t uBaseOffset = uBufferInfo.second;
      id<MTLBuffer> bBuffer = (__bridge id<MTLBuffer>)bBufferInfo.first;
      size_t bBaseOffset = bBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
              "lu_trsmUpperRight_kernel_float");

      if (metalProfilingEnabled()) {
        dispatchKernel(
            sym.commandQueue, pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:uBuffer offset:uBaseOffset atIndex:0];
              [encoder setBytes:&offU length:sizeof(int64_t) atIndex:1];
              [encoder setBuffer:bBuffer offset:bBaseOffset atIndex:2];
              [encoder setBytes:&offB length:sizeof(int64_t) atIndex:3];
              [encoder setBytes:&m length:sizeof(int64_t) atIndex:4];
              [encoder setBytes:&n length:sizeof(int64_t) atIndex:5];
              [encoder setBytes:&ldb length:sizeof(int64_t) atIndex:6];
            },
            1);
        return;
      }

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
          1);
    }
  }

  virtual void saveGemm(int64_t m, int64_t n, int64_t k, const float* L, int64_t offL,
                         int64_t ldL, const float* U, int64_t offU, int64_t ldU, float* C,
                         int64_t offC, int64_t ldC) override {
    @autoreleasepool {
      if (m <= 0 || n <= 0 || k <= 0) return;

      // Profiling path: dispatch individually for per-kernel GPU timestamps
      if (metalProfilingEnabled()) {
        auto lBufferInfo = MetalBufferRegistry::instance().findBuffer(L);
        auto uBufferInfo = MetalBufferRegistry::instance().findBuffer(U);
        auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
        if (!lBufferInfo.first || !uBufferInfo.first || !cBufferInfo.first) {
          throw std::runtime_error("MetalNumericCtx<float>::saveGemm: buffer not found");
        }
        id<MTLBuffer> lBuffer = (__bridge id<MTLBuffer>)lBufferInfo.first;
        size_t lBaseOffset = lBufferInfo.second;
        id<MTLBuffer> uBuffer = (__bridge id<MTLBuffer>)uBufferInfo.first;
        size_t uBaseOffset = uBufferInfo.second;
        id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
        size_t cBaseOffset = cBufferInfo.second;

        id<MTLComputePipelineState> pipeline = getProfiledPipeline(
                "lu_saveGemm_kernel_float");

        int64_t numThreads = m * n;
        dispatchKernel(
            sym.commandQueue, pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:lBuffer offset:lBaseOffset atIndex:0];
              [encoder setBytes:&offL length:sizeof(int64_t) atIndex:1];
              [encoder setBytes:&ldL length:sizeof(int64_t) atIndex:2];
              [encoder setBuffer:uBuffer offset:uBaseOffset atIndex:3];
              [encoder setBytes:&offU length:sizeof(int64_t) atIndex:4];
              [encoder setBytes:&ldU length:sizeof(int64_t) atIndex:5];
              [encoder setBuffer:cBuffer offset:cBaseOffset atIndex:6];
              [encoder setBytes:&offC length:sizeof(int64_t) atIndex:7];
              [encoder setBytes:&ldC length:sizeof(int64_t) atIndex:8];
              [encoder setBytes:&m length:sizeof(int64_t) atIndex:9];
              [encoder setBytes:&n length:sizeof(int64_t) atIndex:10];
              [encoder setBytes:&k length:sizeof(int64_t) atIndex:11];
            },
            (NSUInteger)numThreads);
        sym.luGemmCalls++;
        return;
      }

      // CPU BLAS path: immediate GEMM for small operations
#ifdef BASPACHO_USE_BLAS
      if (k <= getCpuBlasThreshold() && !pendingEncoder_ && !pendingCmdBuf_) {
        // C -= L * U (row-major: in col-major view, C^T -= U^T * L^T)
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    (BLAS_INT)n, (BLAS_INT)m, (BLAS_INT)k, -1.0f,
                    U + offU, (BLAS_INT)ldU, L + offL, (BLAS_INT)ldL,
                    1.0f, C + offC, (BLAS_INT)ldC);
        sym.luGemmCalls++;
        return;
      }
#endif

      // Batched path: buffer work items, flush later in flushPendingGemms()
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

      pendingGemms_.push_back(item);
      sym.luGemmCalls++;
    }
  }

  virtual void applyRowPerm(int64_t* pivots, int64_t n, float* data, int64_t offData, int64_t ld,
                             int64_t numCols) override {
    @autoreleasepool {
      if (n <= 0 || numCols <= 0) return;

      // CPU path: pivots on CPU, no pending GPU work
      if (!pivotsOnGpu_ && !pendingEncoder_ && !pendingCmdBuf_) {
        float* d = data + offData;
        for (int64_t i = 0; i < n; i++) {
          int64_t swapRow = pivots[i];
          if (swapRow != i) {
            for (int64_t c = 0; c < numCols; c++) {
              std::swap(d[i + c * ld], d[swapRow + c * ld]);
            }
          }
        }
        return;
      }

      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::applyRowPerm: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
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

      if (metalProfilingEnabled()) {
        dispatchKernel(
            sym.commandQueue, pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:pivotBuffer offset:pivotByteOffset atIndex:0];
              [encoder setBytes:&n length:sizeof(int64_t) atIndex:1];
              [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:2];
              [encoder setBytes:&offData length:sizeof(int64_t) atIndex:3];
              [encoder setBytes:&ld length:sizeof(int64_t) atIndex:4];
              [encoder setBytes:&numCols length:sizeof(int64_t) atIndex:5];
            },
            numThreads);
        return;
      }

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

  void flush() override {
    commitAndWait();

    // Reset batched state for next factorization
    cachedDataBuffer_ = nil;
    cachedDataBaseOffset_ = 0;
    // Reset append-only GEMM work buffer for next factorization
    gemmWorkBufUsedBytes_ = 0;
    gemmWorkBufInFlight_ = false;
    // Reset pivot state for next factorization
    allPivotsCpuBase_ = nullptr;
    allPivotsCount_ = 0;
    MetalContext::instance().synchronize();
  }

  int64_t deferredPerturbCount() override {
    // Read the accumulated GPU atomic counter (valid after flush/commitAndWait)
    int64_t count = 0;
    if (perturbCountPending_ && perturbCountBuf_) {
      count = *(uint32_t*)[perturbCountBuf_ contents];
      perturbCountPending_ = false;
      // Reset buffer for next factorization
      *(uint32_t*)[perturbCountBuf_ contents] = 0;
    }
    return count;
  }

  MetalSymbolicCtx& sym;
  int64_t numSpans_;
  MetalMirror<float> tempBuffer;
  MetalMirror<int64_t> devSpanToChainOffset;
  std::vector<int64_t> spanToChainOffset;
  MetalMirror<int64_t> devPivots;        // GPU buffer for LU pivots
  MetalMirror<uint32_t> devPivotBuf32;  // GPU buffer for MPS LU pivot output (uint32_t format)
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

  // Encode a kernel dispatch onto a persistent compute encoder within the
  // pending command buffer. Uses a single encoder for all dispatches with
  // memory barriers between them to ensure correct data ordering.
  void encodeKernel(id<MTLComputePipelineState> pipeline,
                    void (^encodeBlock)(id<MTLComputeCommandEncoder>),
                    NSUInteger numThreads) {
    @autoreleasepool {
      if (!pendingCmdBuf_) {
        pendingCmdBuf_ = [sym.commandQueue commandBuffer];
      }
      if (!pendingEncoder_) {
        pendingEncoder_ = [pendingCmdBuf_ computeCommandEncoder];
        pendingDispatchCount_ = 0;
      }

      // Insert memory barrier so previous dispatches' buffer writes are visible
      if (pendingDispatchCount_ > 0) {
        [pendingEncoder_ memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }

      [pendingEncoder_ setComputePipelineState:pipeline];
      encodeBlock(pendingEncoder_);

      NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
      threadGroupSize = MIN(threadGroupSize, numThreads);

      MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
      MTLSize numGroups =
          MTLSizeMake((numThreads + threadGroupSize - 1) / threadGroupSize, 1, 1);

      [pendingEncoder_ dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
      pendingDispatchCount_++;
    }
  }

  // Commit the pending command buffer WITHOUT waiting for GPU completion.
  void commitPending() {
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
  void waitForGpu() {
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
      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
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
      id<MTLComputePipelineState> subDiagPipeline = getProfiledPipeline(
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
      id<MTLComputePipelineState> subDiagPipeline = getProfiledPipeline(
              "sparseElim_subDiagMultT_float");

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

      // Then encode diagonal solve kernel
      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
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
          getProfiledPipeline("sparseElim_subDiagMult_float");

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
      id<MTLComputePipelineState> gatherPipeline =
          getProfiledPipeline("sparseElim_upperGather_float");

      encodeKernel(
          gatherPipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
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
          },
          (NSUInteger)numLumps);

      // Then: diagonal U solve: v[lump] /= U_diagonal
      id<MTLComputePipelineState> diagPipeline =
          getProfiledPipeline("sparseElim_diagDivU_float");

      encodeKernel(
          diagPipeline,
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

  virtual void symm(const float* data, int64_t offset, int64_t n, const float* C, int64_t offC,
                    int64_t ldc, float* D, int64_t ldd, float alpha) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;
      if (pendingEncoder_ || pendingCmdBuf_) {
        commitAndWait();  // Flush GPU work before CPU reads shared buffers
      }

      // Use row-major for data buffer (matches CpuBaseSolveCtx)
      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatK =
          Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;
      using OuterStridedCMajMatM =
          Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      Eigen::Map<const MatRMaj> matA(data + offset, n, n);
      OuterStridedCMajMatK matC(C + offC, n, nRHS, OuterStride(ldc));
      OuterStridedCMajMatM matD(D, n, nRHS, OuterStride(ldd));
      matD.noalias() += alpha * (MatRMaj(matA.template selfadjointView<Eigen::Lower>()) * matC);
    }
  }

  virtual void solveL(const float* data, int64_t offset, int64_t n, float* C, int64_t offC,
                      int64_t ldc) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;
      if (pendingEncoder_ || pendingCmdBuf_) {
        commitAndWait();  // Flush GPU work before CPU reads shared buffers
      }

      // Use row-major for data buffer (matches CpuBaseSolveCtx)
      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatM =
          Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      Eigen::Map<const MatRMaj> matA(data + offset, n, n);
      OuterStridedCMajMatM matC(C + offC, n, nRHS, OuterStride(ldc));
      matA.template triangularView<Eigen::Lower>().solveInPlace(matC);
    }
  }

  virtual void gemv(const float* data, int64_t offset, int64_t nRows, int64_t nCols, const float* A,
                    int64_t offA, int64_t lda, float alpha) override {
    @autoreleasepool {
      if (nRows <= 0 || nCols <= 0 || nRHS <= 0) return;
      if (pendingEncoder_ || pendingCmdBuf_) {
        commitAndWait();  // Flush GPU work before CPU reads shared buffers
      }

      // Use row-major for data buffer (matches CpuBaseSolveCtx)
      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatK =
          Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      tempVecBuffer.resizeToAtLeast(nRows * nRHS);

      Eigen::Map<const MatRMaj> matM(data + offset, nRows, nCols);
      OuterStridedCMajMatK matA(A + offA, nCols, nRHS, OuterStride(lda));
      Eigen::Map<MatRMaj> matC(tempVecBuffer.ptr(), nRows, nRHS);
      matC.noalias() = alpha * (matM * matA);
    }
  }

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, float* C, int64_t ldc) override {
    @autoreleasepool {
      if (numColItems <= 0) return;

      // CPU fallback when no pending GPU work (avoids GPU dispatch overhead for small ops)
      if (!pendingEncoder_ && !pendingCmdBuf_) {
        int64_t startRow =
            (chainColPtr > 0) ? sym.devChainRowsTillEnd.ptr()[chainColPtr - 1] : 0;
        const int64_t* rowsTillEnd = sym.devChainRowsTillEnd.ptr() + chainColPtr;
        const int64_t* toSpan = sym.devChainRowSpan.ptr() + chainColPtr;
        for (int64_t tid = 0; tid < numColItems; tid++) {
          int64_t rowsBefore = (tid > 0) ? (rowsTillEnd[tid - 1] - startRow) : 0;
          int64_t rowsAfter = rowsTillEnd[tid] - startRow;
          int64_t blockRows = rowsAfter - rowsBefore;
          int64_t span = toSpan[tid];
          int64_t spanStart = sym.devSpanStart.ptr()[span];
          const float* srcPtr = tempVecBuffer.ptr() + rowsBefore * nRHS;
          float* dstPtr = C + spanStart;
          for (int rhs = 0; rhs < nRHS; rhs++) {
            for (int64_t i = 0; i < blockRows; i++) {
              dstPtr[i + rhs * ldc] += srcPtr[i * nRHS + rhs];
            }
          }
        }
        return;
      }

      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::assembleVec: buffer not found");
      }
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t cOffset = cBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
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
      if (pendingEncoder_ || pendingCmdBuf_) {
        commitAndWait();  // Flush GPU work before CPU reads shared buffers
      }

      // Use row-major for data buffer (matches CpuBaseSolveCtx)
      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatM =
          Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      Eigen::Map<const MatRMaj> matA(data + offset, n, n);
      OuterStridedCMajMatM matC(C + offC, n, nRHS, OuterStride(ldc));
      matA.template triangularView<Eigen::Lower>().adjoint().solveInPlace(matC);
    }
  }

  virtual void gemvT(const float* data, int64_t offset, int64_t nRows, int64_t nCols, float* A,
                     int64_t offA, int64_t lda, float alpha) override {
    @autoreleasepool {
      if (nRows <= 0 || nCols <= 0 || nRHS <= 0) return;
      if (pendingEncoder_ || pendingCmdBuf_) {
        commitAndWait();  // Flush GPU work before CPU reads shared buffers
      }

      // Use row-major for data buffer (matches CpuBaseSolveCtx)
      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatM =
          Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      Eigen::Map<const MatRMaj> matM(data + offset, nRows, nCols);
      OuterStridedCMajMatM matA(A + offA, nCols, nRHS, OuterStride(lda));
      Eigen::Map<const MatRMaj> matC(tempVecBuffer.ptr(), nRows, nRHS);
      matA.noalias() += alpha * (matM.transpose() * matC);
    }
  }

  virtual void assembleVecT(const float* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    @autoreleasepool {
      if (numColItems <= 0) return;

      // CPU fallback when no pending GPU work (avoids GPU dispatch overhead for small ops)
      if (!pendingEncoder_ && !pendingCmdBuf_) {
        int64_t startRow =
            (chainColPtr > 0) ? sym.devChainRowsTillEnd.ptr()[chainColPtr - 1] : 0;
        const int64_t* rowsTillEnd = sym.devChainRowsTillEnd.ptr() + chainColPtr;
        const int64_t* toSpan = sym.devChainRowSpan.ptr() + chainColPtr;
        for (int64_t tid = 0; tid < numColItems; tid++) {
          int64_t rowsBefore = (tid > 0) ? (rowsTillEnd[tid - 1] - startRow) : 0;
          int64_t rowsAfter = rowsTillEnd[tid] - startRow;
          int64_t blockRows = rowsAfter - rowsBefore;
          int64_t span = toSpan[tid];
          int64_t spanStart = sym.devSpanStart.ptr()[span];
          float* dstPtr = tempVecBuffer.ptr() + rowsBefore * nRHS;
          const float* srcPtr = C + spanStart;
          for (int rhs = 0; rhs < nRHS; rhs++) {
            for (int64_t i = 0; i < blockRows; i++) {
              dstPtr[i * nRHS + rhs] = srcPtr[i + rhs * ldc];
            }
          }
        }
        return;
      }

      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::assembleVecT: buffer not found");
      }
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t cOffset = cBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
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

      // CPU fallback when no pending GPU work (avoids GPU dispatch overhead for small ops)
      if (!pendingEncoder_ && !pendingCmdBuf_) {
        using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
        using OuterStridedCMajMatM =
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                       OuterStride>;
        Eigen::Map<const MatRMaj> matA(data + offM, n, n);
        OuterStridedCMajMatM matC(C + offC, n, nRHS, OuterStride(ldc));
        matA.template triangularView<Eigen::UnitLower>().solveInPlace(matC);
        return;
      }

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!dataBufferInfo.first || !cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::solveLUnit: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;
      size_t cOffset = cBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
              "lu_solveLUnit_direct_kernel_float");

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
          1);
    }
  }

  // Solve U * x = b where U is upper triangular (backward substitution)
  virtual void solveU(const float* data, int64_t offM, int64_t n, float* C, int64_t offC,
                      int64_t ldc) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;

      // CPU fallback when no pending GPU work (avoids GPU dispatch overhead for small ops)
      if (!pendingEncoder_ && !pendingCmdBuf_) {
        using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
        using OuterStridedCMajMatM =
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                       OuterStride>;
        Eigen::Map<const MatRMaj> matA(data + offM, n, n);
        OuterStridedCMajMatM matC(C + offC, n, nRHS, OuterStride(ldc));
        matA.template triangularView<Eigen::Upper>().solveInPlace(matC);
        return;
      }

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!dataBufferInfo.first || !cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::solveU: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;
      size_t cOffset = cBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
              "lu_solveU_direct_kernel_float");

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
          1);
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

      // CPU fallback when no pending GPU work
      if (!pendingEncoder_ && !pendingCmdBuf_) {
        for (int64_t i = 0; i < n; i++) {
          int64_t swapRow = pivots[i];
          if (swapRow != i) {
            for (int rhs = 0; rhs < nRHS; rhs++) {
              std::swap(vec[i + rhs * ldVec], vec[swapRow + rhs * ldVec]);
            }
          }
        }
        return;
      }

      // Compute offset into pre-uploaded pivot buffer, or upload on-demand
      size_t pivotByteOffset = 0;
      if (pivotsBase_ && pivots >= pivotsBase_ && pivots < pivotsBase_ + pivotsSize_) {
        pivotByteOffset = (pivots - pivotsBase_) * sizeof(int64_t);
      } else {
        // Fallback: no pre-upload, sync and copy per-call
        commitAndWait();
        devPivots.resizeToAtLeast(n);
        memcpy(devPivots.ptr(), pivots, n * sizeof(int64_t));
      }

      auto vecBufferInfo = MetalBufferRegistry::instance().findBuffer(vec);
      if (!vecBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::applyRowPermVec: buffer not found");
      }
      id<MTLBuffer> vecBuffer = (__bridge id<MTLBuffer>)vecBufferInfo.first;
      size_t vecBaseOffset = vecBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
              "lu_applyRowPermVec_kernel_float");

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)devPivots.buffer()
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

      // CPU fallback when no pending GPU work
      if (!pendingEncoder_ && !pendingCmdBuf_) {
        for (int64_t i = n - 1; i >= 0; i--) {
          int64_t swapRow = pivots[i];
          if (swapRow != i) {
            for (int rhs = 0; rhs < nRHS; rhs++) {
              std::swap(vec[i + rhs * ldVec], vec[swapRow + rhs * ldVec]);
            }
          }
        }
        return;
      }

      // Compute offset into pre-uploaded pivot buffer, or upload on-demand
      size_t pivotByteOffset = 0;
      if (pivotsBase_ && pivots >= pivotsBase_ && pivots < pivotsBase_ + pivotsSize_) {
        pivotByteOffset = (pivots - pivotsBase_) * sizeof(int64_t);
      } else {
        // Fallback: no pre-upload, sync and copy per-call
        commitAndWait();
        devPivots.resizeToAtLeast(n);
        memcpy(devPivots.ptr(), pivots, n * sizeof(int64_t));
      }

      auto vecBufferInfo = MetalBufferRegistry::instance().findBuffer(vec);
      if (!vecBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::applyRowPermVecInv: buffer not found");
      }
      id<MTLBuffer> vecBuffer = (__bridge id<MTLBuffer>)vecBufferInfo.first;
      size_t vecBaseOffset = vecBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
              "lu_applyRowPermVecInv_kernel_float");

      int64_t nRHS64 = nRHS;
      encodeKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)devPivots.buffer()
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

      // CPU fallback when no pending GPU work (avoids GPU dispatch overhead for small ops)
      if (!pendingEncoder_ && !pendingCmdBuf_) {
        using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
        using OuterStridedCMajMatK =
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
                       0, OuterStride>;
        using OuterStridedCMajMatM =
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                       OuterStride>;
        Eigen::Map<const MatRMaj> matM(data + offset, nRows, nCols);
        OuterStridedCMajMatK matSrc(vec + srcOff, nCols, nRHS, OuterStride(ldVec));
        OuterStridedCMajMatM matDst(vec + dstOff, nRows, nRHS, OuterStride(ldVec));
        matDst.noalias() += alpha * (matM * matSrc);
        return;
      }

      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto vecBufferInfo = MetalBufferRegistry::instance().findBuffer(vec);
      if (!dataBufferInfo.first || !vecBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::gemvDirect: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> vecBuffer = (__bridge id<MTLBuffer>)vecBufferInfo.first;
      size_t dataBaseOffset = dataBufferInfo.second;
      size_t vecBaseOffset = vecBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
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

  void flush() override { commitAndWait(); }

  MetalSymbolicCtx& sym;
  int nRHS;
  MetalMirror<float> tempVecBuffer;
  MetalMirror<int64_t> devPivots;  // GPU buffer for LU pivots
  const int64_t* pivotsBase_ = nullptr;  // Base pointer of pre-uploaded pivots
  int64_t pivotsSize_ = 0;               // Size of pre-uploaded pivot buffer

  // Deferred sync state — batch multiple GPU dispatches into shared command buffers
  id<MTLCommandBuffer> pendingCmdBuf_ = nil;
  id<MTLComputeCommandEncoder> pendingEncoder_ = nil;
  int pendingDispatchCount_ = 0;
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
            getProfiledPipeline("factor_lumps_kernel_float");
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

      // Step 2: Sparse elimination - loop dispatch of non-batched kernel
      if (elim.numBlockPairs > 0) {
        id<MTLComputePipelineState> pipeline =
            getProfiledPipeline("sparse_elim_straight_kernel_float");
        id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];

        // Set structural buffers (same for all batch items)
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                    offset:0
                   atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                    offset:0
                   atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                    offset:0
                   atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                    offset:0
                   atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                    offset:0
                   atIndex:4];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanToLump.buffer()
                    offset:0
                   atIndex:5];
        [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanOffsetInLump.buffer()
                    offset:0
                   atIndex:6];
        [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:8];
        [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:9];
        [encoder setBuffer:(__bridge id<MTLBuffer>)elim.makeBlockPairEnumStraight.buffer()
                    offset:0
                   atIndex:10];
        [encoder setBytes:&elim.numBlockPairs length:sizeof(int64_t) atIndex:11];

        threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        threadGroupSize = MIN(threadGroupSize, (NSUInteger)elim.numBlockPairs);
        threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        numGroups = MTLSizeMake(
            ((NSUInteger)elim.numBlockPairs + threadGroupSize - 1) / threadGroupSize, 1, 1);

        // Dispatch once per batch item, changing only the data buffer
        for (int b = 0; b < batchSize; b++) {
          auto bufferInfo = MetalBufferRegistry::instance().findBuffer((*data)[b]);
          BASPACHO_CHECK_WHAT1(bufferInfo.first,
                               "batched doElimination: data buffer not found");
          [encoder setBuffer:(__bridge id<MTLBuffer>)bufferInfo.first
                      offset:bufferInfo.second
                     atIndex:7];
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

#ifdef BASPACHO_USE_BLAS
      // CPU BLAS fallback: faster than MPS dispatch for small/medium matrices
      if (n <= getCpuBlasThreshold()) {
        for (int b = 0; b < batchSize; b++) {
          LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', n, (*data)[b] + offA, n);
        }
        return;
      }
#endif

      static const int64_t kMpsPotrfMinN = getMpsThreshold("BASPACHO_MPS_POTRF_MIN_N", 4);
      static const int64_t kMpsPotrfMaxN = getMpsThreshold("BASPACHO_MPS_POTRF_MAX_N", 128);

      if (n < kMpsPotrfMinN) {
        using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        for (int b = 0; b < batchSize; b++) {
          Eigen::Map<MatRMaj> matA((*data)[b] + offA, n, n);
          Eigen::LLT<Eigen::Ref<MatRMaj>> llt(matA);
          if (llt.info() != Eigen::Success) {
            fprintf(stderr, "Metal batched potrf: Cholesky failed (batch %d)\n", b);
          }
        }
        return;
      }

#ifdef BASPACHO_USE_BLAS
      if (n > kMpsPotrfMaxN) {
        // Large matrix — multi-threaded CPU BLAS is faster than MPS.
        for (int b = 0; b < batchSize; b++) {
          LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', n, (*data)[b] + offA, n);
        }
        return;
      }
#endif

      // MPS path: encode all batch items into one command buffer
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

#ifdef BASPACHO_USE_BLAS
      // CPU BLAS fallback: faster than MPS for small/medium operations
      if (n <= getCpuBlasThreshold()) {
        for (int b = 0; b < batchSize; b++) {
          float* batchData = (*data)[b];
          cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, n, k,
                      1.0f, batchData + offA, n, batchData + offB, n);
        }
        return;
      }
#endif

      static const int64_t kMpsTrsmThreshold =
          getMpsThreshold("BASPACHO_MPS_TRSM_THRESHOLD", 0);

      if ((int64_t)n * n * k < kMpsTrsmThreshold) {
        using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using MatCMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
        for (int b = 0; b < batchSize; b++) {
          float* batchData = (*data)[b];
          Eigen::Map<const MatCMaj> matA(batchData + offA, n, n);
          Eigen::Map<MatRMaj> matB(batchData + offB, k, n);
          matA.template triangularView<Eigen::Upper>()
              .template solveInPlace<Eigen::OnTheRight>(matB);
        }
        return;
      }

      // MPS path: encode all batch items into one command buffer
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

#ifdef BASPACHO_USE_BLAS
      // CPU BLAS fallback: faster than MPS for small/medium operations
      if (k <= getCpuBlasThreshold()) {
        for (int b = 0; b < batchSize; b++) {
          const float* batchData = (*data)[b];
          // C(n,m) = B(n,k) * A^T(k,m), row-major at data+offset
          // Fortran col-major: C(m,n) = A^T(m,k) * B(k,n)
          cblas_sgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, (BLAS_INT)m, (BLAS_INT)n,
                      (BLAS_INT)k, 1.0f, batchData + offset, (BLAS_INT)k,
                      batchData + offset, (BLAS_INT)k, 0.0f, tempBufPtrs[b], (BLAS_INT)m);
        }
        return;
      }
#endif

      static const int64_t kMpsThreshold =
          getMpsThreshold("BASPACHO_MPS_GEMM_THRESHOLD", 0);
      bool useMps = (m * n * k >= kMpsThreshold);

      if (useMps) {
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
      } else {
        using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        for (int b = 0; b < batchSize; b++) {
          const float* AB = (*data)[b] + offset;
          Eigen::Map<const MatRMaj> matA(AB, m, k);
          Eigen::Map<const MatRMaj> matB(AB, n, k);
          Eigen::Map<MatRMaj> matC(tempBufPtrs[b], n, m);
          matC.noalias() = matB * matA.transpose();
        }
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

      // CPU fallback: scatter-subtract per batch item (avoids GPU dispatch overhead)
      if (numBlockRows * numBlockCols <= getCpuBlasThreshold()) {
        const int64_t* chainRowsTillEnd =
            sym.skel.chainRowsTillEnd.data() + srcColDataOffset;
        const int64_t* pToSpan = sym.skel.chainRowSpan.data() + srcColDataOffset;
        const int64_t* pSpanToChainOffset = spanToChainOffset.data();
        const int64_t* pSpanOffsetInLump = sym.skel.spanOffsetInLump.data();
        for (int b = 0; b < batchSize; b++) {
          float* batchData = (*data)[b];
          const float* matRectPtr = tempBufPtrs[b];
          for (int64_t r = 0; r < numBlockRows; r++) {
            int64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
            int64_t rSize = chainRowsTillEnd[r] - rBegin - rectRowBegin;
            int64_t rParam = pToSpan[r];
            int64_t rOffset = pSpanToChainOffset[rParam];
            const float* matRowPtr = matRectPtr + rBegin * srcRectWidth;
            int64_t cEnd = std::min(numBlockCols, r + 1);
            for (int64_t c = 0; c < cEnd; c++) {
              int64_t cStart = chainRowsTillEnd[c - 1] - rectRowBegin;
              int64_t cSize = chainRowsTillEnd[c] - cStart - rectRowBegin;
              int64_t offset = rOffset + pSpanOffsetInLump[pToSpan[c]];
              float* dst = batchData + offset;
              const float* src = matRowPtr + cStart;
              for (int64_t i = 0; i < rSize; i++)
                for (int64_t j = 0; j < cSize; j++)
                  dst[i * dstStride + j] -= src[i * srcRectWidth + j];
            }
          }
        }
        return;
      }

      // GPU path: upload spanToChainOffset if not yet done
      if (!gpuAssemblyUsed_) {
        memcpy(devSpanToChainOffset.ptr(), spanToChainOffset.data(),
               spanToChainOffset.size() * sizeof(int64_t));
        gpuAssemblyUsed_ = true;
      }

      id<MTLComputePipelineState> pipeline = getProfiledPipeline("assemble_kernel_float");

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
            getProfiledPipeline("sparseElim_diagSolveL_float");
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
            getProfiledPipeline("sparseElim_subDiagMult_float");
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
      {
        id<MTLComputePipelineState> pipeline =
            getProfiledPipeline("sparseElim_subDiagMultT_float");
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
            getProfiledPipeline("sparseElim_diagSolveLt_float");
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

      id<MTLComputePipelineState> pipeline = getProfiledPipeline("assembleVec_kernel_float");

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

      id<MTLComputePipelineState> pipeline = getProfiledPipeline("assembleVecT_kernel_float");

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
