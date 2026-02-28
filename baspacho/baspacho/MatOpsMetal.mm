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
#include <chrono>
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

namespace BaSpaCho {

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

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

// Synchronization ops for Metal
struct MetalSyncOps {
  static void sync() { MetalContext::instance().synchronize(); }
};

// Symbolic elimination context for Metal
struct MetalSymElimCtx : SymElimCtx {
  MetalSymElimCtx() {}
  virtual ~MetalSymElimCtx() override {}

  int64_t numColumns;
  int64_t numBlockPairs;
  MetalMirror<int64_t> makeBlockPairEnumStraight;
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

  // Commit the pending command buffer and wait for all GPU work to complete.
  // This is needed when the CPU must access data that the GPU may have written
  // (e.g., before CPU fallback paths or memcpy to shared buffers).
  void commitAndWait() {
    flushPendingGemms();
    commitPending();
    waitForGpu();
  }

  // Flush buffered saveGemm work items as a single batched kernel dispatch.
  // Must wait for GPU to finish reading devGemmWorkBuf_ before overwriting it.
  void flushPendingGemms() {
    if (pendingGemms_.empty()) return;

    // Wait for GPU to finish reading from devGemmWorkBuf_ before overwriting
    if (gemmWorkBufInFlight_) {
      commitPending();
      waitForGpu();
      gemmWorkBufInFlight_ = false;
    }

    int64_t count = (int64_t)pendingGemms_.size();
    size_t bytesNeeded = count * sizeof(LUGemmWorkItem);
    size_t int64sNeeded = (bytesNeeded + sizeof(int64_t) - 1) / sizeof(int64_t);
    devGemmWorkBuf_.resizeToAtLeast(int64sNeeded);

    // Copy work items to GPU buffer (safe — GPU is not reading it now)
    memcpy(devGemmWorkBuf_.ptr(), pendingGemms_.data(), bytesNeeded);

    id<MTLComputePipelineState> pipeline = getProfiledPipeline(
            "lu_batchedSaveGemm_kernel_float");

    encodeKernel(
        pipeline,
        ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:cachedDataBuffer_ offset:0 atIndex:0];
          [encoder setBuffer:(__bridge id<MTLBuffer>)devGemmWorkBuf_.buffer()
                      offset:0
                     atIndex:1];
          [encoder setBytes:&count length:sizeof(int64_t) atIndex:2];
        },
        (NSUInteger)count);

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

  virtual void potrf(int64_t n, float* data, int64_t offA) override {
    @autoreleasepool {
      if (n <= 0) return;

      static const int64_t kMpsPotrfMinN =
          getMpsThreshold("BASPACHO_MPS_POTRF_MIN_N", 32);

      if (n < kMpsPotrfMinN) {
        // Small matrix — CPU Eigen is faster than MPS dispatch overhead
        commitAndWait();
        using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<MatRMaj> matA(data + offA, n, n);
        Eigen::LLT<Eigen::Ref<MatRMaj>> llt(matA);
        if (llt.info() != Eigen::Success) {
          fprintf(stderr, "Metal potrf: Cholesky failed\n");
        }
        return;
      }

      // Flush pending work, end compute encoder (MPS needs its own encoding)
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

      static const int64_t kMpsTrsmThreshold =
          getMpsThreshold("BASPACHO_MPS_TRSM_THRESHOLD", 64 * 64 * 64);

      if ((int64_t)n * n * k < kMpsTrsmThreshold) {
        // Small — CPU Eigen is faster than MPS dispatch overhead
        commitAndWait();
        using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using MatCMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
        // col-major's upper = (row-major's lower).transpose()
        Eigen::Map<const MatCMaj> matA(data + offA, n, n);
        Eigen::Map<MatRMaj> matB(data + offB, k, n);
        matA.template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(matB);
        return;
      }

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

      // Submit pending assemble work (which reads tempBuffer) without waiting.
      // Metal command queue ordering ensures it completes before the GEMM below.
      commitPending();

      // Ensure temp buffer is large enough
      tempBuffer.resizeToAtLeast(m * n);

      // Use MPS for larger matrices (threshold based on empirical testing)
      // MPS dispatch overhead makes it slower for small matrices
      static const int64_t kMpsThreshold =
          getMpsThreshold("BASPACHO_MPS_GEMM_THRESHOLD", 64 * 64 * 64);
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
        // CPU fallback — need GPU to finish before CPU accesses shared memory
        waitForGpu();

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

      // Profiling fallback: use sync dispatch for per-kernel GPU timestamps
      if (metalProfilingEnabled()) {
        id<MTLComputePipelineState> pipeline = getProfiledPipeline(
                "lu_getrf_kernel_float");
        devPivots.resizeToAtLeast(minMN);
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

      // Commit and wait — need CPU-visible data for pivot conversion below
      commitPending();
      waitForGpu();

      // Convert MPS uint32_t pivots → int64_t (MPS is 0-based, same as BaSpaCho)
      uint32_t* mpsPivots = devPivotBuf32.ptr();
      for (int64_t i = 0; i < minMN; i++) {
        pivots[i] = static_cast<int64_t>(mpsPivots[i]);
      }

      return 0;
    }
  }

  virtual void trsmLowerUnit(int64_t m, int64_t n, const float* L, int64_t offL, float* B,
                              int64_t offB, int64_t ldb) override {
    @autoreleasepool {
      if (m <= 0 || n <= 0) return;

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

      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::applyRowPerm: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
              "lu_applyRowPerm_kernel_float");

      // Copy pivots to GPU buffer (shared memory, safe: all applyRowPerm calls
      // within a lump use the same pivot data, and GPU hasn't started reading
      // until the next commit in getrf's MPS path)
      devPivots.resizeToAtLeast(n);
      memcpy(devPivots.ptr(), pivots, n * sizeof(int64_t));

      // Dispatch as single threadgroup with min(256, numCols) threads
      // (threadgroup_barrier in kernel requires single threadgroup)
      NSUInteger numThreads = (NSUInteger)std::min((int64_t)256, numCols);

      if (metalProfilingEnabled()) {
        dispatchKernel(
            sym.commandQueue, pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:(__bridge id<MTLBuffer>)devPivots.buffer() offset:0 atIndex:0];
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
            [encoder setBuffer:(__bridge id<MTLBuffer>)devPivots.buffer() offset:0 atIndex:0];
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
    gemmWorkBufInFlight_ = false;
    MetalContext::instance().synchronize();
  }

  MetalSymbolicCtx& sym;
  int64_t numSpans_;
  MetalMirror<float> tempBuffer;
  MetalMirror<int64_t> devSpanToChainOffset;
  std::vector<int64_t> spanToChainOffset;
  MetalMirror<int64_t> devPivots;        // GPU buffer for LU pivots (profiling path)
  MetalMirror<uint32_t> devPivotBuf32;  // GPU buffer for MPS LU pivot output (uint32_t format)
  id<MTLBuffer> potrfStatusBuf_ = nil;  // Cached status buffer for MPS Cholesky
  bool assembleWasCalled_ = false;      // Track whether assemble() was called

  // Batched saveGemm state
  std::vector<LUGemmWorkItem> pendingGemms_;   // CPU-side work item accumulator
  MetalMirror<int64_t> devGemmWorkBuf_;        // GPU buffer for work items (reused per flush)
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

  virtual ~MetalSolveCtx() override {}

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

      // Dispatch diagonal solve kernel
      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
              "sparseElim_diagSolveL_float");

      int64_t nRHS64 = nRHS;
      dispatchKernel(
          sym.commandQueue, pipeline,
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

      // Dispatch below-diagonal multiply: matQ -= block * matC
      id<MTLComputePipelineState> subDiagPipeline = getProfiledPipeline(
              "sparseElim_subDiagMult_float");

      dispatchKernel(
          sym.commandQueue, subDiagPipeline,
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

      // Dispatch below-diagonal transpose multiply first: matC -= block^T * matQ
      id<MTLComputePipelineState> subDiagPipeline = getProfiledPipeline(
              "sparseElim_subDiagMultT_float");

      dispatchKernel(
          sym.commandQueue, subDiagPipeline,
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

      // Then dispatch diagonal solve kernel
      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
              "sparseElim_diagSolveLt_float");

      dispatchKernel(
          sym.commandQueue, pipeline,
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
      dispatchKernel(
          sym.commandQueue, pipeline,
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
      dispatchKernel(
          sym.commandQueue, pipeline,
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

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
              "lu_solveLUnit_direct_kernel_float");

      int64_t nRHS64 = nRHS;
      dispatchKernel(
          sym.commandQueue, pipeline,
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
      dispatchKernel(
          sym.commandQueue, pipeline,
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

  // Apply row permutation P to vector: for each i, swap row i with row pivots[i]
  virtual void applyRowPermVec(const int64_t* pivots, int64_t n, float* vec,
                                int64_t ldVec) override {
    @autoreleasepool {
      if (n <= 0) return;

      // Copy pivots to GPU buffer
      devPivots.resizeToAtLeast(n);
      memcpy(devPivots.ptr(), pivots, n * sizeof(int64_t));

      auto vecBufferInfo = MetalBufferRegistry::instance().findBuffer(vec);
      if (!vecBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::applyRowPermVec: buffer not found");
      }
      id<MTLBuffer> vecBuffer = (__bridge id<MTLBuffer>)vecBufferInfo.first;
      size_t vecBaseOffset = vecBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
              "lu_applyRowPermVec_kernel_float");

      int64_t nRHS64 = nRHS;
      dispatchKernel(
          sym.commandQueue, pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)devPivots.buffer() offset:0 atIndex:0];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:1];
            [encoder setBuffer:vecBuffer offset:vecBaseOffset atIndex:2];
            [encoder setBytes:&ldVec length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:4];
          },
          1);  // Must sync because devPivots reused across calls
    }
  }

  // Apply inverse row permutation P^T to vector (reverse order)
  virtual void applyRowPermVecInv(const int64_t* pivots, int64_t n, float* vec,
                                   int64_t ldVec) override {
    @autoreleasepool {
      if (n <= 0) return;

      // Copy pivots to GPU buffer
      devPivots.resizeToAtLeast(n);
      memcpy(devPivots.ptr(), pivots, n * sizeof(int64_t));

      auto vecBufferInfo = MetalBufferRegistry::instance().findBuffer(vec);
      if (!vecBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::applyRowPermVecInv: buffer not found");
      }
      id<MTLBuffer> vecBuffer = (__bridge id<MTLBuffer>)vecBufferInfo.first;
      size_t vecBaseOffset = vecBufferInfo.second;

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
              "lu_applyRowPermVecInv_kernel_float");

      int64_t nRHS64 = nRHS;
      dispatchKernel(
          sym.commandQueue, pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)devPivots.buffer() offset:0 atIndex:0];
            [encoder setBytes:&n length:sizeof(int64_t) atIndex:1];
            [encoder setBuffer:vecBuffer offset:vecBaseOffset atIndex:2];
            [encoder setBytes:&ldVec length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:4];
          },
          1);  // Must sync because devPivots reused across calls
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

      id<MTLComputePipelineState> pipeline = getProfiledPipeline(
              "lu_gemvDirect_kernel_float");

      int64_t nRHS64 = nRHS;
      dispatchKernel(
          sym.commandQueue, pipeline,
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

  void flush() override { MetalContext::instance().synchronize(); }

  MetalSymbolicCtx& sym;
  int nRHS;
  MetalMirror<float> tempVecBuffer;
  MetalMirror<int64_t> devPivots;  // GPU buffer for LU pivots
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

      static const int64_t kMpsPotrfMinN = getMpsThreshold("BASPACHO_MPS_POTRF_MIN_N", 32);

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

      static const int64_t kMpsTrsmThreshold =
          getMpsThreshold("BASPACHO_MPS_TRSM_THRESHOLD", 64 * 64 * 64);

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

      static const int64_t kMpsThreshold =
          getMpsThreshold("BASPACHO_MPS_GEMM_THRESHOLD", 64 * 64 * 64);
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
    if (assembleWasCalled_) {
      MetalContext::instance().synchronize();
      assembleWasCalled_ = false;
    }

    const CoalescedBlockMatrixSkel& skel = sym.skel;
    for (int64_t i = skel.chainColPtr[targetLump], iEnd = skel.chainColPtr[targetLump + 1];
         i < iEnd; i++) {
      spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
    }
    memcpy(devSpanToChainOffset.ptr(), spanToChainOffset.data(),
           spanToChainOffset.size() * sizeof(int64_t));
  }

  virtual void assemble(std::vector<float*>* data, int64_t rectRowBegin, int64_t dstStride,
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    @autoreleasepool {
      if (numBlockRows <= 0 || numBlockCols <= 0) return;
      assembleWasCalled_ = true;

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
