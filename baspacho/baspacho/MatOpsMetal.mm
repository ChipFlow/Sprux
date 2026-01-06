/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <chrono>
#include <iostream>
#include <typeindex>

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

// Synchronization ops for Metal
struct MetalSyncOps {
  static void sync() { MetalContext::instance().synchronize(); }
};

// Symbolic elimination context for Metal
struct MetalSymElimCtx : SymElimCtx {
  MetalSymElimCtx() {}
  virtual ~MetalSymElimCtx() override {}

  // GPU data for elimination
  int64_t numColumns;
  int64_t numBlockPairs;
  MetalMirror<int64_t> makeBlockPairEnumStraight;

  // CPU data for sparse elimination solve (same as CpuBaseSymElimCtx)
  // Needed for the below-diagonal update loop in solve
  int64_t spanRowBegin;
  int64_t maxBufferSize;
  std::vector<int64_t> rowPtr;       // row data pointer (CSR format)
  std::vector<int64_t> colLump;      // column lump for each entry
  std::vector<int64_t> chainColOrd;  // order in column chain elements
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

    // GPU data: block pair enumeration for sparse elimination kernel
    vector<int64_t> makeStraight(lumpsEnd - lumpsBegin + 1);
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

    // CPU data: needed for sparse elimination solve update loop (same as CpuBaseSymbolicCtx)
    int64_t spanRowBegin = skel.lumpToSpan[lumpsEnd];
    int64_t numSpanRows = skel.spanStart.size() - 1 - spanRowBegin;
    elim->spanRowBegin = spanRowBegin;
    elim->rowPtr.assign(numSpanRows + 1, 0);

    // Count entries per row
    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      for (int64_t i = skel.chainColPtr[l], iEnd = skel.chainColPtr[l + 1]; i < iEnd; i++) {
        int64_t s = skel.chainRowSpan[i];
        if (s < spanRowBegin) {
          continue;
        }
        int64_t sRel = s - spanRowBegin;
        elim->rowPtr[sRel]++;
      }
    }
    int64_t totNumChains = cumSumVec(elim->rowPtr);
    elim->colLump.resize(totNumChains);
    elim->chainColOrd.resize(totNumChains);

    // Fill in column and chain order data (must match CpuBaseSymbolicCtx exactly)
    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      for (int64_t iBegin = skel.chainColPtr[l], iEnd = skel.chainColPtr[l + 1], i = iBegin;
           i < iEnd; i++) {
        int64_t s = skel.chainRowSpan[i];
        if (s < spanRowBegin) {
          continue;
        }
        int64_t sRel = s - spanRowBegin;
        elim->colLump[elim->rowPtr[sRel]] = l;
        elim->chainColOrd[elim->rowPtr[sRel]] = i - iBegin;
        elim->rowPtr[sRel]++;  // Post-increment to fill
      }
    }
    rewindVec(elim->rowPtr);  // Restore starting positions after filling

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

// Helper to dispatch a Metal compute kernel using shared command buffer
static void dispatchKernel(id<MTLComputePipelineState> pipeline,
                           void (^encodeBlock)(id<MTLComputeCommandEncoder>), NSUInteger numThreads) {
  @autoreleasepool {
    // Use shared command buffer for batching
    id<MTLCommandBuffer> cmdBuf =
        (__bridge id<MTLCommandBuffer>)MetalContext::instance().getCommandBuffer();
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
    // No commit - batched in shared command buffer, committed on synchronize()
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

  virtual ~MetalNumericCtx() override {}

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
      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "factor_spans_kernel_float");

      int64_t numSpans = spanEnd - spanBegin;
      if (numSpans <= 0) return;

      dispatchKernel(
          pipeline,
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
            pipeline,
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
            pipeline,
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

      // Find the MTLBuffer for data
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::potrf: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      // BaSpaCho stores data row-major but MPS Cholesky expects column-major!
      // For a symmetric matrix, row-major lower = column-major upper (transpose).
      // So we use lower:NO and MPS will compute Cholesky on upper triangle,
      // which gives us the transpose of L in our row-major storage.
      // Then we need to transpose it back to get L in row-major lower.
      //
      // Alternative: Fall back to CPU for now until we implement proper transposition.
      // CPU fallback for correctness - MPS layout issues need more investigation.
      {
        // CRITICAL: Synchronize to ensure any pending GPU work is complete
        // before reading data for CPU operations. This ensures cache coherency
        // on unified memory systems.
        MetalContext::instance().synchronize();

        using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<MatRMaj> mat(data + offA, n, n);

        // Debug: check the matrix before factorization
        float diag0 = mat(0, 0);
        float offdiag = (n > 1) ? mat(1, 0) : 0.0f;
        static int potrf_count = 0;
        if (potrf_count < 5 || (potrf_count % 1000 == 0)) {
          NSLog(@"potrf[%d]: n=%lld, offA=%lld, diag[0]=%.6f, mat[1,0]=%.6f",
                potrf_count, (long long)n, (long long)offA, diag0, offdiag);
        }
        potrf_count++;

        Eigen::LLT<Eigen::Ref<MatRMaj>> llt(mat);
        if (llt.info() != Eigen::Success) {
          // Debug: print more info about the failure
          NSLog(@"potrf FAILED: n=%lld, offA=%lld, diag[0]=%.6f, llt.info=%d",
                (long long)n, (long long)offA, diag0, (int)llt.info());
          // Print first few elements of the matrix
          NSLog(@"Matrix first row: ");
          for (int i = 0; i < std::min((int64_t)5, n); ++i) {
            NSLog(@"  [0,%d]=%.6f", i, mat(0, i));
          }
          throw std::runtime_error("MetalNumericCtx<float>::potrf: Cholesky failed");
        }
        // LLT writes L to lower triangle, which is what we want
        return;
      }

      // MPS Cholesky - fully async, batched with other GPU ops
      MPSMatrixDescriptor* descA =
          [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                columns:n
                                               rowBytes:n * sizeof(float)
                                               dataType:MPSDataTypeFloat32];

      MPSMatrix* mpsA = [[MPSMatrix alloc] initWithBuffer:dataBuffer
                                                   offset:dataBaseOffset + offA * sizeof(float)
                                               descriptor:descA];

      MPSMatrixDecompositionCholesky* cholesky =
          [[MPSMatrixDecompositionCholesky alloc] initWithDevice:sym.device
                                                           lower:YES
                                                           order:n];

      // Use shared command buffer for batching
      id<MTLCommandBuffer> cmdBuf =
          (__bridge id<MTLCommandBuffer>)MetalContext::instance().getCommandBuffer();
      [cholesky encodeToCommandBuffer:cmdBuf
                         sourceMatrix:mpsA
                         resultMatrix:mpsA
                               status:nil];
      // No commit - batched in shared command buffer, committed on synchronize()
    }
  }

  virtual void trsm(int64_t n, int64_t k, float* data, int64_t offA, int64_t offB) override {
    @autoreleasepool {
      if (n <= 0 || k <= 0) return;

      // CPU fallback for correctness - MPS layout issues need more investigation.
      // Solve: B = B * L^{-T} where L is lower triangular at offA
      // B is at offB with dimensions (k rows, n cols), stored row-major
      // Derivation: If X * L^T = B, take transpose: L * X^T = B^T
      // So X^T = L^{-1} * B^T, and X = (L^{-1} * B^T)^T
      {
        using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using MatCMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
        // L is n x n lower triangular, stored row-major
        Eigen::Map<const MatRMaj> matL(data + offA, n, n);
        // B is k x n, stored row-major
        Eigen::Map<MatRMaj> matB(data + offB, k, n);
        // Solve: X = (L^{-1} * B^T)^T
        MatCMaj Bt = matB.transpose();  // B^T
        matL.template triangularView<Eigen::Lower>().solveInPlace(Bt);  // L^{-1} * B^T
        matB = Bt.transpose();  // (L^{-1} * B^T)^T = B * L^{-T}
        return;
      }

      // Find the MTLBuffer for data
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::trsm: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      // MPS triangular solve - fully async
      // Solve: B = B * L^{-T} where L is lower triangular at offA
      // B is at offB with dimensions (k rows, n cols)
      MPSMatrixDescriptor* descL =
          [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                columns:n
                                               rowBytes:n * sizeof(float)
                                               dataType:MPSDataTypeFloat32];

      MPSMatrixDescriptor* descB =
          [MPSMatrixDescriptor matrixDescriptorWithRows:k
                                                columns:n
                                               rowBytes:n * sizeof(float)
                                               dataType:MPSDataTypeFloat32];

      MPSMatrix* mpsL = [[MPSMatrix alloc] initWithBuffer:dataBuffer
                                                   offset:dataBaseOffset + offA * sizeof(float)
                                               descriptor:descL];
      MPSMatrix* mpsB = [[MPSMatrix alloc] initWithBuffer:dataBuffer
                                                   offset:dataBaseOffset + offB * sizeof(float)
                                               descriptor:descB];

      // Solve X * L^T = B (right side, transpose of lower triangular)
      MPSMatrixSolveTriangular* solve =
          [[MPSMatrixSolveTriangular alloc] initWithDevice:sym.device
                                                     right:YES
                                                     upper:NO
                                                 transpose:YES
                                                      unit:NO
                                                     order:n
                                          numberOfRightHandSides:k
                                                     alpha:1.0];

      // Use shared command buffer for batching
      id<MTLCommandBuffer> cmdBuf =
          (__bridge id<MTLCommandBuffer>)MetalContext::instance().getCommandBuffer();
      [solve encodeToCommandBuffer:cmdBuf
                      sourceMatrix:mpsL
               rightHandSideMatrix:mpsB
                    solutionMatrix:mpsB];
      // No commit - batched in shared command buffer, committed on synchronize()
    }
  }

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const float* data,
                            int64_t offset) override {
    @autoreleasepool {
      if (m <= 0 || n <= 0 || k <= 0) return;

      // Ensure temp buffer is large enough
      tempBuffer.resizeToAtLeast(m * n);

      // CPU fallback for all sizes to avoid MPS layout issues
      // TODO: Implement correct MPS path with proper row-major handling
      bool useMps = false;  // Disabled for correctness

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

        // Use shared command buffer for batching
        id<MTLCommandBuffer> cmdBuf =
            (__bridge id<MTLCommandBuffer>)MetalContext::instance().getCommandBuffer();
        [gemm encodeToCommandBuffer:cmdBuf leftMatrix:mpsB rightMatrix:mpsA resultMatrix:mpsC];
        // No commit - batched in shared command buffer, committed on synchronize()
      } else {
        // Use Eigen for small matrices (lower overhead)
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

  // Helper for CPU fallback: subtract strided matrix
  static inline void stridedMatSub(float* dst, int64_t dstStride, const float* src,
                                   int64_t srcStride, int64_t rSize, int64_t cSize) {
    for (int64_t j = 0; j < rSize; j++) {
      for (int64_t i = 0; i < cSize; i++) {
        dst[j * dstStride + i] -= src[j * srcStride + i];
      }
    }
  }

  virtual void assemble(float* data, int64_t rectRowBegin, int64_t dstStride,
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    @autoreleasepool {
      if (numBlockRows <= 0 || numBlockCols <= 0) return;

      // CPU fallback for correctness - Metal kernel has issues
      {
        // Synchronize to ensure previous GPU work (saveSyrkGemm) is complete
        MetalContext::instance().synchronize();

        const CoalescedBlockMatrixSkel& skel = sym.skel;
        const int64_t* chainRowsTillEnd = skel.chainRowsTillEnd.data() + srcColDataOffset;
        const int64_t* pToSpan = skel.chainRowSpan.data() + srcColDataOffset;
        const int64_t* pSpanToChainOffset = spanToChainOffset.data();
        const int64_t* pSpanOffsetInLump = skel.spanOffsetInLump.data();
        const float* matRectPtr = tempBuffer.ptr();  // Direct access to unified memory

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
            stridedMatSub(dst, dstStride, src, srcRectWidth, rSize, cSize);
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
      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "assemble_kernel_float");

      int64_t numThreads = numBlockRows * numBlockCols;

      // Compute startRow = chainRowsTillEnd[srcColDataOffset - 1] (element before chain start)
      // This matches CPU ref where startRow = pChainRowsTillEnd[-1] after offsetting the pointer
      int64_t startRow = (srcColDataOffset > 0) ? sym.skel.chainRowsTillEnd[srcColDataOffset - 1] : 0;

      dispatchKernel(
          pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBytes:&numBlockRows length:sizeof(int64_t) atIndex:0];
            [encoder setBytes:&numBlockCols length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&startRow length:sizeof(int64_t) atIndex:2];
            [encoder setBytes:&srcRectWidth length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&dstStride length:sizeof(int64_t) atIndex:4];
            // pChainRowsTillEnd offset by srcColDataOffset
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

  MetalSymbolicCtx& sym;
  int64_t numSpans_;
  MetalMirror<float> tempBuffer;
  MetalMirror<int64_t> devSpanToChainOffset;
  std::vector<int64_t> spanToChainOffset;
};

// Solve context for float - Metal implementation
template <>
struct MetalSolveCtx<float> : SolveCtx<float> {
  MetalSolveCtx(MetalSymbolicCtx& sym_, int nRHS_) : sym(sym_), nRHS(nRHS_) {
    // Allocate temp buffer for vector assembly
    tempVecBuffer.resizeToAtLeast(sym.skel.order() * nRHS);
  }

  virtual ~MetalSolveCtx() override {}

  // CPU fallback for sparseElimSolveL - matches MatOpsFast.cpp implementation
  void sparseElimSolveL_cpu(const MetalSymElimCtx& elim, const float* data, int64_t lumpsBegin,
                            int64_t lumpsEnd, float* C, int64_t ldc) {
    using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
    using OuterStridedCMajMatM =
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                   OuterStride>;

    const CoalescedBlockMatrixSkel& skel = sym.skel;

    // Part 1: Diagonal solves for each lump
    for (int64_t lump = lumpsBegin; lump < lumpsEnd; lump++) {
      int64_t lumpStart = skel.lumpStart[lump];
      int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
      int64_t colStart = skel.chainColPtr[lump];
      int64_t diagDataPtr = skel.chainData[colStart];

      Eigen::Map<const MatRMaj> diagBlock(data + diagDataPtr, lumpSize, lumpSize);
      OuterStridedCMajMatM matC(C + lumpStart, lumpSize, nRHS, OuterStride(ldc));
      diagBlock.template triangularView<Eigen::Lower>().solveInPlace(matC);
    }

    // Part 2: Below-diagonal updates using elimination context
    int64_t numElimRows = elim.rowPtr.size() - 1;
    for (int64_t sRel = 0L; sRel < numElimRows; sRel++) {
      int64_t rowSpan = sRel + elim.spanRowBegin;
      int64_t rowSpanStart = skel.spanStart[rowSpan];
      int64_t rowSpanSize = skel.spanStart[rowSpan + 1] - rowSpanStart;
      OuterStridedCMajMatM matQ(C + rowSpanStart, rowSpanSize, nRHS, OuterStride(ldc));

      for (int64_t i = elim.rowPtr[sRel], iEnd = elim.rowPtr[sRel + 1]; i < iEnd; i++) {
        int64_t lump = elim.colLump[i];
        int64_t lumpStart = skel.lumpStart[lump];
        int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
        int64_t chainColOrd = elim.chainColOrd[i];

        int64_t ptr = skel.chainColPtr[lump] + chainColOrd;
        int64_t blockPtr = skel.chainData[ptr];

        Eigen::Map<const MatRMaj> block(data + blockPtr, rowSpanSize, lumpSize);
        OuterStridedCMajMatM matC(C + lumpStart, lumpSize, nRHS, OuterStride(ldc));
        matQ.noalias() -= block * matC;
      }
    }
  }

  virtual void sparseElimSolveL(const SymElimCtx& elimData, const float* data, int64_t lumpsBegin,
                                int64_t lumpsEnd, float* C, int64_t ldc) override {
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);
      const MetalSymElimCtx& elim = *pElim;

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      // Use CPU fallback - includes both diagonal solve and update loop
      bool useCpuFallback = true;
      if (useCpuFallback) {
        MetalContext::instance().synchronize();  // Ensure GPU work is done
        sparseElimSolveL_cpu(elim, data, lumpsBegin, lumpsEnd, C, ldc);
        NSLog(@"sparseElimSolveL CPU: lumps %lld-%lld", (long long)lumpsBegin, (long long)lumpsEnd);
        return;
      }

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

      // Dispatch diagonal solve kernel
      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "sparseElim_diagSolveL_float");

      int64_t nRHS64 = nRHS;
      dispatchKernel(
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

      // Synchronize before returning - CPU operations (solveL) may follow immediately
      // and need to see the GPU-modified data.
      NSLog(@"sparseElimSolveL: syncing after GPU kernels, lumps %lld-%lld", (long long)lumpsBegin, (long long)lumpsEnd);
      MetalContext::instance().synchronize();
    }
  }

  // CPU fallback for sparseElimSolveLt - matches MatOpsFast.cpp implementation
  void sparseElimSolveLt_cpu(const float* data, int64_t lumpsBegin, int64_t lumpsEnd, float* C,
                             int64_t ldc) {
    using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
    using OuterStridedCMajMatM =
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                   OuterStride>;

    const CoalescedBlockMatrixSkel& skel = sym.skel;

    for (int64_t lump = lumpsBegin; lump < lumpsEnd; lump++) {
      int64_t lumpStart = skel.lumpStart[lump];
      int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
      int64_t colStart = skel.chainColPtr[lump];
      int64_t colEnd = skel.chainColPtr[lump + 1];
      OuterStridedCMajMatM matC(C + lumpStart, lumpSize, nRHS, OuterStride(ldc));

      // Part 1: Below-diagonal updates - done BEFORE diagonal solve
      for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
        int64_t rowSpan = skel.chainRowSpan[colPtr];
        int64_t rowSpanStart = skel.spanStart[rowSpan];
        int64_t rowSpanSize = skel.spanStart[rowSpan + 1] - rowSpanStart;
        int64_t blockPtr = skel.chainData[colPtr];
        Eigen::Map<const MatRMaj> block(data + blockPtr, rowSpanSize, lumpSize);
        OuterStridedCMajMatM matQ(C + rowSpanStart, rowSpanSize, nRHS, OuterStride(ldc));
        matC.noalias() -= block.transpose() * matQ;
      }

      // Part 2: Diagonal solve with L^T (adjoint of lower triangular)
      int64_t diagDataPtr = skel.chainData[colStart];
      Eigen::Map<const MatRMaj> diagBlock(data + diagDataPtr, lumpSize, lumpSize);
      diagBlock.template triangularView<Eigen::Lower>().adjoint().solveInPlace(matC);
    }
  }

  virtual void sparseElimSolveLt(const SymElimCtx& elimData, const float* data, int64_t lumpsBegin,
                                 int64_t lumpsEnd, float* C, int64_t ldc) override {
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      // Use CPU fallback for debugging
      bool useCpuFallback = true;
      if (useCpuFallback) {
        MetalContext::instance().synchronize();  // Ensure GPU work is done
        sparseElimSolveLt_cpu(data, lumpsBegin, lumpsEnd, C, ldc);
        NSLog(@"sparseElimSolveLt CPU: lumps %lld-%lld", (long long)lumpsBegin, (long long)lumpsEnd);
        return;
      }

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

      // Dispatch diagonal solve kernel
      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "sparseElim_diagSolveLt_float");

      int64_t nRHS64 = nRHS;
      dispatchKernel(
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

      // Synchronize before returning - CPU operations (solveLt) may follow immediately
      // and need to see the GPU-modified data.
      NSLog(@"sparseElimSolveLt: syncing after GPU kernels, lumps %lld-%lld", (long long)lumpsBegin, (long long)lumpsEnd);
      MetalContext::instance().synchronize();
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

      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "assembleVec_kernel_float");

      // startRow = chainRowsTillEnd[chainColPtr - 1] (element before chain start)
      int64_t startRow =
          (chainColPtr > 0) ? sym.devChainRowsTillEnd.ptr()[chainColPtr - 1] : 0;

      int64_t nRHS64 = nRHS;
      dispatchKernel(
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

      // Synchronize to ensure GPU kernel completes before subsequent CPU operations
      // (solveL, gemv) read from the C buffer. Without this, the solve produces wrong results.
      MetalContext::instance().synchronize();
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

      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "assembleVecT_kernel_float");

      // startRow = chainRowsTillEnd[chainColPtr - 1] (element before chain start)
      int64_t startRow =
          (chainColPtr > 0) ? sym.devChainRowsTillEnd.ptr()[chainColPtr - 1] : 0;

      int64_t nRHS64 = nRHS;
      dispatchKernel(
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

      // Synchronize to ensure GPU kernel completes before subsequent CPU operations
      // (gemvT) read from the tempVecBuffer. Without this, the solve produces wrong results.
      MetalContext::instance().synchronize();
    }
  }

  MetalSymbolicCtx& sym;
  int nRHS;
  MetalMirror<float> tempVecBuffer;
};

// Batched numeric context for float
template <>
struct MetalNumericCtx<std::vector<float*>> : NumericCtx<std::vector<float*>> {
  MetalNumericCtx(MetalSymbolicCtx& sym_, int64_t tempBufSize, int64_t numSpans, int batchSize_)
      : sym(sym_), numSpans_(numSpans), batchSize(batchSize_) {
    tempBuffer.resizeToAtLeast(tempBufSize * batchSize);
  }

  virtual ~MetalNumericCtx() override {}

  virtual void pseudoFactorSpans(std::vector<float*>* data, int64_t spanBegin,
                                 int64_t spanEnd) override {
    throw std::runtime_error(
        "MetalNumericCtx<std::vector<float*>>::pseudoFactorSpans not yet implemented");
  }

  virtual void doElimination(const SymElimCtx& elimData, std::vector<float*>* data,
                             int64_t lumpsBegin, int64_t lumpsEnd) override {
    throw std::runtime_error(
        "MetalNumericCtx<std::vector<float*>>::doElimination not yet implemented");
  }

  virtual void potrf(int64_t n, std::vector<float*>* data, int64_t offA) override {
    throw std::runtime_error("MetalNumericCtx<std::vector<float*>>::potrf not yet implemented");
  }

  virtual void trsm(int64_t n, int64_t k, std::vector<float*>* data, int64_t offA,
                    int64_t offB) override {
    throw std::runtime_error("MetalNumericCtx<std::vector<float*>>::trsm not yet implemented");
  }

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const std::vector<float*>* data,
                            int64_t offset) override {
    throw std::runtime_error(
        "MetalNumericCtx<std::vector<float*>>::saveSyrkGemm not yet implemented");
  }

  virtual void prepareAssemble(int64_t targetLump) override {}

  virtual void assemble(std::vector<float*>* data, int64_t rectRowBegin, int64_t dstStride,
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    throw std::runtime_error("MetalNumericCtx<std::vector<float*>>::assemble not yet implemented");
  }

  MetalSymbolicCtx& sym;
  int64_t numSpans_;
  int batchSize;
  MetalMirror<float> tempBuffer;
};

// Batched solve context for float
template <>
struct MetalSolveCtx<std::vector<float*>> : SolveCtx<std::vector<float*>> {
  MetalSolveCtx(MetalSymbolicCtx& sym_, int nRHS_, int batchSize_)
      : sym(sym_), nRHS(nRHS_), batchSize(batchSize_) {}

  virtual ~MetalSolveCtx() override {}

  virtual void sparseElimSolveL(const SymElimCtx& elimData, const std::vector<float*>* data,
                                int64_t lumpsBegin, int64_t lumpsEnd, std::vector<float*>* C,
                                int64_t ldc) override {
    throw std::runtime_error(
        "MetalSolveCtx<std::vector<float*>>::sparseElimSolveL not yet implemented");
  }

  virtual void sparseElimSolveLt(const SymElimCtx& elimData, const std::vector<float*>* data,
                                 int64_t lumpsBegin, int64_t lumpsEnd, std::vector<float*>* C,
                                 int64_t ldc) override {
    throw std::runtime_error(
        "MetalSolveCtx<std::vector<float*>>::sparseElimSolveLt not yet implemented");
  }

  virtual void symm(const std::vector<float*>* data, int64_t offset, int64_t n,
                    const std::vector<float*>* C, int64_t offC, int64_t ldc, std::vector<float*>* D,
                    int64_t ldd, float alpha) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::symm not yet implemented");
  }

  virtual void solveL(const std::vector<float*>* data, int64_t offset, int64_t n,
                      std::vector<float*>* C, int64_t offC, int64_t ldc) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::solveL not yet implemented");
  }

  virtual void gemv(const std::vector<float*>* data, int64_t offset, int64_t nRows, int64_t nCols,
                    const std::vector<float*>* A, int64_t offA, int64_t lda, float alpha) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::gemv not yet implemented");
  }

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, std::vector<float*>* C,
                           int64_t ldc) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::assembleVec not yet implemented");
  }

  virtual void solveLt(const std::vector<float*>* data, int64_t offset, int64_t n,
                       std::vector<float*>* C, int64_t offC, int64_t ldc) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::solveLt not yet implemented");
  }

  virtual void gemvT(const std::vector<float*>* data, int64_t offset, int64_t nRows, int64_t nCols,
                     std::vector<float*>* A, int64_t offA, int64_t lda, float alpha) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::gemvT not yet implemented");
  }

  virtual void assembleVecT(const std::vector<float*>* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::assembleVecT not yet implemented");
  }

  MetalSymbolicCtx& sym;
  int nRHS;
  int batchSize;
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
