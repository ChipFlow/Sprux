/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <iostream>
#include <typeindex>

#include <Eigen/Dense>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/baspacho/WebGPUDefs.h"

namespace BaSpaCho {

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

// Synchronization ops for WebGPU
struct WebGPUSyncOps {
  static void sync() { WebGPUContext::instance().synchronize(); }
};

// Symbolic elimination context for WebGPU
struct WebGPUSymElimCtx : SymElimCtx {
  WebGPUSymElimCtx() {}
  virtual ~WebGPUSymElimCtx() override {}

  int64_t numColumns;
  int64_t numBlockPairs;
  WebGPUMirror<int64_t> makeBlockPairEnumStraight;
};

// Forward declarations
struct WebGPUSymbolicCtx;

template <typename T>
struct WebGPUNumericCtx;

template <typename T>
struct WebGPUSolveCtx;

// Symbolic context for WebGPU operations
struct WebGPUSymbolicCtx : SymbolicCtx {
  WebGPUSymbolicCtx(const CoalescedBlockMatrixSkel& skel_, const std::vector<int64_t>& permutation)
      : skel(skel_) {
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
  }

  virtual ~WebGPUSymbolicCtx() override {}

  virtual PermutedCoalescedAccessor deviceAccessor() override {
    PermutedCoalescedAccessor retv;
    retv.init(devSpanStart.ptr(), devSpanToLump.ptr(), devLumpStart.ptr(), devSpanOffsetInLump.ptr(),
              devChainColPtr.ptr(), devChainRowSpan.ptr(), devChainData.ptr(), devPermutation.ptr());
    return retv;
  }

  virtual SymElimCtxPtr prepareElimination(int64_t lumpsBegin, int64_t lumpsEnd) override {
    WebGPUSymElimCtx* elim = new WebGPUSymElimCtx;

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

  // Device buffers (mirrors of skeleton data)
  WebGPUMirror<int64_t> devLumpToSpan;
  WebGPUMirror<int64_t> devChainRowsTillEnd;
  WebGPUMirror<int64_t> devChainRowSpan;
  WebGPUMirror<int64_t> devSpanOffsetInLump;
  WebGPUMirror<int64_t> devLumpStart;
  WebGPUMirror<int64_t> devChainColPtr;
  WebGPUMirror<int64_t> devChainData;
  WebGPUMirror<int64_t> devBoardColPtr;
  WebGPUMirror<int64_t> devBoardChainColOrd;
  WebGPUMirror<int64_t> devSpanStart;
  WebGPUMirror<int64_t> devSpanToLump;
  WebGPUMirror<int64_t> devPermutation;
};

// WebGPU operations factory
struct WebGPUOps : Ops {
  virtual SymbolicCtxPtr createSymbolicCtx(const CoalescedBlockMatrixSkel& skel,
                                           const std::vector<int64_t>& permutation) override {
    return SymbolicCtxPtr(new WebGPUSymbolicCtx(skel, permutation));
  }
};

// Numeric context for float - WebGPU implementation
// Uses CPU fallback for BLAS operations (like Metal does for complex ops)
template <>
struct WebGPUNumericCtx<float> : NumericCtx<float> {
  WebGPUNumericCtx(WebGPUSymbolicCtx& sym_, int64_t tempBufSize, int64_t numSpans)
      : sym(sym_), numSpans_(numSpans), spanToChainOffset(numSpans) {
    tempBuffer.resizeToAtLeast(tempBufSize);
    devSpanToChainOffset.resizeToAtLeast(numSpans);
  }

  virtual ~WebGPUNumericCtx() override {}

  virtual void pseudoFactorSpans(float* data, int64_t spanBegin, int64_t spanEnd) override {
    // CPU fallback - using Eigen for now
    // Full GPU implementation would use compute shaders
    for (int64_t s = spanBegin; s < spanEnd; s++) {
      int64_t lump = sym.skel.spanToLump[s];
      int64_t spanOff = sym.skel.spanOffsetInLump[s];
      int64_t lumpSize = sym.skel.lumpStart[lump + 1] - sym.skel.lumpStart[lump];
      int64_t spanSize = sym.skel.spanStart[s + 1] - sym.skel.spanStart[s];
      int64_t colStart = sym.skel.chainColPtr[lump];
      int64_t dataPtr = sym.skel.chainData[colStart];

      // Pointer to start of this span within diagonal block
      float* spanDiag = data + dataPtr + spanOff * (lumpSize + 1);

      // Cholesky on span diagonal
      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      Eigen::Map<MatRMaj> matA(spanDiag, spanSize, lumpSize);
      auto subBlock = matA.block(0, 0, spanSize, spanSize);
      Eigen::LLT<Eigen::Ref<Eigen::MatrixXf>> llt(subBlock);
    }
  }

  virtual void doElimination(const SymElimCtx& elimData, float* data, int64_t lumpsBegin,
                             int64_t lumpsEnd) override {
    const WebGPUSymElimCtx* pElim = dynamic_cast<const WebGPUSymElimCtx*>(&elimData);
    BASPACHO_CHECK_NOTNULL(pElim);
    const WebGPUSymElimCtx& elim = *pElim;

    int64_t numLumps = lumpsEnd - lumpsBegin;
    if (numLumps <= 0) return;

    // CPU fallback for now - full GPU implementation would dispatch compute shaders
    // Step 1: Factor lumps (Cholesky on diagonal blocks + below-diagonal solve)
    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      int64_t lumpSize = sym.skel.lumpStart[l + 1] - sym.skel.lumpStart[l];
      int64_t colStart = sym.skel.chainColPtr[l];
      int64_t dataPtr = sym.skel.chainData[colStart];

      // Cholesky on diagonal block
      float* diagBlock = data + dataPtr;
      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      Eigen::Map<MatRMaj> matA(diagBlock, lumpSize, lumpSize);
      Eigen::LLT<Eigen::Ref<MatRMaj>> llt(matA);

      // Below-diagonal solve
      int64_t gatheredStart = sym.skel.boardColPtr[l];
      int64_t gatheredEnd = sym.skel.boardColPtr[l + 1];
      if (gatheredEnd > gatheredStart + 1) {
        int64_t rowDataStart = sym.skel.boardChainColOrd[gatheredStart + 1];
        int64_t rowDataEnd = sym.skel.boardChainColOrd[gatheredEnd - 1];
        int64_t belowDiagStart = sym.skel.chainData[colStart + rowDataStart];
        int64_t numRows = sym.skel.chainRowsTillEnd[colStart + rowDataEnd - 1] -
                          sym.skel.chainRowsTillEnd[colStart + rowDataStart - 1];

        if (numRows > 0) {
          float* belowDiag = data + belowDiagStart;
          Eigen::Map<MatRMaj> matB(belowDiag, numRows, lumpSize);
          using MatCMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
          Eigen::Map<MatCMaj> matL(diagBlock, lumpSize, lumpSize);
          matL.template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(
              matB);
        }
      }
    }

    // Step 2: Sparse elimination (SYRK/GEMM updates)
    // This is where the GPU kernel would be used for parallel updates
    // For now using CPU fallback via tempBuffer and assembly
  }

  virtual void potrf(int64_t n, float* data, int64_t offA) override {
    if (n <= 0) return;

    // CPU fallback using Eigen
    using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<MatRMaj> matA(data + offA, n, n);
    Eigen::LLT<Eigen::Ref<MatRMaj>> llt(matA);

    if (llt.info() != Eigen::Success) {
      fprintf(stderr, "WebGPU potrf: Cholesky failed\n");
    }
  }

  virtual void trsm(int64_t n, int64_t k, float* data, int64_t offA, int64_t offB) override {
    if (n <= 0 || k <= 0) return;

    // CPU fallback using Eigen
    using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatCMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    Eigen::Map<const MatCMaj> matA(data + offA, n, n);
    Eigen::Map<MatRMaj> matB(data + offB, k, n);
    matA.template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(matB);
  }

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const float* data,
                            int64_t offset) override {
    if (m <= 0 || n <= 0 || k <= 0) return;

    // CPU fallback using Eigen
    using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const float* srcPtr = data + offset;

    // Source is row-major: m1 rows, k columns for the "inner" part
    Eigen::Map<const MatRMaj> matL(srcPtr, m, k);

    // Compute symmetric part: temp1 = L * L^T
    Eigen::MatrixXf temp1 = matL * matL.transpose();

    // If n > m, also have a rectangular gemm part
    if (n > m) {
      Eigen::Map<const MatRMaj> matR(srcPtr + m * k, n - m, k);
      Eigen::MatrixXf temp2 = matR * matL.transpose();

      // Store to temp buffer
      float* tempPtr = tempBuffer.ptr();
      Eigen::Map<MatRMaj> dst1(tempPtr, m, m);
      dst1 = temp1;
      Eigen::Map<MatRMaj> dst2(tempPtr + m * m, n - m, m);
      dst2 = temp2;
    } else {
      float* tempPtr = tempBuffer.ptr();
      Eigen::Map<MatRMaj> dst(tempPtr, m, m);
      dst = temp1;
    }
  }

  virtual void prepareAssemble(int64_t targetLump) override {
    // Prepare chain offsets for assembly
    int64_t lumpSize = sym.skel.lumpStart[targetLump + 1] - sym.skel.lumpStart[targetLump];
    int64_t colStart = sym.skel.chainColPtr[targetLump];
    int64_t colEnd = sym.skel.chainColPtr[targetLump + 1];

    for (int64_t c = colStart; c < colEnd; c++) {
      int64_t span = sym.skel.chainRowSpan[c];
      spanToChainOffset[span] = sym.skel.chainData[c];
    }
  }

  virtual void assemble(float* data, int64_t rectRowBegin, int64_t dstStride, int64_t srcColDataOffset,
                        int64_t srcRectWidth, int64_t numBlockRows, int64_t numBlockCols) override {
    // CPU fallback - copy from temp buffer to destination with proper strides
    const float* tempPtr = tempBuffer.ptr();

    for (int64_t r = 0; r < numBlockRows; r++) {
      for (int64_t c = 0; c <= r && c < numBlockCols; c++) {
        // Get block bounds (simplified - actual implementation needs chain lookup)
        // This is a placeholder - full implementation requires chain traversal
      }
    }
  }

  WebGPUSymbolicCtx& sym;
  int64_t numSpans_;
  std::vector<int64_t> spanToChainOffset;
  WebGPUMirror<float> tempBuffer;
  WebGPUMirror<int64_t> devSpanToChainOffset;
};

// Double precision is not supported on WebGPU (like Metal)
template <>
struct WebGPUNumericCtx<double> : NumericCtx<double> {
  WebGPUNumericCtx(WebGPUSymbolicCtx& /*sym*/, int64_t /*tempBufSize*/, int64_t /*numSpans*/) {
    throw std::runtime_error(
        "WebGPU backend does not support double precision. "
        "WebGPU/WGSL has limited double-precision support across GPU backends. "
        "Use float precision with WebGPU, or use BackendFast (CPU) or BackendCuda for double.");
  }

  virtual ~WebGPUNumericCtx() override {}

  // All methods throw - should never be called
  virtual void pseudoFactorSpans(double*, int64_t, int64_t) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
  virtual void doElimination(const SymElimCtx&, double*, int64_t, int64_t) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
  virtual void potrf(int64_t, double*, int64_t) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
  virtual void trsm(int64_t, int64_t, double*, int64_t, int64_t) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
  virtual void saveSyrkGemm(int64_t, int64_t, int64_t, const double*, int64_t) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
  virtual void prepareAssemble(int64_t) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
  virtual void assemble(double*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
};

// Solve context for float - WebGPU implementation
template <>
struct WebGPUSolveCtx<float> : SolveCtx<float> {
  WebGPUSolveCtx(WebGPUSymbolicCtx& sym_, int nRHS_) : sym(sym_), nRHS(nRHS_) {}

  virtual ~WebGPUSolveCtx() override {}

  virtual void sparseElimSolveL(const SymElimCtx& elimData, const float* data, int64_t lumpsBegin,
                                int64_t lumpsEnd, float* C, int64_t ldc) override {
    // CPU fallback
    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      int64_t lumpStart = sym.skel.lumpStart[l];
      int64_t lumpSize = sym.skel.lumpStart[l + 1] - lumpStart;
      int64_t colStart = sym.skel.chainColPtr[l];
      int64_t diagDataPtr = sym.skel.chainData[colStart];

      const float* diagBlock = data + diagDataPtr;

      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using MatCMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

      for (int rhs = 0; rhs < nRHS; rhs++) {
        float* v = C + lumpStart + ldc * rhs;
        Eigen::Map<Eigen::VectorXf> vecV(v, lumpSize);
        Eigen::Map<const MatCMaj> matL(diagBlock, lumpSize, lumpSize);
        matL.template triangularView<Eigen::Upper>().transpose().solveInPlace(vecV);
      }
    }
  }

  virtual void sparseElimSolveLt(const SymElimCtx& elimData, const float* data, int64_t lumpsBegin,
                                 int64_t lumpsEnd, float* C, int64_t ldc) override {
    // CPU fallback
    for (int64_t l = lumpsEnd - 1; l >= lumpsBegin; l--) {
      int64_t lumpStart = sym.skel.lumpStart[l];
      int64_t lumpSize = sym.skel.lumpStart[l + 1] - lumpStart;
      int64_t colStart = sym.skel.chainColPtr[l];
      int64_t diagDataPtr = sym.skel.chainData[colStart];

      const float* diagBlock = data + diagDataPtr;

      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using MatCMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

      for (int rhs = 0; rhs < nRHS; rhs++) {
        float* v = C + lumpStart + ldc * rhs;
        Eigen::Map<Eigen::VectorXf> vecV(v, lumpSize);
        Eigen::Map<const MatCMaj> matL(diagBlock, lumpSize, lumpSize);
        matL.template triangularView<Eigen::Upper>().solveInPlace(vecV);
      }
    }
  }

  virtual void symm(const float* data, int64_t offset, int64_t n, const float* C, int64_t offC,
                    int64_t ldc, float* D, int64_t ldd, float alpha) override {
    // CPU fallback
    using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatCMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    Eigen::Map<const MatRMaj> matA(data + offset, n, n);
    Eigen::Map<const MatCMaj> matC(C + offC, n, nRHS);
    Eigen::Map<MatCMaj> matD(D, n, nRHS);

    // D += alpha * A * C (where A is symmetric, stored as lower triangle row-major)
    matD += alpha * matA.template selfadjointView<Eigen::Lower>() * matC;
  }

  virtual void solveL(const float* data, int64_t offset, int64_t n, float* C, int64_t offC,
                      int64_t ldc) override {
    // CPU fallback
    using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatCMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    Eigen::Map<const MatRMaj> matL(data + offset, n, n);
    Eigen::Map<MatCMaj> matC(C + offC, n, nRHS);

    matL.template triangularView<Eigen::Lower>().solveInPlace(matC);
  }

  virtual void solveLt(const float* data, int64_t offset, int64_t n, float* C, int64_t offC,
                       int64_t ldc) override {
    // CPU fallback
    using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatCMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    Eigen::Map<const MatRMaj> matL(data + offset, n, n);
    Eigen::Map<MatCMaj> matC(C + offC, n, nRHS);

    matL.template triangularView<Eigen::Lower>().transpose().solveInPlace(matC);
  }

  virtual void gemv(const float* data, int64_t offset, int64_t nRows, int64_t nCols, const float* A,
                    int64_t offA, int64_t lda, float alpha) override {
    // CPU fallback
    using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatCMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    Eigen::Map<const MatRMaj> matM(data + offset, nRows, nCols);
    Eigen::Map<const MatCMaj> matA(A + offA, nCols, nRHS);

    // Result accumulated in tempVec
    tempVec.conservativeResize(nRows, nRHS);
    tempVec += alpha * matM * matA;
  }

  virtual void gemvT(const float* data, int64_t offset, int64_t nRows, int64_t nCols, float* A,
                     int64_t offA, int64_t lda, float alpha) override {
    // CPU fallback
    using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatCMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    Eigen::Map<const MatRMaj> matM(data + offset, nRows, nCols);
    Eigen::Map<MatCMaj> matA(A + offA, nCols, nRHS);

    // A += alpha * M^T * tempVec
    matA += alpha * matM.transpose() * tempVec.topRows(nRows);
  }

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, float* C, int64_t ldc) override {
    // CPU fallback - simplified
  }

  virtual void assembleVecT(const float* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    // CPU fallback - simplified
  }

  WebGPUSymbolicCtx& sym;
  int nRHS;
  Eigen::MatrixXf tempVec;
};

// Double precision solve context - not supported
template <>
struct WebGPUSolveCtx<double> : SolveCtx<double> {
  WebGPUSolveCtx(WebGPUSymbolicCtx& /*sym*/, int /*nRHS*/) {
    throw std::runtime_error(
        "WebGPU backend does not support double precision. "
        "Use float precision with WebGPU, or use BackendFast (CPU) or BackendCuda for double.");
  }

  virtual ~WebGPUSolveCtx() override {}

  // All methods throw
  virtual void sparseElimSolveL(const SymElimCtx&, const double*, int64_t, int64_t, double*,
                                int64_t) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
  virtual void sparseElimSolveLt(const SymElimCtx&, const double*, int64_t, int64_t, double*,
                                 int64_t) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
  virtual void symm(const double*, int64_t, int64_t, const double*, int64_t, int64_t, double*,
                    int64_t, double) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
  virtual void solveL(const double*, int64_t, int64_t, double*, int64_t, int64_t) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
  virtual void solveLt(const double*, int64_t, int64_t, double*, int64_t, int64_t) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
  virtual void gemv(const double*, int64_t, int64_t, int64_t, const double*, int64_t, int64_t,
                    double) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
  virtual void gemvT(const double*, int64_t, int64_t, int64_t, double*, int64_t, int64_t,
                     double) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
  virtual void assembleVec(int64_t, int64_t, double*, int64_t) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
  virtual void assembleVecT(const double*, int64_t, int64_t, int64_t) override {
    throw std::runtime_error("WebGPU backend does not support double precision");
  }
};

// Factory methods for creating contexts
NumericCtxBase* WebGPUSymbolicCtx::createNumericCtxForType(type_index tIdx, int64_t tempBufSize,
                                                           int batchSize) {
  (void)batchSize;  // WebGPU doesn't support batched operations yet

  static const type_index floatIdx(typeid(float));
  static const type_index doubleIdx(typeid(double));

  if (tIdx == floatIdx) {
    return new WebGPUNumericCtx<float>(*this, tempBufSize, skel.numSpans());
  } else if (tIdx == doubleIdx) {
    return new WebGPUNumericCtx<double>(*this, tempBufSize, skel.numSpans());
  }

  BASPACHO_CHECK(false) << "Unsupported type for WebGPU numeric context";
  return nullptr;
}

SolveCtxBase* WebGPUSymbolicCtx::createSolveCtxForType(type_index tIdx, int nRHS, int batchSize) {
  (void)batchSize;

  static const type_index floatIdx(typeid(float));
  static const type_index doubleIdx(typeid(double));

  if (tIdx == floatIdx) {
    return new WebGPUSolveCtx<float>(*this, nRHS);
  } else if (tIdx == doubleIdx) {
    return new WebGPUSolveCtx<double>(*this, nRHS);
  }

  BASPACHO_CHECK(false) << "Unsupported type for WebGPU solve context";
  return nullptr;
}

// Public factory function
OpsPtr webgpuOps() { return OpsPtr(new WebGPUOps()); }

}  // end namespace BaSpaCho
