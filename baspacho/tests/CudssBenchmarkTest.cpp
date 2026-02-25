/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// cuDSS vs BaSpaCho profiling benchmark.
// Loads c6288 Jacobian (Matrix Market) and solves with both solvers.
// Run under `nsys profile` to compare GPU execution patterns.

#ifdef BASPACHO_HAVE_CUDSS

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cudss.h>
#include <nvtx3/nvToolsExt.h>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/CudaDefs.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using Vector = Eigen::Vector<T, Eigen::Dynamic>;

// ============================================================================
// Matrix Market parser
// ============================================================================

struct CsrMatrix {
  int64_t nRows;
  int64_t nCols;
  int64_t nnz;
  vector<int64_t> rowPtr;  // size nRows+1
  vector<int64_t> colInd;  // size nnz
  vector<double> values;   // size nnz
};

// Parse a Matrix Market coordinate file into CSR format.
// Handles: %%MatrixMarket matrix coordinate real general
CsrMatrix readMatrixMarket(const string& path) {
  ifstream f(path);
  EXPECT_TRUE(f.is_open()) << "Cannot open: " << path;

  string line;
  // Read header line
  getline(f, line);
  EXPECT_TRUE(line.find("%%MatrixMarket") != string::npos) << "Not a MatrixMarket file: " << path;
  EXPECT_TRUE(line.find("coordinate") != string::npos) << "Only coordinate format supported";

  // Skip comment lines
  while (getline(f, line)) {
    if (line.empty() || line[0] == '%') continue;
    break;
  }

  // Parse dimensions: rows cols nnz
  int64_t nRows, nCols, nnz;
  {
    istringstream iss(line);
    iss >> nRows >> nCols >> nnz;
  }

  // Read COO triplets
  struct Triplet {
    int64_t row, col;
    double val;
  };
  vector<Triplet> triplets;
  triplets.reserve(nnz);

  for (int64_t i = 0; i < nnz; i++) {
    Triplet t;
    f >> t.row >> t.col >> t.val;
    t.row--;  // 1-indexed -> 0-indexed
    t.col--;
    triplets.push_back(t);
  }

  EXPECT_EQ((int64_t)triplets.size(), nnz) << "Triplet count mismatch";

  // Sort by (row, col) for CSR construction
  sort(triplets.begin(), triplets.end(), [](const Triplet& a, const Triplet& b) {
    return a.row < b.row || (a.row == b.row && a.col < b.col);
  });

  // Build CSR
  CsrMatrix csr;
  csr.nRows = nRows;
  csr.nCols = nCols;
  csr.nnz = nnz;
  csr.rowPtr.resize(nRows + 1, 0);
  csr.colInd.resize(nnz);
  csr.values.resize(nnz);

  for (int64_t i = 0; i < nnz; i++) {
    csr.rowPtr[triplets[i].row + 1]++;
  }
  for (int64_t i = 0; i < nRows; i++) {
    csr.rowPtr[i + 1] += csr.rowPtr[i];
  }

  for (int64_t i = 0; i < nnz; i++) {
    csr.colInd[i] = triplets[i].col;
    csr.values[i] = triplets[i].val;
  }

  return csr;
}

// Read a dense vector from Matrix Market format (Nx1 coordinate or array)
Vector<double> readRhsVector(const string& path) {
  ifstream f(path);
  EXPECT_TRUE(f.is_open()) << "Cannot open: " << path;

  string line;
  getline(f, line);
  EXPECT_TRUE(line.find("%%MatrixMarket") != string::npos) << "Not a MatrixMarket file";

  bool isCoordinate = line.find("coordinate") != string::npos;

  // Skip comments
  while (getline(f, line)) {
    if (line.empty() || line[0] == '%') continue;
    break;
  }

  int64_t nRows, nCols;
  if (isCoordinate) {
    int64_t nnz;
    istringstream iss(line);
    iss >> nRows >> nCols >> nnz;
    EXPECT_EQ(nCols, 1) << "RHS must be a column vector";

    Vector<double> rhs = Vector<double>::Zero(nRows);
    for (int64_t i = 0; i < nnz; i++) {
      int64_t row, col;
      double val;
      f >> row >> col >> val;
      rhs(row - 1) = val;
    }
    return rhs;
  } else {
    // Array format
    istringstream iss(line);
    iss >> nRows >> nCols;
    EXPECT_EQ(nCols, 1) << "RHS must be a column vector";

    Vector<double> rhs(nRows);
    for (int64_t i = 0; i < nRows; i++) {
      f >> rhs(i);
    }
    return rhs;
  }
}

// ============================================================================
// Get the test data directory
// ============================================================================

static string getMtxDir() {
  // Check --mtx command-line arg (gtest doesn't parse custom args, use env)
  const char* envDir = getenv("BASPACHO_MTX_DIR");
  if (envDir) return string(envDir);

  // Default: test_data/c6288_jacobian relative to repo root
  // Try a few common locations
  vector<string> candidates = {
      "test_data/c6288_jacobian",
      "../test_data/c6288_jacobian",
      "../../test_data/c6288_jacobian",
  };

  for (const auto& dir : candidates) {
    ifstream test(dir + "/jacobian.mtx");
    if (test.good()) return dir;
  }

  // Last resort: absolute path
  return "test_data/c6288_jacobian";
}

// ============================================================================
// CSR -> CSC conversion (cuDSS uses CSR, UMFPACK-style needs CSC)
// ============================================================================

struct CscMatrix {
  int64_t nRows, nCols, nnz;
  vector<int64_t> colPtr;
  vector<int64_t> rowInd;
  vector<double> values;
};

CscMatrix csrToCsc(const CsrMatrix& csr) {
  CscMatrix csc;
  csc.nRows = csr.nRows;
  csc.nCols = csr.nCols;
  csc.nnz = csr.nnz;
  csc.colPtr.resize(csr.nCols + 1, 0);
  csc.rowInd.resize(csr.nnz);
  csc.values.resize(csr.nnz);

  // Count entries per column
  for (int64_t i = 0; i < csr.nnz; i++) {
    csc.colPtr[csr.colInd[i] + 1]++;
  }
  for (int64_t j = 0; j < csr.nCols; j++) {
    csc.colPtr[j + 1] += csc.colPtr[j];
  }

  // Fill values
  vector<int64_t> colCursor = csc.colPtr;
  for (int64_t i = 0; i < csr.nRows; i++) {
    for (int64_t k = csr.rowPtr[i]; k < csr.rowPtr[i + 1]; k++) {
      int64_t col = csr.colInd[k];
      int64_t pos = colCursor[col]++;
      csc.rowInd[pos] = i;
      csc.values[pos] = csr.values[k];
    }
  }

  return csc;
}

// ============================================================================
// Sparse residual computation: ||Ax - b|| / ||b|| using CSR
// ============================================================================

double computeResidual(const CsrMatrix& A, const Vector<double>& x, const Vector<double>& b) {
  Vector<double> Ax = Vector<double>::Zero(A.nRows);
  for (int64_t i = 0; i < A.nRows; i++) {
    for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      Ax(i) += A.values[k] * x(A.colInd[k]);
    }
  }
  return (Ax - b).norm() / b.norm();
}

// ============================================================================
// cuDSS error checking macro
// ============================================================================

#define cudssCHECK(call)                                                          \
  do {                                                                            \
    cudssStatus_t status = (call);                                                \
    if (status != CUDSS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "[%s:%d] cuDSS Error: %d\n", __FILE__, __LINE__, (int)status); \
      FAIL() << "cuDSS call failed with status " << (int)status;                  \
    }                                                                             \
  } while (0)

// ============================================================================
// BaSpaCho CUDA LU test: load c6288 Jacobian, solve on GPU
// ============================================================================

TEST(CudssBenchmark, BaSpaCho_LU) {
  string mtxDir = getMtxDir();
  cout << "Loading matrix from: " << mtxDir << endl;

  CsrMatrix A = readMatrixMarket(mtxDir + "/jacobian.mtx");
  Vector<double> b = readRhsVector(mtxDir + "/rhs.mtx");

  cout << "Matrix: " << A.nRows << " x " << A.nCols << ", nnz=" << A.nnz << endl;
  ASSERT_EQ(A.nRows, A.nCols) << "Matrix must be square";
  ASSERT_EQ(b.size(), A.nRows) << "RHS size mismatch";

  int64_t n = A.nRows;

  // Build BaSpaCho structure from CSR.
  // For a scalar matrix, each row/col is its own span/lump (paramSize=1 for all).
  // We need the sparsity structure as block columns.

  // Build column sets from CSR (transpose gives CSC-like column access)
  vector<set<int64_t>> colBlocks(n);
  for (int64_t i = 0; i < n; i++) {
    for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      int64_t j = A.colInd[k];
      // BaSpaCho CSC structure: column j has row i
      colBlocks[j].insert(i);
    }
  }

  nvtxRangePush("BaSpaCho_Analysis");

  // Create SparseStructure (CSC format internally)
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();

  // Scalar blocks: each span/lump is size 1
  vector<int64_t> paramSize(n, 1);
  vector<int64_t> spanStart(n + 1);
  iota(spanStart.begin(), spanStart.end(), 0);
  vector<int64_t> lumpToSpan(n + 1);
  iota(lumpToSpan.begin(), lumpToSpan.end(), 0);

  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  factorSkel.initUpperTriangle();

  // Build dense-ish representation for filling block data.
  // For scalar blocks, the factorSkel data is a flat array matching the sparse structure.
  vector<double> data(factorSkel.totalDataSize(), 0.0);

  // Fill from CSR values into BaSpaCho block storage
  int64_t numLumps = factorSkel.numLumps();

  // Build a lookup: (row, col) -> value from CSR
  // For large matrices this is more efficient using the CSR directly
  // We iterate the skeleton and look up values from the CSR structure
  for (int64_t l = 0; l < numLumps; l++) {
    int64_t chainStart = factorSkel.chainColPtr[l];
    int64_t chainEnd = factorSkel.chainColPtr[l + 1];
    int64_t lumpStartCol = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStartCol;

    for (int64_t c = chainStart; c < chainEnd; c++) {
      int64_t rowSpan = factorSkel.chainRowSpan[c];
      int64_t rowStart = factorSkel.spanStart[rowSpan];
      int64_t rowSize = factorSkel.spanStart[rowSpan + 1] - rowStart;
      int64_t dataOffset = factorSkel.chainData[c];

      for (int64_t r = 0; r < rowSize; r++) {
        int64_t globalRow = rowStart + r;
        for (int64_t col = 0; col < lumpSize; col++) {
          int64_t globalCol = lumpStartCol + col;
          // Look up (globalRow, globalCol) in CSR
          for (int64_t k = A.rowPtr[globalRow]; k < A.rowPtr[globalRow + 1]; k++) {
            if (A.colInd[k] == globalCol) {
              data[dataOffset + r * lumpSize + col] = A.values[k];
              break;
            }
          }
        }
      }
    }
  }

  // Fill upper triangle from CSR
  if (!factorSkel.upperChainData.empty()) {
    int64_t upperDataBase = factorSkel.dataSize();
    for (int64_t l = 0; l < numLumps; l++) {
      int64_t upperRowStart = factorSkel.upperChainRowPtr[l];
      int64_t upperRowEnd = factorSkel.upperChainRowPtr[l + 1];
      int64_t lumpStartRow = factorSkel.lumpStart[l];
      int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStartRow;

      for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
        int64_t colSpan = factorSkel.upperChainColSpan[i];
        int64_t colStart = factorSkel.spanStart[colSpan];
        int64_t colSize = factorSkel.spanStart[colSpan + 1] - colStart;
        int64_t upperDataOffset = upperDataBase + factorSkel.upperChainData[i];

        for (int64_t r = 0; r < lumpSize; r++) {
          int64_t globalRow = lumpStartRow + r;
          for (int64_t c = 0; c < colSize; c++) {
            int64_t globalCol = colStart + c;
            for (int64_t k = A.rowPtr[globalRow]; k < A.rowPtr[globalRow + 1]; k++) {
              if (A.colInd[k] == globalCol) {
                data[upperDataOffset + r * colSize + c] = A.values[k];
                break;
              }
            }
          }
        }
      }
    }
  }

  Solver solver(std::move(factorSkel), {}, {}, cudaOps());
  nvtxRangePop();  // BaSpaCho_Analysis

  vector<int64_t> pivots(n);

  nvtxRangePush("BaSpaCho_Factor");
  auto factorStart = hrc::now();
  solver.factorLU(data.data(), pivots.data());
  double factorMs = tdelta(hrc::now() - factorStart).count() * 1000;
  nvtxRangePop();

  Vector<double> x = b;

  nvtxRangePush("BaSpaCho_Solve");
  auto solveStart = hrc::now();
  solver.solveLU(data.data(), pivots.data(), x.data(), n, 1);
  double solveMs = tdelta(hrc::now() - solveStart).count() * 1000;
  nvtxRangePop();

  double residual = computeResidual(A, x, b);

  cout << "\n=== BaSpaCho CUDA LU ===" << endl;
  cout << "  Factor: " << fixed << setprecision(2) << factorMs << " ms" << endl;
  cout << "  Solve:  " << solveMs << " ms" << endl;
  cout << "  Residual: " << scientific << setprecision(4) << residual << endl;

  EXPECT_LT(residual, 1e-8) << "BaSpaCho residual too large";
}

// ============================================================================
// cuDSS LU test: load c6288 Jacobian, solve on GPU via cuDSS
// ============================================================================

TEST(CudssBenchmark, CuDSS_LU) {
  string mtxDir = getMtxDir();
  cout << "Loading matrix from: " << mtxDir << endl;

  CsrMatrix A = readMatrixMarket(mtxDir + "/jacobian.mtx");
  Vector<double> b = readRhsVector(mtxDir + "/rhs.mtx");

  cout << "Matrix: " << A.nRows << " x " << A.nCols << ", nnz=" << A.nnz << endl;
  ASSERT_EQ(A.nRows, A.nCols) << "Matrix must be square";
  ASSERT_EQ(b.size(), A.nRows) << "RHS size mismatch";

  int64_t n = A.nRows;

  // cuDSS uses CSR format with int32 indices
  // Convert int64_t -> int for cuDSS
  vector<int> csrRowPtr32(A.rowPtr.begin(), A.rowPtr.end());
  vector<int> csrColInd32(A.colInd.begin(), A.colInd.end());

  // Allocate GPU memory
  int* d_rowPtr = nullptr;
  int* d_colInd = nullptr;
  double* d_values = nullptr;
  double* d_b = nullptr;
  double* d_x = nullptr;

  cuCHECK(cudaMalloc(&d_rowPtr, (n + 1) * sizeof(int)));
  cuCHECK(cudaMalloc(&d_colInd, A.nnz * sizeof(int)));
  cuCHECK(cudaMalloc(&d_values, A.nnz * sizeof(double)));
  cuCHECK(cudaMalloc(&d_b, n * sizeof(double)));
  cuCHECK(cudaMalloc(&d_x, n * sizeof(double)));

  cuCHECK(cudaMemcpy(d_rowPtr, csrRowPtr32.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  cuCHECK(cudaMemcpy(d_colInd, csrColInd32.data(), A.nnz * sizeof(int), cudaMemcpyHostToDevice));
  cuCHECK(cudaMemcpy(d_values, A.values.data(), A.nnz * sizeof(double), cudaMemcpyHostToDevice));
  cuCHECK(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));

  // Create cuDSS handle and objects
  cudssHandle_t handle = nullptr;
  cudssCHECK(cudssCreate(&handle));

  cudssConfig_t config = nullptr;
  cudssCHECK(cudssConfigCreate(&config));

  cudssData_t data = nullptr;
  cudssCHECK(cudssDataCreate(handle, &data));

  // Create matrix descriptor (CSR, general/unsymmetric)
  cudssMatrix_t cudssA = nullptr;
  int64_t nRows64 = n;
  int64_t nCols64 = n;
  int64_t nnz64 = A.nnz;
  cudssCHECK(cudssMatrixCreateCsr(
      &cudssA, nRows64, nCols64, nnz64,
      d_rowPtr, nullptr,  // row end offsets (null = derive from rowPtr)
      d_colInd, d_values,
      CUDA_R_32I, CUDA_R_64F,
      CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
      CUDSS_BASE_ZERO));

  // Create dense RHS and solution vectors
  cudssMatrix_t cudssB = nullptr;
  cudssMatrix_t cudssX = nullptr;
  int64_t nrhs = 1;
  int64_t ldb = n;
  int64_t ldx = n;
  cudssCHECK(cudssMatrixCreateDn(&cudssB, nRows64, nrhs, ldb, d_b, CUDA_R_64F,
                                  CUDSS_LAYOUT_COL_MAJOR));
  cudssCHECK(cudssMatrixCreateDn(&cudssX, nRows64, nrhs, ldx, d_x, CUDA_R_64F,
                                  CUDSS_LAYOUT_COL_MAJOR));

  // Analysis phase
  nvtxRangePush("CuDSS_Analysis");
  auto analysisStart = hrc::now();
  cudssCHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, cudssA, cudssX, cudssB));
  cuCHECK(cudaDeviceSynchronize());
  double analysisMs = tdelta(hrc::now() - analysisStart).count() * 1000;
  nvtxRangePop();

  // Factorization phase
  nvtxRangePush("CuDSS_Factor");
  auto factorStart = hrc::now();
  cudssCHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, cudssA, cudssX, cudssB));
  cuCHECK(cudaDeviceSynchronize());
  double factorMs = tdelta(hrc::now() - factorStart).count() * 1000;
  nvtxRangePop();

  // Solve phase
  nvtxRangePush("CuDSS_Solve");
  auto solveStart = hrc::now();
  cudssCHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, cudssA, cudssX, cudssB));
  cuCHECK(cudaDeviceSynchronize());
  double solveMs = tdelta(hrc::now() - solveStart).count() * 1000;
  nvtxRangePop();

  // Copy solution back
  Vector<double> x(n);
  cuCHECK(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

  double residual = computeResidual(A, x, b);

  cout << "\n=== cuDSS LU ===" << endl;
  cout << "  Analysis: " << fixed << setprecision(2) << analysisMs << " ms" << endl;
  cout << "  Factor:   " << factorMs << " ms" << endl;
  cout << "  Solve:    " << solveMs << " ms" << endl;
  cout << "  Residual: " << scientific << setprecision(4) << residual << endl;

  EXPECT_LT(residual, 1e-8) << "cuDSS residual too large";

  // Cleanup
  cudssMatrixDestroy(cudssA);
  cudssMatrixDestroy(cudssB);
  cudssMatrixDestroy(cudssX);
  cudssDataDestroy(handle, data);
  cudssConfigDestroy(config);
  cudssDestroy(handle);

  cudaFree(d_rowPtr);
  cudaFree(d_colInd);
  cudaFree(d_values);
  cudaFree(d_b);
  cudaFree(d_x);
}

#endif  // BASPACHO_HAVE_CUDSS
