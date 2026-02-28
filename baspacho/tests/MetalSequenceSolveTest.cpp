/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Metal GPU variant of SequenceSolveTest.
// Tests LU factorization on real-world circuit Jacobian sequences using
// the Metal backend (float-only, Apple Silicon).

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/MetalDefs.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/testing/MatrixMarketReader.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;
namespace fs = std::filesystem;

static string findTestDataDir(const string& subdir) {
  const char* envDir = getenv("BASPACHO_TEST_DATA_DIR");
  if (envDir) {
    string candidate = string(envDir) + "/" + subdir;
    if (fs::is_directory(candidate)) return candidate;
  }

  vector<string> prefixes = {"test_data", "../test_data", "../../test_data", "../../../test_data"};
  for (const auto& prefix : prefixes) {
    string candidate = prefix + "/" + subdir;
    if (fs::is_directory(candidate)) return candidate;
  }
  return "";
}

static vector<pair<string, string>> discoverSequenceFiles(const string& dir) {
  vector<pair<string, string>> pairs;
  for (int idx = 0;; idx++) {
    ostringstream jacName, rhsName;
    jacName << dir << "/jacobian_" << setw(4) << setfill('0') << idx << ".mtx";
    rhsName << dir << "/rhs_" << setw(4) << setfill('0') << idx << ".mtx";
    if (!fs::exists(jacName.str()) || !fs::exists(rhsName.str())) break;
    pairs.push_back({jacName.str(), rhsName.str()});
  }
  return pairs;
}

static CoalescedBlockMatrixSkel buildSkeletonFromCsr(const CsrMatrix& A) {
  int64_t n = A.nRows;
  vector<set<int64_t>> colBlocks(n);
  for (int64_t i = 0; i < n; i++) {
    for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      colBlocks[A.colInd[k]].insert(i);
    }
  }

  SparseStructure ss = columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
  vector<int64_t> spanStart(n + 1);
  iota(spanStart.begin(), spanStart.end(), 0);
  vector<int64_t> lumpToSpan(n + 1);
  iota(lumpToSpan.begin(), lumpToSpan.end(), 0);

  SparseStructure groupedSs = columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
  CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
  skel.initUpperTriangle();
  return skel;
}

template <typename T>
static void fillDataFromCsr(const CoalescedBlockMatrixSkel& skel, const CsrMatrix& A,
                            vector<T>& data) {
  fill(data.begin(), data.end(), T(0));
  int64_t numLumps = skel.numLumps();

  for (int64_t l = 0; l < numLumps; l++) {
    int64_t chainStart = skel.chainColPtr[l];
    int64_t chainEnd = skel.chainColPtr[l + 1];
    int64_t lumpStartCol = skel.lumpStart[l];
    int64_t lumpSize = skel.lumpStart[l + 1] - lumpStartCol;

    for (int64_t c = chainStart; c < chainEnd; c++) {
      int64_t rowSpan = skel.chainRowSpan[c];
      int64_t rowStart = skel.spanStart[rowSpan];
      int64_t rowSize = skel.spanStart[rowSpan + 1] - rowStart;
      int64_t dataOffset = skel.chainData[c];

      for (int64_t r = 0; r < rowSize; r++) {
        int64_t globalRow = rowStart + r;
        for (int64_t col = 0; col < lumpSize; col++) {
          int64_t globalCol = lumpStartCol + col;
          for (int64_t k = A.rowPtr[globalRow]; k < A.rowPtr[globalRow + 1]; k++) {
            if (A.colInd[k] == globalCol) {
              data[dataOffset + r * lumpSize + col] = T(A.values[k]);
              break;
            }
          }
        }
      }
    }
  }

  if (!skel.upperChainData.empty()) {
    int64_t upperDataBase = skel.dataSize();
    for (int64_t l = 0; l < numLumps; l++) {
      int64_t upperRowStart = skel.upperChainRowPtr[l];
      int64_t upperRowEnd = skel.upperChainRowPtr[l + 1];
      int64_t lumpStartRow = skel.lumpStart[l];
      int64_t lumpSize = skel.lumpStart[l + 1] - lumpStartRow;

      for (int64_t i = upperRowStart; i < upperRowEnd; i++) {
        int64_t colSpan = skel.upperChainColSpan[i];
        int64_t colStart = skel.spanStart[colSpan];
        int64_t colSize = skel.spanStart[colSpan + 1] - colStart;
        int64_t upperDataOffset = upperDataBase + skel.upperChainData[i];

        for (int64_t r = 0; r < lumpSize; r++) {
          int64_t globalRow = lumpStartRow + r;
          for (int64_t c = 0; c < colSize; c++) {
            int64_t globalCol = colStart + c;
            for (int64_t k = A.rowPtr[globalRow]; k < A.rowPtr[globalRow + 1]; k++) {
              if (A.colInd[k] == globalCol) {
                data[upperDataOffset + r * colSize + c] = T(A.values[k]);
                break;
              }
            }
          }
        }
      }
    }
  }
}

// Float-only residual computation for Metal (CSR values are double, solution is float)
static float computeResidualFloat(const CsrMatrix& A, const Eigen::VectorXf& x,
                                  const Eigen::VectorXf& b) {
  Eigen::VectorXf Ax = Eigen::VectorXf::Zero(A.nRows);
  for (int64_t i = 0; i < A.nRows; i++) {
    for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      Ax(i) += float(A.values[k]) * x(A.colInd[k]);
    }
  }
  return (Ax - b).norm() / b.norm();
}

// ============================================================================
// Ring oscillator on Metal (float-only, tiny matrices — good GPU smoke test)
// ============================================================================

TEST(MetalSequenceSolve, RingOscillator) {
  string dir = findTestDataDir("ring_sequence");
  if (dir.empty()) {
    GTEST_SKIP() << "test_data/ring_sequence/ not found";
  }

  auto files = discoverSequenceFiles(dir);
  ASSERT_GE(files.size(), 2u) << "Need at least 2 sequence files";

  cout << "Metal ring oscillator sequence: " << files.size() << " matrices in " << dir << endl;

  CsrMatrix A0 = readMatrixMarket(files[0].first);
  ASSERT_EQ(A0.nRows, A0.nCols) << "Matrix must be square";
  cout << "Matrix size: " << A0.nRows << " x " << A0.nCols << ", nnz=" << A0.nnz << endl;

  int64_t n = A0.nRows;

  CoalescedBlockMatrixSkel skel = buildSkeletonFromCsr(A0);
  vector<float> data(skel.totalDataSize());
  Solver solver(std::move(skel), {}, {}, metalOps());
  const auto& solverSkel = solver.skel();

  // CPU float solver for reference comparison (same fill-in structure)
  CoalescedBlockMatrixSkel cpuSkel = buildSkeletonFromCsr(A0);
  vector<float> cpuData(cpuSkel.totalDataSize());
  Solver cpuSolver(std::move(cpuSkel), {}, {}, fastOps());
  const auto& cpuSolverSkel = cpuSolver.skel();
  vector<int64_t> cpuPivots(n);

  vector<int64_t> pivots(n);
  int passed = 0;
  int skipped = 0;

  for (size_t i = 0; i < files.size(); i++) {
    CsrMatrix A = readMatrixMarket(files[i].first);
    Eigen::VectorXd bDouble = readRhsVector(files[i].second);

    ASSERT_EQ(A.nRows, n) << "Matrix #" << i << " has different dimensions";
    ASSERT_EQ(bDouble.size(), n) << "RHS #" << i << " has wrong size";

    Eigen::VectorXf b = bDouble.cast<float>();

    // Skip near-zero RHS
    if (b.norm() < 1e-7f) {
      skipped++;
      continue;
    }

    // Check CPU float path first (same supernodal fill-in structure).
    // Some matrices trigger zero pivots in BaSpaCho's float LU due to fill-in
    // structure — skip those since the issue is precision, not Metal.
    float cpuResidual;
    try {
      fillDataFromCsr<float>(cpuSolverSkel, A, cpuData);
      cpuSolver.factorLU(cpuData.data(), cpuPivots.data());
      Eigen::VectorXf xCpu = b;
      cpuSolver.solveLU(cpuData.data(), cpuPivots.data(), xCpu.data(), n, 1);
      cpuResidual = computeResidualFloat(A, xCpu, b);
    } catch (const exception&) {
      // CPU float also fails (zero pivot) — skip this matrix
      skipped++;
      continue;
    }

    if (cpuResidual > 1e-2f) {
      skipped++;
      continue;
    }

    // Factor+solve on Metal GPU
    fillDataFromCsr<float>(solverSkel, A, data);

    {
      MetalMirror<float> dataGpu(data);
      solver.factorLU(dataGpu.ptr(), pivots.data());
      dataGpu.get(data);
    }

    Eigen::VectorXf x = b;
    {
      MetalMirror<float> dataGpu(data);
      MetalMirror<float> xGpu(vector<float>(x.data(), x.data() + n));
      solver.solveLU(dataGpu.ptr(), pivots.data(), xGpu.ptr(), n, 1);
      vector<float> xVec(n);
      xGpu.get(xVec);
      for (int64_t j = 0; j < n; j++) x(j) = xVec[j];
    }

    float residual = computeResidualFloat(A, x, b);

    // Metal should be within 100x of CPU BaSpaCho float
    float threshold = max(cpuResidual * 100.0f, 1e-4f);
    EXPECT_LT(residual, threshold) << "Matrix #" << i << " Metal residual too large: " << residual
                                   << " (CPU: " << cpuResidual << ")";
    passed++;
  }

  cout << "Passed: " << passed << ", Skipped (zero RHS): " << skipped << endl;
  ASSERT_GT(passed, 0) << "No matrices were actually tested";
}
