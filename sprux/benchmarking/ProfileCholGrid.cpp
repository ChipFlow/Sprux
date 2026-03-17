/*
 * Copyright (c) Robert Taylor, 2026. All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Profile Cholesky factor on GRID problem to understand dispatch overhead
// Usage: build/sprux/benchmarking/profile_chol_grid

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "sprux/sprux/CoalescedBlockMatrix.h"
#include "sprux/sprux/Solver.h"
#include "sprux/testing/TestingMatGen.h"
#include "sprux/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace testing_utils;
using namespace std;

int main() {
  // Generate GRID problem: 100x100 grid, block size 3
  auto gen = SparseMatGenerator::genGrid(100, 100, 1.0, 2, 42);
  SparseStructure struct_grid = columnsToCscStruct(gen.columns).transpose();
  int64_t nnz = struct_grid.ptrs.back();

  cout << "GRID 100x100 problem: nnz=" << nnz << endl;

  // Create Metal solver
  Settings settingsMetal;
  settingsMetal.backend = BackendMetal;
  vector<int64_t> paramSize(gen.columns.size(), 3);  // block size 3
  
  SolverPtr solverMetal = createSolver(settingsMetal, paramSize, struct_grid);
  int64_t dataSize = solverMetal->dataSize();

  // Generate random data for matrix
  vector<float> dataMetal(dataSize, 0.0f);
  for (int64_t i = 0; i < dataSize; i++) {
    dataMetal[i] = 1.0f + (i % 7) * 0.1f;  // Simple test values
  }

  // Measure Metal factorization
  auto t0 = chrono::high_resolution_clock::now();
  solverMetal->factor(dataMetal.data(), false);
  auto t1 = chrono::high_resolution_clock::now();

  double metalMs = chrono::duration<double, milli>(t1 - t0).count();
  cout << "Metal Cholesky factor time: " << fixed << setprecision(1) << metalMs << " ms" << endl;

  // Create CPU solver
  Settings settingsCpu;
  settingsCpu.backend = BackendFast;
  SolverPtr solverCpu = createSolver(settingsCpu, paramSize, struct_grid);

  vector<float> dataCpu = dataMetal;  // Copy

  // Measure CPU factorization
  t0 = chrono::high_resolution_clock::now();
  solverCpu->factor(dataCpu.data(), false);
  t1 = chrono::high_resolution_clock::now();

  double cpuMs = chrono::duration<double, milli>(t1 - t0).count();
  cout << "CPU BLAS factor time:       " << fixed << setprecision(1) << cpuMs << " ms" << endl;
  cout << "Metal vs CPU: " << fixed << setprecision(2) << (metalMs / cpuMs) << "x ("
       << (metalMs > cpuMs ? "slower" : "faster") << ")" << endl;

  return 0;
}
