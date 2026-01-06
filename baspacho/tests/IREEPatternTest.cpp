/*
 * Test that follows the exact IREE wrapper pattern to debug integration issues.
 * This test mimics how IREE calls BaSpaCho to identify where the precision loss occurs.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include "baspacho/baspacho/MetalDefs.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"

using namespace BaSpaCho;
using namespace std;

/**
 * Create 2D Poisson matrix in CSR format (FULL symmetric matrix, not lower triangular).
 * Returns: {row_ptr, col_idx, values, n}
 */
struct CsrMatrix {
  std::vector<int64_t> row_ptr;
  std::vector<int64_t> col_idx;
  std::vector<float> values;
  int64_t n;
};

CsrMatrix createPoissonCSR(int64_t gridSize) {
  int64_t n = gridSize * gridSize;
  CsrMatrix csr;
  csr.n = n;
  csr.row_ptr.resize(n + 1);
  csr.row_ptr[0] = 0;

  for (int64_t i = 0; i < gridSize; ++i) {
    for (int64_t j = 0; j < gridSize; ++j) {
      int64_t row = i * gridSize + j;

      // Collect neighbors for this row (sorted by column)
      std::vector<std::pair<int64_t, float>> entries;

      // Left neighbor
      if (j > 0) entries.emplace_back(row - 1, -1.0f);
      // Top neighbor
      if (i > 0) entries.emplace_back(row - gridSize, -1.0f);
      // Diagonal
      entries.emplace_back(row, 4.0f);
      // Bottom neighbor
      if (i < gridSize - 1) entries.emplace_back(row + gridSize, -1.0f);
      // Right neighbor
      if (j < gridSize - 1) entries.emplace_back(row + 1, -1.0f);

      // Sort by column
      std::sort(entries.begin(), entries.end());

      for (const auto& e : entries) {
        csr.col_idx.push_back(e.first);
        csr.values.push_back(e.second);
      }
      csr.row_ptr[row + 1] = csr.col_idx.size();
    }
  }

  return csr;
}

/**
 * Test following the exact IREE wrapper pattern:
 * 1. Receive full CSR matrix
 * 2. Extract lower triangular structure
 * 3. Create solver with that structure
 * 4. Load values using lower triangular extraction
 * 5. Factor and solve
 */
TEST(IREEPattern, Poisson2D_Grid10) {
  // Step 1: Create full CSR matrix (as IREE receives from JAX)
  CsrMatrix csr = createPoissonCSR(10);  // 100x100 matrix
  int64_t n = csr.n;
  int64_t original_nnz = csr.values.size();

  std::cout << "Original CSR: n=" << n << ", nnz=" << original_nnz << std::endl;
  std::cout << "First 10 values: ";
  for (int i = 0; i < std::min(10, (int)csr.values.size()); ++i) {
    std::cout << csr.values[i] << " ";
  }
  std::cout << std::endl;

  // Step 2: Extract lower triangular (exactly as IREE does)
  std::vector<int64_t> lower_row_ptr(n + 1);
  std::vector<int64_t> lower_col_idx;
  std::vector<int64_t> lower_to_original_idx;

  lower_row_ptr[0] = 0;
  for (int64_t row = 0; row < n; ++row) {
    for (int64_t ptr = csr.row_ptr[row]; ptr < csr.row_ptr[row + 1]; ++ptr) {
      int64_t col = csr.col_idx[ptr];
      if (col <= row) {  // Lower triangular (including diagonal)
        lower_col_idx.push_back(col);
        lower_to_original_idx.push_back(ptr);
      }
    }
    lower_row_ptr[row + 1] = lower_col_idx.size();
  }

  int64_t lower_nnz = lower_col_idx.size();
  std::cout << "Lower triangular: nnz=" << lower_nnz << std::endl;

  // Step 3: Create sparse structure and solver (as IREE does)
  SparseStructure ss;
  ss.ptrs = lower_row_ptr;
  ss.inds = lower_col_idx;

  std::vector<int64_t> block_sizes(n, 1);  // Scalar blocks

  Settings settings;
  settings.backend = BackendMetal;
  settings.numThreads = 8;
  settings.addFillPolicy = AddFillComplete;
  settings.findSparseEliminationRanges = true;

  SolverPtr solver = createSolver(settings, block_sizes, ss);
  ASSERT_TRUE(solver != nullptr);

  // Get permutation
  const auto& permutation = solver->paramToSpan();
  std::cout << "Permutation first 10: ";
  for (int i = 0; i < std::min(10, (int)n); ++i) {
    std::cout << permutation[i] << " ";
  }
  std::cout << std::endl;

  // Step 4: Extract lower triangular values (as IREE does)
  std::vector<float> lower_values(lower_nnz);
  for (int64_t i = 0; i < lower_nnz; ++i) {
    lower_values[i] = csr.values[lower_to_original_idx[i]];
  }

  std::cout << "Lower values first 10: ";
  for (int i = 0; i < std::min(10, (int)lower_nnz); ++i) {
    std::cout << lower_values[i] << " ";
  }
  std::cout << std::endl;

  // Step 5: Load values into factor data (as IREE does)
  int64_t data_size = solver->dataSize();
  std::vector<float> factorData(data_size, 0.0f);

  solver->loadFromCsr(lower_row_ptr.data(), lower_col_idx.data(),
                      block_sizes.data(), lower_values.data(), factorData.data());

  std::cout << "Factor data after loadFromCsr first 10: ";
  for (int i = 0; i < std::min(10, (int)data_size); ++i) {
    std::cout << factorData[i] << " ";
  }
  std::cout << std::endl;

  // Step 6: Create known solution and RHS
  std::vector<float> x_true(n, 1.0f);  // All ones
  std::vector<float> b(n, 0.0f);

  // b = A * x_true (using full matrix)
  for (int64_t row = 0; row < n; ++row) {
    for (int64_t ptr = csr.row_ptr[row]; ptr < csr.row_ptr[row + 1]; ++ptr) {
      b[row] += csr.values[ptr] * x_true[csr.col_idx[ptr]];
    }
  }

  std::cout << "RHS first 10: ";
  for (int i = 0; i < std::min(10, (int)n); ++i) {
    std::cout << b[i] << " ";
  }
  std::cout << std::endl;

  // Step 7: Factor and solve (as IREE does for Metal)
  {
    MetalMirror<float> dataGpu(factorData);
    MetalMirror<float> permuted;
    permuted.resizeToAtLeast(n);

    std::cout << "Factor data ptr: " << (void*)dataGpu.ptr() << std::endl;
    std::cout << "Permuted ptr: " << (void*)permuted.ptr() << std::endl;

    // Factor
    solver->factor(dataGpu.ptr());
    MetalContext::instance().synchronize();

    std::cout << "Factor data after factor first 10: ";
    for (int i = 0; i < std::min(10, (int)data_size); ++i) {
      std::cout << dataGpu.ptr()[i] << " ";
    }
    std::cout << std::endl;

    // Apply permutation to RHS (scatter)
    float* permutedPtr = permuted.ptr();
    for (int64_t i = 0; i < n; ++i) {
      permutedPtr[permutation[i]] = b[i];
    }

    std::cout << "Permuted RHS first 10: ";
    for (int i = 0; i < std::min(10, (int)n); ++i) {
      std::cout << permutedPtr[i] << " ";
    }
    std::cout << std::endl;

    // Solve
    solver->solve(dataGpu.ptr(), permutedPtr, n, 1);
    MetalContext::instance().synchronize();

    std::cout << "Permuted after solve first 10: ";
    for (int i = 0; i < std::min(10, (int)n); ++i) {
      std::cout << permutedPtr[i] << " ";
    }
    std::cout << std::endl;

    // Apply inverse permutation (gather)
    std::vector<float> solution(n);
    for (int64_t i = 0; i < n; ++i) {
      solution[i] = permutedPtr[permutation[i]];
    }

    std::cout << "Solution first 10: ";
    for (int i = 0; i < std::min(10, (int)n); ++i) {
      std::cout << solution[i] << " ";
    }
    std::cout << std::endl;

    // Compute error
    float diff = 0, ref = 0;
    for (int64_t i = 0; i < n; ++i) {
      float d = solution[i] - x_true[i];
      diff += d * d;
      ref += x_true[i] * x_true[i];
    }
    float relError = std::sqrt(diff) / std::sqrt(ref);
    std::cout << "IREE Pattern relative error: " << relError << std::endl;

    EXPECT_LT(relError, 1e-4) << "IREE pattern should achieve good precision";
  }
}

/**
 * Compare with the working BaSpaCho test pattern side by side.
 */
TEST(IREEPattern, Poisson2D_Grid100_IREE_Scale) {
  // Test at the same scale as the IREE test (100x100 = 10,000 elements)
  CsrMatrix csr = createPoissonCSR(100);
  int64_t n = csr.n;
  int64_t original_nnz = csr.values.size();

  std::cout << "Original CSR: n=" << n << ", nnz=" << original_nnz << std::endl;

  // Extract lower triangular
  std::vector<int64_t> lower_row_ptr(n + 1);
  std::vector<int64_t> lower_col_idx;
  std::vector<int64_t> lower_to_original_idx;

  lower_row_ptr[0] = 0;
  for (int64_t row = 0; row < n; ++row) {
    for (int64_t ptr = csr.row_ptr[row]; ptr < csr.row_ptr[row + 1]; ++ptr) {
      int64_t col = csr.col_idx[ptr];
      if (col <= row) {
        lower_col_idx.push_back(col);
        lower_to_original_idx.push_back(ptr);
      }
    }
    lower_row_ptr[row + 1] = lower_col_idx.size();
  }

  int64_t lower_nnz = lower_col_idx.size();
  std::cout << "Lower triangular: nnz=" << lower_nnz << std::endl;

  // Create solver
  SparseStructure ss;
  ss.ptrs = lower_row_ptr;
  ss.inds = lower_col_idx;

  std::vector<int64_t> block_sizes(n, 1);

  // First try CPU backend to verify matrix is correct
  Settings cpu_settings;
  cpu_settings.backend = BackendFast;  // CPU
  cpu_settings.numThreads = 8;
  cpu_settings.addFillPolicy = AddFillComplete;
  cpu_settings.findSparseEliminationRanges = true;

  SolverPtr cpu_solver = createSolver(cpu_settings, block_sizes, ss);
  ASSERT_TRUE(cpu_solver != nullptr);

  const auto& permutation = cpu_solver->paramToSpan();

  // Extract lower triangular values
  std::vector<float> lower_values(lower_nnz);
  for (int64_t i = 0; i < lower_nnz; ++i) {
    lower_values[i] = csr.values[lower_to_original_idx[i]];
  }

  // Load into factor data
  int64_t data_size = cpu_solver->dataSize();
  std::vector<float> factorData(data_size, 0.0f);

  cpu_solver->loadFromCsr(lower_row_ptr.data(), lower_col_idx.data(),
                      block_sizes.data(), lower_values.data(), factorData.data());

  // Create RHS using sin pattern (same as IREE test)
  std::vector<float> x_true(n);
  for (int64_t i = 0; i < n; ++i) {
    x_true[i] = std::sin(2.0f * M_PI * i / n);
  }
  std::vector<float> b(n, 0.0f);
  for (int64_t row = 0; row < n; ++row) {
    for (int64_t ptr = csr.row_ptr[row]; ptr < csr.row_ptr[row + 1]; ++ptr) {
      b[row] += csr.values[ptr] * x_true[csr.col_idx[ptr]];
    }
  }

  std::cout << "x_true first 5: ";
  for (int i = 0; i < 5; ++i) std::cout << x_true[i] << " ";
  std::cout << std::endl;
  std::cout << "RHS first 5: ";
  for (int i = 0; i < 5; ++i) std::cout << b[i] << " ";
  std::cout << std::endl;

  // Factor and solve on CPU first
  {
    std::vector<float> cpu_data = factorData;
    cpu_solver->factor(cpu_data.data());

    std::cout << "CPU factor data first 10: ";
    for (int i = 0; i < 10; ++i) std::cout << cpu_data[i] << " ";
    std::cout << std::endl;

    std::vector<float> permuted(n);
    for (int64_t i = 0; i < n; ++i) {
      permuted[permutation[i]] = b[i];
    }

    cpu_solver->solve(cpu_data.data(), permuted.data(), n, 1);

    std::vector<float> solution(n);
    for (int64_t i = 0; i < n; ++i) {
      solution[i] = permuted[permutation[i]];
    }

    std::cout << "CPU Solution first 5: ";
    for (int i = 0; i < 5; ++i) std::cout << solution[i] << " ";
    std::cout << std::endl;

    float diff = 0, ref = 0;
    for (int64_t i = 0; i < n; ++i) {
      float d = solution[i] - x_true[i];
      diff += d * d;
      ref += x_true[i] * x_true[i];
    }
    float relError = std::sqrt(diff) / std::sqrt(ref);
    std::cout << "CPU IREE Scale (10,000) relative error: " << relError << std::endl;

    EXPECT_LT(relError, 1e-3) << "CPU IREE scale test should achieve reasonable precision";
  }
}

/**
 * Test Metal backend at 100x100 scale (same as IREE test)
 */
TEST(IREEPattern, Poisson2D_Grid100_Metal) {
  // Test at the same scale as the IREE test (100x100 = 10,000 elements)
  CsrMatrix csr = createPoissonCSR(100);
  int64_t n = csr.n;

  std::cout << "Metal test at 10K scale" << std::endl;

  // Extract lower triangular
  std::vector<int64_t> lower_row_ptr(n + 1);
  std::vector<int64_t> lower_col_idx;
  std::vector<int64_t> lower_to_original_idx;

  lower_row_ptr[0] = 0;
  for (int64_t row = 0; row < n; ++row) {
    for (int64_t ptr = csr.row_ptr[row]; ptr < csr.row_ptr[row + 1]; ++ptr) {
      int64_t col = csr.col_idx[ptr];
      if (col <= row) {
        lower_col_idx.push_back(col);
        lower_to_original_idx.push_back(ptr);
      }
    }
    lower_row_ptr[row + 1] = lower_col_idx.size();
  }

  int64_t lower_nnz = lower_col_idx.size();

  // Create solver with METAL backend (same as IREE)
  SparseStructure ss;
  ss.ptrs = lower_row_ptr;
  ss.inds = lower_col_idx;

  std::vector<int64_t> block_sizes(n, 1);

  Settings settings;
  settings.backend = BackendMetal;  // Same as IREE uses
  settings.numThreads = 8;
  settings.addFillPolicy = AddFillComplete;
  settings.findSparseEliminationRanges = true;

  SolverPtr solver = createSolver(settings, block_sizes, ss);
  ASSERT_TRUE(solver != nullptr);

  const auto& permutation = solver->paramToSpan();
  std::cout << "Permutation first 10: ";
  for (int i = 0; i < 10; ++i) std::cout << permutation[i] << " ";
  std::cout << std::endl;

  // Extract lower triangular values
  std::vector<float> lower_values(lower_nnz);
  for (int64_t i = 0; i < lower_nnz; ++i) {
    lower_values[i] = csr.values[lower_to_original_idx[i]];
  }

  // Load into factor data
  int64_t data_size = solver->dataSize();
  std::vector<float> factorData(data_size, 0.0f);

  solver->loadFromCsr(lower_row_ptr.data(), lower_col_idx.data(),
                      block_sizes.data(), lower_values.data(), factorData.data());

  // Create RHS using sin pattern (same as IREE test)
  std::vector<float> x_true(n);
  for (int64_t i = 0; i < n; ++i) {
    x_true[i] = std::sin(2.0f * M_PI * i / n);
  }
  std::vector<float> b(n, 0.0f);
  for (int64_t row = 0; row < n; ++row) {
    for (int64_t ptr = csr.row_ptr[row]; ptr < csr.row_ptr[row + 1]; ++ptr) {
      b[row] += csr.values[ptr] * x_true[csr.col_idx[ptr]];
    }
  }

  std::cout << "x_true first 5: ";
  for (int i = 0; i < 5; ++i) std::cout << x_true[i] << " ";
  std::cout << std::endl;
  std::cout << "RHS first 5: ";
  for (int i = 0; i < 5; ++i) std::cout << b[i] << " ";
  std::cout << std::endl;

  // Factor and solve on Metal (matching IREE wrapper pattern)
  {
    MetalMirror<float> dataGpu(factorData);
    MetalMirror<float> permuted;
    permuted.resizeToAtLeast(n);

    // Apply permutation to RHS (scatter)
    float* permutedPtr = permuted.ptr();
    for (int64_t i = 0; i < n; ++i) {
      permutedPtr[permutation[i]] = b[i];
    }

    std::cout << "Permuted RHS first 5: ";
    for (int i = 0; i < 5; ++i) std::cout << permutedPtr[i] << " ";
    std::cout << std::endl;

    // Factor with exception handling
    try {
      solver->factor(dataGpu.ptr());
      MetalContext::instance().synchronize();
      std::cout << "Factor succeeded!" << std::endl;
    } catch (const std::exception& e) {
      std::cout << "Factor exception: " << e.what() << std::endl;
      // Continue anyway to compare with IREE
    }

    std::cout << "Factor data after factor first 10: ";
    for (int i = 0; i < 10; ++i) std::cout << dataGpu.ptr()[i] << " ";
    std::cout << std::endl;

    // Solve with exception handling
    try {
      solver->solve(dataGpu.ptr(), permutedPtr, n, 1);
      MetalContext::instance().synchronize();
      std::cout << "Solve succeeded!" << std::endl;
    } catch (const std::exception& e) {
      std::cout << "Solve exception: " << e.what() << std::endl;
    }

    std::cout << "Permuted after solve first 5: ";
    for (int i = 0; i < 5; ++i) std::cout << permutedPtr[i] << " ";
    std::cout << std::endl;

    // Apply inverse permutation (gather)
    std::vector<float> solution(n);
    for (int64_t i = 0; i < n; ++i) {
      solution[i] = permutedPtr[permutation[i]];
    }

    std::cout << "Solution first 5: ";
    for (int i = 0; i < 5; ++i) std::cout << solution[i] << " ";
    std::cout << std::endl;

    float diff = 0, ref = 0;
    for (int64_t i = 0; i < n; ++i) {
      float d = solution[i] - x_true[i];
      diff += d * d;
      ref += x_true[i] * x_true[i];
    }
    float relError = std::sqrt(diff) / std::sqrt(ref);
    std::cout << "Metal 10K relative error: " << relError << std::endl;

    EXPECT_LT(relError, 1e-3) << "Metal at 10K scale should achieve reasonable precision";
  }
}

TEST(IREEPattern, Poisson2D_Grid10_Reference) {
  // This follows the working MetalScalingTest pattern exactly
  int64_t gridSize = 10;
  int64_t n = gridSize * gridSize;

  // Create full CSR to get structure (same as above)
  CsrMatrix csr = createPoissonCSR(gridSize);

  // Extract lower triangular structure
  SparseStructure ss;
  ss.ptrs.resize(n + 1);
  ss.ptrs[0] = 0;

  for (int64_t i = 0; i < n; ++i) {
    int64_t count = 0;
    for (int64_t ptr = csr.row_ptr[i]; ptr < csr.row_ptr[i + 1]; ++ptr) {
      if (csr.col_idx[ptr] <= i) count++;
    }
    ss.ptrs[i + 1] = ss.ptrs[i] + count;
  }

  ss.inds.resize(ss.ptrs[n]);
  int64_t idx = 0;
  std::vector<float> csrValues;
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t ptr = csr.row_ptr[i]; ptr < csr.row_ptr[i + 1]; ++ptr) {
      if (csr.col_idx[ptr] <= i) {
        ss.inds[idx++] = csr.col_idx[ptr];
        csrValues.push_back(csr.values[ptr]);
      }
    }
  }

  // Create solver
  std::vector<int64_t> blockSizes(n, 1);
  Settings settings;
  settings.backend = BackendMetal;
  settings.numThreads = 8;
  settings.addFillPolicy = AddFillComplete;
  settings.findSparseEliminationRanges = true;

  SolverPtr solver = createSolver(settings, blockSizes, ss);

  // Load data
  int64_t dataSize = solver->dataSize();
  std::vector<float> factorData(dataSize, 0.0f);

  // Get permutation
  const auto& permutation = solver->paramToSpan();

  // Convert row pointers to int64_t for loadFromCsr
  std::vector<int64_t> rowPtr(ss.ptrs.begin(), ss.ptrs.end());
  std::vector<int64_t> colIdx(ss.inds.begin(), ss.inds.end());

  solver->loadFromCsr(rowPtr.data(), colIdx.data(), blockSizes.data(),
                      csrValues.data(), factorData.data());

  // Create RHS
  std::vector<float> x_true(n, 1.0f);
  std::vector<float> b(n, 0.0f);
  for (int64_t row = 0; row < n; ++row) {
    for (int64_t ptr = csr.row_ptr[row]; ptr < csr.row_ptr[row + 1]; ++ptr) {
      b[row] += csr.values[ptr] * x_true[csr.col_idx[ptr]];
    }
  }

  // Factor and solve on Metal GPU
  std::vector<float> solution(n);
  {
    MetalMirror<float> dataGpu(factorData);
    MetalMirror<float> rhsGpu;
    rhsGpu.resizeToAtLeast(n);

    // Apply permutation to RHS: permuted[p[i]] = b[i]
    float* rhsPtr = rhsGpu.ptr();
    for (int64_t i = 0; i < n; ++i) {
      rhsPtr[permutation[i]] = b[i];
    }

    // Factor
    solver->factor(dataGpu.ptr());

    // Solve
    solver->solve(dataGpu.ptr(), rhsPtr, n, 1);

    // Sync and get result
    MetalContext::instance().synchronize();

    // Apply inverse permutation: solution[i] = permuted[p[i]]
    for (int64_t i = 0; i < n; ++i) {
      solution[i] = rhsPtr[permutation[i]];
    }
  }

  // Compute error
  float diff = 0, ref = 0;
  for (int64_t i = 0; i < n; ++i) {
    float d = solution[i] - x_true[i];
    diff += d * d;
    ref += x_true[i] * x_true[i];
  }
  float relError = std::sqrt(diff) / std::sqrt(ref);
  std::cout << "Reference pattern relative error: " << relError << std::endl;

  EXPECT_LT(relError, 1e-4) << "Reference pattern should achieve good precision";
}
