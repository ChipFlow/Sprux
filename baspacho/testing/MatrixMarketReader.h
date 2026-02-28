/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace BaSpaCho::testing_utils {

struct CsrMatrix {
  int64_t nRows;
  int64_t nCols;
  int64_t nnz;
  std::vector<int64_t> rowPtr;  // size nRows+1
  std::vector<int64_t> colInd;  // size nnz
  std::vector<double> values;   // size nnz
};

// Parse a Matrix Market coordinate file into CSR format.
// Handles: %%MatrixMarket matrix coordinate real general
inline CsrMatrix readMatrixMarket(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("Cannot open: " + path);
  }

  std::string line;
  // Read header line
  std::getline(f, line);
  if (line.find("%%MatrixMarket") == std::string::npos) {
    throw std::runtime_error("Not a MatrixMarket file: " + path);
  }
  if (line.find("coordinate") == std::string::npos) {
    throw std::runtime_error("Only coordinate format supported: " + path);
  }

  // Skip comment lines
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '%') continue;
    break;
  }

  // Parse dimensions: rows cols nnz
  int64_t nRows, nCols, nnz;
  {
    std::istringstream iss(line);
    iss >> nRows >> nCols >> nnz;
  }

  // Read COO triplets
  struct Triplet {
    int64_t row, col;
    double val;
  };
  std::vector<Triplet> triplets;
  triplets.reserve(nnz);

  for (int64_t i = 0; i < nnz; i++) {
    Triplet t;
    f >> t.row >> t.col >> t.val;
    t.row--;  // 1-indexed -> 0-indexed
    t.col--;
    triplets.push_back(t);
  }

  if ((int64_t)triplets.size() != nnz) {
    throw std::runtime_error("Triplet count mismatch in: " + path);
  }

  // Sort by (row, col) for CSR construction
  std::sort(triplets.begin(), triplets.end(), [](const Triplet& a, const Triplet& b) {
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
inline Eigen::VectorXd readRhsVector(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("Cannot open: " + path);
  }

  std::string line;
  std::getline(f, line);
  if (line.find("%%MatrixMarket") == std::string::npos) {
    throw std::runtime_error("Not a MatrixMarket file: " + path);
  }

  bool isCoordinate = line.find("coordinate") != std::string::npos;

  // Skip comments
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '%') continue;
    break;
  }

  int64_t nRows, nCols;
  if (isCoordinate) {
    int64_t nnz;
    std::istringstream iss(line);
    iss >> nRows >> nCols >> nnz;
    if (nCols != 1) {
      throw std::runtime_error("RHS must be a column vector: " + path);
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nRows);
    for (int64_t i = 0; i < nnz; i++) {
      int64_t row, col;
      double val;
      f >> row >> col >> val;
      rhs(row - 1) = val;
    }
    return rhs;
  } else {
    // Array format
    std::istringstream iss(line);
    iss >> nRows >> nCols;
    if (nCols != 1) {
      throw std::runtime_error("RHS must be a column vector: " + path);
    }

    Eigen::VectorXd rhs(nRows);
    for (int64_t i = 0; i < nRows; i++) {
      f >> rhs(i);
    }
    return rhs;
  }
}

// Sparse residual computation: ||Ax - b|| / ||b|| using CSR
inline double computeResidual(const CsrMatrix& A, const Eigen::VectorXd& x,
                              const Eigen::VectorXd& b) {
  Eigen::VectorXd Ax = Eigen::VectorXd::Zero(A.nRows);
  for (int64_t i = 0; i < A.nRows; i++) {
    for (int64_t k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      Ax(i) += A.values[k] * x(A.colInd[k]);
    }
  }
  return (Ax - b).norm() / b.norm();
}

}  // end namespace BaSpaCho::testing_utils
