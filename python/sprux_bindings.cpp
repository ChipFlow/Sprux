/*
 * Copyright (c) Robert Taylor, 2026. All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Sprux Python bindings using pybind11
// This provides a Python interface for sparse LU factorization
// compatible with Spineax's solver interface.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "sprux/sprux/Solver.h"

namespace py = pybind11;

// Wrapper class that manages Sprux solver state
class PySpruxSolver {
public:
    PySpruxSolver(
        py::array_t<int64_t> indptr,
        py::array_t<int64_t> indices,
        int64_t n,
        const std::string& backend = "auto",
        const std::string& matrix_type = "general"
    ) : n_(n) {
        auto indptr_buf = indptr.request();
        auto indices_buf = indices.request();

        int64_t nnz = indices_buf.size;

        // Create solver based on backend selection
        BaSpaCho::OpsPtr ops;
        if (backend == "metal") {
#ifdef SPRUX_USE_METAL
            ops = BaSpaCho::createMetalOps();
#else
            throw std::runtime_error("Sprux was not built with Metal support");
#endif
        } else if (backend == "cuda") {
#ifdef SPRUX_USE_CUDA
            ops = BaSpaCho::createCudaOps();
#else
            throw std::runtime_error("Sprux was not built with CUDA support");
#endif
        } else if (backend == "opencl") {
#ifdef SPRUX_USE_OPENCL
            ops = BaSpaCho::createOpenCLOps();
#else
            throw std::runtime_error("Sprux was not built with OpenCL support");
#endif
        } else {
            // CPU/auto - use Eigen backend
            ops = BaSpaCho::createEigenOps();
        }

        // Create sparse structure from CSR format
        std::vector<int64_t> row_ptr(
            static_cast<int64_t*>(indptr_buf.ptr),
            static_cast<int64_t*>(indptr_buf.ptr) + n + 1
        );
        std::vector<int64_t> col_idx(
            static_cast<int64_t*>(indices_buf.ptr),
            static_cast<int64_t*>(indices_buf.ptr) + nnz
        );

        // Create solver with symbolic analysis
        auto sparseStruct = BaSpaCho::SparseStructure::fromCSR(n, row_ptr, col_idx);

        // Create solver - will perform fill-reducing ordering and symbolic factorization
        solver_ = BaSpaCho::createSolver(sparseStruct, std::move(ops));

        // Allocate factor storage
        factor_nnz_ = solver_->factorDataSize();
        factor_data_.resize(factor_nnz_);
        pivots_.resize(n);
    }

    // Solve Ax = b using LU factorization
    // Returns (x, inertia) where inertia = [positive_eigenvalues, negative_eigenvalues]
    std::tuple<py::array_t<double>, py::array_t<int32_t>> solve(
        py::array_t<double> b,
        py::array_t<double> data
    ) {
        auto b_buf = b.request();
        auto data_buf = data.request();

        if (b_buf.size != n_) {
            throw std::runtime_error("RHS size mismatch");
        }

        // Copy data to factor storage
        double* values = static_cast<double*>(data_buf.ptr);
        std::memcpy(factor_data_.data(), values,
                    std::min((size_t)data_buf.size, factor_data_.size()) * sizeof(double));

        // Perform LU factorization
        solver_->factorLU(factor_data_.data(), pivots_.data());

        // Solve
        py::array_t<double> x(n_);
        auto x_buf = x.request();
        double* x_ptr = static_cast<double*>(x_buf.ptr);
        double* b_ptr = static_cast<double*>(b_buf.ptr);

        // Copy b to x (solve is in-place)
        std::memcpy(x_ptr, b_ptr, n_ * sizeof(double));

        // Apply permutation and solve
        solver_->solveLU(factor_data_.data(), pivots_.data(), x_ptr, n_, 1);

        // Compute inertia from pivot signs
        int32_t positive = 0, negative = 0;
        for (int64_t i = 0; i < n_; ++i) {
            if (factor_data_[i] > 0) positive++;
            else if (factor_data_[i] < 0) negative++;
        }

        py::array_t<int32_t> inertia(2);
        auto inertia_buf = inertia.request();
        int32_t* inertia_ptr = static_cast<int32_t*>(inertia_buf.ptr);
        inertia_ptr[0] = positive;
        inertia_ptr[1] = negative;

        return std::make_tuple(x, inertia);
    }

    int64_t n() const { return n_; }
    int64_t factor_nnz() const { return factor_nnz_; }

private:
    int64_t n_;
    int64_t factor_nnz_;
    std::unique_ptr<BaSpaCho::Solver> solver_;
    std::vector<double> factor_data_;
    std::vector<int64_t> pivots_;
};


PYBIND11_MODULE(sprux_py, m) {
    m.doc() = "Sprux sparse LU solver Python bindings";

    py::class_<PySpruxSolver>(m, "SpruxSolver")
        .def(py::init<py::array_t<int64_t>, py::array_t<int64_t>, int64_t,
                      const std::string&, const std::string&>(),
             py::arg("indptr"),
             py::arg("indices"),
             py::arg("n"),
             py::arg("backend") = "auto",
             py::arg("matrix_type") = "general",
             "Create a Sprux solver for a CSR sparsity pattern")
        .def("solve", &PySpruxSolver::solve,
             py::arg("b"),
             py::arg("data"),
             "Solve Ax = b, returns (x, inertia)")
        .def_property_readonly("n", &PySpruxSolver::n)
        .def_property_readonly("factor_nnz", &PySpruxSolver::factor_nnz);

    // Expose backend availability
    m.def("is_metal_available", []() {
#ifdef SPRUX_USE_METAL
        return true;
#else
        return false;
#endif
    });

    m.def("is_cuda_available", []() {
#ifdef SPRUX_USE_CUDA
        return true;
#else
        return false;
#endif
    });

    m.def("is_opencl_available", []() {
#ifdef SPRUX_USE_OPENCL
        return true;
#else
        return false;
#endif
    });
}
