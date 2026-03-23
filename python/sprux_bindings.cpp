/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Robert Taylor, 2026. All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Sprux Python bindings using pybind11
//
// Provides a Python interface for sparse direct solving (Cholesky, LU, LDL^T).
// The API mirrors the C++ Solver interface documented in docs/api-guide.md.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "sprux/sprux/Solver.h"

namespace py = pybind11;

namespace {

// Map Python backend string to C++ BackendType enum.
Sprux::BackendType parseBackend(const std::string& backend) {
    if (backend == "cpu" || backend == "fast") return Sprux::BackendFast;
    if (backend == "cuda") return Sprux::BackendCuda;
    if (backend == "metal") return Sprux::BackendMetal;
    if (backend == "opencl") return Sprux::BackendOpenCL;
    if (backend == "auto") return Sprux::BackendAuto;
    throw std::invalid_argument(
        "Unknown backend '" + backend + "'. "
        "Must be one of: 'cpu', 'cuda', 'metal', 'opencl', 'auto'");
}

// Map Python matrix type string to C++ MatrixType enum.
Sprux::MatrixType parseMatrixType(const std::string& matrixType) {
    if (matrixType == "spd") return Sprux::MTYPE_SPD;
    if (matrixType == "general") return Sprux::MTYPE_GENERAL;
    throw std::invalid_argument(
        "Unknown matrix_type '" + matrixType + "'. "
        "Must be one of: 'spd', 'general'");
}

}  // namespace


/**
 * Python wrapper around Sprux::Solver.
 *
 * Manages solver lifetime and provides numpy-based factor/solve methods.
 * Factorization and solve are separate calls (not fused) so callers can
 * re-solve with different RHS vectors without re-factoring.
 */
class PySpruxSolver {
public:
    PySpruxSolver(
        std::vector<int64_t> param_sizes,
        py::array_t<int64_t, py::array::c_style> row_ptrs,
        py::array_t<int64_t, py::array::c_style> col_inds,
        const std::string& matrix_type = "general",
        const std::string& backend = "auto",
        double static_pivot_threshold = -1.0
    ) {
        // Build SparseStructure from CSR arrays
        auto ptrs_buf = row_ptrs.request();
        auto inds_buf = col_inds.request();
        Sprux::SparseStructure ss(
            std::vector<int64_t>(
                static_cast<int64_t*>(ptrs_buf.ptr),
                static_cast<int64_t*>(ptrs_buf.ptr) + ptrs_buf.size),
            std::vector<int64_t>(
                static_cast<int64_t*>(inds_buf.ptr),
                static_cast<int64_t*>(inds_buf.ptr) + inds_buf.size)
        );

        // Configure settings
        Sprux::Settings settings;
        settings.backend = parseBackend(backend);
        settings.matrixType = parseMatrixType(matrix_type);
        settings.staticPivotThreshold = static_pivot_threshold;

        // Perform symbolic analysis
        solver_ = Sprux::createSolver(settings, param_sizes, ss);
    }

    // -- Query methods --

    /// Matrix order (sum of all parameter block sizes).
    int64_t order() const { return solver_->order(); }

    /// Lower-triangle data size (Cholesky / LDL^T storage).
    int64_t data_size() const { return solver_->dataSize(); }

    /// Total data size (lower + upper for LU, lower only for SPD).
    int64_t total_data_size() const { return solver_->totalDataSize(); }

    /// Number of spans (parameter blocks after reordering). Pivot arrays must be this size.
    int64_t num_spans() const { return solver_->skel().numSpans(); }

    /// Matrix type string ("spd" or "general").
    std::string matrix_type() const {
        return solver_->matrixType() == Sprux::MTYPE_GENERAL ? "general" : "spd";
    }

    /// Reordering permutation: paramToSpan[i] is the internal span index for user parameter i.
    py::array_t<int64_t> param_to_span() const {
        const auto& perm = solver_->paramToSpan();
        return py::array_t<int64_t>(perm.size(), perm.data());
    }

    // -- Cholesky (SPD) --

    /// Cholesky factorization: A = L * L^T. Modifies data in place.
    void factor(py::array_t<double, py::array::c_style> data, bool verbose = false) {
        auto buf = data.request();
        checkDataSize(buf.size, solver_->dataSize(), "factor");
        solver_->factor(static_cast<double*>(buf.ptr), verbose);
    }

    /// Solve using Cholesky factor: L * L^T * x = b. Modifies rhs in place.
    void solve(
        py::array_t<double, py::array::c_style> data,
        py::array_t<double, py::array::c_style> rhs,
        int nrhs = 1
    ) {
        auto data_buf = data.request();
        auto rhs_buf = rhs.request();
        int64_t stride = solver_->order();
        solver_->solve(
            static_cast<const double*>(data_buf.ptr),
            static_cast<double*>(rhs_buf.ptr),
            stride, nrhs);
    }

    // -- LU --

    /// LU factorization with partial pivoting: A = P * L * U. Modifies data in place.
    void factor_lu(
        py::array_t<double, py::array::c_style> data,
        py::array_t<int64_t, py::array::c_style> pivots,
        bool verbose = false
    ) {
        auto data_buf = data.request();
        auto piv_buf = pivots.request();
        checkDataSize(data_buf.size, solver_->totalDataSize(), "factor_lu");
        checkDataSize(piv_buf.size, solver_->skel().numSpans(), "factor_lu (pivots)");
        solver_->factorLU(
            static_cast<double*>(data_buf.ptr),
            static_cast<int64_t*>(piv_buf.ptr),
            verbose);
    }

    /// Solve using LU factor: P * L * U * x = b. Modifies rhs in place.
    void solve_lu(
        py::array_t<double, py::array::c_style> data,
        py::array_t<int64_t, py::array::c_style> pivots,
        py::array_t<double, py::array::c_style> rhs,
        int nrhs = 1
    ) {
        auto data_buf = data.request();
        auto piv_buf = pivots.request();
        auto rhs_buf = rhs.request();
        int64_t stride = solver_->order();
        solver_->solveLU(
            static_cast<const double*>(data_buf.ptr),
            static_cast<const int64_t*>(piv_buf.ptr),
            static_cast<double*>(rhs_buf.ptr),
            stride, nrhs);
    }

    // -- LDL^T --

    /// LDL^T factorization for symmetric indefinite matrices. Modifies data in place.
    void factor_ldlt(py::array_t<double, py::array::c_style> data, bool verbose = false) {
        auto buf = data.request();
        checkDataSize(buf.size, solver_->dataSize(), "factor_ldlt");
        solver_->factorLDLT(static_cast<double*>(buf.ptr), verbose);
    }

    /// Solve using LDL^T factor. Modifies rhs in place.
    void solve_ldlt(
        py::array_t<double, py::array::c_style> data,
        py::array_t<double, py::array::c_style> rhs,
        int nrhs = 1
    ) {
        auto data_buf = data.request();
        auto rhs_buf = rhs.request();
        int64_t stride = solver_->order();
        solver_->solveLDLT(
            static_cast<const double*>(data_buf.ptr),
            static_cast<double*>(rhs_buf.ptr),
            stride, nrhs);
    }

    // -- CSR load/extract (for populating internal format from standard CSR) --

    /// Load block CSR values into internal coalesced format.
    void load_from_csr(
        py::array_t<int64_t, py::array::c_style> csr_row_ptrs,
        py::array_t<int64_t, py::array::c_style> csr_col_inds,
        py::array_t<int64_t, py::array::c_style> block_sizes,
        py::array_t<double, py::array::c_style> csr_values,
        py::array_t<double, py::array::c_style> data
    ) {
        auto ptrs_buf = csr_row_ptrs.request();
        auto inds_buf = csr_col_inds.request();
        auto bs_buf = block_sizes.request();
        auto val_buf = csr_values.request();
        auto data_buf = data.request();

        // Zero the full buffer before loading
        std::memset(data_buf.ptr, 0, data_buf.size * sizeof(double));

        solver_->loadFromCsr(
            static_cast<const int64_t*>(ptrs_buf.ptr),
            static_cast<const int64_t*>(inds_buf.ptr),
            static_cast<const int64_t*>(bs_buf.ptr),
            static_cast<const double*>(val_buf.ptr),
            static_cast<double*>(data_buf.ptr));
    }

    /// Extract internal format back to block CSR values.
    void extract_to_csr(
        py::array_t<int64_t, py::array::c_style> csr_row_ptrs,
        py::array_t<int64_t, py::array::c_style> csr_col_inds,
        py::array_t<int64_t, py::array::c_style> block_sizes,
        py::array_t<double, py::array::c_style> data,
        py::array_t<double, py::array::c_style> csr_values
    ) {
        auto ptrs_buf = csr_row_ptrs.request();
        auto inds_buf = csr_col_inds.request();
        auto bs_buf = block_sizes.request();
        auto data_buf = data.request();
        auto val_buf = csr_values.request();

        solver_->extractToCsr(
            static_cast<const int64_t*>(ptrs_buf.ptr),
            static_cast<const int64_t*>(inds_buf.ptr),
            static_cast<const int64_t*>(bs_buf.ptr),
            static_cast<const double*>(data_buf.ptr),
            static_cast<double*>(val_buf.ptr));
    }

    // -- Statistics --

    void enable_stats(bool enabled = true) { solver_->enableStats(enabled); }
    void print_stats() const { solver_->printStats(); }
    void reset_stats() { solver_->resetStats(); }

    /// Number of pivots perturbed by static pivoting during last factorLU call.
    int64_t static_pivot_perturb_count() const { return solver_->staticPivotPerturbCount(); }

private:
    Sprux::SolverPtr solver_;

    static void checkDataSize(int64_t actual, int64_t expected, const char* method) {
        if (actual < expected) {
            throw std::invalid_argument(
                std::string(method) + ": buffer too small (got " +
                std::to_string(actual) + ", need " + std::to_string(expected) + ")");
        }
    }
};


// Module-level factory function matching the documented API:
//   solver = sprux.create_solver(param_sizes, row_ptrs, col_inds, ...)
static std::unique_ptr<PySpruxSolver> create_solver(
    std::vector<int64_t> param_sizes,
    py::array_t<int64_t, py::array::c_style> row_ptrs,
    py::array_t<int64_t, py::array::c_style> col_inds,
    const std::string& matrix_type = "general",
    const std::string& backend = "auto",
    double static_pivot_threshold = -1.0
) {
    return std::make_unique<PySpruxSolver>(
        std::move(param_sizes), row_ptrs, col_inds,
        matrix_type, backend, static_pivot_threshold);
}


PYBIND11_MODULE(sprux_py, m) {
    m.doc() = R"doc(
Sprux: sparse direct solver with GPU acceleration.

Supports Cholesky (SPD), LU with partial pivoting (general),
and LDL^T (symmetric indefinite) factorizations.

Example:
    import sprux_py as sprux
    import numpy as np

    # 4 scalar blocks, CSR sparsity pattern
    solver = sprux.create_solver(
        param_sizes=[1, 1, 1, 1],
        row_ptrs=np.array([0, 2, 4, 7, 9]),
        col_inds=np.array([0, 2, 1, 3, 0, 2, 3, 1, 3]),
        matrix_type="general",
        backend="auto"
    )

    # Allocate buffers
    data = np.zeros(solver.total_data_size)
    pivots = np.zeros(solver.num_spans, dtype=np.int64)

    # ... fill data via solver.load_from_csr() or accessor ...
    solver.factor_lu(data, pivots)

    rhs = np.array([1.0, 2.0, 3.0, 4.0])
    solver.solve_lu(data, pivots, rhs)
    # rhs now contains the solution
)doc";

    // Factory function (recommended entry point)
    m.def("create_solver", &create_solver,
        py::arg("param_sizes"),
        py::arg("row_ptrs"),
        py::arg("col_inds"),
        py::arg("matrix_type") = "general",
        py::arg("backend") = "auto",
        py::arg("static_pivot_threshold") = -1.0,
        R"doc(
Create a Sprux solver from a block-CSR sparsity pattern.

Performs fill-reducing reordering and symbolic factorization.

Args:
    param_sizes: Size of each parameter block (list of ints).
    row_ptrs: CSR row pointers [num_blocks + 1] (numpy int64 array).
    col_inds: CSR column indices (numpy int64 array).
    matrix_type: "spd" (Cholesky/LDL^T) or "general" (LU). Default "general".
    backend: "cpu", "cuda", "metal", "opencl", or "auto". Default "auto".
    static_pivot_threshold: <0 disabled, 0 auto, >0 manual. Default -1.0.

Returns:
    SpruxSolver instance ready for numeric factorization.
)doc");

    // Solver class
    py::class_<PySpruxSolver>(m, "SpruxSolver",
        "Sparse direct solver. Use sprux.create_solver() to construct.")

        // Query properties
        .def_property_readonly("order", &PySpruxSolver::order,
            "Matrix order (sum of all parameter block sizes).")
        .def_property_readonly("data_size", &PySpruxSolver::data_size,
            "Lower-triangle data size (for Cholesky / LDL^T).")
        .def_property_readonly("total_data_size", &PySpruxSolver::total_data_size,
            "Total data size (lower + upper for LU, lower only for SPD).")
        .def_property_readonly("num_spans", &PySpruxSolver::num_spans,
            "Number of spans (reordered parameter blocks). Pivot arrays must be this size.")
        .def_property_readonly("matrix_type", &PySpruxSolver::matrix_type,
            "Matrix type: 'spd' or 'general'.")
        .def("param_to_span", &PySpruxSolver::param_to_span,
            "Reordering permutation: param_to_span[i] is the internal span index for parameter i.")

        // Cholesky
        .def("factor", &PySpruxSolver::factor,
            py::arg("data"), py::arg("verbose") = false,
            "Cholesky factorization (A = L * L^T). Modifies data in place.")
        .def("solve", &PySpruxSolver::solve,
            py::arg("data"), py::arg("rhs"), py::arg("nrhs") = 1,
            "Solve with Cholesky factor. Modifies rhs in place.")

        // LU
        .def("factor_lu", &PySpruxSolver::factor_lu,
            py::arg("data"), py::arg("pivots"), py::arg("verbose") = false,
            "LU factorization with partial pivoting (A = P*L*U). Modifies data in place.")
        .def("solve_lu", &PySpruxSolver::solve_lu,
            py::arg("data"), py::arg("pivots"), py::arg("rhs"), py::arg("nrhs") = 1,
            "Solve with LU factor. Modifies rhs in place.")

        // LDL^T
        .def("factor_ldlt", &PySpruxSolver::factor_ldlt,
            py::arg("data"), py::arg("verbose") = false,
            "LDL^T factorization for symmetric indefinite matrices. Modifies data in place.")
        .def("solve_ldlt", &PySpruxSolver::solve_ldlt,
            py::arg("data"), py::arg("rhs"), py::arg("nrhs") = 1,
            "Solve with LDL^T factor. Modifies rhs in place.")

        // CSR load/extract
        .def("load_from_csr", &PySpruxSolver::load_from_csr,
            py::arg("csr_row_ptrs"), py::arg("csr_col_inds"),
            py::arg("block_sizes"), py::arg("csr_values"), py::arg("data"),
            "Load block CSR values into internal coalesced format.")
        .def("extract_to_csr", &PySpruxSolver::extract_to_csr,
            py::arg("csr_row_ptrs"), py::arg("csr_col_inds"),
            py::arg("block_sizes"), py::arg("data"), py::arg("csr_values"),
            "Extract internal format back to block CSR values.")

        // Statistics
        .def("enable_stats", &PySpruxSolver::enable_stats,
            py::arg("enabled") = true,
            "Enable/disable timing statistics collection.")
        .def("print_stats", &PySpruxSolver::print_stats,
            "Print collected timing statistics to stdout.")
        .def("reset_stats", &PySpruxSolver::reset_stats,
            "Reset collected statistics.")
        .def_property_readonly("static_pivot_perturb_count",
            &PySpruxSolver::static_pivot_perturb_count,
            "Number of pivots perturbed by static pivoting during last factorLU call.");

    // Backend availability queries
    m.def("is_metal_available", []() -> bool {
#ifdef SPRUX_USE_METAL
        return true;
#else
        return false;
#endif
    }, "True if Sprux was compiled with Metal GPU support.");

    m.def("is_cuda_available", []() -> bool {
#ifdef SPRUX_USE_CUDA
        return true;
#else
        return false;
#endif
    }, "True if Sprux was compiled with CUDA GPU support.");

    m.def("is_opencl_available", []() -> bool {
#ifdef SPRUX_USE_OPENCL
        return true;
#else
        return false;
#endif
    }, "True if Sprux was compiled with OpenCL GPU support.");

    m.def("detect_best_backend", []() -> std::string {
        auto backend = Sprux::detectBestBackend();
        switch (backend) {
            case Sprux::BackendCuda: return "cuda";
            case Sprux::BackendMetal: return "metal";
            case Sprux::BackendOpenCL: return "opencl";
            default: return "cpu";
        }
    }, "Detect the best available backend for this system.");
}
