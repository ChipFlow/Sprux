#!/bin/bash
# Sprux GPU Test Runner for Cloud Run
# This script is executed inside the Cloud Run container

set -euo pipefail

echo "=== Sprux GPU Test Runner ==="
echo "Date: $(date)"
echo "Commit: ${GITHUB_SHA:-unknown}"
echo ""

# GPU diagnostics
echo "=== GPU Diagnostics ==="
nvidia-smi || echo "nvidia-smi not available"
echo ""

# Clone repository
echo "=== Cloning Repository ==="
REPO_URL="https://github.com/${GITHUB_REPOSITORY:-facebookresearch/baspacho}.git"
if [ -n "${GITHUB_TOKEN:-}" ]; then
    REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
fi

git clone --depth=1 "$REPO_URL" /workspace/sprux
cd /workspace/sprux

# Checkout specific commit if provided
if [ -n "${GITHUB_SHA:-}" ]; then
    echo "Fetching commit: $GITHUB_SHA"
    git fetch --depth=1 origin "$GITHUB_SHA"
    git checkout "$GITHUB_SHA"
fi

echo ""

# Start sccache and show initial stats
echo "=== sccache Stats (before build) ==="
sccache --start-server 2>/dev/null || true
sccache --show-stats || echo "sccache not available"
echo ""

# Configure and build
echo "=== Configuring CMake ==="
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DSPRUX_USE_CUBLAS=ON \
    -DSPRUX_USE_METAL=OFF \
    -DSPRUX_USE_OPENCL=OFF \
    -DSPRUX_BUILD_TESTS=ON \
    -DSPRUX_BUILD_EXAMPLES=ON

echo ""
echo "=== Building ==="
cmake --build build -- -j$(nproc)

echo ""
echo "=== sccache Stats (after build) ==="
sccache --show-stats || true

echo ""

# Run tests
echo "=== Running Tests ==="
cd build
ctest --output-on-failure -j$(nproc)
TEST_RESULT=$?

echo ""

# Profile CUDA LU operations to verify all ops are on GPU
if [ $TEST_RESULT -eq 0 ] && command -v nsys &> /dev/null; then
    echo "=== Profiling CUDA LU (Nsight Systems) ==="
    nsys profile --trace=cuda --stats=true --force-overwrite=true \
        --output /tmp/cuda_lu_profile \
        ./sprux/tests/CudaLUTest --gtest_filter="CudaLU.BlockSparse_double" 2>&1 | \
        grep -E "cublas|cusolver|Kernel|cudaMemcpy|CUDA API|GPU" || true
    echo ""
fi

# Run benchmarks if tests passed
if [ $TEST_RESULT -eq 0 ]; then
    echo "=== Running Benchmarks ==="

    # Quick benchmark with a few problems
    echo "--- CUDA Backend ---"
    ./sprux/benchmarking/bench -S "CUDA" -n 3 2>&1 || echo "CUDA benchmark completed with warnings"

    echo ""
    echo "--- BLAS CPU Backend ---"
    ./sprux/benchmarking/bench -S "BLAS" -n 3 2>&1 || echo "BLAS benchmark completed with warnings"
fi

echo ""
echo "=== GPU Test Complete ==="
exit $TEST_RESULT
