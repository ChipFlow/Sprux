/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file WebGPUDefs.h
 * @brief WebGPU GPU backend definitions for BaSpaCho
 *
 * This file provides the WebGPU compute backend using Dawn (Google's WebGPU implementation).
 *
 * ## Precision Limitation
 *
 * **The WebGPU backend only supports single-precision (float) operations.**
 *
 * WebGPU/WGSL does not have widespread double-precision support across all GPU backends.
 * Attempting to use double precision with the WebGPU backend will result in a
 * runtime error with a clear message. Use BackendFast (CPU) or BackendCuda
 * (NVIDIA GPU) for double precision requirements.
 *
 * ## Usage
 *
 * ```cpp
 * #include "baspacho/baspacho/Solver.h"
 *
 * // Create solver with WebGPU backend (float only)
 * Settings settings;
 * settings.backend = BackendWebGPU;
 * auto solver = createSolver<float>(paramSize, structure, settings);
 *
 * // Use WebGPUMirror for GPU memory
 * WebGPUMirror<float> dataGpu(hostData);
 * solver.factor(dataGpu.ptr());
 * dataGpu.get(hostData);  // Copy back to CPU
 * ```
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Dawn WebGPU C++ API
#include <webgpu/webgpu_cpp.h>

namespace BaSpaCho {

// Error checking macro for WebGPU operations
#define wgpuCHECK(condition, msg)                                                \
  do {                                                                           \
    if (!(condition)) {                                                          \
      fprintf(stderr, "[%s:%d] WebGPU Error: %s\n", __FILE__, __LINE__, (msg));  \
      abort();                                                                   \
    }                                                                            \
  } while (0)

#define CHECK_WEBGPU_ALLOCATION(buffer, size)                                \
  if (buffer == nullptr) {                                                   \
    fprintf(stderr, "WebGPU: allocation of block of %ld bytes failed\n",     \
            static_cast<long>(size));                                        \
    abort();                                                                  \
  }

// WebGPU context singleton - manages device, command queue, and shader module
class WebGPUContext {
 public:
  static WebGPUContext& instance();

  // Get WebGPU objects
  wgpu::Device device() const { return device_; }
  wgpu::Queue queue() const { return queue_; }
  wgpu::ShaderModule shaderModule() const { return shaderModule_; }

  // Wait for all GPU operations to complete
  void synchronize();

  // Get a compute pipeline for a kernel function
  wgpu::ComputePipeline getPipeline(const std::string& entryPoint);

  // Submit a command buffer
  void submit(wgpu::CommandBuffer commandBuffer);

 private:
  WebGPUContext();
  ~WebGPUContext();
  WebGPUContext(const WebGPUContext&) = delete;
  WebGPUContext& operator=(const WebGPUContext&) = delete;

  void initDevice();
  void loadShaderModule();

  wgpu::Instance instance_;
  wgpu::Adapter adapter_;
  wgpu::Device device_;
  wgpu::Queue queue_;
  wgpu::ShaderModule shaderModule_;

  mutable std::mutex pipelineMutex_;
  std::unordered_map<std::string, wgpu::ComputePipeline> pipelineCache_;
};

// Buffer registry for mapping raw pointers back to their WGPUBuffers
// This is needed because compute operations require WGPUBuffer objects
class WebGPUBufferRegistry {
 public:
  static WebGPUBufferRegistry& instance();

  // Register a buffer with its base pointer and size
  void registerBuffer(void* ptr, wgpu::Buffer buffer, size_t sizeBytes);

  // Unregister a buffer
  void unregisterBuffer(void* ptr);

  // Find the buffer containing a given pointer, returns {buffer, byteOffset}
  // Returns {nullptr, 0} if not found
  std::pair<wgpu::Buffer, size_t> findBuffer(const void* ptr) const;

 private:
  WebGPUBufferRegistry() = default;
  ~WebGPUBufferRegistry() = default;
  WebGPUBufferRegistry(const WebGPUBufferRegistry&) = delete;
  WebGPUBufferRegistry& operator=(const WebGPUBufferRegistry&) = delete;

  struct BufferInfo {
    wgpu::Buffer buffer;
    void* basePtr;      // Base pointer from mapped buffer
    size_t sizeBytes;   // Size in bytes
  };
  mutable std::mutex mutex_;
  std::vector<BufferInfo> buffers_;
};

// Utility class to mirror an std::vector on the GPU via WebGPU buffer
// Uses MapMode for CPU/GPU access
template <typename T>
class WebGPUMirror {
 public:
  WebGPUMirror() : ptr_(nullptr), allocSize_(0), mappedPtr_(nullptr) {}

  explicit WebGPUMirror(const std::vector<T>& vec)
      : ptr_(nullptr), allocSize_(0), mappedPtr_(nullptr) {
    load(vec);
  }

  ~WebGPUMirror() { clear(); }

  // Non-copyable
  WebGPUMirror(const WebGPUMirror&) = delete;
  WebGPUMirror& operator=(const WebGPUMirror&) = delete;

  // Movable
  WebGPUMirror(WebGPUMirror&& other) noexcept
      : buffer_(std::move(other.buffer_)),
        ptr_(other.ptr_),
        allocSize_(other.allocSize_),
        mappedPtr_(other.mappedPtr_) {
    other.ptr_ = nullptr;
    other.allocSize_ = 0;
    other.mappedPtr_ = nullptr;
  }

  WebGPUMirror& operator=(WebGPUMirror&& other) noexcept {
    if (this != &other) {
      clear();
      buffer_ = std::move(other.buffer_);
      ptr_ = other.ptr_;
      allocSize_ = other.allocSize_;
      mappedPtr_ = other.mappedPtr_;
      other.ptr_ = nullptr;
      other.allocSize_ = 0;
      other.mappedPtr_ = nullptr;
    }
    return *this;
  }

  void clear();
  void resizeToAtLeast(size_t size);
  void load(const std::vector<T>& vec);

  // Copy data from GPU back to a vector
  void get(std::vector<T>& vec) const;

  // Get raw pointer for host access (mapped buffer)
  T* ptr() const { return ptr_; }

  // Sync host writes to GPU (call before GPU compute)
  void put();

  // Get WebGPU buffer handle (for binding to compute pass)
  wgpu::Buffer buffer() const { return buffer_; }

  size_t allocSize() const { return allocSize_; }

 private:
  wgpu::Buffer buffer_;
  T* ptr_;             // CPU-visible pointer from mapped buffer
  size_t allocSize_;
  mutable void* mappedPtr_;  // For async mapping operations
};

// Utility class to mirror an std::vector of pointers, applying an offset
// Used for batched operations
template <typename T>
class WebGPUPtrMirror {
 public:
  WebGPUPtrMirror() : ptr_(nullptr), allocSize_(0) {}

  WebGPUPtrMirror(const std::vector<T*>& vec, int64_t offset = 0)
      : ptr_(nullptr), allocSize_(0) {
    load(vec, offset);
  }

  ~WebGPUPtrMirror() { clear(); }

  // Non-copyable
  WebGPUPtrMirror(const WebGPUPtrMirror&) = delete;
  WebGPUPtrMirror& operator=(const WebGPUPtrMirror&) = delete;

  void clear();
  void load(const std::vector<T*>& vec, int64_t offset = 0);

  T** ptr() const { return ptr_; }
  wgpu::Buffer buffer() const { return buffer_; }
  size_t allocSize() const { return allocSize_; }

 private:
  wgpu::Buffer buffer_;
  T** ptr_;
  size_t allocSize_;
};

// Explicit template instantiations declared (defined in WebGPUDefs.cpp)
extern template class WebGPUMirror<float>;
extern template class WebGPUMirror<double>;
extern template class WebGPUMirror<int64_t>;
extern template class WebGPUMirror<int32_t>;

extern template class WebGPUPtrMirror<float>;
extern template class WebGPUPtrMirror<double>;

}  // end namespace BaSpaCho
