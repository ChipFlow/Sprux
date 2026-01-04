/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/WebGPUDefs.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace BaSpaCho {

// ============================================================================
// WebGPUContext implementation
// ============================================================================

WebGPUContext& WebGPUContext::instance() {
  static WebGPUContext ctx;
  return ctx;
}

WebGPUContext::WebGPUContext() {
  initDevice();
  loadShaderModule();
}

WebGPUContext::~WebGPUContext() {
  // Release in reverse order of creation
  pipelineCache_.clear();
  shaderModule_ = nullptr;
  queue_ = nullptr;
  device_ = nullptr;
  adapter_ = nullptr;
  instance_ = nullptr;
}

void WebGPUContext::initDevice() {
  // Create instance
  wgpu::InstanceDescriptor instanceDesc{};
  instance_ = wgpu::CreateInstance(&instanceDesc);
  wgpuCHECK(instance_ != nullptr, "Failed to create WebGPU instance");

  // Request adapter synchronously using CallbackInfo2
  wgpu::RequestAdapterOptions adapterOpts{};
  adapterOpts.powerPreference = wgpu::PowerPreference::HighPerformance;

  bool adapterReceived = false;
  wgpu::Adapter receivedAdapter;

  wgpu::RequestAdapterCallbackInfo2 adapterCallbackInfo{};
  adapterCallbackInfo.mode = wgpu::CallbackMode::WaitAnyOnly;
  adapterCallbackInfo.callback = [](WGPURequestAdapterStatus status,
                                    WGPUAdapter adapter,
                                    WGPUStringView message,
                                    void* userdata1,
                                    void* userdata2) {
    auto* received = static_cast<bool*>(userdata1);
    auto* adapterPtr = static_cast<wgpu::Adapter*>(userdata2);
    if (status == WGPURequestAdapterStatus_Success) {
      *received = true;
      *adapterPtr = wgpu::Adapter::Acquire(adapter);
    } else {
      fprintf(stderr, "WebGPU: Failed to get adapter: %.*s\n",
              static_cast<int>(message.length), message.data ? message.data : "unknown error");
      *received = false;
    }
  };
  adapterCallbackInfo.userdata1 = &adapterReceived;
  adapterCallbackInfo.userdata2 = &receivedAdapter;

  wgpu::Future adapterFuture = instance_.RequestAdapter(&adapterOpts, adapterCallbackInfo);

  // Wait for adapter
  wgpu::InstanceWaitStatus waitStatus =
      instance_.WaitAny(adapterFuture, std::numeric_limits<uint64_t>::max());
  wgpuCHECK(waitStatus == wgpu::InstanceWaitStatus::Success, "Failed to wait for adapter");
  wgpuCHECK(adapterReceived, "Adapter callback not received");

  adapter_ = receivedAdapter;
  wgpuCHECK(adapter_ != nullptr, "Failed to get WebGPU adapter");

  // Request device synchronously using CallbackInfo2
  wgpu::DeviceDescriptor deviceDesc{};
  deviceDesc.label = "BaSpaCho Device";

  // Request features we need
  std::vector<wgpu::FeatureName> requiredFeatures;
  // No special features required for basic compute

  deviceDesc.requiredFeatureCount = requiredFeatures.size();
  deviceDesc.requiredFeatures = requiredFeatures.data();

  // Set device lost callback
  deviceDesc.SetDeviceLostCallback(
      wgpu::CallbackMode::AllowSpontaneous,
      [](const wgpu::Device& device, wgpu::DeviceLostReason reason, WGPUStringView message) {
        (void)device;
        fprintf(stderr, "WebGPU device lost: reason=%d, message=%.*s\n",
                static_cast<int>(reason),
                static_cast<int>(message.length), message.data ? message.data : "unknown");
      });

  // Set uncaptured error callback
  deviceDesc.SetUncapturedErrorCallback(
      [](const wgpu::Device& device, wgpu::ErrorType type, WGPUStringView message) {
        (void)device;
        fprintf(stderr, "WebGPU error: type=%d, message=%.*s\n",
                static_cast<int>(type),
                static_cast<int>(message.length), message.data ? message.data : "unknown");
      });

  bool deviceReceived = false;
  wgpu::Device receivedDevice;

  wgpu::RequestDeviceCallbackInfo2 deviceCallbackInfo{};
  deviceCallbackInfo.mode = wgpu::CallbackMode::WaitAnyOnly;
  deviceCallbackInfo.callback = [](WGPURequestDeviceStatus status,
                                   WGPUDevice device,
                                   WGPUStringView message,
                                   void* userdata1,
                                   void* userdata2) {
    auto* received = static_cast<bool*>(userdata1);
    auto* devicePtr = static_cast<wgpu::Device*>(userdata2);
    if (status == WGPURequestDeviceStatus_Success) {
      *received = true;
      *devicePtr = wgpu::Device::Acquire(device);
    } else {
      fprintf(stderr, "WebGPU: Failed to get device: %.*s\n",
              static_cast<int>(message.length), message.data ? message.data : "unknown error");
      *received = false;
    }
  };
  deviceCallbackInfo.userdata1 = &deviceReceived;
  deviceCallbackInfo.userdata2 = &receivedDevice;

  wgpu::Future deviceFuture = adapter_.RequestDevice(&deviceDesc, deviceCallbackInfo);

  // Wait for device
  waitStatus = instance_.WaitAny(deviceFuture, std::numeric_limits<uint64_t>::max());
  wgpuCHECK(waitStatus == wgpu::InstanceWaitStatus::Success, "Failed to wait for device");
  wgpuCHECK(deviceReceived, "Device callback not received");

  device_ = receivedDevice;
  wgpuCHECK(device_ != nullptr, "Failed to get WebGPU device");

  // Get queue
  queue_ = device_.GetQueue();
  wgpuCHECK(queue_ != nullptr, "Failed to get WebGPU queue");
}

void WebGPUContext::loadShaderModule() {
#ifdef BASPACHO_WEBGPU_KERNEL_PATH
  // Load WGSL source from file
  std::ifstream file(BASPACHO_WEBGPU_KERNEL_PATH);
  if (!file.is_open()) {
    fprintf(stderr, "WebGPU: Failed to open kernel file: %s\n", BASPACHO_WEBGPU_KERNEL_PATH);
    abort();
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string wgslSource = buffer.str();

  wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
  wgslDesc.code = wgslSource.c_str();

  wgpu::ShaderModuleDescriptor moduleDesc{};
  moduleDesc.nextInChain = &wgslDesc;
  moduleDesc.label = "BaSpaCho Kernels";

  shaderModule_ = device_.CreateShaderModule(&moduleDesc);
  wgpuCHECK(shaderModule_ != nullptr, "Failed to create shader module");
#else
  fprintf(stderr, "WebGPU: BASPACHO_WEBGPU_KERNEL_PATH not defined\n");
  abort();
#endif
}

void WebGPUContext::synchronize() {
  // Submit an empty command buffer and wait for it to complete
  wgpu::CommandEncoderDescriptor encoderDesc{};
  wgpu::CommandEncoder encoder = device_.CreateCommandEncoder(&encoderDesc);
  wgpu::CommandBuffer cmdBuffer = encoder.Finish();
  queue_.Submit(1, &cmdBuffer);

  // Wait for completion using OnSubmittedWorkDone
  bool done = false;
  queue_.OnSubmittedWorkDone(
      [](WGPUQueueWorkDoneStatus status, void* userdata) {
        (void)status;
        *reinterpret_cast<bool*>(userdata) = true;
      },
      &done);

  while (!done) {
    instance_.ProcessEvents();
  }
}

wgpu::ComputePipeline WebGPUContext::getPipeline(const std::string& entryPoint) {
  std::lock_guard<std::mutex> lock(pipelineMutex_);

  auto it = pipelineCache_.find(entryPoint);
  if (it != pipelineCache_.end()) {
    return it->second;
  }

  // Create compute pipeline
  wgpu::ComputePipelineDescriptor pipelineDesc{};
  pipelineDesc.label = entryPoint.c_str();
  pipelineDesc.compute.module = shaderModule_;
  pipelineDesc.compute.entryPoint = entryPoint.c_str();

  wgpu::ComputePipeline pipeline = device_.CreateComputePipeline(&pipelineDesc);
  wgpuCHECK(pipeline != nullptr, ("Failed to create pipeline for: " + entryPoint).c_str());

  pipelineCache_[entryPoint] = pipeline;
  return pipeline;
}

void WebGPUContext::submit(wgpu::CommandBuffer commandBuffer) {
  queue_.Submit(1, &commandBuffer);
}

void WebGPUContext::processEvents() {
  instance_.ProcessEvents();
}

// ============================================================================
// WebGPUBufferRegistry implementation
// ============================================================================

WebGPUBufferRegistry& WebGPUBufferRegistry::instance() {
  static WebGPUBufferRegistry registry;
  return registry;
}

void WebGPUBufferRegistry::registerBuffer(void* ptr, wgpu::Buffer buffer, size_t sizeBytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  buffers_.push_back({buffer, ptr, sizeBytes});
}

void WebGPUBufferRegistry::unregisterBuffer(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = std::find_if(buffers_.begin(), buffers_.end(),
                         [ptr](const BufferInfo& info) { return info.basePtr == ptr; });
  if (it != buffers_.end()) {
    buffers_.erase(it);
  }
}

std::pair<wgpu::Buffer, size_t> WebGPUBufferRegistry::findBuffer(const void* ptr) const {
  std::lock_guard<std::mutex> lock(mutex_);

  for (const auto& info : buffers_) {
    const char* basePtr = reinterpret_cast<const char*>(info.basePtr);
    const char* endPtr = basePtr + info.sizeBytes;
    const char* queryPtr = reinterpret_cast<const char*>(ptr);

    if (queryPtr >= basePtr && queryPtr < endPtr) {
      size_t offset = static_cast<size_t>(queryPtr - basePtr);
      return {info.buffer, offset};
    }
  }

  return {nullptr, 0};
}

// ============================================================================
// WebGPUMirror template implementation
// ============================================================================

template <typename T>
void WebGPUMirror<T>::clear() {
  if (buffer_ != nullptr) {
    WebGPUBufferRegistry::instance().unregisterBuffer(ptr_);
    buffer_.Unmap();
    buffer_ = nullptr;
  }
  ptr_ = nullptr;
  allocSize_ = 0;
  mappedPtr_ = nullptr;
}

template <typename T>
void WebGPUMirror<T>::resizeToAtLeast(size_t size) {
  if (size <= allocSize_) {
    return;
  }

  clear();

  // Round up to 256-byte alignment (WebGPU requirement)
  size_t alignedSize = ((size * sizeof(T) + 255) / 256) * 256;
  if (alignedSize < 256) alignedSize = 256;  // Minimum buffer size

  wgpu::BufferDescriptor bufferDesc{};
  bufferDesc.size = alignedSize;
  bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst |
                     wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::MapRead |
                     wgpu::BufferUsage::MapWrite;
  bufferDesc.mappedAtCreation = true;

  buffer_ = WebGPUContext::instance().device().CreateBuffer(&bufferDesc);
  CHECK_WEBGPU_ALLOCATION(buffer_, alignedSize);

  ptr_ = reinterpret_cast<T*>(buffer_.GetMappedRange());
  CHECK_WEBGPU_ALLOCATION(ptr_, alignedSize);

  allocSize_ = size;

  // Register with buffer registry
  WebGPUBufferRegistry::instance().registerBuffer(ptr_, buffer_, alignedSize);
}

template <typename T>
void WebGPUMirror<T>::load(const std::vector<T>& vec) {
  if (vec.empty()) {
    clear();
    return;
  }

  resizeToAtLeast(vec.size());

  // Copy data to mapped buffer
  std::memcpy(ptr_, vec.data(), vec.size() * sizeof(T));

  // Unmap to make buffer usable by GPU
  buffer_.Unmap();

  // Re-map for CPU access (WebGPU requires explicit mapping after GPU use)
  // For now, we keep it unmapped until get() is called
}

template <typename T>
void WebGPUMirror<T>::put() {
  // Ensure any host writes are visible to GPU
  // In WebGPU, we need to unmap the buffer before GPU can use it
  if (buffer_ != nullptr) {
    buffer_.Unmap();
  }
}

template <typename T>
void WebGPUMirror<T>::get(std::vector<T>& vec) const {
  if (buffer_ == nullptr || allocSize_ == 0) {
    return;
  }

  // Synchronize GPU operations
  WebGPUContext::instance().synchronize();

  // Map buffer for reading
  bool mapped = false;
  const void* mapPtr = nullptr;

  buffer_.MapAsync(
      wgpu::MapMode::Read, 0, allocSize_ * sizeof(T),
      [](WGPUBufferMapAsyncStatus status, void* userdata) {
        *reinterpret_cast<bool*>(userdata) = (status == WGPUBufferMapAsyncStatus_Success);
      },
      &mapped);

  // Wait for mapping
  while (!mapped) {
    WebGPUContext::instance().processEvents();
  }

  mapPtr = buffer_.GetConstMappedRange();
  if (mapPtr != nullptr) {
    vec.resize(allocSize_);
    std::memcpy(vec.data(), mapPtr, allocSize_ * sizeof(T));
  }

  buffer_.Unmap();
}

// ============================================================================
// WebGPUPtrMirror template implementation
// ============================================================================

template <typename T>
void WebGPUPtrMirror<T>::clear() {
  if (buffer_ != nullptr) {
    buffer_.Unmap();
    buffer_ = nullptr;
  }
  ptr_ = nullptr;
  allocSize_ = 0;
}

template <typename T>
void WebGPUPtrMirror<T>::load(const std::vector<T*>& vec, int64_t offset) {
  clear();

  if (vec.empty()) {
    return;
  }

  size_t size = vec.size();

  // Round up to 256-byte alignment
  size_t alignedSize = ((size * sizeof(T*) + 255) / 256) * 256;
  if (alignedSize < 256) alignedSize = 256;

  wgpu::BufferDescriptor bufferDesc{};
  bufferDesc.size = alignedSize;
  bufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
  bufferDesc.mappedAtCreation = true;

  buffer_ = WebGPUContext::instance().device().CreateBuffer(&bufferDesc);
  CHECK_WEBGPU_ALLOCATION(buffer_, alignedSize);

  ptr_ = reinterpret_cast<T**>(buffer_.GetMappedRange());
  CHECK_WEBGPU_ALLOCATION(ptr_, alignedSize);

  // Copy pointers with offset applied
  for (size_t i = 0; i < size; ++i) {
    ptr_[i] = vec[i] + offset;
  }

  buffer_.Unmap();
  allocSize_ = size;
}

// Explicit template instantiations
template class WebGPUMirror<float>;
template class WebGPUMirror<double>;
template class WebGPUMirror<int64_t>;
template class WebGPUMirror<int32_t>;

template class WebGPUPtrMirror<float>;
template class WebGPUPtrMirror<double>;

}  // end namespace BaSpaCho
