// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include <ATen/cuda/CUDAContext.h>

#include <c10/core/Allocator.h>
#include <mutex>

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"

namespace at::cuda {

namespace {

inline void ensureDeviceContextPoolInitialized() {
  static std::once_flag init_pool_once;
  std::call_once(init_pool_once, []() {
    if (phi::DeviceContextPool::IsInitialized()) {
      return;
    }

    std::vector<phi::Place> places;
    int gpu_count = phi::backends::gpu::GetGPUDeviceCount();
    for (int device = 0; device < gpu_count; ++device) {
      places.emplace_back(phi::GPUPlace(device));
    }
    places.emplace_back(phi::CPUPlace());
    places.emplace_back(phi::GPUPinnedPlace());
    phi::DeviceContextPool::Init(places);
  });
}

/// Returns the GPUContext for the current device.
inline phi::GPUContext* getCurrentGPUContext() {
  ensureDeviceContextPoolInitialized();
  int device_id = phi::backends::gpu::GetCurrentDeviceId();
  return static_cast<phi::GPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::GPUPlace(device_id)));
}

/// Frees a phi::Allocation that was released with .release() during allocate().
static void deletePaddleCUDAAllocation(void* p) {
  delete static_cast<phi::Allocation*>(p);
}

/// Adapter class that wraps Paddle's AllocatorFacade as a c10::Allocator.
/// This provides a bridge between Paddle's allocation interface and PyTorch's
/// c10::Allocator interface for the CUDA compatibility layer.
class PaddleCUDAAllocatorAdapter : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t n) override {
    int device_id = phi::backends::gpu::GetCurrentDeviceId();
    if (n == 0) {
      // Return a DataPtr that carries the current CUDA device without
      // allocating any memory.  Callers that probe device identity via
      // DataPtr::device() (e.g. zero-byte tensor construction) will therefore
      // observe the correct CUDA device rather than a default CPU device.
      // NOTE: For HIP/ROCm builds, PyTorch's compatibility layer still
      // exposes DeviceType::CUDA (kCUDA) rather than a separate HIP device
      // type, so we follow the same convention here.
      return c10::DataPtr(nullptr,
                          nullptr,
                          nullptr,
                          c10::Device(c10::DeviceType::CUDA, device_id));
    }
    auto* alloc = paddle::memory::allocation::AllocatorFacade::Instance()
                      .GetAllocator(phi::GPUPlace(device_id))
                      .get();
    auto phi_alloc = alloc->Allocate(n);
    void* ptr = phi_alloc->ptr();
    phi::Place place = phi_alloc->place();
    // Transfer ownership of phi_alloc to the DataPtr's context.
    auto* raw_alloc = phi_alloc.release();
    return c10::DataPtr(
        ptr, raw_alloc, deletePaddleCUDAAllocation, c10::Device(place));
  }

  void copy_data(void* dst, const void* src, size_t n) const override {
    if (n == 0) return;
      // Use GPU device-to-device copy.  std::memcpy is not valid for device
      // memory; callers such as c10::Allocator::clone() rely on this method to
      // perform correct D2D copies on CUDA/HIP memory.
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpy(dst, src, n, hipMemcpyDeviceToDevice));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice));
#endif
  }

  c10::DeleterFnPtr raw_deleter() const override {
    // allocate() returns data=device_ptr, context=phi::Allocation*, so
    // get() != get_context() and the raw_allocate/raw_deallocate API is
    // unsafe for this allocator.  Returning nullptr signals that.
    return nullptr;
  }
};

}  // namespace

CUDAContextDeviceProp* getCurrentDeviceProperties() {
  int device = phi::backends::gpu::GetCurrentDeviceId();
  return getDeviceProperties(device);
}

int warp_size() { return getCurrentDeviceProperties()->warpSize; }

CUDAContextDeviceProp* getDeviceProperties(c10::DeviceIndex device) {
  return const_cast<CUDAContextDeviceProp*>(
      &phi::backends::gpu::GetDeviceProperties(device));
}

bool canDeviceAccessPeer(c10::DeviceIndex device,
                         c10::DeviceIndex peer_device) {
  int can_access = 0;
#ifdef PADDLE_WITH_HIP
  hipDeviceCanAccessPeer(&can_access, device, peer_device);
#else
  cudaDeviceCanAccessPeer(&can_access, device, peer_device);
#endif
  return can_access != 0;
}

/* Handles */

CUDAContextSparseHandle getCurrentCUDASparseHandle() {
  return getCurrentGPUContext()->cusparse_handle();
}

CUDAContextBlasHandle getCurrentCUDABlasHandle() {
  return getCurrentGPUContext()->cublas_handle();
}

CUDAContextBlasLtHandle getCurrentCUDABlasLtHandle() {
  return getCurrentGPUContext()->cublaslt_handle();
}

void clearCublasWorkspaces() {
  // Workspaces are owned and managed by phi::GPUContext; no explicit
  // cleanup is required here.
}

WorkspaceMapWithMutex& cublas_handle_stream_to_workspace() {
  static WorkspaceMapWithMutex workspace_map;
  return workspace_map;
}

WorkspaceMapWithMutex& cublaslt_handle_stream_to_workspace() {
  static WorkspaceMapWithMutex workspace_map;
  return workspace_map;
}

// Default workspace size consistent with PyTorch's chosen default (32 MiB).
static constexpr size_t kDefaultWorkspaceSize = 32UL * 1024UL * 1024UL;

size_t getChosenWorkspaceSize() { return kDefaultWorkspaceSize; }

size_t getCUDABlasLtWorkspaceSize() {
  // Probe the context with the default size and return what was actually
  // allocated.
  auto [ptr, size] =
      getCurrentGPUContext()->cublaslt_workspace(kDefaultWorkspaceSize);
  (void)ptr;
  return size;
}

void* getCUDABlasLtWorkspace() {
  return getCurrentGPUContext()
      ->cublaslt_workspace(kDefaultWorkspaceSize)
      .first;
}

CUDAContextSolverHandle getCurrentCUDASolverDnHandle() {
  return getCurrentGPUContext()->cusolver_dn_handle();
}

#if defined(USE_CUDSS)
cudssHandle_t getCurrentCudssHandle() {
  // cudss is not yet integrated into phi::GPUContext; not implemented.
  PADDLE_THROW(
      common::errors::Unimplemented("getCurrentCudssHandle() is not "
                                    "implemented in the Paddle compat layer."));
  return nullptr;
}
#endif  // USE_CUDSS

c10::Allocator* getCUDADeviceAllocator() {
  static PaddleCUDAAllocatorAdapter adapter;
  return &adapter;
}

}  // namespace at::cuda

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP
