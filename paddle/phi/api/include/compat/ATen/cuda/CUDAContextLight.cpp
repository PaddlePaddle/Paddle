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

#include <ATen/cuda/CUDAContextLight.h>

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"

namespace at::cuda {

namespace {

/// Returns the GPUContext for the current device.
inline phi::GPUContext* getCurrentGPUContext() {
  int device_id = phi::backends::gpu::GetCurrentDeviceId();
  return static_cast<phi::GPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::GPUPlace(device_id)));
}

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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
CUDAContextSolverHandle getCurrentCUDASolverDnHandle() {
  return getCurrentGPUContext()->cusolver_dn_handle();
}
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

#if defined(USE_CUDSS)
cudssHandle_t getCurrentCudssHandle() {
  // cudss is not yet integrated into phi::GPUContext; not implemented.
  PADDLE_THROW(
      common::errors::Unimplemented("getCurrentCudssHandle() is not "
                                    "implemented in the Paddle compat layer."));
  return nullptr;
}
#endif  // USE_CUDSS

}  // namespace at::cuda

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP
