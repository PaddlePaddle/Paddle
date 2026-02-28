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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include "gtest/gtest.h"
#include "paddle/phi/backends/gpu/gpu_info.h"

// ---------------------------------------------------------------------------
// CUDAFunctions.h — covers the 2 missing lines:
//   c10::cuda::device_synchronize() and c10::cuda::stream_synchronize()
// ---------------------------------------------------------------------------

TEST(CUDAFunctionsTest, DeviceSynchronize) {
  // Exercises the PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize()) branch
  ASSERT_NO_THROW(c10::cuda::device_synchronize());
}

TEST(CUDAFunctionsTest, StreamSynchronize) {
  // Exercises phi::backends::gpu::GpuStreamSync()
  auto stream = c10::cuda::getCurrentCUDAStream();
  ASSERT_NO_THROW(c10::cuda::stream_synchronize(stream));
}

TEST(CUDAFunctionsTest, AtNamespaceAliases) {
  // Exercises the using aliases in at::cuda namespace
  ASSERT_NO_THROW(at::cuda::device_synchronize());
  auto stream = c10::cuda::getCurrentCUDAStream();
  ASSERT_NO_THROW(at::cuda::stream_synchronize(stream));
}

// ---------------------------------------------------------------------------
// CUDAContextLight.h — covers the 1 missing line: is_available()
// ---------------------------------------------------------------------------

TEST(CUDAContextLightTest, IsAvailable) {
  // With GPU compilation and at least one device, this must be true.
  int gpu_count = phi::backends::gpu::GetGPUDeviceCount();
  ASSERT_EQ(at::cuda::is_available(), gpu_count > 0);
}

// ---------------------------------------------------------------------------
// CUDAContextLight.cpp — covers all 42 missing lines
// ---------------------------------------------------------------------------

// getNumGPUs() delegages to c10::cuda::device_count()
TEST(CUDAContextLightTest, GetNumGPUs) {
  int64_t n = at::cuda::getNumGPUs();
  ASSERT_GE(n, 1);
}

// getCurrentDeviceProperties() / getDeviceProperties()
TEST(CUDAContextLightTest, DeviceProperties) {
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  ASSERT_NE(prop, nullptr);
  // Sanity-check a few well-known fields
  ASSERT_GT(prop->multiProcessorCount, 0);
  ASSERT_GT(prop->totalGlobalMem, 0UL);

  // getDeviceProperties(explicit device id) must return the same struct
  int device_id = phi::backends::gpu::GetCurrentDeviceId();
  cudaDeviceProp* prop2 = at::cuda::getDeviceProperties(device_id);
  ASSERT_EQ(prop, prop2);
}

// warp_size()
TEST(CUDAContextLightTest, WarpSize) {
  int ws = at::cuda::warp_size();
  // All NVIDIA and AMD GPU architectures have warp size of 32 or 64
  ASSERT_TRUE(ws == 32 || ws == 64);
}

// canDeviceAccessPeer() — a device cannot peer-access itself
TEST(CUDAContextLightTest, CanDeviceAccessPeer) {
  int device_id = phi::backends::gpu::GetCurrentDeviceId();
  // Self-to-self peer access is always false per CUDA spec
  bool self_peer = at::cuda::canDeviceAccessPeer(device_id, device_id);
  ASSERT_FALSE(self_peer);
}

// getCUDADeviceAllocator() — returns nullptr in the Paddle compat layer
TEST(CUDAContextLightTest, GetCUDADeviceAllocator) {
  ASSERT_EQ(at::cuda::getCUDADeviceAllocator(), nullptr);
}

// Handle accessors — all must return non-null handles
TEST(CUDAContextLightTest, GetCurrentCUDABlasHandle) {
  cublasHandle_t h = at::cuda::getCurrentCUDABlasHandle();
  ASSERT_NE(h, nullptr);
}

TEST(CUDAContextLightTest, GetCurrentCUDABlasLtHandle) {
  cublasLtHandle_t h = at::cuda::getCurrentCUDABlasLtHandle();
  ASSERT_NE(h, nullptr);
}

TEST(CUDAContextLightTest, GetCurrentCUDASparseHandle) {
  cusparseHandle_t h = at::cuda::getCurrentCUDASparseHandle();
  ASSERT_NE(h, nullptr);
}

#if defined(CUDART_VERSION) || defined(USE_ROCM)
TEST(CUDAContextLightTest, GetCurrentCUDASolverDnHandle) {
  cusolverDnHandle_t h = at::cuda::getCurrentCUDASolverDnHandle();
  ASSERT_NE(h, nullptr);
}
#endif

// clearCublasWorkspaces() — must not crash (no-op in the compat layer)
TEST(CUDAContextLightTest, ClearCublasWorkspaces) {
  ASSERT_NO_THROW(at::cuda::clearCublasWorkspaces());
}

// cublas_handle_stream_to_workspace() — must return a stable reference
TEST(CUDAContextLightTest, CublasHandleStreamToWorkspace) {
  at::cuda::WorkspaceMapWithMutex& wm =
      at::cuda::cublas_handle_stream_to_workspace();
  // The map should start empty
  ASSERT_TRUE(wm.map.empty());
  // Two calls must return the same singleton
  ASSERT_EQ(&wm, &at::cuda::cublas_handle_stream_to_workspace());
}

// cublaslt_handle_stream_to_workspace() — same contract
TEST(CUDAContextLightTest, CublasLtHandleStreamToWorkspace) {
  at::cuda::WorkspaceMapWithMutex& wm =
      at::cuda::cublaslt_handle_stream_to_workspace();
  ASSERT_TRUE(wm.map.empty());
  ASSERT_EQ(&wm, &at::cuda::cublaslt_handle_stream_to_workspace());
}

// getChosenWorkspaceSize() — must be 32 MiB
TEST(CUDAContextLightTest, GetChosenWorkspaceSize) {
  constexpr size_t kExpected = 32UL * 1024UL * 1024UL;
  ASSERT_EQ(at::cuda::getChosenWorkspaceSize(), kExpected);
}

// getCUDABlasLtWorkspaceSize() / getCUDABlasLtWorkspace()
TEST(CUDAContextLightTest, CUDABlasLtWorkspace) {
  size_t sz = at::cuda::getCUDABlasLtWorkspaceSize();
  ASSERT_GT(sz, 0UL);

  void* ptr = at::cuda::getCUDABlasLtWorkspace();
  ASSERT_NE(ptr, nullptr);
}

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP
