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

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/cuda.h>

#include "gtest/gtest.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "paddle/phi/backends/gpu/gpu_info.h"
#endif

// Platform-specific definitions for memory operations
#if defined(PADDLE_WITH_HIP)
#include <hip/hip_runtime.h>
#define MEMCPY_FN hipMemcpy
#define MEMCPY_HOST_TO_DEVICE hipMemcpyHostToDevice
#define MEMCPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
#define SUCCESS_CODE hipSuccess
#define DEVICE_SYNCHRONIZE_FN hipDeviceSynchronize
#elif defined(PADDLE_WITH_CUDA)
#define MEMCPY_FN cudaMemcpy
#define MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define SUCCESS_CODE cudaSuccess
#define DEVICE_SYNCHRONIZE_FN cudaDeviceSynchronize
#endif

// ---------------------------------------------------------------------------
// CUDAFunctions.h — covers the 2 missing lines:
//   c10::cuda::device_synchronize() and c10::cuda::stream_synchronize()
// ---------------------------------------------------------------------------

TEST(CUDAFunctionsTest, DeviceSynchronize) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (!at::cuda::is_available()) {
    return;
  }
  // Exercises the PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize()) branch
  ASSERT_NO_THROW(c10::cuda::device_synchronize());
#else
  // In CPU-only builds, device_synchronize throws
  ASSERT_THROW(c10::cuda::device_synchronize(), std::exception);
#endif
}

// CPU-only: torch::cuda::synchronize must report "No CUDA GPUs are available"
// rather than the older "Cannot visit device count" produced by device_count().
// Matches PyTorch behavior where device_count() returns 0 in CPU-only builds
// and the synchronize() pre-check is the single source of the GPU-missing
// error message.
TEST(CUDAFunctionsTest, SynchronizeReportsNoGpuMessageInCpuOnly) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // Only relevant in CPU-only builds
  return;
#else
  try {
    torch::cuda::synchronize();
    FAIL() << "expected exception";
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("No CUDA GPUs are available"), std::string::npos) << msg;
    EXPECT_EQ(msg.find("Cannot visit device count"), std::string::npos) << msg;
  }
#endif
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(CUDAFunctionsTest, StreamSynchronize) {
  if (!at::cuda::is_available()) {
    return;
  }
  // Exercises phi::backends::gpu::GpuStreamSync()
  auto stream = c10::cuda::getCurrentCUDAStream();
  ASSERT_NO_THROW(c10::cuda::stream_synchronize(stream));
}
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(CUDAFunctionsTest, AtNamespaceAliases) {
  if (!at::cuda::is_available()) {
    return;
  }
  // Exercises the using aliases in at::cuda namespace
  ASSERT_NO_THROW(at::cuda::device_synchronize());
  auto stream = c10::cuda::getCurrentCUDAStream();
  ASSERT_NO_THROW(at::cuda::stream_synchronize(stream));
}

TEST(CUDAFunctionsTest, TorchSynchronizePreservesCurrentDevice) {
  if (!torch::cuda::is_available()) {
    return;
  }
  if (torch::cuda::device_count() < 2) {
    return;
  }

  constexpr int current_device = 0;
  constexpr int other_device = 1;
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(current_device));
  ASSERT_EQ(phi::backends::gpu::GetCurrentDeviceId(), current_device);

  ASSERT_NO_THROW(torch::cuda::synchronize(other_device));
  EXPECT_EQ(phi::backends::gpu::GetCurrentDeviceId(), current_device);
}

TEST(CUDAFunctionsTest, SynchronizeRejectsInvalidNegativeDevice) {
  if (!torch::cuda::is_available()) {
    return;
  }
  ASSERT_THROW(torch::cuda::synchronize(-2), std::exception);
}

TEST(CUDAFunctionsTest, CUDAGuardRestoresOriginalDeviceAfterMultipleSwitches) {
  if (!torch::cuda::is_available()) {
    return;
  }
  if (torch::cuda::device_count() < 2) {
    return;
  }

  constexpr int original_device = 0;
  constexpr int intermediate_device = 1;
  phi::backends::gpu::SetDeviceId(original_device);
  ASSERT_EQ(phi::backends::gpu::GetCurrentDeviceId(), original_device);

  {
    c10::cuda::CUDAGuard guard(
        static_cast<c10::DeviceIndex>(intermediate_device));
    ASSERT_EQ(phi::backends::gpu::GetCurrentDeviceId(), intermediate_device);
    guard.set_index(static_cast<c10::DeviceIndex>(original_device));
    ASSERT_EQ(phi::backends::gpu::GetCurrentDeviceId(), original_device);
    guard.set_index(static_cast<c10::DeviceIndex>(intermediate_device));
    ASSERT_EQ(phi::backends::gpu::GetCurrentDeviceId(), intermediate_device);
  }

  EXPECT_EQ(phi::backends::gpu::GetCurrentDeviceId(), original_device);
}

TEST(CUDAFunctionsTest,
     CUDAGuardRestoresOriginalDeviceAfterReturnToOriginalThenExit) {
  if (!torch::cuda::is_available()) {
    return;
  }
  if (torch::cuda::device_count() < 2) {
    return;
  }

  constexpr int original_device = 0;
  constexpr int intermediate_device = 1;
  phi::backends::gpu::SetDeviceId(original_device);
  ASSERT_EQ(phi::backends::gpu::GetCurrentDeviceId(), original_device);

  {
    c10::cuda::CUDAGuard guard(
        static_cast<c10::DeviceIndex>(intermediate_device));
    ASSERT_EQ(phi::backends::gpu::GetCurrentDeviceId(), intermediate_device);

    guard.set_index(static_cast<c10::DeviceIndex>(original_device));
    ASSERT_EQ(phi::backends::gpu::GetCurrentDeviceId(), original_device);
  }

  EXPECT_EQ(phi::backends::gpu::GetCurrentDeviceId(), original_device);
}

TEST(CUDAFunctionsTest,
     OptionalCUDAGuardResetRestoresOriginalDeviceAfterReturnToOriginal) {
  if (!torch::cuda::is_available()) {
    return;
  }
  if (torch::cuda::device_count() < 2) {
    return;
  }

  constexpr int original_device = 0;
  constexpr int intermediate_device = 1;
  phi::backends::gpu::SetDeviceId(original_device);
  ASSERT_EQ(phi::backends::gpu::GetCurrentDeviceId(), original_device);

  c10::cuda::OptionalCUDAGuard guard;
  guard.set_index(static_cast<c10::DeviceIndex>(intermediate_device));
  ASSERT_EQ(phi::backends::gpu::GetCurrentDeviceId(), intermediate_device);

  guard.set_index(static_cast<c10::DeviceIndex>(original_device));
  ASSERT_EQ(phi::backends::gpu::GetCurrentDeviceId(), original_device);

  guard.reset();

  EXPECT_EQ(phi::backends::gpu::GetCurrentDeviceId(), original_device);
  EXPECT_FALSE(guard.original_device().has_value());
  EXPECT_FALSE(guard.current_device().has_value());
}
#endif

// ---------------------------------------------------------------------------
// CUDAContextLight.h — covers the 1 missing line: is_available()
// ---------------------------------------------------------------------------

TEST(CUDAContextLightTest, IsAvailable) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // With GPU compilation and at least one device, this must be true.
  int gpu_count = phi::backends::gpu::GetGPUDeviceCount();
  ASSERT_EQ(at::cuda::is_available(), gpu_count > 0);
#else
  // In CPU-only builds, is_available() should return false
  ASSERT_FALSE(at::cuda::is_available());
#endif
}

// ---------------------------------------------------------------------------
// CUDAContextLight.cpp — covers all 42 missing lines
// ---------------------------------------------------------------------------

// getNumGPUs() delegages to c10::cuda::device_count()
TEST(CUDAContextLightTest, GetNumGPUs) {
  int64_t n = at::cuda::getNumGPUs();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  ASSERT_EQ(n, c10::cuda::device_count());
  ASSERT_GE(n, 0);
#else
  // In CPU-only builds, device_count() returns 0
  ASSERT_EQ(n, 0);
#endif
}

// CPU-only: device_count() must return 0 instead of throwing, matching the
// PyTorch contract that device_count() is a non-throwing query.
TEST(CUDAContextLightTest, DeviceCountReturnsZeroInCpuOnly) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // Only relevant in CPU-only builds
  return;
#else
  ASSERT_NO_THROW({
    EXPECT_EQ(c10::cuda::device_count(), 0);
    EXPECT_EQ(torch::cuda::device_count(), 0);
  });
#endif
}

// CPU-only: is_available() must be false and not throw, matching PyTorch.
TEST(CUDAContextLightTest, IsAvailableFalseAndNoThrowInCpuOnly) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // Only relevant in CPU-only builds
  return;
#else
  ASSERT_NO_THROW({
    EXPECT_FALSE(at::cuda::is_available());
    EXPECT_FALSE(torch::cuda::is_available());
  });
#endif
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

// The following tests require CUDA runtime and can only run in CUDA builds

// getCurrentDeviceProperties() / getDeviceProperties()
TEST(CUDAContextLightTest, DeviceProperties) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::cuda::CUDAContextDeviceProp* prop =
      at::cuda::getCurrentDeviceProperties();
  ASSERT_NE(prop, nullptr);
  // Sanity-check a few well-known fields
  ASSERT_GT(prop->multiProcessorCount, 0);
  ASSERT_GT(prop->totalGlobalMem, 0UL);

  // getDeviceProperties(explicit device id) must return the same struct
  int device_id = phi::backends::gpu::GetCurrentDeviceId();
  at::cuda::CUDAContextDeviceProp* prop2 =
      at::cuda::getDeviceProperties(device_id);
  ASSERT_EQ(prop, prop2);
}

// warp_size()
TEST(CUDAContextLightTest, WarpSize) {
  if (!at::cuda::is_available()) {
    return;
  }
  int ws = at::cuda::warp_size();
  // All NVIDIA and AMD GPU architectures have warp size of 32 or 64
  ASSERT_TRUE(ws == 32 || ws == 64);
}

// canDeviceAccessPeer() — a device cannot peer-access itself
TEST(CUDAContextLightTest, CanDeviceAccessPeer) {
  if (!at::cuda::is_available()) {
    return;
  }
  int device_id = phi::backends::gpu::GetCurrentDeviceId();
  // Self-to-self peer access is always false per CUDA spec
  bool self_peer = at::cuda::canDeviceAccessPeer(device_id, device_id);
  ASSERT_FALSE(self_peer);
}

// Handle accessors — all must return non-null handles
TEST(CUDAContextLightTest, GetCurrentCUDABlasHandle) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::cuda::CUDAContextBlasHandle h = at::cuda::getCurrentCUDABlasHandle();
  ASSERT_NE(h, nullptr);
}

TEST(CUDAContextLightTest, GetCurrentCUDABlasLtHandle) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::cuda::CUDAContextBlasLtHandle h = at::cuda::getCurrentCUDABlasLtHandle();
  ASSERT_NE(h, nullptr);
}

TEST(CUDAContextLightTest, GetCurrentCUDASparseHandle) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::cuda::CUDAContextSparseHandle h = at::cuda::getCurrentCUDASparseHandle();
  ASSERT_NE(h, nullptr);
}

#if defined(CUDART_VERSION) || defined(USE_ROCM)
TEST(CUDAContextLightTest, GetCurrentCUDASolverDnHandle) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::cuda::CUDAContextSolverHandle h =
      at::cuda::getCurrentCUDASolverDnHandle();
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
  if (!at::cuda::is_available()) {
    return;
  }
  size_t sz = at::cuda::getCUDABlasLtWorkspaceSize();
  ASSERT_GT(sz, 0UL);

  void* ptr = at::cuda::getCUDABlasLtWorkspace();
  ASSERT_NE(ptr, nullptr);
}

TEST(CUDAContextLightTest, CUDADeviceAllocatorSingleton) {
  if (!at::cuda::is_available()) {
    return;
  }
  c10::Allocator* a0 = at::cuda::getCUDADeviceAllocator();
  c10::Allocator* a1 = at::cuda::getCUDADeviceAllocator();
  ASSERT_NE(a0, nullptr);
  ASSERT_EQ(a0, a1);
}

TEST(CUDAContextLightTest, CUDADeviceAllocatorCloneAndCopyData) {
  if (!at::cuda::is_available()) {
    return;
  }
  c10::Allocator* alloc = at::cuda::getCUDADeviceAllocator();
  ASSERT_NE(alloc, nullptr);

  constexpr size_t kBytes = 32;
  c10::DataPtr src = alloc->allocate(kBytes);
  ASSERT_NE(src.get(), nullptr);

  uint8_t h_src[kBytes];
  uint8_t h_dst[kBytes];
  for (size_t i = 0; i < kBytes; ++i) {
    h_src[i] = static_cast<uint8_t>(i + 1);
    h_dst[i] = 0;
  }

  ASSERT_EQ(MEMCPY_FN(src.get(), h_src, kBytes, MEMCPY_HOST_TO_DEVICE),
            SUCCESS_CODE);

  c10::DataPtr cloned = alloc->clone(src.get(), kBytes);
  ASSERT_NE(cloned.get(), nullptr);

  ASSERT_EQ(MEMCPY_FN(h_dst, cloned.get(), kBytes, MEMCPY_DEVICE_TO_HOST),
            SUCCESS_CODE);
  ASSERT_EQ(DEVICE_SYNCHRONIZE_FN(), SUCCESS_CODE);

  for (size_t i = 0; i < kBytes; ++i) {
    ASSERT_EQ(h_dst[i], h_src[i]);
  }
}

TEST(CUDAContextLightTest, CUDADeviceAllocatorCloneZeroBytes) {
  if (!at::cuda::is_available()) {
    return;
  }
  c10::Allocator* alloc = at::cuda::getCUDADeviceAllocator();
  ASSERT_NE(alloc, nullptr);

  c10::DataPtr src = alloc->allocate(0);
  ASSERT_EQ(src.get(), nullptr);

  c10::DataPtr cloned = alloc->clone(src.get(), 0);
  ASSERT_EQ(cloned.get(), nullptr);
  ASSERT_EQ(cloned.device().type(), c10::DeviceType::CUDA);
}

TEST(CUDAContextLightTest, AllocatorZeroSizeAndNoopCopyBranches) {
  if (!at::cuda::is_available()) {
    return;
  }
  c10::Allocator* alloc = at::cuda::getCUDADeviceAllocator();
  ASSERT_NE(alloc, nullptr);

  c10::DataPtr zero = alloc->allocate(0);
  ASSERT_EQ(zero.device().type(), c10::DeviceType::CUDA);
  ASSERT_EQ(alloc->raw_deleter(), nullptr);

  // n==0 branch should early-return without touching pointers.
  alloc->copy_data(nullptr, nullptr, 0);
}

#if defined(USE_CUDSS)
TEST(CUDAContextLightTest, CudssHandleIsUnimplemented) {
  ASSERT_THROW((void)at::cuda::getCurrentCudssHandle(), std::exception);
}
#endif

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP
