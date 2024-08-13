// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include "gtest/gtest.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/malloc.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace memory {

__global__ void write_kernel(int* data, uint64_t n, uint64_t step) {
  int thread_num = gridDim.x * blockDim.x;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint64_t i = thread_id; i * step < n; i += thread_num) {
    *(data + i * step) = 1;
  }
}

__global__ void sum_kernel(int* data, uint64_t n, uint64_t step, int* sum) {
  int thread_num = gridDim.x * blockDim.x;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint64_t i = thread_id; i * step < n; i += thread_num) {
    atomicAdd(sum, *(data + i * step));
  }
}

TEST(ManagedMemoryTest, H2DTest) {
  if (!platform::IsGPUManagedMemorySupported(0)) {
    return;
  }

  uint64_t n_data = 1024;
  uint64_t step = 1;
  allocation::AllocationPtr allocation =
      Alloc(phi::GPUPlace(0), n_data * sizeof(int));
  int* data = static_cast<int*>(allocation->ptr());

  memset(data, 0, n_data * sizeof(int));          // located on host memory
  write_kernel<<<1, 1024>>>(data, n_data, step);  // trans to device memory

#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif

  int sum = 0;
  for (uint64_t i = 0; i < n_data; ++i) {
    sum += *(data + i);
  }
  EXPECT_EQ(sum, n_data / step);
  allocation = nullptr;
}

TEST(ManagedMemoryTest, D2HTest) {
  if (!platform::IsGPUManagedMemorySupported(0)) {
    return;
  }

  uint64_t n_data = 1024;
  uint64_t step = 1;
  AllocationPtr allocation = Alloc(phi::GPUPlace(0), n_data * sizeof(int));
  int* data = static_cast<int*>(allocation->ptr());

  write_kernel<<<1, 1024>>>(data, n_data, step);  // located on device memory

#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif

  memset(data, 0, n_data * sizeof(int));  // trans to host memory

  int sum = 0;
  for (uint64_t i = 0; i < n_data; ++i) {
    sum += *(data + i);
  }
  EXPECT_EQ(sum, 0);
}

TEST(ManagedMemoryTest, OversubscribeGPUMemoryTest) {
  if (!platform::IsGPUManagedMemoryOversubscriptionSupported(0)) {
    return;
  }

  uint64_t available_mem = platform::GpuAvailableMemToAlloc();
  uint64_t n_data = available_mem * 2 / sizeof(int) +
                    1;  // requires more than 2 * available_mem bytes
  uint64_t step = std::max(n_data / 1024, static_cast<uint64_t>(1));
  AllocationPtr data_allocation = Alloc(phi::GPUPlace(0), n_data * sizeof(int));
  AllocationPtr sum_allocation = Alloc(phi::GPUPlace(0), sizeof(int));
  int* data = static_cast<int*>(data_allocation->ptr());
  int* sum = static_cast<int*>(sum_allocation->ptr());
  (*sum) = 0;

  write_kernel<<<1, 1024>>>(data, n_data, step);
  sum_kernel<<<1, 1024>>>(data, n_data, step, sum);

#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif

  EXPECT_EQ(*sum, (n_data + step - 1) / step);
}

TEST(ManagedMemoryTest, OOMExceptionTest) {
  if (!platform::IsGPUManagedMemorySupported(0)) {
    return;
  }
  EXPECT_THROW(Alloc(phi::GPUPlace(0), size_t(1) << 60),
               memory::allocation::BadAlloc);
}

}  // namespace memory
}  // namespace paddle
