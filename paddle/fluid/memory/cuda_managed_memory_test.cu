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

#include <cuda.h>
#include <cuda_runtime.h>
#include "gtest/gtest.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/place.h"

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
  uint64_t n_data = 1024;
  uint64_t step = 1;
  AllocationPtr allocation = Alloc(platform::CUDAPlace(), n_data * sizeof(int));
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
}

TEST(UnifiledMemoryTest, D2HTest) {
  uint64_t n_data = 1024;
  uint64_t step = 1;
  AllocationPtr allocation = Alloc(platform::CUDAPlace(), n_data * sizeof(int));
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

TEST(UnifiledMemoryTest, OversubscribeGPUMemoryTest) {
  // requires 16G int data with 4 bytes for each one
  uint64_t n_data = (uint64_t(1)) << 34;
  uint64_t step = 1024;
  AllocationPtr data_allocation =
      Alloc(platform::CUDAPlace(), n_data * sizeof(int));
  AllocationPtr sum_allocation = Alloc(platform::CUDAPlace(), sizeof(int));
  int* data = static_cast<int*>(data_allocation->ptr());
  int* sum = static_cast<int*>(sum_allocation->ptr());
  (*sum) = 0;

  write_kernel<<<5120, 1024>>>(data, n_data, step);
  sum_kernel<<<5120, 1024>>>(data, n_data, step, sum);

#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif

  EXPECT_EQ(*sum, n_data / step);
}

}  // namespace memory
}  // namespace paddle
