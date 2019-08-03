// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/memory/allocation/cuda_device_context_allocator.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace memory {
namespace allocation {

const int NUM_STREAMS = 8;
const int N = 1 << 20;

__global__ void kernel(float *x, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    x[i] = 3.14159 * i;
  }
}

TEST(CUDADeviceContextAllocator, multi_stream) {
  auto place = platform::CUDAPlace(0);

  CUDADeviceContextAllocator allocator(place);
  EXPECT_TRUE(cudaSuccess == cudaSetDevice(0));

  AllocationPtr main_stream_alloc_ptr = Alloc(place, N * sizeof(float));
  EXPECT_EQ(main_stream_alloc_ptr->size(), N * sizeof(float));
  float *main_stream_data =
      reinterpret_cast<float *>(main_stream_alloc_ptr->ptr());

  float *data[NUM_STREAMS];

  for (int i = 0; i < NUM_STREAMS; ++i) {
    // default stream
    kernel<<<1, 64>>>(main_stream_data, N);

    paddle::platform::CUDADeviceContext dev_ctx(place);
    AllocationPtr allocation_ptr = Alloc(dev_ctx, N * sizeof(float));
    VLOG(4) << "Get ptr";
    EXPECT_EQ(allocation_ptr->size(), N * sizeof(float));
    data[i] = reinterpret_cast<float *>(allocation_ptr->ptr());

    // multi-streams
    kernel<<<1, 64, 0, dev_ctx.stream()>>>(data[i], N);
  }
  EXPECT_TRUE(cudaSuccess == cudaDeviceReset());
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
