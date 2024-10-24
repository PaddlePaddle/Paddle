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

#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/memory/malloc.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/stream.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

namespace paddle {
namespace memory {

const int NUM_STREAMS = 8;
const int N = 2;
const float DELTA = 1e-1;

__global__ void kernel(float *x, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    x[i] = 3.14159 * i;
  }
}

void CheckKernelOutput(const AllocationPtr &x, int n) {
  auto host_x = std::unique_ptr<float[]>(new float[n]);
  for (int i = 0; i < n; ++i) {
#ifdef PADDLE_WITH_HIP
    EXPECT_TRUE(hipSuccess == hipMemcpy(host_x.get(),
                                        (x->ptr()),
                                        n * sizeof(float),
                                        hipMemcpyDeviceToHost));
#else
    EXPECT_TRUE(cudaSuccess == cudaMemcpy(host_x.get(),
                                          (x->ptr()),
                                          n * sizeof(float),
                                          cudaMemcpyDeviceToHost));
#endif
    EXPECT_GE(host_x[i] + DELTA, 3.14159f * i);
    EXPECT_LE(host_x[i] - DELTA, 3.14159f * i);
  }
}

void MultiStreamCompute(const AllocationPtr &first_data,
                        const AllocationPtr &second_data,
                        phi::GPUContext *ctx) {
  // multi-streams
  EXPECT_GE(first_data->size(), N * sizeof(float));

#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL((kernel),
                     dim3(1),
                     dim3(64),
                     0,
                     ctx->stream(),
                     reinterpret_cast<float *>(first_data->ptr()),
                     N);
#else
  kernel<<<1, 64, 0, ctx->stream()>>>(
      reinterpret_cast<float *>(first_data->ptr()), N);
#endif

  EXPECT_GE(second_data->size(), N * sizeof(float));
  // allocate and compute on same stream again

#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL((kernel),
                     dim3(1),
                     dim3(64),
                     0,
                     ctx->stream(),
                     reinterpret_cast<float *>(second_data->ptr()),
                     N);
#else
  kernel<<<1, 64, 0, ctx->stream()>>>(
      reinterpret_cast<float *>(second_data->ptr()), N);
#endif
}

TEST(Malloc, GPUContextMultiStream) {
  auto place = phi::GPUPlace(0);
  platform::SetDeviceId(0);

  AllocationPtr main_stream_alloc_ptr = Alloc(place, N * sizeof(float));
  EXPECT_GE(main_stream_alloc_ptr->size(), N * sizeof(float));

  AllocationPtr first_data[NUM_STREAMS], second_data[NUM_STREAMS];
  std::vector<phi::GPUContext *> dev_ctx;

// default stream
#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL((kernel),
                     dim3(1),
                     dim3(64),
                     0,
                     0,
                     reinterpret_cast<float *>(main_stream_alloc_ptr->ptr()),
                     N);
#else
  kernel<<<1, 64>>>(reinterpret_cast<float *>(main_stream_alloc_ptr->ptr()), N);
#endif
  main_stream_alloc_ptr.reset();

  for (int i = 0; i < NUM_STREAMS; ++i) {
    auto ctx = new phi::GPUContext(place);
    ctx->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                          .GetAllocator(place, ctx->stream())
                          .get());
    ctx->SetHostAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::CPUPlace())
            .get());
    ctx->SetZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(place)
            .get());
    ctx->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::GPUPinnedPlace())
            .get());
    ctx->PartialInitWithAllocator();
    dev_ctx.emplace_back(ctx);
    first_data[i] =
        Alloc(ctx->GetPlace(),
              N * sizeof(float),
              phi::Stream(reinterpret_cast<phi::StreamId>(ctx->stream())));
    second_data[i] =
        Alloc(ctx->GetPlace(),
              N * sizeof(float),
              phi::Stream(reinterpret_cast<phi::StreamId>(ctx->stream())));
    MultiStreamCompute(first_data[i], second_data[i], ctx);
  }

#ifdef PADDLE_WITH_HIP
  EXPECT_TRUE(hipSuccess == hipDeviceSynchronize());
#else
  EXPECT_TRUE(cudaSuccess == cudaDeviceSynchronize());
#endif

  for (int i = 0; i < NUM_STREAMS; ++i) {
    CheckKernelOutput(first_data[i], N);
    CheckKernelOutput(second_data[i], N);
  }

  // For cudaMallocAsyncAllocator, cudaFreeAsync is executed on _malloc_stream,
  // which is the stream passed at Alloc(). Therefore, the stream must be
  // postponed until the the memory is freed. Otherwise, the stream would be
  // destroyed before the cudaFreeAsync is called.
  for (int i = 0; i < NUM_STREAMS; i++) {
    first_data[i].release();
    second_data[i].release();
    delete dev_ctx[i];
  }
}

TEST(Malloc, GPUContextMultiThreadMultiStream) {
  auto place = phi::GPUPlace(0);
  platform::SetDeviceId(0);

  AllocationPtr main_stream_alloc_ptr = Alloc(place, N * sizeof(float));
  EXPECT_GE(main_stream_alloc_ptr->size(), N * sizeof(float));

  AllocationPtr first_data[NUM_STREAMS], second_data[NUM_STREAMS];
  std::vector<phi::GPUContext *> dev_ctx;

// default stream
#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL((kernel),
                     dim3(1),
                     dim3(64),
                     0,
                     0,
                     reinterpret_cast<float *>(main_stream_alloc_ptr->ptr()),
                     N);
#else
  kernel<<<1, 64>>>(reinterpret_cast<float *>(main_stream_alloc_ptr->ptr()), N);
#endif
  main_stream_alloc_ptr.reset();
  std::vector<std::thread> threads;

  for (int i = 0; i < NUM_STREAMS; ++i) {
    auto ctx = new phi::GPUContext(place);
    ctx->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                          .GetAllocator(place, ctx->stream())
                          .get());
    ctx->SetHostAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::CPUPlace())
            .get());
    ctx->SetZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(place)
            .get());
    ctx->SetHostZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(phi::CPUPlace())
            .get());
    ctx->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::GPUPinnedPlace())
            .get());
    ctx->PartialInitWithAllocator();
    dev_ctx.emplace_back(ctx);
    first_data[i] =
        Alloc(ctx->GetPlace(),
              N * sizeof(float),
              phi::Stream(reinterpret_cast<phi::StreamId>(ctx->stream())));
    second_data[i] =
        Alloc(ctx->GetPlace(),
              N * sizeof(float),
              phi::Stream(reinterpret_cast<phi::StreamId>(ctx->stream())));
    threads.emplace_back(MultiStreamCompute,
                         std::ref(first_data[i]),
                         std::ref(second_data[i]),
                         ctx);
  }

  for (int i = 0; i < NUM_STREAMS; ++i) {
    threads[i].join();
  }

#ifdef PADDLE_WITH_HIP
  EXPECT_TRUE(hipSuccess == hipDeviceSynchronize());
#else
  EXPECT_TRUE(cudaSuccess == cudaDeviceSynchronize());
#endif

  for (int i = 0; i < NUM_STREAMS; ++i) {
    CheckKernelOutput(first_data[i], N);
    CheckKernelOutput(second_data[i], N);
  }

  // There are dependencies on the pointer deconstructing. Manually
  // release the pointers would resolve the conflict.
  for (int i = 0; i < NUM_STREAMS; i++) {
    first_data[i].release();
    second_data[i].release();
    delete dev_ctx[i];
  }
}

TEST(Malloc, AllocZero) {
  auto place = phi::GPUPlace(0);
  AllocationPtr allocation_ptr = Alloc(place, 0);
  EXPECT_GE(allocation_ptr->size(), 0);
}

TEST(Malloc, AllocWithStream) {
  size_t size = 1024;
  AllocationPtr allocation = Alloc(phi::GPUPlace(), size, phi::Stream(0));
  EXPECT_EQ(allocation->size(), 1024);
}

}  // namespace memory
}  // namespace paddle
