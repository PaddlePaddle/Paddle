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

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/gpu_info.h"

#if defined(PADDLE_WITH_CUDA)
DECLARE_int64(gpu_allocator_retry_time);
#endif

namespace paddle {
namespace memory {

const int NUM_STREAMS = 8;
const int N = 2;
const float DELTA = 1e-1;

using CudaDevCtxVec = std::vector<std::unique_ptr<platform::CUDADeviceContext>>;

__global__ void kernel(float *x, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    x[i] = 3.14159 * i;
  }
}

void CheckKernelOutput(float *x, int n) {
  auto host_x = std::unique_ptr<float[]>(new float[n]);
  for (int i = 0; i < n; ++i) {
#ifdef PADDLE_WITH_HIP
    EXPECT_TRUE(hipSuccess == hipMemcpy(host_x.get(), x, n * sizeof(float),
                                        hipMemcpyDeviceToHost));
#else
    EXPECT_TRUE(cudaSuccess == cudaMemcpy(host_x.get(), x, n * sizeof(float),
                                          cudaMemcpyDeviceToHost));
#endif
    EXPECT_GE(host_x[i] + DELTA, 3.14159f * i);
    EXPECT_LE(host_x[i] - DELTA, 3.14159f * i);
  }
}

void MultiStreamCompute(float **data, float **second_data,
                        const platform::CUDADeviceContext &ctx) {
  // multi-streams
  AllocationPtr allocation_ptr = Alloc(ctx, N * sizeof(float));
  EXPECT_GE(allocation_ptr->size(), N * sizeof(float));
  *data = reinterpret_cast<float *>(allocation_ptr->ptr());
#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL((kernel), dim3(1), dim3(64), 0, ctx.stream(), *data, N);
#else
  kernel<<<1, 64, 0, ctx.stream()>>>(*data, N);
#endif

  // allocate and compute on same stream again
  allocation_ptr = Alloc(ctx, N * sizeof(float));
  EXPECT_GE(allocation_ptr->size(), N * sizeof(float));
  *second_data = reinterpret_cast<float *>(allocation_ptr->ptr());
#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL((kernel), dim3(1), dim3(64), 0, ctx.stream(), *second_data,
                     N);
#else
  kernel<<<1, 64, 0, ctx.stream()>>>(*second_data, N);
#endif
}

TEST(Malloc, CUDADeviceContextMultiStream) {
  auto place = platform::CUDAPlace(0);
  platform::SetDeviceId(0);

  AllocationPtr main_stream_alloc_ptr = Alloc(place, N * sizeof(float));
  EXPECT_GE(main_stream_alloc_ptr->size(), N * sizeof(float));
  float *main_stream_data =
      reinterpret_cast<float *>(main_stream_alloc_ptr->ptr());

  float *data[NUM_STREAMS];
  float *second_data[NUM_STREAMS];
  CudaDevCtxVec dev_ctx;

// default stream
#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL((kernel), dim3(1), dim3(64), 0, 0, main_stream_data, N);
#else
  kernel<<<1, 64>>>(main_stream_data, N);
#endif
  main_stream_alloc_ptr.reset();

  for (int i = 0; i < NUM_STREAMS; ++i) {
    dev_ctx.push_back(std::unique_ptr<platform::CUDADeviceContext>(
        new platform::CUDADeviceContext(place)));
    MultiStreamCompute(&data[i], &second_data[i], *dev_ctx[i]);
  }

#ifdef PADDLE_WITH_HIP
  EXPECT_TRUE(hipSuccess == hipDeviceSynchronize());
#else
  EXPECT_TRUE(cudaSuccess == cudaDeviceSynchronize());
#endif
  for (int i = 0; i < NUM_STREAMS; ++i) {
    CheckKernelOutput(data[i], N);
    CheckKernelOutput(second_data[i], N);
  }
}

TEST(Malloc, CUDADeviceContextMultiThreadMultiStream) {
  auto place = platform::CUDAPlace(0);
  platform::SetDeviceId(0);

  AllocationPtr main_stream_alloc_ptr = Alloc(place, N * sizeof(float));
  EXPECT_GE(main_stream_alloc_ptr->size(), N * sizeof(float));
  float *main_stream_data =
      reinterpret_cast<float *>(main_stream_alloc_ptr->ptr());

  float *data[NUM_STREAMS];
  float *second_data[NUM_STREAMS];
  CudaDevCtxVec dev_ctx;
  std::vector<std::thread> threads;

// default stream
#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL((kernel), dim3(1), dim3(64), 0, 0, main_stream_data, N);
#else
  kernel<<<1, 64>>>(main_stream_data, N);
#endif
  main_stream_alloc_ptr.reset();

  for (int i = 0; i < NUM_STREAMS; ++i) {
    dev_ctx.push_back(std::unique_ptr<platform::CUDADeviceContext>(
        new platform::CUDADeviceContext(place)));
    threads.push_back(std::thread(MultiStreamCompute, &data[i], &second_data[i],
                                  std::cref(*dev_ctx[i])));
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
    CheckKernelOutput(data[i], N);
    CheckKernelOutput(second_data[i], N);
  }
}

TEST(Malloc, AllocZero) {
  auto place = platform::CUDAPlace(0);
  AllocationPtr allocation_ptr = Alloc(place, 0);
  EXPECT_GE(allocation_ptr->size(), 0);
}

TEST(Malloc, CUDAAllocRetry) {
  platform::Place place = platform::CUDAPlace();
  size_t available_size = platform::GpuAvailableMemToAlloc();
  // alloc_size < available_size < 2 * alloc_size
  size_t alloc_size = available_size / 4 * 3;

  auto alloc_fun = [&place, alloc_size]() {
    return AllocShared(place, alloc_size);
  };
  std::shared_ptr<Allocation> allocation = alloc_fun();
  auto start_time = std::chrono::steady_clock::now();
  std::thread th(alloc_fun);
  std::this_thread::sleep_for(std::chrono::seconds(1));
  allocation.reset();
  th.join();
  auto end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> time = end_time - start_time;
  VLOG(10) << "time cost = " << time.count() << " s";
  EXPECT_LE(time.count() * 1000, FLAGS_gpu_allocator_retry_time);
}

__global__ void add_kernel(int *x, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    atomicAdd(x + i, tid);
  }
}

class StreamSafeCUDAAllocTest : public ::testing::Test {
 protected:
  void SetUp() override {
    place_ = platform::CUDAPlace();
    stream_num_ = 64;
    grid_num_ = 1;
    block_num_ = 64;
    data_num_ = 64;
    default_stream = nullptr;

    streams_.reserve(stream_num_);
    streams_.emplace_back(default_stream);
    for (size_t i = 1; i < stream_num_; ++i) {
      cudaStream_t stream;
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamCreate(&stream));
      streams_.emplace_back(stream);
    }

    for (size_t i = 0; i < stream_num_; ++i) {
      size_t allocation_size = data_num_ * sizeof(int);
      std::shared_ptr<Allocation> allocation =
          AllocShared(place_, streams_[i], allocation_size);
      PADDLE_ENFORCE_CUDA_SUCCESS(
          cudaMemset(allocation->ptr(), 0, allocation->size()));
      allocations_.emplace_back(allocation);
    }
  }

  void SingleStreamRun(size_t idx) {
    for (size_t i = 0; i < stream_num_; ++i) {
      int *x = reinterpret_cast<int *>(allocations_[i]->ptr());
      add_kernel<<<grid_num_, block_num_, 0, streams_[idx]>>>(x, data_num_);
      if (i != idx) {
        RecordStream(allocations_[i].get(), streams_[idx]);
      }
    }
  }

  void MultiStreamRun() {
    for (int i = 0; i < stream_num_; ++i) {
      SingleStreamRun(i);
    }
    allocations_.clear();  // fast_gc
  }

  void MultiThreadMUltiStreamRun() {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < stream_num_; ++i) {
      threads.push_back(
          std::thread(&StreamSafeCUDAAllocTest::SingleStreamRun, this, i));
    }
    for (size_t i = 0; i < stream_num_; ++i) {
      threads[i].join();
    }
    allocations_.clear();  // fast_gc
  }

  void CheckResult() {
    auto host_x = std::unique_ptr<int[]>(new int[data_num_]);
    size_t thread_num = grid_num_ * block_num_;
    for (int i = 0; i < stream_num_; ++i) {
      // tricky code, the allocations are still accessible even though
      // allocations_.clear() has been called
      PADDLE_ENFORCE_CUDA_SUCCESS(
          cudaMemcpy(host_x.get(), allocations_[i]->ptr(),
                     data_num_ * sizeof(int), cudaMemcpyDeviceToHost));
      for (int j = 0; j < data_num_; ++j) {
        EXPECT_TRUE(host_x[j] == (j % thread_num) * stream_num_);
      }
    }
  }

  void TearDown() override {
    cudaDeviceSynchronize();
    for (cudaStream_t stream : streams_) {
      Release(place_, stream);
    }

    for (size_t i = 1; i < stream_num_; ++i) {
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(streams_[i]));
    }

    uint64_t cuda_malloc_size =
        platform::RecordedCudaMallocSize(place_.GetDeviceId());
    ASSERT_EQ(cuda_malloc_size, 0) << "Found " << cuda_malloc_size
                                   << " bytes memory that not released yet, "
                                      "there may be a memory leak problem.";
  }

  size_t stream_num_;
  size_t grid_num_;
  size_t block_num_;
  size_t data_num_;
  platform::CUDAPlace place_;
  cudaStream_t default_stream;
  std::vector<cudaStream_t> streams_;
  std::vector<std::shared_ptr<Allocation>> allocations_;
};

TEST_F(StreamSafeCUDAAllocTest, CUDAMutilStream) {
  MultiStreamRun();
  CheckResult();
}

TEST_F(StreamSafeCUDAAllocTest, CUDAMutilThreadMutilStream) {
  MultiThreadMUltiStreamRun();
  CheckResult();
}

}  // namespace memory
}  // namespace paddle
