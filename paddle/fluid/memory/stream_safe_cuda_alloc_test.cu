// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace memory {

__global__ void add_kernel(int *x, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    atomicAdd(x + i, tid);
  }
}

void CheckMemLeak(const platform::CUDAPlace &place) {
  uint64_t cuda_malloc_size =
      platform::RecordedGpuMallocSize(place.GetDeviceId());
  ASSERT_EQ(cuda_malloc_size, 0) << "Found " << cuda_malloc_size
                                 << " bytes memory that not released yet,"
                                 << " there may be a memory leak problem";
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
      gpuStream_t stream;
#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&stream));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreate(&stream));
#endif
      streams_.emplace_back(stream);
    }

    for (size_t i = 0; i < stream_num_; ++i) {
      size_t allocation_size = data_num_ * sizeof(int);
      std::shared_ptr<Allocation> allocation =
          AllocShared(place_, allocation_size, streams_[i]);
#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemset(allocation->ptr(), 0, allocation->size()));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemset(allocation->ptr(), 0, allocation->size()));
#endif
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
#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemcpy(host_x.get(), allocations_[i]->ptr(),
                     data_num_ * sizeof(int), cudaMemcpyDeviceToHost));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpy(host_x.get(), allocations_[i]->ptr(),
                                           data_num_ * sizeof(int),
                                           hipMemcpyDeviceToHost));
#endif
      for (int j = 0; j < data_num_; ++j) {
        EXPECT_TRUE(host_x[j] == (j % thread_num) * stream_num_);
      }
    }
  }

  void TearDown() override {
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else
    PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif
    for (gpuStream_t stream : streams_) {
      Release(place_, stream);
    }

    for (size_t i = 1; i < stream_num_; ++i) {
#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(streams_[i]));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(streams_[i]));
#endif
    }

    CheckMemLeak(place_);
  }

  size_t stream_num_;
  size_t grid_num_;
  size_t block_num_;
  size_t data_num_;
  platform::CUDAPlace place_;
  gpuStream_t default_stream;
  std::vector<gpuStream_t> streams_;
  std::vector<std::shared_ptr<Allocation>> allocations_;
};

TEST_F(StreamSafeCUDAAllocTest, CUDAMutilStreamTest) {
  MultiStreamRun();
  CheckResult();
}

TEST_F(StreamSafeCUDAAllocTest, CUDAMutilThreadMutilStreamTest) {
  MultiThreadMUltiStreamRun();
  CheckResult();
}

TEST(StreamSafeCUDAAllocInterfaceTest, AllocInterfaceTest) {
  platform::CUDAPlace place = platform::CUDAPlace();
  size_t alloc_size = 256;

  std::shared_ptr<Allocation> allocation_implicit_stream =
      AllocShared(place, alloc_size);
  EXPECT_GE(allocation_implicit_stream->size(), alloc_size);

  void *address = allocation_implicit_stream->ptr();
  allocation_implicit_stream.reset();

  gpuStream_t default_stream = nullptr;
  allocation::AllocationPtr allocation_unique =
      Alloc(place, alloc_size, default_stream);
  EXPECT_GE(allocation_unique->size(), alloc_size);
  EXPECT_EQ(allocation_unique->ptr(), address);
  allocation_unique.reset();

  Release(place);
  CheckMemLeak(place);
}

TEST(StreamSafeCUDAAllocInterfaceTest, GetAllocatorInterfaceTest) {
  platform::CUDAPlace place = platform::CUDAPlace();
  auto &instance = allocation::AllocatorFacade::Instance();
  const std::shared_ptr<Allocator> &allocator = instance.GetAllocator(place);

  size_t alloc_size = 256;
  std::shared_ptr<Allocation> allocation_from_allocator =
      allocator->Allocate(alloc_size);
  EXPECT_GE(allocation_from_allocator->size(), alloc_size);
  void *address = allocation_from_allocator->ptr();
  allocation_from_allocator.reset();

  std::shared_ptr<Allocation> allocation_implicit_stream =
      AllocShared(place, alloc_size);
  EXPECT_GE(allocation_implicit_stream->size(), alloc_size);
  EXPECT_EQ(allocation_implicit_stream->ptr(), address);
  allocation_implicit_stream.reset();

  Release(place);
  CheckMemLeak(place);
}

#ifdef PADDLE_WITH_CUDA
TEST(StreamSafeCUDAAllocInterfaceTest, CUDAGraphExceptionTest) {
  platform::CUDAPlace place = platform::CUDAPlace();
  size_t alloc_size = 1;
  std::shared_ptr<Allocation> allocation = AllocShared(place, alloc_size);

  platform::BeginCUDAGraphCapture(place, cudaStreamCaptureModeGlobal);
  EXPECT_THROW(AllocShared(place, alloc_size), paddle::platform::EnforceNotMet);
  EXPECT_THROW(Alloc(place, alloc_size), paddle::platform::EnforceNotMet);
  EXPECT_THROW(Release(place), paddle::platform::EnforceNotMet);
  EXPECT_THROW(allocation::AllocatorFacade::Instance().GetAllocator(place),
               paddle::platform::EnforceNotMet);
  EXPECT_THROW(AllocShared(place, alloc_size, nullptr),
               paddle::platform::EnforceNotMet);
  EXPECT_THROW(Alloc(place, alloc_size, nullptr),
               paddle::platform::EnforceNotMet);
  EXPECT_THROW(Release(place, nullptr), paddle::platform::EnforceNotMet);
  EXPECT_THROW(RecordStream(allocation.get(), nullptr),
               paddle::platform::EnforceNotMet);
  platform::EndCUDAGraphCapture();

  allocation.reset();
  Release(place);
  CheckMemLeak(place);
}
#endif

TEST(StreamSafeCUDAAllocRetryTest, RetryTest) {
  platform::CUDAPlace place = platform::CUDAPlace();
  gpuStream_t stream1, stream2;
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&stream1));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&stream2));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreate(&stream1));
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreate(&stream2));
#endif
  size_t available_size = platform::GpuAvailableMemToAlloc();
  // alloc_size < available_size < 2 * alloc_size
  size_t alloc_size = available_size / 4 * 3;

  std::shared_ptr<Allocation> allocation1 =
      AllocShared(place, alloc_size, stream1);
  std::shared_ptr<Allocation> allocation2;

  std::thread th([&allocation2, &place, &stream2, alloc_size]() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    allocation2 = AllocShared(place, alloc_size, stream2);
  });
  allocation1.reset();  // free but not release
  th.join();
  EXPECT_GE(allocation2->size(), alloc_size);
  allocation2.reset();

#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif

  Release(place, stream1);
  Release(place, stream2);
  CheckMemLeak(place);
}

}  // namespace memory
}  // namespace paddle
