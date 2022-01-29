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
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/stream/stream.h"

namespace paddle {
namespace memory {

__global__ void add_kernel(int *x, int n) {
  int thread_num = gridDim.x * blockDim.x;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = thread_id; i < n; i += thread_num) {
    atomicAdd(x + i, thread_id);
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
    block_num_ = 32;
    data_num_ = 131072;
    workspace_size_ = data_num_ * sizeof(int);

    // alloc workspace for each stream
    for (size_t i = 0; i < stream_num_; ++i) {
      gpuStream_t stream;
#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&stream));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreate(&stream));
#endif

      std::shared_ptr<Allocation> allocation = AllocShared(
          place_, workspace_size_,
          platform::Stream(reinterpret_cast<platform::StreamId>(stream)));
#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemset(allocation->ptr(), 0, allocation->size()));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemset(allocation->ptr(), 0, allocation->size()));
#endif

      streams_.emplace_back(stream);
      workspaces_.emplace_back(allocation);
    }

    result_ = Alloc(place_, stream_num_ * workspace_size_);
  }

  void SingleStreamRun(size_t idx) {
    // for all stream i,
    // stream idx lauch a kernel to add (j % thread_num) to workspaces_[i][j]
    for (size_t i = 0; i < stream_num_; ++i) {
      int *x = reinterpret_cast<int *>(workspaces_[i]->ptr());
      add_kernel<<<grid_num_, block_num_, 0, streams_[idx]>>>(x, data_num_);
      RecordStream(workspaces_[i], streams_[idx]);
    }
  }

  void CopyResultAsync() {
    for (size_t i = 0; i < stream_num_; ++i) {
#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
          reinterpret_cast<int *>(result_->ptr()) + i * data_num_,
          workspaces_[i]->ptr(), workspace_size_, cudaMemcpyDeviceToDevice));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpyAsync(
          reinterpret_cast<int *>(result_->ptr()) + i * data_num_,
          workspaces_[i]->ptr(), workspace_size_, hipMemcpyDeviceToDevice));
#endif
    }
  }

  void MultiStreamRun() {
    for (size_t i = 0; i < stream_num_; ++i) {
      SingleStreamRun(i);
    }
    CopyResultAsync();
    workspaces_.clear();  // fast_gc
    cudaDeviceSynchronize();
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
    CopyResultAsync();
    workspaces_.clear();  // fast_gc
    cudaDeviceSynchronize();
  }

  void CheckResult() {
    auto result_host = std::unique_ptr<int[]>(new int[result_->size()]);
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(result_host.get(), result_->ptr(),
                                          result_->size(),
                                          cudaMemcpyDeviceToHost));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpy(result_host.get(), result_->ptr(),
                                         result_->size(),
                                         hipMemcpyDeviceToHost));
#endif
    size_t thread_num = grid_num_ * block_num_;
    for (size_t i = 0; i < stream_num_; ++i) {
      for (size_t j = 0; j < data_num_; ++j) {
        EXPECT_TRUE(result_host[i * stream_num_ + j] ==
                    (j % thread_num) * stream_num_);
      }
    }
    result_.reset();
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
  size_t workspace_size_;
  platform::CUDAPlace place_;
  std::vector<gpuStream_t> streams_;
  std::vector<std::shared_ptr<Allocation>> workspaces_;
  allocation::AllocationPtr result_;
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

  gpuStream_t default_stream =
      dynamic_cast<platform::CUDADeviceContext *>(
          paddle::platform::DeviceContextPool::Instance().Get(place))
          ->stream();
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
  size_t alloc_size = 256;

  allocation::AllocationPtr allocation_implicit_stream =
      Alloc(place, alloc_size);
  EXPECT_GE(allocation_implicit_stream->size(), alloc_size);
  void *address = allocation_implicit_stream->ptr();
  allocation_implicit_stream.reset();

  auto &instance = allocation::AllocatorFacade::Instance();
  const std::shared_ptr<Allocator> &allocator = instance.GetAllocator(place);

  allocation::AllocationPtr allocation_from_allocator =
      allocator->Allocate(alloc_size);
  EXPECT_GE(allocation_from_allocator->size(), alloc_size);
  EXPECT_EQ(allocation_from_allocator->ptr(), address);
  allocation_from_allocator.reset();

  Release(place);
  CheckMemLeak(place);
}

TEST(StreamSafeCUDAAllocInterfaceTest, ZeroSizeRecordStreamTest) {
  platform::CUDAPlace place = platform::CUDAPlace();
  std::shared_ptr<Allocation> zero_size_allocation = AllocShared(place, 0);
  EXPECT_EQ(zero_size_allocation->ptr(), nullptr);

  gpuStream_t stream;
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&stream));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreate(&stream));
#endif

  EXPECT_NO_THROW(RecordStream(zero_size_allocation, stream));

#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(stream));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(stream));
#endif
}

TEST(StreamSafeCUDAAllocInterfaceTest, GetStreamInterfaceTest) {
  platform::CUDAPlace place = platform::CUDAPlace();
  size_t alloc_size = 256;

  gpuStream_t default_stream =
      dynamic_cast<platform::CUDADeviceContext *>(
          paddle::platform::DeviceContextPool::Instance().Get(place))
          ->stream();
  std::shared_ptr<Allocation> allocation_implicit_stream =
      AllocShared(place, alloc_size);
  EXPECT_EQ(GetStream(allocation_implicit_stream), default_stream);

  gpuStream_t new_stream;
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&new_stream));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreate(&new_stream));
#endif

  std::shared_ptr<Allocation> allocation_new_stream = AllocShared(
      place, alloc_size,
      platform::Stream(reinterpret_cast<platform::StreamId>(new_stream)));
  EXPECT_EQ(GetStream(allocation_new_stream), new_stream);

#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(new_stream));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(new_stream));
#endif

  allocation_implicit_stream.reset();
  allocation_new_stream.reset();
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
  EXPECT_THROW(AllocShared(place, alloc_size,
                           platform::Stream(
                               reinterpret_cast<platform::StreamId>(nullptr))),
               paddle::platform::EnforceNotMet);
  EXPECT_THROW(Alloc(place, alloc_size, nullptr),
               paddle::platform::EnforceNotMet);
  EXPECT_THROW(Release(place, nullptr), paddle::platform::EnforceNotMet);
  EXPECT_THROW(RecordStream(allocation, nullptr),
               paddle::platform::EnforceNotMet);
  EXPECT_THROW(GetStream(allocation), paddle::platform::EnforceNotMet);
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
  // alloc_size < available_size < 2 * alloc_size,
  // so the second alloc will fail and retry
  size_t alloc_size = available_size / 4 * 3;

  allocation::AllocationPtr allocation1 = Alloc(place, alloc_size, stream1);
  allocation::AllocationPtr allocation2;

  std::thread th([&allocation2, &place, &stream2, alloc_size]() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    allocation2 = Alloc(place, alloc_size, stream2);
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
