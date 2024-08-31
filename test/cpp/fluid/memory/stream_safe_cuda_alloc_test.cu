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

#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/memory/memory.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/stream.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include "paddle/phi/core/platform/cuda_graph_with_memory_pool.h"
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

#define RETURN_IF_NOT_ENABLED                            \
  {                                                      \
    if (!memory::allocation::AllocatorFacade::Instance() \
             .IsStreamSafeCUDAAllocatorUsed()) {         \
      return;                                            \
    }                                                    \
  }

namespace paddle {
namespace memory {

// y += (x + 1)
__global__ void add_kernel(int *x, int *y, int n) {
  int thread_num = gridDim.x * blockDim.x;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = thread_id; i < n; i += thread_num) {
    y[i] += x[i] + 1;
  }
}

void CheckMemLeak(const phi::GPUPlace &place) {
  uint64_t cuda_malloc_size =
      platform::RecordedGpuMallocSize(place.GetDeviceId());
  ASSERT_EQ(cuda_malloc_size, 0)
      << "Found " << cuda_malloc_size << " bytes memory that not released yet,"
      << " there may be a memory leak problem";
}

TEST(StreamSafeCUDAAllocInterfaceTest, AllocInterfaceTest) {
  RETURN_IF_NOT_ENABLED;

  phi::GPUPlace place = phi::GPUPlace();
  size_t alloc_size = 256;

  std::shared_ptr<Allocation> allocation_implicit_stream =
      AllocShared(place, alloc_size);
  EXPECT_GE(allocation_implicit_stream->size(), alloc_size);

  void *address = allocation_implicit_stream->ptr();
  allocation_implicit_stream.reset();

  gpuStream_t default_stream =
      dynamic_cast<phi::GPUContext *>(
          phi::DeviceContextPool::Instance().Get(place))
          ->stream();
  allocation::AllocationPtr allocation_unique =
      Alloc(place,
            alloc_size,
            phi::Stream(reinterpret_cast<phi::StreamId>(default_stream)));
  EXPECT_GE(allocation_unique->size(), alloc_size);
  EXPECT_EQ(allocation_unique->ptr(), address);
  allocation_unique.reset();

  Release(place);
  CheckMemLeak(place);
}

TEST(StreamSafeCUDAAllocInterfaceTest, GetAllocatorInterfaceTest) {
  RETURN_IF_NOT_ENABLED;

  phi::GPUPlace place = phi::GPUPlace();
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

TEST(StreamSafeCUDAAllocInterfaceTest, GetAllocatorWithDefaultStreamTest) {
  RETURN_IF_NOT_ENABLED;

  auto &instance = allocation::AllocatorFacade::Instance();
  phi::GPUPlace place = phi::GPUPlace();
  const std::shared_ptr<Allocator> allocator_implicit_stream =
      instance.GetAllocator(place);
  const std::shared_ptr<Allocator> allocator_default_stream =
      instance.GetAllocator(place,
                            static_cast<phi::GPUContext *>(
                                phi::DeviceContextPool::Instance().Get(place))
                                ->stream());
  EXPECT_EQ(allocator_implicit_stream.get(), allocator_default_stream.get());
}

TEST(StreamSafeCUDAAllocInterfaceTest, ZeroSizeRecordStreamTest) {
  RETURN_IF_NOT_ENABLED;

  phi::GPUPlace place = phi::GPUPlace();
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
  RETURN_IF_NOT_ENABLED;

  phi::GPUPlace place = phi::GPUPlace();
  size_t alloc_size = 256;

  gpuStream_t default_stream =
      dynamic_cast<phi::GPUContext *>(
          phi::DeviceContextPool::Instance().Get(place))
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

  std::shared_ptr<Allocation> allocation_new_stream =
      AllocShared(place,
                  alloc_size,
                  phi::Stream(reinterpret_cast<phi::StreamId>(new_stream)));
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

TEST(StreamSafeCUDAAllocRetryTest, RetryTest) {
  RETURN_IF_NOT_ENABLED;

  phi::GPUPlace place = phi::GPUPlace();
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

  allocation::AllocationPtr allocation1 = Alloc(
      place, alloc_size, phi::Stream(reinterpret_cast<phi::StreamId>(stream1)));
  allocation::AllocationPtr allocation2;

  std::thread th([&allocation2, &place, &stream2, alloc_size]() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    allocation2 = Alloc(place,
                        alloc_size,
                        phi::Stream(reinterpret_cast<phi::StreamId>(stream2)));
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

class StreamSafeCUDAAllocTest : public ::testing::Test {
 protected:
  void SetUp() override {
    place_ = phi::GPUPlace();
    stream_num_ = 64;
    grid_num_ = 1;
    block_num_ = 32;
    data_num_ = 131072;
    workspace_size_ = data_num_ * sizeof(int);

    for (size_t i = 0; i < stream_num_; ++i) {
      gpuStream_t stream;
#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&stream));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreate(&stream));
#endif

      std::shared_ptr<phi::Allocation> workspace_allocation =
          AllocShared(place_,
                      workspace_size_,
                      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
      std::shared_ptr<phi::Allocation> result_allocation =
          AllocShared(place_,
                      workspace_size_,
                      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
      std::shared_ptr<phi::Allocation> host_result_allocation =
          AllocShared(phi::CPUPlace(), workspace_size_);

#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemset(
          workspace_allocation->ptr(), 0, workspace_allocation->size()));
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemset(result_allocation->ptr(), 0, result_allocation->size()));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(hipMemset(
          workspace_allocation->ptr(), 0, workspace_allocation->size()));
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemset(result_allocation->ptr(), 0, result_allocation->size()));
#endif

      streams_.emplace_back(stream);
      workspaces_.emplace_back(workspace_allocation);
      results_.emplace_back(result_allocation);
      host_results_.emplace_back(host_result_allocation);
    }
  }

  void SingleStreamRun(size_t idx) {
    int *y = reinterpret_cast<int *>(results_[idx]->ptr());
    int neighbouring_idx = idx > 0 ? idx - 1 : idx;

    add_kernel<<<grid_num_, block_num_, 0, streams_[idx]>>>(
        reinterpret_cast<int *>(workspaces_[idx]->ptr()), y, data_num_);
    add_kernel<<<grid_num_, block_num_, 0, streams_[idx]>>>(
        reinterpret_cast<int *>(workspaces_[neighbouring_idx]->ptr()),
        y,
        data_num_);
    RecordStream(workspaces_[neighbouring_idx], streams_[idx]);
  }

  void MultiStreamRun() {
    // Must run in reverse order, or the workspace_[i - 1] will be released
    // before streams_[i]'s kernel launch
    for (int i = stream_num_ - 1; i >= 0; --i) {
      SingleStreamRun(i);
      workspaces_[i].reset();  // fast GC
    }
  }

  void MultiThreadMultiStreamRun() {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < stream_num_; ++i) {
      threads.emplace_back(&StreamSafeCUDAAllocTest::SingleStreamRun, this, i);
    }
    for (size_t i = 0; i < stream_num_; ++i) {
      threads[i].join();
    }
    workspaces_.clear();
  }

  void CUDAGraphRun() {
    testing_cuda_graph_ = true;
    platform::BeginCUDAGraphCapture(phi::GPUPlace(),
                                    cudaStreamCaptureModeGlobal);

    std::shared_ptr<Allocation> data_allocation =
        AllocShared(phi::GPUPlace(), workspace_size_);
    std::shared_ptr<Allocation> result_allocation =
        AllocShared(phi::GPUPlace(), workspace_size_);

    int *data = static_cast<int *>(data_allocation->ptr());
    int *result = static_cast<int *>(result_allocation->ptr());

    gpuStream_t main_stream = GetStream(data_allocation);
    gpuStream_t other_stream;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&other_stream));

    add_kernel<<<grid_num_, block_num_, 0, main_stream>>>(
        data, result, data_num_);
    RecordStream(data_allocation, other_stream);

    std::unique_ptr<phi::backends::gpu::CUDAGraph> cuda_graph =
        platform::EndCUDAGraphCapture();

    int replay_times = 10;
    for (int i = 0; i < replay_times; ++i) {
      cuda_graph->Replay();
    }

    std::shared_ptr<Allocation> host_result_allocation =
        AllocShared(phi::CPUPlace(), workspace_size_);
    Copy(host_result_allocation->place(),
         host_result_allocation->ptr(),
         result_allocation->place(),
         result_allocation->ptr(),
         workspace_size_,
         main_stream);
    cudaStreamSynchronize(main_stream);

    int *host_result = static_cast<int *>(host_result_allocation->ptr());
    for (int i = 0; i < data_num_; ++i) {
      EXPECT_EQ(host_result[i], replay_times);
    }

    data_allocation.reset();
    result_allocation.reset();
    cuda_graph.release();
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(other_stream));
  }

  void CheckResult() {
    for (size_t i = 0; i < stream_num_; ++i) {
      Copy(host_results_[i]->place(),
           host_results_[i]->ptr(),
           results_[i]->place(),
           results_[i]->ptr(),
           workspace_size_,
           streams_[i]);
    }
    cudaDeviceSynchronize();

    size_t thread_num = grid_num_ * block_num_;
    for (size_t i = 0; i < stream_num_; ++i) {
      int *result = static_cast<int *>(host_results_[i]->ptr());
      for (size_t j = 0; j < data_num_; ++j) {
        EXPECT_EQ(result[j], 2);
      }
    }
  }

  void TearDown() override {
    workspaces_.clear();
    results_.clear();
    host_results_.clear();
    for (gpuStream_t stream : streams_) {
      Release(place_, stream);
    }

    for (size_t i = 0; i < stream_num_; ++i) {
#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(streams_[i]));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(streams_[i]));
#endif
    }

    // Memory release for CUDA Graph memory pool is forbidden
    if (!testing_cuda_graph_) {
      CheckMemLeak(place_);
    }
  }

  bool testing_cuda_graph_{0};
  size_t stream_num_;
  size_t grid_num_;
  size_t block_num_;
  size_t data_num_;
  size_t workspace_size_;
  phi::GPUPlace place_;
  std::vector<gpuStream_t> streams_;
  std::vector<std::shared_ptr<phi::Allocation>> workspaces_;
  std::vector<std::shared_ptr<phi::Allocation>> results_;
  std::vector<std::shared_ptr<phi::Allocation>> host_results_;
};

TEST_F(StreamSafeCUDAAllocTest, CUDAMutilStreamTest) {
  RETURN_IF_NOT_ENABLED;

  MultiStreamRun();
  CheckResult();
}

TEST_F(StreamSafeCUDAAllocTest, CUDAMutilThreadMutilStreamTest) {
  RETURN_IF_NOT_ENABLED;

  MultiThreadMultiStreamRun();
  CheckResult();
}

#if (defined(PADDLE_WITH_CUDA) && (CUDA_VERSION >= 11000))
TEST_F(StreamSafeCUDAAllocTest, CUDAGraphTest) {
  RETURN_IF_NOT_ENABLED;

  MultiStreamRun();
  CUDAGraphRun();
  CheckResult();
}
#endif

}  // namespace memory
}  // namespace paddle
