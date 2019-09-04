/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/memory/detail/buddy_allocator.h"

#include <memory>

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/gpu_info.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>

DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_uint64(initial_gpu_memory_in_mb);
DECLARE_uint64(reallocate_gpu_memory_in_mb);
#endif

namespace paddle {
namespace memory {
namespace detail {

constexpr static int TEST_GPU_ID = 0;

int* TestBuddyAllocator(BuddyAllocator* allocator, size_t size_bytes,
                        bool use_system_allocator = false,
                        bool free_ptr = true) {
  bool freed = false;
  size_t used_bytes = allocator->Used();

  if (size_bytes > 0) {
    void* p = allocator->Alloc(size_bytes);

    EXPECT_NE(p, nullptr);

#ifdef PADDLE_WITH_CUDA
    if (size_bytes < allocator->GetMaxChunkSize()) {
#else
    if (size_bytes < allocator->GetMaxChunkSize()) {
#endif
      // Not allocate from SystemAllocator
      EXPECT_FALSE(use_system_allocator);
      EXPECT_GE(allocator->Used(), used_bytes + size_bytes);
    } else {
      // Allocate from SystemAllocator doesn't count in Used()
      EXPECT_TRUE(use_system_allocator);
      EXPECT_EQ(allocator->Used(), used_bytes);
    }

    int* intp = static_cast<int*>(p);
    if (!free_ptr) {
      return intp;
    }
    std::shared_ptr<int> ptr(intp, [&](void* p) {
      allocator->Free(intp);
      freed = true;
    });
  } else {
    freed = true;
  }

  EXPECT_EQ(used_bytes, allocator->Used());
  EXPECT_TRUE(freed);
  return nullptr;
}

#ifdef PADDLE_WITH_CUDA
TEST(BuddyAllocator, GpuFraction) {
  // In a 16 GB machine, the pool size will be about 160 MB
  FLAGS_fraction_of_gpu_memory_to_use = 0.01;
  FLAGS_initial_gpu_memory_in_mb = 0;
  FLAGS_reallocate_gpu_memory_in_mb = 0;

  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  // Less than pool size
  TestBuddyAllocator(&buddy_allocator, 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 20);

  // Greater than max chunk size
  TestBuddyAllocator(&buddy_allocator, 499 << 20,
                     /* use_system_allocator = */ true);
  TestBuddyAllocator(&buddy_allocator, 2 * static_cast<size_t>(1 << 30),
                     /* use_system_allocator = */ true);
}

TEST(BuddyAllocator, InitRealloc) {
  FLAGS_initial_gpu_memory_in_mb = 100;
  FLAGS_reallocate_gpu_memory_in_mb = 50;

  EXPECT_EQ(platform::GpuMaxChunkSize(), static_cast<size_t>(100 << 20));

  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  // Less then initial size and reallocate size
  TestBuddyAllocator(&buddy_allocator, 10 << 20);
  // Between initial size and reallocate size and not exceed pool
  TestBuddyAllocator(&buddy_allocator, 80 << 20);
  TestBuddyAllocator(&buddy_allocator, 99 << 20);
  // Greater than max chunk size
  TestBuddyAllocator(&buddy_allocator, 101 << 20,
                     /* use_system_allocator = */ true);
  TestBuddyAllocator(&buddy_allocator, 2 * static_cast<size_t>(1 << 30),
                     /* use_system_allocator = */ true);
}

TEST(BuddyAllocator, ReallocSizeGreaterThanInit) {
  FLAGS_initial_gpu_memory_in_mb = 5;
  FLAGS_reallocate_gpu_memory_in_mb = 10;

  EXPECT_EQ(platform::GpuMaxChunkSize(), static_cast<size_t>(10 << 20));

  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  // Less than initial size and reallocate size
  TestBuddyAllocator(&buddy_allocator, 1 << 20);
  // Between initial size and reallocate size and exceed pool
  TestBuddyAllocator(&buddy_allocator, 6 << 20);
  TestBuddyAllocator(&buddy_allocator, 8 << 20);
  TestBuddyAllocator(&buddy_allocator, 9 << 20);
  // Greater than max trunk size
  TestBuddyAllocator(&buddy_allocator, 11 << 20,
                     /* use_system_allocator = */ true);
  TestBuddyAllocator(&buddy_allocator, 2 * static_cast<size_t>(1 << 30),
                     /* use_system_allocator = */ true);
}

TEST(BuddyAllocator, FractionRefillPool) {
  FLAGS_fraction_of_gpu_memory_to_use = 0.6;
  FLAGS_initial_gpu_memory_in_mb = 0;
  FLAGS_reallocate_gpu_memory_in_mb = 0;

  size_t max_chunk_size = platform::GpuMaxChunkSize();
  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), max_chunk_size);

  // Less than pool size
  int* p0 = TestBuddyAllocator(&buddy_allocator, max_chunk_size - 1000,
                               /* use_system_allocator = */ false,
                               /* free_ptr = */ false);
  // Max chunk size should be same during allocation
  EXPECT_EQ(max_chunk_size, buddy_allocator.GetMaxChunkSize());

  size_t alloc =
      platform::GpuAvailableMemToAlloc() * FLAGS_fraction_of_gpu_memory_to_use;
  // Exceed pool trigger refilling size of fraction of avaiable gpu, and should
  // be able to alloc 60% of the remaining GPU
  int* p1 = TestBuddyAllocator(&buddy_allocator, alloc,
                               /* use_system_allocator = */ false,
                               /* free_ptr = */ false);
  // Max chunk size should be same during allocation
  EXPECT_EQ(max_chunk_size, buddy_allocator.GetMaxChunkSize());

  alloc =
      platform::GpuAvailableMemToAlloc() * FLAGS_fraction_of_gpu_memory_to_use;
  // Exceed pool trigger refilling size of fraction of avaiable gpu, and should
  // be able to alloc 60% of the remaining GPU
  TestBuddyAllocator(&buddy_allocator, alloc,
                     /* use_system_allocator = */ false);
  // Max chunk size should be same during allocation
  EXPECT_EQ(max_chunk_size, buddy_allocator.GetMaxChunkSize());

  buddy_allocator.Free(p0);
  buddy_allocator.Free(p1);
}

TEST(BuddyAllocator, AllocFromAvailable) {
  FLAGS_fraction_of_gpu_memory_to_use = 0.7;
  FLAGS_initial_gpu_memory_in_mb = 0;
  FLAGS_reallocate_gpu_memory_in_mb = 0;

  size_t total = 0, available = 0;
  platform::SetDeviceId(TEST_GPU_ID);
  platform::GpuMemoryUsage(&available, &total);

  // Take half of available GPU
  void* p;
  cudaError_t result = cudaMalloc(&p, available >> 1);
  EXPECT_TRUE(result == cudaSuccess);

  // BuddyAllocator should be able to alloc the remaining GPU
  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  TestBuddyAllocator(&buddy_allocator, 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 20);
  TestBuddyAllocator(&buddy_allocator, static_cast<size_t>(1 << 30));

  if (p) {
    EXPECT_TRUE(cudaFree(p) == cudaSuccess);
  }
}

TEST(BuddyAllocator, AllocFromAvailableWhenFractionIsOne) {
  FLAGS_fraction_of_gpu_memory_to_use = 1.0;
  FLAGS_initial_gpu_memory_in_mb = 0;
  FLAGS_reallocate_gpu_memory_in_mb = 0;

  void* p = nullptr;
  EXPECT_TRUE(cudaMalloc(&p, static_cast<size_t>(4) << 30) == cudaSuccess);

  // BuddyAllocator should be able to alloc the remaining GPU
  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  TestBuddyAllocator(&buddy_allocator, static_cast<size_t>(1) << 30);
  TestBuddyAllocator(&buddy_allocator, static_cast<size_t>(5) << 30);

  if (p) {
    EXPECT_TRUE(cudaFree(p) == cudaSuccess);
  }
}

#endif

}  // namespace detail
}  // namespace memory
}  // namespace paddle
