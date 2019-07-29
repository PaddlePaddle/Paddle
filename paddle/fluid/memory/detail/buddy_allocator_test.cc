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
constexpr static bool USE_SYSTEM_ALLOCATOR = true;

void TestBuddyAllocator(BuddyAllocator* allocator, size_t size_bytes,
                        bool use_system_allocator = false) {
  bool freed = false;
  size_t used_bytes = allocator->Used();

  if (size_bytes > 0) {
    void* p = allocator->Alloc(size_bytes);

    EXPECT_NE(p, nullptr);

#ifdef PADDLE_WITH_CUDA
    if (size_bytes < platform::GpuMaxChunkSize()) {
#else
    if (size_bytes < platform::CpuMaxChunkSize()) {
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
    std::shared_ptr<int> ptr(intp, [&](void* p) {
      allocator->Free(intp);
      freed = true;
    });
  } else {
    freed = true;
  }

  EXPECT_EQ(used_bytes, allocator->Used());
  EXPECT_TRUE(freed);
}

#ifdef PADDLE_WITH_CUDA
TEST(BuddyAllocator, GpuFraction) {
  // In a 16 GB machine, the pool size will be about 160 MB
  platform::ResetGpuMaxChunkSize();
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
  TestBuddyAllocator(&buddy_allocator, 499 << 20, USE_SYSTEM_ALLOCATOR);
  TestBuddyAllocator(&buddy_allocator, 2 * static_cast<size_t>(1 << 30),
                     USE_SYSTEM_ALLOCATOR);
}

TEST(BuddyAllocator, InitRealloc) {
  platform::ResetGpuMaxChunkSize();
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
  TestBuddyAllocator(&buddy_allocator, 101 << 20, USE_SYSTEM_ALLOCATOR);
  TestBuddyAllocator(&buddy_allocator, 2 * static_cast<size_t>(1 << 30),
                     USE_SYSTEM_ALLOCATOR);
}

TEST(BuddyAllocator, ReallocSizeGreaterThanInit) {
  platform::ResetGpuMaxChunkSize();
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
  TestBuddyAllocator(&buddy_allocator, 11 << 20, USE_SYSTEM_ALLOCATOR);
  TestBuddyAllocator(&buddy_allocator, 2 * static_cast<size_t>(1 << 30),
                     USE_SYSTEM_ALLOCATOR);
}

TEST(BuddyAllocator, LargeFractionRealloc) {
  platform::ResetGpuMaxChunkSize();
  FLAGS_fraction_of_gpu_memory_to_use = 0.92;
  FLAGS_initial_gpu_memory_in_mb = 0;
  FLAGS_reallocate_gpu_memory_in_mb = 0;

  size_t max_chunk_size = platform::GpuMaxChunkSize();
  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), max_chunk_size);

  // Less than pool size
  TestBuddyAllocator(&buddy_allocator, max_chunk_size - 1000);
  // Max chunk size should be same during allocation
  EXPECT_EQ(max_chunk_size, platform::GpuMaxChunkSize());

  // Exceed pool trigger refilling size of fraction of avaiable gpu
  TestBuddyAllocator(&buddy_allocator, 2000);
  // Max chunk size should be same during allocation
  EXPECT_EQ(max_chunk_size, platform::GpuMaxChunkSize());
}

TEST(BuddyAllocator, AllocFromAvailable) {
  platform::ResetGpuMaxChunkSize();
  FLAGS_fraction_of_gpu_memory_to_use = 0.7;
  FLAGS_initial_gpu_memory_in_mb = 0;
  FLAGS_reallocate_gpu_memory_in_mb = 0;

  size_t total = 0, available = 0;
  platform::SetDeviceId(TEST_GPU_ID);
  platform::GpuMemoryUsage(&available, &total);

  // Taken half of available GPU
  void* p;
  cudaError_t result = cudaMalloc(&p, available >> 1);
  EXPECT_TRUE(result == cudaSuccess);

  // BuddyAllocator should be able to alloc remaining GPU
  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(TEST_GPU_ID)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  TestBuddyAllocator(&buddy_allocator, 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 20);
  TestBuddyAllocator(&buddy_allocator, static_cast<size_t>(1 << 30));
}

#endif

}  // namespace detail
}  // namespace memory
}  // namespace paddle
