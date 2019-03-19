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
DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_uint64(initial_gpu_memory_in_mb);
DECLARE_uint64(reallocate_gpu_memory_in_mb);
#endif

namespace paddle {
namespace memory {
namespace detail {

constexpr static int test_gpu_id = 0;

void TestBuddyAllocator(BuddyAllocator* allocator, size_t size_bytes) {
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
      EXPECT_GE(allocator->Used(), used_bytes + size_bytes);
    } else {
      // Allocate from SystemAllocator doesn't count in Used()
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
  FLAGS_fraction_of_gpu_memory_to_use = 0.01;

  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(test_gpu_id)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  TestBuddyAllocator(&buddy_allocator, 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 10);
  TestBuddyAllocator(&buddy_allocator, 10 << 20);
  TestBuddyAllocator(&buddy_allocator, 2 * static_cast<size_t>(1 << 30));
}

TEST(BuddyAllocator, InitRealloc) {
  FLAGS_initial_gpu_memory_in_mb = 100;
  FLAGS_reallocate_gpu_memory_in_mb = 50;

  EXPECT_EQ(platform::GpuMaxChunkSize(), static_cast<size_t>(100 << 20));

  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(test_gpu_id)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  // Less then initial size and reallocate size
  TestBuddyAllocator(&buddy_allocator, 10 << 20);
  // Between initial size and reallocate size and not exceed pool
  TestBuddyAllocator(&buddy_allocator, 80 << 20);
  // Less then reallocate size and exceed pool
  TestBuddyAllocator(&buddy_allocator, 40 << 20);
  // Greater then reallocate size and exceed pool
  TestBuddyAllocator(&buddy_allocator, 80 << 20);
  // Greater then initial size and reallocate size
  TestBuddyAllocator(&buddy_allocator, 2 * static_cast<size_t>(1 << 30));
}

TEST(BuddyAllocator, ReallocSizeGreaterThanInit) {
  FLAGS_initial_gpu_memory_in_mb = 5;
  FLAGS_reallocate_gpu_memory_in_mb = 10;

  EXPECT_EQ(platform::GpuMaxChunkSize(), static_cast<size_t>(10 << 20));

  BuddyAllocator buddy_allocator(
      std::unique_ptr<SystemAllocator>(new GPUAllocator(test_gpu_id)),
      platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

  // Less then initial size and reallocate size
  TestBuddyAllocator(&buddy_allocator, 1 << 20);
  // Between initial size and reallocate size and not exceed pool
  TestBuddyAllocator(&buddy_allocator, 3 << 20);
  // Less then initial size and exceed pool
  TestBuddyAllocator(&buddy_allocator, 3 << 20);
  // Less then reallocate size and not exceed pool (now pool is 15 MB, used 7
  // MB)
  TestBuddyAllocator(&buddy_allocator, 7 << 20);
  // Less then reallocate size and exceed pool
  TestBuddyAllocator(&buddy_allocator, 8 << 20);
  // Greater then initial size and reallocate size
  TestBuddyAllocator(&buddy_allocator, 2 * static_cast<size_t>(1 << 30));
}
#endif

}  // namespace detail
}  // namespace memory
}  // namespace paddle
