/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/memory/memory.h"
#include "paddle/memory/detail/memory_block.h"
#include "paddle/memory/detail/meta_data.h"

#include "paddle/platform/cpu_info.h"
#include "paddle/platform/gpu_info.h"
#include "paddle/platform/place.h"

#include <gtest/gtest.h>
#include <unordered_map>

inline bool is_aligned(void const *p) {
  return 0 == (reinterpret_cast<uintptr_t>(p) & 0x3);
}

size_t align(size_t size, paddle::platform::CPUPlace place) {
  size += sizeof(paddle::memory::detail::Metadata);
  size_t alignment = paddle::platform::CpuMinChunkSize();
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}

TEST(BuddyAllocator, CPUAllocation) {
  void *p = nullptr;

  EXPECT_EQ(p, nullptr);

  paddle::platform::CPUPlace cpu;
  p = paddle::memory::Alloc(cpu, 4096);

  EXPECT_NE(p, nullptr);

  paddle::memory::Free(cpu, p);
}

TEST(BuddyAllocator, CPUMultAlloc) {
  paddle::platform::CPUPlace cpu;

  std::unordered_map<void *, size_t> ps;

  size_t total_size = paddle::memory::Used(cpu);
  EXPECT_EQ(total_size, 0UL);

  for (auto size :
       {128, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304}) {
    ps[paddle::memory::Alloc(cpu, size)] = size;

    // Buddy Allocator doesn't manage too large memory chunk
    if (paddle::memory::Used(cpu) == total_size) continue;

    size_t aligned_size = align(size, cpu);
    total_size += aligned_size;
    EXPECT_EQ(total_size, paddle::memory::Used(cpu));
  }

  for (auto p : ps) {
    EXPECT_EQ(is_aligned(p.first), true);
    paddle::memory::Free(cpu, p.first);

    // Buddy Allocator doesn't manage too large memory chunk
    if (paddle::memory::Used(cpu) == total_size) continue;

    size_t aligned_size = align(p.second, cpu);
    total_size -= aligned_size;
    EXPECT_EQ(total_size, paddle::memory::Used(cpu));
  }
}

#ifdef PADDLE_WITH_CUDA

size_t align(size_t size, paddle::platform::GPUPlace place) {
  size += sizeof(paddle::memory::detail::Metadata);
  size_t alignment = paddle::platform::GpuMinChunkSize();
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}

TEST(BuddyAllocator, GPUAllocation) {
  void *p = nullptr;

  EXPECT_EQ(p, nullptr);

  paddle::platform::GPUPlace gpu(0);
  p = paddle::memory::Alloc(gpu, 4096);

  EXPECT_NE(p, nullptr);

  paddle::memory::Free(gpu, p);
}

TEST(BuddyAllocator, GPUMultAlloc) {
  paddle::platform::GPUPlace gpu;

  std::unordered_map<void *, size_t> ps;

  size_t total_size = paddle::memory::Used(gpu);
  EXPECT_EQ(total_size, 0UL);

  for (auto size :
       {128, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304}) {
    ps[paddle::memory::Alloc(gpu, size)] = size;

    // Buddy Allocator doesn't manage too large memory chunk
    if (paddle::memory::Used(gpu) == total_size) continue;

    size_t aligned_size = align(size, gpu);
    total_size += aligned_size;
    EXPECT_EQ(total_size, paddle::memory::Used(gpu));
  }

  for (auto p : ps) {
    EXPECT_EQ(is_aligned(p.first), true);
    paddle::memory::Free(gpu, p.first);

    // Buddy Allocator doesn't manage too large memory chunk
    if (paddle::memory::Used(gpu) == total_size) continue;

    size_t aligned_size = align(p.second, gpu);
    total_size -= aligned_size;
    EXPECT_EQ(total_size, paddle::memory::Used(gpu));
  }
}

#endif  // PADDLE_ONLY_CPU
