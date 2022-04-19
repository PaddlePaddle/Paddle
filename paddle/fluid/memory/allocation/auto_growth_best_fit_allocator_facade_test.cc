// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include <condition_variable>  // NOLINT
#include <mutex>               // NOLINT
#include <random>
#include <thread>  // NOLINT
#include "gflags/gflags.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_double(fraction_of_cuda_pinned_memory_to_use);
DECLARE_int64(gpu_allocator_retry_time);
#endif

DECLARE_string(allocator_strategy);

namespace paddle {
namespace memory {
namespace allocation {

static inline size_t AlignTo(size_t size, size_t alignment) {
  auto remaining = size % alignment;
  return remaining == 0 ? size : size + alignment - remaining;
}

TEST(allocator, allocator) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  FLAGS_fraction_of_gpu_memory_to_use = 0.01;
  FLAGS_gpu_allocator_retry_time = 500;
  FLAGS_fraction_of_cuda_pinned_memory_to_use = 0.5;
#endif

  FLAGS_allocator_strategy = "auto_growth";

  auto &instance = AllocatorFacade::Instance();
  size_t size = 1024;
  platform::Place place;

  {
    place = platform::CPUPlace();
    size = 1024;
    auto cpu_allocation = instance.Alloc(place, size);
    ASSERT_NE(cpu_allocation, nullptr);
    ASSERT_NE(cpu_allocation->ptr(), nullptr);
    ASSERT_EQ(cpu_allocation->place(), place);
    ASSERT_EQ(cpu_allocation->size(), AlignedSize(size, 1024));
  }

#if (defined PADDLE_WITH_CUDA || defined PADDLE_WITH_HIP)
  {
    place = platform::CUDAPlace(0);
    size = 1024;
    auto gpu_allocation = instance.Alloc(place, size);
    ASSERT_NE(gpu_allocation, nullptr);
    ASSERT_NE(gpu_allocation->ptr(), nullptr);
    ASSERT_EQ(gpu_allocation->place(), place);
    ASSERT_GE(gpu_allocation->size(),
              AlignedSize(size, platform::GpuMinChunkSize()));
  }

  {
    // Allocate 2GB gpu memory
    place = platform::CUDAPlace(0);
    size = 2 * static_cast<size_t>(1 << 30);
    auto gpu_allocation = instance.Alloc(place, size);
    ASSERT_NE(gpu_allocation, nullptr);
    ASSERT_NE(gpu_allocation->ptr(), nullptr);
    ASSERT_EQ(gpu_allocation->place(), place);
    ASSERT_GE(gpu_allocation->size(),
              AlignedSize(size, platform::GpuMinChunkSize()));
  }

  {
    place = platform::CUDAPinnedPlace();
    size = (1 << 20);
    auto cuda_pinned_allocation =
        instance.Alloc(platform::CUDAPinnedPlace(), 1 << 20);
    ASSERT_NE(cuda_pinned_allocation, nullptr);
    ASSERT_NE(cuda_pinned_allocation->ptr(), nullptr);
    ASSERT_EQ(cuda_pinned_allocation->place(), place);
    ASSERT_GE(cuda_pinned_allocation->size(), AlignedSize(size, 1 << 20));
  }
#endif
}

TEST(multithread_allocate, test_segfault) {
  FLAGS_allocator_strategy = "auto_growth";
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::mutex mtx;
  std::condition_variable cv;
  bool flag = false;

  auto alloc_func = [&](int dev_id, unsigned int seed) {
    auto &instance = AllocatorFacade::Instance();

    std::mt19937 gen(seed);
    std::uniform_int_distribution<size_t> dist(1 << 20, 1 << 25);

    {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] { return flag; });
    }

    for (int i = 0; i < 50; i++) {
      size_t size = dist(gen);
      for (int j = 0; j < 10; j++) {
        instance.Alloc(platform::CUDAPlace(dev_id), size);
      }
    }
  };

  std::vector<std::thread> ths;
  for (size_t i = 0; i < 50; ++i) {
    std::random_device rd;
    ths.emplace_back(alloc_func, 0, rd());
  }

  {
    std::lock_guard<std::mutex> guard(mtx);
    flag = true;
  }
  cv.notify_all();

  for (auto &th : ths) {
    th.join();
  }
#endif
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
