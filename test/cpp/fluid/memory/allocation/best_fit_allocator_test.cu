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

#include <memory>
#include <random>
#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/memory/allocation/best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/cuda_allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/for_range.h"
namespace paddle {
namespace memory {
namespace allocation {

struct ForEachFill {
  size_t* ptr_;

  explicit ForEachFill(size_t* ptr) : ptr_(ptr) {}

  __device__ void operator()(size_t i) { ptr_[i] = i; }
};

TEST(BestFitAllocator, concurrent_cuda) {
  CUDAAllocator allocator(platform::CUDAPlace(0));
  // 256 MB
  auto cuda_allocation = allocator.Allocate(256U * 1024 * 1024);
  BestFitAllocator concurrent_allocator(cuda_allocation.get());

  platform::CUDAPlace gpu(0);
  phi::GPUContext dev_ctx(gpu);
  dev_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(gpu, dev_ctx.stream())
                           .get());
  dev_ctx.PartialInitWithAllocator();

  auto th_main = [&](std::random_device::result_type seed) {
    std::default_random_engine engine(seed);
    std::uniform_int_distribution<size_t> dist(1U, 1024U);
    std::array<size_t, 1024> buf;

    for (size_t i = 0; i < 128; ++i) {
      size_t allocate_size = dist(engine);

      auto allocation =
          concurrent_allocator.Allocate(sizeof(size_t) * allocate_size);

      size_t* data = reinterpret_cast<size_t*>(allocation->ptr());

      ForEachFill fill(data);
      platform::ForRange<phi::GPUContext> for_range(dev_ctx, allocate_size);
      for_range(fill);

      memory::Copy(platform::CPUPlace(),
                   buf.data(),
                   gpu,
                   data,
                   sizeof(size_t) * allocate_size,
                   dev_ctx.stream());

      dev_ctx.Wait();
      for (size_t j = 0; j < allocate_size; ++j) {
        ASSERT_EQ(buf[j], j);
      }
      allocation = nullptr;
    }
  };

  {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < 1024; ++i) {
      std::random_device dev;
      threads.emplace_back(th_main, dev());
    }
    for (auto& th : threads) {
      th.join();
    }
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
