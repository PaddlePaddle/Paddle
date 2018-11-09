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

#include "paddle/fluid/memory/allocation/naive_managed_allocator.h"
#include <atomic>  // NOLINT
#include <random>
#include <thread>  // NOLINT
#include <vector>
#include "gtest/gtest.h"

namespace paddle {
namespace memory {
namespace allocation {

class StubAllocator : public UnmanagedAllocator {
 public:
  std::unique_ptr<Allocation> Allocate(size_t size,
                                       Attr attr = kDefault) override {
    counter_.fetch_add(1);
    return std::unique_ptr<Allocation>(
        new Allocation(nullptr, size, platform::CPUPlace()));
  }
  void FreeUniquePtr(std::unique_ptr<Allocation> allocation) override {
    counter_.fetch_sub(1);
  }
  bool IsAllocThreadSafe() const override { return true; }

  std::atomic<int> counter_{0};
};

TEST(NaiveManagedAllocator, main) {
  auto allocator = NaiveManagedAllocator::Create(
      std::unique_ptr<Allocator>(new StubAllocator()));

  auto th_main = [=] {
    std::random_device dev;
    std::default_random_engine engine(dev());
    std::uniform_int_distribution<int> dist(0, 1);

    std::vector<std::shared_ptr<Allocation>> allocations;

    for (int j = 0; j < 1024; ++j) {
      bool to_insert = static_cast<bool>(dist(engine));
      if (to_insert) {
        allocations.emplace_back(allocator->AllocateShared(10));
      } else {
        if (!allocations.empty()) {
          allocations.pop_back();
        }
      }
    }
  };

  {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < 1024; ++i) {
      threads.emplace_back(th_main);
    }
    for (auto& th : threads) {
      th.join();
    }
  }
  ASSERT_EQ(reinterpret_cast<StubAllocator&>(
                std::dynamic_pointer_cast<NaiveManagedAllocator>(allocator)
                    ->UnderlyingAllocator())
                .counter_,
            0);
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
