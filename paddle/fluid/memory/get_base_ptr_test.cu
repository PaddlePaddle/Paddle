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

#include <random>
#include "gtest/gtest.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {

class CUDAAllocatoionBasePtrTest : public ::testing::Test {
 public:
  void SetUp() override {
    place_ = platform::CUDAPlace();
    alloc_times_ = 100;
    batch_size_ = 10;
    max_alloc_size_ = platform::GpuMaxAllocSize() / alloc_times_;
    random_engine_ = std::default_random_engine(time(NULL));
    dis_ = std::uniform_int_distribution<int>(0, max_alloc_size_);
  }

  void OneByOneAllocTest() {
    for (size_t i = 0; i < alloc_times_; ++i) {
      size_t size = dis_(random_engine_);
      auto allocation = AllocShared(place_, size);

      void* base_ptr = GetBasePtr(allocation);
      void* system_ptr =
          platform::GetGpuBasePtr(allocation->ptr(), place_.GetDeviceId());
      EXPECT_EQ(base_ptr, system_ptr);
    }

    Release(place_);
  }

  void BatchByBatchAllocTest() {
    std::vector<std::shared_ptr<phi::Allocation>> allocations;
    allocations.reserve(batch_size_);
    size_t batch_num = alloc_times_ / batch_size_;

    for (size_t i = 0; i < batch_num; ++i) {
      for (size_t j = 0; j < batch_size_; ++j) {
        size_t size = dis_(random_engine_);
        auto allocation = AllocShared(place_, size);

        void* base_ptr = GetBasePtr(allocation);
        void* system_ptr =
            platform::GetGpuBasePtr(allocation->ptr(), place_.GetDeviceId());
        EXPECT_EQ(base_ptr, system_ptr);

        allocations.emplace_back(allocation);
      }
      allocations.clear();
    }

    Release(place_);
  }

  void ContinuousAllocTest() {
    std::vector<std::shared_ptr<phi::Allocation>> allocations;
    allocations.reserve(alloc_times_);

    for (size_t i = 0; i < alloc_times_; ++i) {
      size_t size = dis_(random_engine_);
      auto allocation = AllocShared(place_, size);

      void* base_ptr = GetBasePtr(allocation);
      void* system_ptr =
          platform::GetGpuBasePtr(allocation->ptr(), place_.GetDeviceId());
      EXPECT_EQ(base_ptr, system_ptr);

      allocations.emplace_back(allocation);
    }

    allocations.clear();
    Release(place_);
  }

  void ZeroSizeAllocTest() {
    auto allocation = AllocShared(place_, 0);
    void* base_ptr = GetBasePtr(allocation);
    void* system_ptr =
        platform::GetGpuBasePtr(allocation->ptr(), place_.GetDeviceId());
    EXPECT_EQ(base_ptr, system_ptr);
  }

 private:
  platform::CUDAPlace place_;
  size_t max_alloc_size_;
  size_t alloc_times_;
  size_t batch_size_;
  std::default_random_engine random_engine_;
  std::uniform_int_distribution<int> dis_;
};

TEST_F(CUDAAllocatoionBasePtrTest, base_ptr_test) {
  OneByOneAllocTest();
  BatchByBatchAllocTest();
  ContinuousAllocTest();
  ZeroSizeAllocTest();
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
