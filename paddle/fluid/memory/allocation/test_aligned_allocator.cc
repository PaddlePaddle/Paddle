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

#include "gtest/gtest.h"
#include "paddle/fluid/memory/allocation/aligned_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

TEST(aligned, aligned_size) {
  ASSERT_EQ(AlignedSize(1024, 1024), 1024);
  ASSERT_EQ(AlignedSize(1023, 1024), 1024);
  ASSERT_EQ(AlignedSize(1025, 1024), 2048);
}

struct StubAllocator : public Allocator {
 public:
  StubAllocator() = default;

  size_t AllocNum() const { return alloc_num_; }

 protected:
  Allocation *AllocateImpl(size_t size) override {
    ++alloc_num_;
    return new Allocation(new uint8_t[size], size, platform::CPUPlace());
  }

  void FreeImpl(Allocation *allocation) override {
    delete[] static_cast<uint8_t *>(allocation->ptr());
    delete allocation;
    --alloc_num_;
  }

 private:
  size_t alloc_num_{0};
};

bool IsAligned(const AllocationPtr &alloc, size_t alignment) {
  return reinterpret_cast<uintptr_t>(alloc->ptr()) % alignment == 0;
}

TEST(aligned_allocator, aligned_allocator) {
  size_t alignment = 1024;
  auto allocator = std::make_shared<StubAllocator>();
  auto aligned_allocator =
      std::make_shared<AlignedAllocator>(allocator, alignment);

  auto alloc1 = aligned_allocator->Allocate(1345);
  ASSERT_EQ(allocator->AllocNum(), 1);
  ASSERT_TRUE(IsAligned(alloc1, alignment));
  alloc1.reset();
  ASSERT_EQ(allocator->AllocNum(), 0);

  {
    auto alloc2 = aligned_allocator->Allocate(200);
    ASSERT_TRUE(IsAligned(alloc2, alignment));
    ASSERT_EQ(allocator->AllocNum(), 1);

    auto alloc3 = aligned_allocator->Allocate(3021);
    ASSERT_TRUE(IsAligned(alloc3, alignment));
    ASSERT_EQ(allocator->AllocNum(), 2);
  }

  ASSERT_EQ(allocator->AllocNum(), 0);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
