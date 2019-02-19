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

#include "paddle/fluid/memory/detail/bud_allocator.h"
#include <gtest/gtest.h>

namespace paddle {
namespace memory {
namespace detail {

TEST(BudAllocator, Normalize) {
  ASSERT_EQ(512, BudAllocator::Normalize(300));
  ASSERT_EQ(512, BudAllocator::Normalize(511));
  ASSERT_EQ(512, BudAllocator::Normalize(512));
  ASSERT_NE(512, BudAllocator::Normalize(513));
  ASSERT_EQ(1024, BudAllocator::Normalize(513));
}

TEST(BudAllocator, GetLevel) {
  ASSERT_EQ(8, BudAllocator::GetLevel(300));
  ASSERT_EQ(9, BudAllocator::GetLevel(BudAllocator::Normalize(300)));
  ASSERT_EQ(8, BudAllocator::GetLevel(511));
  ASSERT_EQ(9, BudAllocator::GetLevel(513));
}

TEST(BudAllocator, CPUAllocateAndFree) {
  auto allocator = new BudAllocator(
      std::unique_ptr<detail::SystemAllocator>(new detail::CPUAllocator()));
  allocator->InitBySize(5000);
  ASSERT_EQ(1, allocator->NumOfBlocks());
  ASSERT_EQ(8192, allocator->FreeSize());

  auto ptr1 = allocator->Alloc(300);
  ASSERT_EQ(5, allocator->NumOfBlocks());

  auto ptr2 = allocator->Alloc(200);
  ASSERT_EQ(6, allocator->NumOfBlocks());
  ASSERT_EQ(7424, allocator->FreeSize());

  allocator->Free(ptr1);
  ASSERT_EQ(6, allocator->NumOfBlocks());

  allocator->Free(ptr2);
  ASSERT_EQ(1, allocator->NumOfBlocks());
  ASSERT_EQ(8192, allocator->FreeSize());
}

#ifdef PADDLE_WITH_CUDA
TEST(BudAllocator, GPUAllocateAndFree) {
  auto allocator = new BudAllocator(
      std::unique_ptr<detail::SystemAllocator>(new detail::GPUAllocator(0)));
  allocator->InitBySize(5000);
  ASSERT_EQ(1, allocator->NumOfBlocks());
  ASSERT_EQ(8192, allocator->FreeSize());

  auto ptr1 = allocator->Alloc(300);
  ASSERT_EQ(5, allocator->NumOfBlocks());

  auto ptr2 = allocator->Alloc(200);
  ASSERT_EQ(6, allocator->NumOfBlocks());
  ASSERT_EQ(7424, allocator->FreeSize());

  allocator->Free(ptr1);
  ASSERT_EQ(6, allocator->NumOfBlocks());

  allocator->Free(ptr2);
  ASSERT_EQ(1, allocator->NumOfBlocks());
  ASSERT_EQ(8192, allocator->FreeSize());
}
#endif

}  // namespace detail
}  // namespace memory
}  // namespace paddle
