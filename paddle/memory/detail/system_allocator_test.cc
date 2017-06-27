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

#include "paddle/memory/detail/system_allocator.h"

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

template <typename Allocator>
void TestAllocator(void* p) {
  p = Allocator::Alloc(1024);

  int* i = static_cast<int*>(p);
  std::shared_ptr<int> ptr(i, [](int* p) { Allocator::Free(p, 1024); });

  EXPECT_NE(p, nullptr);
}

TEST(CPUAllocator, NoLockMem) {
  void* p = nullptr;
  FLAGS_uses_pinned_memory = false;
  TestAllocator<paddle::memory::detail::CPUAllocator>(p);
  EXPECT_EQ(p, nullptr);
}

TEST(CPUAllocator, LockMem) {
  void* p = nullptr;
  FLAGS_uses_pinned_memory = true;
  TestAllocator<paddle::memory::detail::CPUAllocator>(p);
  EXPECT_EQ(p, nullptr);
}

#ifndef PADDLE_ONLY_CPU
TEST(GPUAllocator, NoStaging) {
  void* p = nullptr;
  FLAGS_uses_pinned_memory = false;
  TestAllocator<paddle::memory::detail::GPUAllocator>(p);
  EXPECT_EQ(p, nullptr);
}
TEST(GPUAllocator, Staging) {
  void* p = nullptr;
  FLAGS_uses_pinned_memory = true;
  TestAllocator<paddle::memory::detail::GPUAllocator>(p);
  EXPECT_EQ(p, nullptr);
}
#endif  // PADDLE_ONLY_CPU
