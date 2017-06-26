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
#include "gtest/gtest.h"

TEST(CPUAllocator, NoLockMem) {
  paddle::memory::detail::CPUAllocator<false> a;
  void* p = a.Alloc(4096);
  EXPECT_NE(p, nullptr);
  a.Free(p, 4096);
}

TEST(CPUAllocator, LockMem) {
  paddle::memory::detail::CPUAllocator<true> a;
  void* p = a.Alloc(4096);
  EXPECT_NE(p, nullptr);
  a.Free(p, 4096);
}

#ifndef PADDLE_ONLY_CPU

TEST(GPUAllocator, NonStaging) {
  paddle::memory::detail::GPUAllocator<false> a;
  void* p = a.Alloc(4096);
  EXPECT_NE(p, nullptr);
  a.Free(p, 4096);
}

TEST(GPUAllocator, Staging) {
  paddle::memory::detail::GPUAllocator<true> a;
  void* p = a.Alloc(4096);
  EXPECT_NE(p, nullptr);
  a.Free(p, 4096);
}

#endif  // PADDLE_ONLY_CPU
