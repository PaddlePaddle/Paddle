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
#include "paddle/platform/place.h"

#include "gtest/gtest.h"

TEST(BuddyAllocator, CPUAllocation) {
  void *p = nullptr;

  EXPECT_EQ(p, nullptr);

  paddle::platform::CPUPlace cpu;
  p = paddle::memory::Alloc(cpu, 4096);

  EXPECT_NE(p, nullptr);

  paddle::memory::Free(cpu, p);
}

#ifndef PADDLE_ONLY_CPU

TEST(BuddyAllocator, GPUAllocation) {
  void *p = nullptr;

  EXPECT_EQ(p, nullptr);

  paddle::platform::GPUPlace gpu(0);
  p = paddle::memory::Alloc(gpu, 4096);

  EXPECT_NE(p, nullptr);

  paddle::memory::Free(gpu, p);
}

#endif  // PADDLE_ONLY_CPU
