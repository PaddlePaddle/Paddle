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

#ifndef _WIN32

#include "paddle/fluid/memory/allocation/mmap_allocator.h"

#include <sys/types.h>

#include "gtest/gtest.h"

namespace paddle {
namespace memory {
namespace allocation {

TEST(MemoryMapAllocator, test_allocation) {
  size_t data_size = 4UL * 1024;
  // 1. allocate memoruy
  MemoryMapAllocator allocator;
  auto mmap_allocation = allocator.Allocate(data_size);
  auto* mmap_ptr = static_cast<int32_t*>(mmap_allocation->ptr());
  // 2. allocate memoruy & copy
  for (int32_t i = 0; i < 1024; ++i) {
    mmap_ptr[i] = i;
  }
  // 3. create child process
  pid_t fpid = fork();
  if (fpid == 0) {
    std::stringstream ss;
    for (int32_t i = 0; i < 1024; ++i) {
      ASSERT_EQ(mmap_ptr[i], i);
    }
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
