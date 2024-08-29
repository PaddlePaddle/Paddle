// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/mmap_allocator.h"

#include "gtest/gtest.h"

namespace paddle {
namespace memory {
namespace allocation {

TEST(MemoryMapAllocation, test_allocation_base) {
  size_t data_size = 4UL * 1024;

  // 1. allocate writer holder
  auto mmap_writer_holder = AllocateMemoryMapWriterAllocation(data_size);
  std::string ipc_name = mmap_writer_holder->ipc_name();
  // 2. write data
  auto* writer_ptr = static_cast<int32_t*>(mmap_writer_holder->ptr());
  for (int32_t i = 0; i < 1024; ++i) {
    writer_ptr[i] = i;
  }
  // 3. create child process
  pid_t fpid = fork();
  if (fpid == 0) {
    // 4. rebuild reader holder
    auto mmap_reader_holder =
        RebuildMemoryMapReaderAllocation(ipc_name, data_size);
    auto* reader_ptr = static_cast<int32_t*>(mmap_reader_holder->ptr());
    for (int32_t i = 0; i < 1024; ++i) {
      ASSERT_EQ(reader_ptr[i], i);
    }
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
