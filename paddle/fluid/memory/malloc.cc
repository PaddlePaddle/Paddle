/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/memory/malloc.h"

DEFINE_bool(init_allocated_mem, false,
            "It is a mistake that the values of the memory allocated by "
            "BuddyAllocator are always zeroed in some op's implementation. "
            "To find this error in time, we use init_allocated_mem to indicate "
            "that initializing the allocated memory with a small value "
            "during unit testing.");
DECLARE_double(fraction_of_gpu_memory_to_use);

namespace paddle {
namespace memory {

std::shared_ptr<Allocation> AllocShared(const platform::Place& place,
                                        size_t size, Allocator::Attr attr) {
  return allocation::AllocatorFacade::Instance().AllocShared(place, size, attr);
}

std::unique_ptr<Allocation> Alloc(const platform::Place& place, size_t size,
                                  Allocator::Attr attr) {
  return allocation::AllocatorFacade::Instance().Alloc(place, size, attr);
}
}  // namespace memory
}  // namespace paddle
