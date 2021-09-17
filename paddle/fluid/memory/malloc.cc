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

#include "paddle/fluid/memory/malloc.h"

#include <thread>
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {

static void LogAllocation(Allocation *alloc, const platform::Place &place) {
  if (alloc == nullptr) {
    VLOG(10) << "Allocate 0 "
             << " on " << place << " addr " << static_cast<void *>(nullptr)
             << " " << std::this_thread::get_id();
  } else {
    VLOG(10) << "Allocate " << alloc->size() << " on " << alloc->place()
             << " addr " << alloc->ptr() << " " << std::this_thread::get_id();
  }
}

std::shared_ptr<Allocation> AllocShared(const platform::Place &place,
                                        size_t size) {
  auto alloc = allocation::AllocatorFacade::Instance().AllocShared(place, size);
  LogAllocation(alloc.get(), place);
  return alloc;
}

AllocationPtr Alloc(const platform::Place &place, size_t size) {
  auto alloc = allocation::AllocatorFacade::Instance().Alloc(place, size);
  LogAllocation(alloc.get(), place);
  return alloc;
}

uint64_t Release(const platform::Place &place) {
  return allocation::AllocatorFacade::Instance().Release(place);
}

}  // namespace memory
}  // namespace paddle
