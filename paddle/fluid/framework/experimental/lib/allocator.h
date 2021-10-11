/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/tcmpt/core/allocator.h"

namespace paddle {
namespace experimental {

class DefaultAllocator : public tcmpt::Allocator {
 public:
  explicit DefaultAllocator(const platform::Place& place) : place_(place) {}

  static void Delete(void* data) {
    deleter_(static_cast<memory::Allocation*>(data));
  }

  tcmpt::Allocation Allocate(size_t bytes_size) override {
    memory::AllocationPtr a = memory::Alloc(place_, bytes_size);
    void* ptr = a->ptr();
    return tcmpt::Allocation(ptr, a.release(), &Delete, place_);
  }

 private:
  platform::Place place_;
  static memory::Allocator::AllocationDeleter deleter_;
};

}  // namespace experimental
}  // namespace paddle
