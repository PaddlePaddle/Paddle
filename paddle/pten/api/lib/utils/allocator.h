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
#include "paddle/pten/core/allocator.h"
#include "paddle/pten/core/storage.h"

namespace paddle {
namespace experimental {

class DefaultAllocator : public pten::Allocator {
 public:
  explicit DefaultAllocator(const paddle::platform::Place& place)
      : place_(place) {}

  AllocationPtr Allocate(size_t bytes_size) override {
    return memory::Alloc(place_, bytes_size);
  }

 private:
  paddle::platform::Place place_;
};

class StringAllocator : public pten::Allocator {
 public:
  explicit StringAllocator(const paddle::platform::Place& place)
      : place_(place) {}

  AllocationPtr Allocate(size_t bytes_size) override {
    paddle::memory::AllocationPtr a = memory::Alloc(place_, bytes_size);
    void* ptr = a->ptr();
    if (paddle::platform::is_cpu_place(place_)) {
      std::memset(ptr, 0, bytes_size);
    } else if (paddle::platform::is_gpu_place(place_)) {
#ifdef PADDLE_WITH_HIP
      hipMemset(ptr, 0, bytes_size);
#else
      cudaMemset(ptr, 0, bytes_size);
#endif
    }
    return a;
  }

 private:
  paddle::platform::Place place_;
};

}  // namespace experimental
}  // namespace paddle
