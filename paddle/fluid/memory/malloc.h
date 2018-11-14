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

#pragma once

#include <memory>
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/place.h"
namespace paddle {
namespace memory {
using allocation::Allocation;
using allocation::Allocator;
using allocation::AllocationPtr;

extern std::shared_ptr<Allocation> AllocShared(
    const platform::Place& place, size_t size,
    Allocator::Attr attr = Allocator::kDefault);

extern AllocationPtr Alloc(const platform::Place& place, size_t size,
                           Allocator::Attr attr = Allocator::kDefault);

namespace legacy {

template <typename Place>
void* Alloc(const Place& place, size_t size);

template <typename Place>
void Free(const Place& place, void* p);

template <typename Place>
size_t Used(const Place& place);

struct Usage : public boost::static_visitor<size_t> {
  size_t operator()(const platform::CPUPlace& cpu) const;
  size_t operator()(const platform::CUDAPlace& gpu) const;
  size_t operator()(const platform::CUDAPinnedPlace& cuda_pinned) const;
};

size_t memory_usage(const platform::Place& p);

}  // namespace legacy

}  // namespace memory
}  // namespace paddle
