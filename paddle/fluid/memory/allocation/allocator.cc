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

#include "paddle/fluid/memory/allocation/allocator.h"

#include <functional>

namespace paddle {
namespace memory {
namespace allocation {

std::mutex Allocation::s_mutex_global;

std::unordered_map<int, std::pair<uint64_t, uint64_t>>
    Allocation::s_memory_allocated;
std::unordered_map<int, std::unique_ptr<std::mutex>> Allocation::s_mutex_map;

void Allocation::CheckMutex(const int& device_id) {
  if (s_mutex_map.find(device_id) == s_mutex_map.end()) {
    s_mutex_global.lock();
    if (s_mutex_map.find(device_id) == s_mutex_map.end())
      s_mutex_map[device_id] =
          std::move(std::unique_ptr<std::mutex>(new std::mutex()));
    s_mutex_global.unlock();
  }
}

Allocator::~Allocator() {}

bool Allocator::IsAllocThreadSafe() const { return false; }

AllocationPtr Allocator::Allocate(size_t size, Allocator::Attr attr) {
  auto ptr = AllocateImpl(size, attr);
  ptr->set_allocator(this);
  return AllocationPtr(ptr);
}

void Allocator::Free(Allocation* allocation) { delete allocation; }

const char* BadAlloc::what() const noexcept { return msg_.c_str(); }

void AllocationDeleter::operator()(Allocation* allocation) const {
  auto* allocator = allocation->allocator();
  allocator->Free(allocation);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
