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

#include <paddle/memory/cpu/pinning.h>
#include <paddle/memory/cpu/system_allocator.h>

#include <cassert>
#include <cstdlib>
#include <memory>
#include <vector>

namespace paddle {
namespace memory {
namespace cpu {

class Allocator {
public:
  virtual ~Allocator() {}

public:
  virtual void* malloc(size_t) = 0;
  virtual void free(void*, size_t) = 0;
};

class PinnedAllocator : public Allocator {
public:
  virtual void* malloc(size_t size) {
    void* address = std::malloc(size);

    if (address == nullptr) {
      return address;
    }

    memory::cpu::pin_memory(address, size);
    return address;
  }

  virtual void free(void* address, size_t size) {
    memory::cpu::unpin_memory(address, size);
    std::free(address);
  }
};

class DefaultAllocator : public Allocator {
public:
  virtual void* malloc(size_t size) { return std::malloc(size); }

  virtual void free(void* address, size_t size) { return std::free(address); }
};

static std::vector<std::unique_ptr<Allocator>> system_allocators;

void* SystemAllocator::malloc(size_t& index, size_t size) {
  index = 0;

  for (auto& allocator : system_allocators) {
    void* address = allocator->malloc(size);

    if (address == nullptr) {
      ++index;
      continue;
    }

    return address;
  }

  return nullptr;
}

void SystemAllocator::free(void* address, size_t size, size_t index) {
  assert(index < system_allocators.size());

  system_allocators[index]->free(address, size);
}

size_t SystemAllocator::index_count() { return system_allocators.size(); }

void SystemAllocator::init() {
  assert(system_allocators.empty());

  // make sure no copies occur
  system_allocators.reserve(2);

  // add the pinned allocator
  system_allocators.push_back(std::unique_ptr<Allocator>(new PinnedAllocator));

  // add the default allocator
  system_allocators.push_back(std::unique_ptr<Allocator>(new DefaultAllocator));
}

void SystemAllocator::shutdown() {
  assert(!system_allocators.empty());

  // destroy all allocators
  system_allocators.clear();
}

bool SystemAllocator::uses_gpu() { return false; }

}  // namespace cpu
}  // namespace memory
}  // namespace paddle
