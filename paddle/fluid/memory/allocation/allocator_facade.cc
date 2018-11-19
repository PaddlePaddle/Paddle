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

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include <map>
#include <string>
#include "paddle/fluid/memory/allocation/allocator_factory.h"
#include "paddle/fluid/memory/allocation/allocator_factory_registry.h"

USE_ALLOCATOR_FACTORY(legacy);
USE_ALLOCATOR_FACTORY(NaiveBestFit);

namespace paddle {
namespace memory {
namespace allocation {

class AllocatorFacadePrivate {
 public:
  std::map<platform::Place, std::unique_ptr<Allocator>> allocators_;

  ~AllocatorFacadePrivate() = default;

  AllocatorFacadePrivate()
      : allocators_(AllocatorFactoryRegistry::Instance().Get().Build()) {}
};

// Pimpl. Make interface clean.
AllocatorFacade::AllocatorFacade() : m_(new AllocatorFacadePrivate()) {}
AllocatorFacade::~AllocatorFacade() { delete m_; }

AllocatorFacade& AllocatorFacade::Instance() {
  static AllocatorFacade instance;
  return instance;
}

std::shared_ptr<Allocation> AllocatorFacade::AllocShared(
    const platform::Place& place, size_t size, Allocator::Attr attr) {
  return std::shared_ptr<Allocation>(Alloc(place, size, attr).release(),
                                     AllocationDeleter());
}

AllocationPtr AllocatorFacade::Alloc(const platform::Place& place, size_t size,
                                     Allocator::Attr attr) {
  auto it = m_->allocators_.find(place);
  if (it == m_->allocators_.end()) {
    throw BadAlloc(
        string::Sprintf("No such allocator for the place, %s", place));
  }
  return m_->allocators_.at(place)->Allocate(size, attr);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
