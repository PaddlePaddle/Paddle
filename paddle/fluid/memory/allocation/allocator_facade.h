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

#pragma once
#include <memory>
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {

// Allocator Facade is the interface exposed to other modules.
// All the configuration or dirty code under development should
// be hidden behind this facade.
//
// NOTE(yy): This class is a singleton class.
// NOTE(yy): To create a stable ABI and make compilation faster. Here we use
// a Pimpl trick;
class AllocatorFacadePrivate;
class AllocatorFacade {
 public:
  ~AllocatorFacade();
  AllocatorFacade(const AllocatorFacade& o) = delete;
  const AllocatorFacade& operator=(const AllocatorFacade& o) = delete;

  static AllocatorFacade& Instance();

  // Allocate a shared allocation.
  std::shared_ptr<Allocation> AllocShared(const platform::Place& place,
                                          size_t size);

  // Allocate a unique allocation.
  AllocationPtr Alloc(const platform::Place& place, size_t size);

  // Release unused memory pool.
  uint64_t Release(const platform::Place& place);

  // TODO(yy): Allocate a Copy-On-Write allocation?
 private:
  AllocatorFacade();
  AllocatorFacadePrivate* m_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
