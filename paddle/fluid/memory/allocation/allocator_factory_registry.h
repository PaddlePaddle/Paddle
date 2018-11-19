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
#include <list>
#include <memory>
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "paddle/fluid/platform/variant.h"
namespace paddle {
namespace memory {
namespace allocation {
class AllocatorFactory;
class AllocatorFactoryRegistry {
 private:
  AllocatorFactoryRegistry() = default;

 public:
  static AllocatorFactoryRegistry& Instance() {
    static AllocatorFactoryRegistry registry;
    return registry;
  }

  template <typename T>
  void Register() {
    factories_.emplace_back(new T());
  }

  AllocatorFactory& Get();

 private:
  std::list<std::unique_ptr<AllocatorFactory>> factories_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#define REGISTER_ALLOCATOR_FACTORY(id, TYPENAME)                       \
  int __allocator_init_func_##id##__() {                               \
    ::paddle::memory::allocation::AllocatorFactoryRegistry::Instance() \
        .template Register<TYPENAME>();                                \
    return 0;                                                          \
  }

#define USE_ALLOCATOR_FACTORY(id)                                \
  extern int __allocator_init_func_##id##__();                   \
  static UNUSED int __allocator_factory_regestry_item_##id##__ = \
      __allocator_init_func_##id##__();
