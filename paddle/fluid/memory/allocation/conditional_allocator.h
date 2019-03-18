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
#include <functional>
#include <utility>
#include <vector>
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

// A composite allocator who will dispatch the allocation request by registered
// condition.
//
// For example:
//
// auto* cond_allocator = new ConditionalAllocator();
// cond_allocator->AddAllocator([](size_t size, Attr attr){
//   // if size > 10
//   return size > 10;
// }, allocator_a).AddAllocator([](size_t size, Attr attr){
//   // elif attr is kDefault
//   return attr == kDefault;
// }, allocator_b).AddAllocator([](size_t size, Attr attr){
//   // else
//   return true;
// }, allocator_c);
class ConditionalAllocator : public Allocator {
 public:
  ConditionalAllocator() = default;

  ConditionalAllocator& AddAllocator(std::function<bool(size_t, Attr)> func,
                                     std::shared_ptr<Allocator> allocator);

  bool IsAllocThreadSafe() const override;

 protected:
  Allocation* AllocateImpl(size_t size, Allocator::Attr attr) override;

 private:
  using AllocatorWithCond =
      std::pair<std::function<bool(size_t, Attr)>, std::shared_ptr<Allocator>>;
  std::vector<AllocatorWithCond> underlying_allocators_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
