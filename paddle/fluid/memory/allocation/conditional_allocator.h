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

class ConditionalAllocator : public ManagedAllocator {
 public:
  ConditionalAllocator() = default;

  ConditionalAllocator& AddAllocator(
      std::function<bool(size_t, Attr)> func,
      std::shared_ptr<ManagedAllocator> allocator);
  std::unique_ptr<Allocation> Allocate(size_t size, Attr attr) override;
  std::shared_ptr<Allocation> AllocateShared(size_t size, Attr attr) override;
  bool IsAllocThreadSafe() const override;

 private:
  template <typename Callback>
  inline typename std::result_of<Callback(ManagedAllocator&)>::type
  SelectAndInvoke(size_t size, Attr attr, Callback callback) {
    for (auto& pair : underlying_allocators_) {
      if (pair.first(size, attr)) {
        return callback(*pair.second);
      }
    }
    PADDLE_THROW("No suitable allocator");
  }

  std::vector<std::pair<std::function<bool(size_t, Attr)>,
                        std::shared_ptr<ManagedAllocator>>>
      underlying_allocators_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
