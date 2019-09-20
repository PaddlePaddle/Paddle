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
#include <utility>
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

class AlignedAllocator : public Allocator {
 public:
  AlignedAllocator(const std::shared_ptr<Allocator>& underlying_allocator,
                   size_t alignment);

  bool IsAllocThreadSafe() const override;

 protected:
  Allocation* AllocateImpl(size_t size) override;

  void FreeImpl(Allocation* allocation) override;

 private:
  std::shared_ptr<Allocator> underlying_allocator_;
  size_t alignment_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
