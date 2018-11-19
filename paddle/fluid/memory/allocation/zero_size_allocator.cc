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

#include "paddle/fluid/memory/allocation/zero_size_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

bool ZeroSizeAllocator::IsAllocThreadSafe() const {
  return underlying_allocator_->IsAllocThreadSafe();
}

Allocation *ZeroSizeAllocator::AllocateImpl(size_t size, Allocator::Attr attr) {
  if (size == 0) {
    return new ZeroSizeAllocation(place_);
  } else {
    return underlying_allocator_->Allocate(size, attr).release();
  }
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
