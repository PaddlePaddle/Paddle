// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/memory_utils.h"

namespace phi {

namespace memory_utils {

Allocator::AllocationPtr Alloc(const phi::GPUPlace& place,
                               size_t size,
                               const phi::Stream& stream) {
  return MemoryUtils::Instance().Alloc(place, size, stream);
}

Allocator::AllocationPtr Alloc(const phi::Place& place, size_t size) {
  return MemoryUtils::Instance().Alloc(place, size);
}

std::shared_ptr<Allocation> AllocShared(const phi::Place& place,
                                        size_t size,
                                        const phi::Stream& stream) {
  return MemoryUtils::Instance().AllocShared(place, size, stream);
}

std::shared_ptr<Allocation> AllocShared(const phi::Place& place, size_t size) {
  return MemoryUtils::Instance().AllocShared(place, size);
}

bool InSameStream(const std::shared_ptr<Allocation>& allocation,
                  const phi::Stream& stream) {
  return MemoryUtils::Instance().InSameStream(allocation, stream);
}

void AllocationDeleter(Allocation* allocation) {
  MemoryUtils::Instance().AllocationDeleter(allocation);
}

}  // namespace memory_utils

}  // namespace phi
