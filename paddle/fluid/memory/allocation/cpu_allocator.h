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
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {
// CPU system allocator and allocation.
//
// NOTE(yy): Should we just use `malloc` here since there is an
// aligned_allocator.
//
// NOTE(yy): It is no need to use `BestFitAllocator` in CPU. We can import
// an open-sourced allocator into Paddle.
class CPUAllocator;
class CPUAllocation : public MannualFreeAllocation {
 public:
  CPUAllocation(CPUAllocator* allocator, void* ptr, size_t size);
};

class CPUAllocator : public MannualFreeAllocator {
 public:
  constexpr static size_t kAlignment = 64u;
  bool IsAllocThreadSafe() const override;

 protected:
  void Free(MannualFreeAllocation* allocation) override;
  MannualFreeAllocation* AllocateImpl(size_t size,
                                      Allocator::Attr attr) override;
};
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
