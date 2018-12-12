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
#include "paddle/fluid/platform/lock_guard_ptr.h"

namespace paddle {
namespace platform {

class TemporalAllocation : public memory::allocation::Allocation {
 public:
#ifdef PADDLE_WITH_CUDA
  TemporalAllocation(memory::allocation::AllocationPtr &&underlying_allocation,
                     const cudaStream_t &stream);
#endif
  explicit TemporalAllocation(
      memory::allocation::AllocationPtr &&underlying_allocation);

  memory::allocation::AllocationPtr underlying_allocation_;
#ifdef PADDLE_WITH_CUDA
  cudaStream_t stream_;
#endif
};

class TemporalAllocator : public memory::allocation::Allocator {
 public:
  explicit TemporalAllocator(platform::Place place);

  bool IsAllocThreadSafe() const override;

 protected:
  void Free(memory::allocation::Allocation *allocation) override;

  memory::allocation::Allocation *AllocateImpl(
      size_t size, memory::allocation::Allocator::Attr attr) override;

 private:
  platform::Place place_;
#ifdef PADDLE_WITH_CUDA
  cudaStream_t stream_;
#endif
};

}  // namespace platform
}  // namespace paddle
