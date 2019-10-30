// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/numpy_allocator.h"
#include <stdlib.h>
#include <string>

namespace paddle {
namespace memory {
namespace allocation {
Allocation* NumpyAllocator::AllocateImpl(size_t size) {
  VLOG(3) << "NumpyAllocator::AllocateImpl, numpy_allocator: " << this;
  auto res = new NumpyAllocation(data_ptr, size, deleter);
  VLOG(3) << "NumpyAllocator::AllocateImpl, numpy_allocation: " << res;
  return res;
}
void NumpyAllocator::FreeImpl(Allocation* allocation) {
  VLOG(3) << "NumpyAllocator::FreeImpl";
  reinterpret_cast<NumpyAllocation*>(allocation)->callDeleter();
}
bool NumpyAllocator::IsAllocThreadSafe() const { return true; }
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
