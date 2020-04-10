// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/thread_local_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

class ThreadLocalAllocation : public Allocation {
 public:
  ThreadLocalAllocation(void* ptr, size_t size, platform::Place place)
      : Allocation(ptr, size, place) {}

  void SetThreadLocalAllocator(std::shared_ptr<Allocator> allocator) {
    allocator_ = allocator;
  }

 private:
  std::shared_ptr<Allocator> allocator_;
};

Allocation* ThreadLocalAllocator::AllocateImpl(size_t size) {
  void* ptr = buddy_allocator_->Alloc(size);
  auto* tl_allocation = new ThreadLocalAllocation(ptr, size, place_);
  tl_allocation->SetThreadLocalAllocator(shared_from_this());
  return tl_allocation;
}

void ThreadLocalAllocator::FreeImpl(Allocation* allocation) {
  auto* tl_allocation = static_cast<ThreadLocalAllocation*>(allocation);
  buddy_allocator_->Free(tl_allocation->ptr());
  delete tl_allocation;
}

const std::shared_ptr<Allocator>& GetThreadLocalAllocator() {
  static thread_local std::shared_ptr<Allocator> allocator(
      new ThreadLocalAllocator(platform::CUDAPlace()));
  return allocator;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
