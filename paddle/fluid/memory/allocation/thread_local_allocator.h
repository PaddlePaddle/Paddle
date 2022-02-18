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

#pragma once

#include <memory>
#include <vector>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/detail/buddy_allocator.h"
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {

class ThreadLocalAllocatorImpl;

class ThreadLocalAllocation : public Allocation {
 public:
  ThreadLocalAllocation(void* ptr, size_t size, platform::Place place)
      : Allocation(ptr, size, place) {}

  void SetThreadLocalAllocatorImpl(
      std::shared_ptr<ThreadLocalAllocatorImpl> allocator) {
    allocator_ = allocator;
  }

  std::shared_ptr<ThreadLocalAllocatorImpl> GetAllocator() {
    return allocator_;
  }

 private:
  std::shared_ptr<ThreadLocalAllocatorImpl> allocator_;
};

class ThreadLocalAllocatorImpl
    : public std::enable_shared_from_this<ThreadLocalAllocatorImpl> {
 public:
  explicit ThreadLocalAllocatorImpl(const platform::Place& p);
  ThreadLocalAllocation* AllocateImpl(size_t size);
  void FreeImpl(ThreadLocalAllocation* allocation);
  uint64_t ReleaseImpl();

 private:
  std::unique_ptr<memory::detail::BuddyAllocator> buddy_allocator_;
  platform::Place place_;
};

class ThreadLocalCUDAAllocatorPool {
 public:
  static ThreadLocalCUDAAllocatorPool& Instance() {
    static thread_local ThreadLocalCUDAAllocatorPool pool;
    return pool;
  }

  std::shared_ptr<ThreadLocalAllocatorImpl> Get(int gpu_id);

 private:
  ThreadLocalCUDAAllocatorPool();
  std::vector<int> devices_;
  std::vector<std::unique_ptr<std::once_flag>> init_flags_;
  std::vector<std::shared_ptr<ThreadLocalAllocatorImpl>> allocators_;
};

class ThreadLocalCUDAAllocator : public Allocator {
 public:
  explicit ThreadLocalCUDAAllocator(const platform::CUDAPlace& p)
      : gpu_id_(p.device) {}

  bool IsAllocThreadSafe() const override { return true; }

 protected:
  pten::Allocation* AllocateImpl(size_t size) override {
    return ThreadLocalCUDAAllocatorPool::Instance().Get(gpu_id_)->AllocateImpl(
        size);
  }
  void FreeImpl(pten::Allocation* allocation) override {
    auto* tl_allocation = static_cast<ThreadLocalAllocation*>(allocation);
    auto allocator_impl = tl_allocation->GetAllocator();
    allocator_impl->FreeImpl(tl_allocation);
  }
  uint64_t ReleaseImpl(const platform::Place& p) override {
    return ThreadLocalCUDAAllocatorPool::Instance().Get(gpu_id_)->ReleaseImpl();
  }

 private:
  int gpu_id_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
