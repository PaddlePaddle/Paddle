/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/memory/detail/system_allocator.h"

namespace paddle {
namespace memory {
namespace detail {

template<typename Allocator>
class BuddyAllocator {
  public:
    // TODO(gangliao): This is a draft, add Buddy Allocator Algorithm soon
    BuddyAllocator() {}
    ~BuddyAllocator() {}

  public:
    void* Alloc(size_t size) {
        return Allocator::Alloc(size); 
    }
    void Free(void*) {
      // Because all info like size are stored in meta data,
      // thus it's duplicate if add the parameter `size` in
      // `Free(void*)` interface.
    }
    size_t Used();

  public:
    BuddyAllocator(const BuddyAllocator&) = delete;
    BuddyAllocator& operator=(const BuddyAllocator&) = delete;

  private:
    size_t min_alloc_size_;
    size_t max_alloc_size_;

  private:
    std::mutex mutex_;
};

BuddyAllocator<CPUAllocator>* GetCPUBuddyAllocator() {
  static BuddyAllocator<CPUAllocator>* a = nullptr;
  if (a == nullptr) {
    a = new BuddyAllocator<CPUAllocator>();
  }
  return a;
}

#ifndef PADDLE_ONLY_CPU  // The following code are for CUDA.

BuddyAllocator<GPUAllocator>* GetGPUBuddyAllocator(int gpu_id) {
  static BuddyAllocator<GPUAllocator>** as = NULL;
  if (as == NULL) {
    int gpu_num = platform::GetDeviceCount(); 
    as = new BuddyAllocator<GPUAllocator>*[gpu_num];
    for (int gpu = 0; gpu < gpu_num; gpu++) {
        as[gpu] = new BuddyAllocator<GPUAllocator>();
    }
  }
  return as[gpu_id];
}

#endif  // PADDLE_ONLY_CPU 

}  // namespace detail
}  // namespace memory
}  // namespace paddle
