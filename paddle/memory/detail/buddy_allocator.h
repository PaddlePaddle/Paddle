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

#include <mutex>
#include <vector>

namespace paddle {
namespace memory {
namespace detail {

class BuddyAllocator {
 public:
  BuddyAllocator(size_t pool_size, size_t max_pools,
                 SystemAllocator* system_allocator);
  ~BuddyAllocator();

  void* Alloc(size_t size);
  void Free(void*);
  size_t Used();

 private:
  struct Block {
    size_t size_;
    Block* left_;   // left buddy
    Block* right_;  // right buddy
  };

  // Initially, there is only one pool.  If a Alloc founds not enough
  // memory from that pool, and there has not been max_num_pools_,
  // create a new pool by calling system_allocator_.Alloc(pool_size_).
  std::vector<void*> pools_;

  size_t pool_size_;      // the size of each pool;
  size_t max_num_pools_;  // the size of all pools;

  SystemAllocator* system_allocator_;

  std::mutex mutex_;

  // Disable copy and assignment.
  BuddyAllocator(const BuddyAllocator&) = delete;
  BuddyAllocator& operator=(const BuddyAllocator&) = delete;
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
