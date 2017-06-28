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

#include "paddle/memory/detail/buddy_allocator.h"

namespace paddle {
namespace memory {
namespace detail {

BuddyAllocator::BuddyAllocator(size_t pool_size, size_t max_pools,
                               SystemAllocator* system_allocator)
    : pool_size_(pool_size),
      max_pools_(max_pools),
      system_allocator_(system_allocator) {
  PADDLE_ASSERT(pool_size > 0);
  PADDLE_ASSERT(max_pools > 0);
  PADDLE_ASSERT(system_allocator != nullptr);
}

}  // namespace detail
}  // namespace memory
}  // namespace paddle
