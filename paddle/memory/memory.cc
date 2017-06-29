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

#include "paddle/memory/memory.h"
#include "paddle/memory/detail/buddy_allocator.h"
#include "paddle/memory/detail/system_allocator.h"
#include "paddle/platform/assert.h"

#include <boost/variant.hpp>

namespace paddle {
namespace memory {

void* Alloc(platform::Place pl, size_t size) {
#ifndef PADDLE_ONLY_CPU
  if (paddle::platform::is_gpu_place(pl)) {
    size_t gpu_id = boost::get<platform::GPUPlace>(pl).device;
    return detail::GetGPUBuddyAllocator(gpu_id)->Alloc(size);
  }
#endif  // PADDLE_ONLY_CPU
  PADDLE_ASSERT(paddle::platform::is_cpu_place(pl));
  return detail::GetCPUBuddyAllocator()->Alloc(size);
}

void Free(paddle::platform::Place pl, void* p) {
#ifndef PADDLE_ONLY_CPU
  if (paddle::platform::is_gpu_place(pl)) {
    size_t gpu_id = boost::get<platform::GPUPlace>(pl).device;
    detail::GetGPUBuddyAllocator(gpu_id)->Free(p);
  }
#endif  // PADDLE_ONLY_CPU
  PADDLE_ASSERT(paddle::platform::is_cpu_place(pl));
  detail::GetCPUBuddyAllocator()->Free(p);
}

size_t Used(paddle::platform::Place pl) {
#ifndef PADDLE_ONLY_CPU
  if (paddle::platform::is_gpu_place(pl)) {
    size_t gpu_id = boost::get<platform::GPUPlace>(pl).device;
    return detail::GetGPUBuddyAllocator(gpu_id)->Used();
  }
#endif  // PADDLE_ONLY_CPU
  PADDLE_ASSERT(paddle::platform::is_cpu_place(pl));
  return detail::GetCPUBuddyAllocator()->Used();
}

}  // namespace memory
}  // namespace paddle
