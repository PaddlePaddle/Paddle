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

namespace paddle {
namespace memory {

detail::BuddyAllocator* GetCPUBuddyAllocator() {
  static detail::BuddyAllocator* a = nullptr;
  if (a == nullptr) {
    a = new detail::BuddyAllocator(new detail::CPUAllocator,
                                   platform::CpuMinChunkSize(),
                                   platform::CpuMaxChunkSize());
  }
  return a;
}

template <>
void* Alloc<platform::CPUPlace>(platform::CPUPlace place, size_t size) {
  return GetCPUBuddyAllocator()->Alloc(size);
}

template <>
void Free<platform::CPUPlace>(platform::CPUPlace place, void* p) {
  GetCPUBuddyAllocator()->Free(p);
}

template <>
size_t Used<platform::CPUPlace>(platform::CPUPlace place) {
  return GetCPUBuddyAllocator()->Used();
}

#ifndef PADDLE_ONLY_CPU

detail::BuddyAllocator* GetGPUBuddyAllocator(int gpu_id) {
  static detail::BuddyAllocator** as = NULL;
  if (as == NULL) {
    int gpu_num = platform::GetDeviceCount();
    as = new detail::BuddyAllocator*[gpu_num];
    for (int gpu = 0; gpu < gpu_num; gpu++) {
      platform::SetDeviceId(gpu);
      as[gpu] = new detail::BuddyAllocator(new detail::GPUAllocator,
                                           platform::GpuMinChunkSize(),
                                           platform::GpuMaxChunkSize());
    }
  }
  return as[gpu_id];
}

template <>
void* Alloc<platform::GPUPlace>(platform::GPUPlace place, size_t size) {
  return GetGPUBuddyAllocator(place.device)->Alloc(size);
}

template <>
void Free<platform::GPUPlace>(platform::GPUPlace place, void* p) {
  GetGPUBuddyAllocator(place.device)->Free(p);
}

template <>
size_t Used<platform::GPUPlace>(platform::GPUPlace place) {
  return GetGPUBuddyAllocator(place.device)->Used();
}

#endif  // PADDLE_ONLY_CPU

}  // namespace memory
}  // namespace paddle
