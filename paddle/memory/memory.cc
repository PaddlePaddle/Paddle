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

#include <algorithm>  // for transform
#include <cstring>    // for memcpy
#include <memory>     // for unique_ptr
#include <mutex>      // for call_once

#include "glog/logging.h"

#include "paddle/memory/detail/buddy_allocator.h"
#include "paddle/memory/detail/system_allocator.h"
#include "paddle/platform/gpu_info.h"

DECLARE_double(fraction_of_gpu_memory_to_use);

namespace paddle {
namespace memory {

using BuddyAllocator = detail::BuddyAllocator;

std::once_flag cpu_allocator_flag;
std::once_flag gpu_allocator_flag;

BuddyAllocator* GetCPUBuddyAllocator() {
  static std::unique_ptr<BuddyAllocator> a{nullptr};

  std::call_once(cpu_allocator_flag, [&]() {
    a.reset(new BuddyAllocator(new detail::CPUAllocator,
                               platform::CpuMinChunkSize(),
                               platform::CpuMaxChunkSize()));
  });

  return a.get();
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

#ifdef PADDLE_WITH_CUDA

BuddyAllocator* GetGPUBuddyAllocator(int gpu_id) {
  using BuddyAllocVec = std::vector<BuddyAllocator*>;
  static std::unique_ptr<BuddyAllocVec, void (*)(BuddyAllocVec * p)> as{
      new BuddyAllocVec, [](BuddyAllocVec* p) {
        std::for_each(p->begin(), p->end(),
                      [](BuddyAllocator* p) { delete p; });
      }};

  // GPU buddy allocators
  auto& allocators = *as.get();

  // GPU buddy allocator initialization
  std::call_once(gpu_allocator_flag, [&]() {
    int gpu_num = platform::GetCUDADeviceCount();
    allocators.reserve(gpu_num);
    for (int gpu = 0; gpu < gpu_num; gpu++) {
      platform::SetDeviceId(gpu);
      allocators.emplace_back(new BuddyAllocator(new detail::GPUAllocator,
                                                 platform::GpuMinChunkSize(),
                                                 platform::GpuMaxChunkSize()));
    }
    VLOG(3) << "\n\nNOTE: each GPU device use "
            << FLAGS_fraction_of_gpu_memory_to_use * 100 << "% of GPU memory.\n"
            << "You can set environment variable '"
            << platform::kEnvFractionGpuMemoryToUse
            << "' to change the fraction of GPU usage.\n\n";
  });

  platform::SetDeviceId(gpu_id);
  return allocators[gpu_id];
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
