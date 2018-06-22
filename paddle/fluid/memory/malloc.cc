/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/memory/malloc.h"

#include "glog/logging.h"

#include "paddle/fluid/memory/detail/buddy_allocator.h"
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"

DECLARE_double(fraction_of_gpu_memory_to_use);

namespace paddle {
namespace memory {

using BuddyAllocator = detail::BuddyAllocator;

BuddyAllocator* GetCPUBuddyAllocator() {
  static detail::BuddyAllocator* a = nullptr;
  if (a == nullptr) {
    a = new detail::BuddyAllocator(new detail::CPUAllocator,
                                   platform::CpuMinChunkSize(),
                                   platform::CpuMaxChunkSize());
  }
  return a;
}

#ifdef PADDLE_WITH_CUDA

BuddyAllocator* GetGPUBuddyAllocator(int gpu_id) {
  static BuddyAllocator** as = NULL;
  if (as == NULL) {
    int gpu_num = platform::GetCUDADeviceCount();
    as = new BuddyAllocator*[gpu_num];
    for (int gpu = 0; gpu < gpu_num; gpu++) {
      as[gpu] = nullptr;
    }
  }
  platform::SetDeviceId(gpu_id);
  if (!as[gpu_id]) {
    as[gpu_id] = new BuddyAllocator(new detail::GPUAllocator(gpu_id),
                                    platform::GpuMinChunkSize(),
                                    platform::GpuMaxChunkSize());
    VLOG(10) << "\n\nNOTE: each GPU device use "
             << FLAGS_fraction_of_gpu_memory_to_use * 100
             << "% of GPU memory.\n"
             << "You can set GFlags environment variable '"
             << "FLAGS_fraction_of_gpu_memory_to_use"
             << "' to change the fraction of GPU usage.\n\n";
  }
  return as[gpu_id];
}

BuddyAllocator* GetCUDAPinnedBuddyAllocator() {
  static BuddyAllocator* ba = NULL;
  if (ba == NULL) {
    ba = new BuddyAllocator(new detail::CUDAPinnedAllocator,
                            platform::CUDAPinnedMinChunkSize(),
                            platform::CUDAPinnedMaxChunkSize());
  }
  return ba;
}

#endif  // PADDLE_WITH_CUDA

void* Alloc(const platform::Place& place, size_t size) {
  void* ptr = nullptr;

  if (platform::is_cpu_place(place)) {
    VLOG(10) << "Allocate " << size << " bytes on " << platform::Place(place);
    ptr = GetCPUBuddyAllocator()->Alloc(size);
    VLOG(10) << "  pointer=" << ptr;
  }
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(place)) {
    auto* buddy_allocator = GetGPUBuddyAllocator(
        dynamic_cast<const platform::CUDAPlace&>(place).device);
    ptr = buddy_allocator->Alloc(size);
    if (ptr == nullptr) {
      int cur_dev = platform::GetCurrentDeviceId();
      const platform::CUDAPlace& cuda_place =
          dynamic_cast<const platform::CUDAPlace&>(place);
      platform::SetDeviceId(cuda_place.device);
      size_t avail, total;
      platform::GpuMemoryUsage(&avail, &total);
      LOG(WARNING) << "Cannot allocate " << size << " bytes in GPU "
                   << cuda_place.device << ", available " << avail << " bytes";
      LOG(WARNING) << "total " << total;
      LOG(WARNING) << "GpuMinChunkSize " << platform::GpuMinChunkSize();
      LOG(WARNING) << "GpuMaxChunkSize " << platform::GpuMaxChunkSize();
      LOG(WARNING) << "GPU memory used: " << Used(cuda_place);
      platform::SetDeviceId(cur_dev);
    }
  } else if (platform::is_cuda_pinned_place(place)) {
    auto* buddy_allocator = GetCUDAPinnedBuddyAllocator();
    ptr = buddy_allocator->Alloc(size);
    if (ptr == nullptr) {
      LOG(WARNING) << "cudaMallocHost Cannot allocate " << size
                   << " bytes in CUDAPinnedPlace";
    }
  }
#endif  // PADDLE_WITH_CUDA

  return ptr;
}

void Free(const platform::Place& place, void* p) {
  VLOG(10) << "Free pointer=" << p << " on " << place;

  if (platform::is_cpu_place(place)) {
    GetCPUBuddyAllocator()->Free(p);
  }
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(place)) {
    GetGPUBuddyAllocator(dynamic_cast<const platform::CUDAPlace&>(place).device)
        ->Free(p);
  } else if (platform::is_cuda_pinned_place(place)) {
    GetCUDAPinnedBuddyAllocator()->Free(p);
  }
#endif  // PADDLE_WITH_CUDA
}

size_t Used(const platform::Place& place) {
  size_t r = 0;

  if (platform::is_cpu_place(place)) {
    r = GetCPUBuddyAllocator()->Used();
  }
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(place)) {
    r = GetGPUBuddyAllocator(
            dynamic_cast<const platform::CUDAPlace&>(place).device)
            ->Used();
  } else if (platform::is_cuda_pinned_place(place)) {
    r = GetCUDAPinnedBuddyAllocator()->Used();
  }
#endif  // PADDLE_WITH_CUDA

  return r;
}

size_t memory_usage(const platform::Place& p) { return Used(p); }

}  // namespace memory
}  // namespace paddle
