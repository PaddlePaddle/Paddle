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

template <>
void* Alloc<platform::CPUPlace>(platform::CPUPlace place, size_t size) {
  VLOG(10) << "Allocate " << size << " bytes on " << platform::Place(place);
  void* p = GetCPUBuddyAllocator()->Alloc(size);
  VLOG(10) << "  pointer=" << p;
  return p;
}

template <>
void Free<platform::CPUPlace>(platform::CPUPlace place, void* p) {
  VLOG(10) << "Free pointer=" << p << " on " << platform::Place(place);
  GetCPUBuddyAllocator()->Free(p);
}

template <>
size_t Used<platform::CPUPlace>(platform::CPUPlace place) {
  return GetCPUBuddyAllocator()->Used();
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

template <>
size_t Used<platform::CUDAPlace>(platform::CUDAPlace place) {
  return GetGPUBuddyAllocator(place.device)->Used();
}

template <>
void* Alloc<platform::CUDAPlace>(platform::CUDAPlace place, size_t size) {
  auto* buddy_allocator = GetGPUBuddyAllocator(place.device);
  auto* ptr = buddy_allocator->Alloc(size);
  if (ptr == nullptr) {
    int cur_dev = platform::GetCurrentDeviceId();
    platform::SetDeviceId(place.device);
    size_t avail, total;
    platform::GpuMemoryUsage(&avail, &total);
    LOG(WARNING) << "Cannot allocate " << size << " bytes in GPU "
                 << place.device << ", available " << avail << " bytes";
    LOG(WARNING) << "total " << total;
    LOG(WARNING) << "GpuMinChunkSize " << platform::GpuMinChunkSize();
    LOG(WARNING) << "GpuMaxChunkSize " << platform::GpuMaxChunkSize();
    LOG(WARNING) << "GPU memory used: " << Used<platform::CUDAPlace>(place);
    platform::SetDeviceId(cur_dev);
  }
  return ptr;
}

template <>
void Free<platform::CUDAPlace>(platform::CUDAPlace place, void* p) {
  GetGPUBuddyAllocator(place.device)->Free(p);
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

template <>
size_t Used<platform::CUDAPinnedPlace>(platform::CUDAPinnedPlace place) {
  return GetCUDAPinnedBuddyAllocator()->Used();
}

template <>
void* Alloc<platform::CUDAPinnedPlace>(platform::CUDAPinnedPlace place,
                                       size_t size) {
  auto* buddy_allocator = GetCUDAPinnedBuddyAllocator();
  void* ptr = buddy_allocator->Alloc(size);

  if (ptr == nullptr) {
    LOG(WARNING) << "cudaMallocHost Cannot allocate " << size
                 << " bytes in CUDAPinnedPlace";
  }
  return ptr;
}

template <>
void Free<platform::CUDAPinnedPlace>(platform::CUDAPinnedPlace place, void* p) {
  GetCUDAPinnedBuddyAllocator()->Free(p);
}
#endif

size_t Usage::operator()(const platform::CPUPlace& cpu) const {
  return Used(cpu);
}

size_t Usage::operator()(const platform::CUDAPlace& gpu) const {
#ifdef PADDLE_WITH_CUDA
  return Used(gpu);
#else
  PADDLE_THROW("'CUDAPlace' is not supported in CPU only device.");
#endif
}

size_t Usage::operator()(const platform::CUDAPinnedPlace& cuda_pinned) const {
#ifdef PADDLE_WITH_CUDA
  return Used(cuda_pinned);
#else
  PADDLE_THROW("'CUDAPinnedPlace' is not supported in CPU only device.");
#endif
}

size_t memory_usage(const platform::Place& p) {
  return boost::apply_visitor(Usage(), p);
}

}  // namespace memory
}  // namespace paddle
