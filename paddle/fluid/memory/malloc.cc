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

#include <vector>

#include "paddle/fluid/memory/malloc.h"

#include "glog/logging.h"

#include "paddle/fluid/memory/detail/buddy_allocator.h"
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/gpu_info.h"

DEFINE_bool(init_allocated_mem, false,
            "It is a mistake that the values of the memory allocated by "
            "BuddyAllocator are always zeroed in some op's implementation. "
            "To find this error in time, we use init_allocated_mem to indicate "
            "that initializing the allocated memory with a small value "
            "during unit testing.");
DECLARE_double(fraction_of_gpu_memory_to_use);

namespace paddle {
namespace memory {

using BuddyAllocator = detail::BuddyAllocator;

BuddyAllocator* GetCPUBuddyAllocator() {
  // We tried thread_local for inference::RNN1 model, but that not works much
  // for multi-thread test.
  static std::once_flag init_flag;
  static detail::BuddyAllocator* a = nullptr;

  std::call_once(init_flag, []() {
    a = new detail::BuddyAllocator(
        std::unique_ptr<detail::SystemAllocator>(new detail::CPUAllocator),
        platform::CpuMinChunkSize(), platform::CpuMaxChunkSize());
  });

  return a;
}

// We compared the NaiveAllocator with BuddyAllocator in CPU memory allocation,
// seems they are almost the same overhead.
struct NaiveAllocator {
  void* Alloc(size_t size) { return malloc(size); }

  void Free(void* p) {
    PADDLE_ENFORCE(p);
    free(p);
  }

  static NaiveAllocator* Instance() {
    static NaiveAllocator x;
    return &x;
  }

 private:
  std::mutex lock_;
};

template <>
void* Alloc<platform::CPUPlace>(platform::CPUPlace place, size_t size) {
  VLOG(10) << "Allocate " << size << " bytes on " << platform::Place(place);
  void* p = GetCPUBuddyAllocator()->Alloc(size);
  if (FLAGS_init_allocated_mem) {
    memset(p, 0xEF, size);
  }
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
  static std::once_flag init_flag;
  static detail::BuddyAllocator** a_arr = nullptr;

  std::call_once(init_flag, [gpu_id]() {
    int gpu_num = platform::GetCUDADeviceCount();
    PADDLE_ENFORCE(gpu_id < gpu_num, "gpu_id:%d should < gpu_num:%d", gpu_id,
                   gpu_num);

    a_arr = new BuddyAllocator*[gpu_num];
    for (int i = 0; i < gpu_num; i++) {
      a_arr[i] = nullptr;
      platform::SetDeviceId(i);
      a_arr[i] = new BuddyAllocator(
          std::unique_ptr<detail::SystemAllocator>(new detail::GPUAllocator(i)),
          platform::GpuMinChunkSize(), platform::GpuMaxChunkSize());

      VLOG(10) << "\n\nNOTE: each GPU device use "
               << FLAGS_fraction_of_gpu_memory_to_use * 100
               << "% of GPU memory.\n"
               << "You can set GFlags environment variable '"
               << "FLAGS_fraction_of_gpu_memory_to_use"
               << "' to change the fraction of GPU usage.\n\n";
    }
  });

  platform::SetDeviceId(gpu_id);
  return a_arr[gpu_id];
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
    LOG(WARNING) << "GpuMinChunkSize " << buddy_allocator->GetMinChunkSize();
    LOG(WARNING) << "GpuMaxChunkSize " << buddy_allocator->GetMaxChunkSize();
    LOG(WARNING) << "GPU memory used: " << Used<platform::CUDAPlace>(place);
    platform::SetDeviceId(cur_dev);
  }
  if (FLAGS_init_allocated_mem) {
    cudaMemset(ptr, 0xEF, size);
  }
  return ptr;
}

template <>
void Free<platform::CUDAPlace>(platform::CUDAPlace place, void* p) {
  GetGPUBuddyAllocator(place.device)->Free(p);
}

BuddyAllocator* GetCUDAPinnedBuddyAllocator() {
  static std::once_flag init_flag;
  static BuddyAllocator* ba = nullptr;

  std::call_once(init_flag, []() {
    ba = new BuddyAllocator(std::unique_ptr<detail::SystemAllocator>(
                                new detail::CUDAPinnedAllocator),
                            platform::CUDAPinnedMinChunkSize(),
                            platform::CUDAPinnedMaxChunkSize());
  });

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
  if (FLAGS_init_allocated_mem) {
    memset(ptr, 0xEF, size);
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
