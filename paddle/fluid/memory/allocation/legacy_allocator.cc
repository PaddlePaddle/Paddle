// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/legacy_allocator.h"
#include <string>
#include "glog/logging.h"
#include "paddle/fluid/memory/detail/buddy_allocator.h"
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/string/printf.h"

DEFINE_bool(init_allocated_mem, false,
            "It is a mistake that the values of the memory allocated by "
            "BuddyAllocator are always zeroed in some op's implementation. "
            "To find this error in time, we use init_allocated_mem to indicate "
            "that initializing the allocated memory with a small value "
            "during unit testing.");
DECLARE_double(fraction_of_gpu_memory_to_use);

namespace paddle {
namespace memory {
namespace legacy {
template <typename Place>
void *Alloc(const Place &place, size_t size);

template <typename Place>
void Free(const Place &place, void *p);

template <typename Place>
size_t Used(const Place &place);

struct Usage : public boost::static_visitor<size_t> {
  size_t operator()(const platform::CPUPlace &cpu) const;
  size_t operator()(const platform::CUDAPlace &gpu) const;
  size_t operator()(const platform::CUDAPinnedPlace &cuda_pinned) const;
};

size_t memory_usage(const platform::Place &p);

using BuddyAllocator = detail::BuddyAllocator;

BuddyAllocator *GetCPUBuddyAllocator() {
  // We tried thread_local for inference::RNN1 model, but that not works much
  // for multi-thread test.
  static std::once_flag init_flag;
  static detail::BuddyAllocator *a = nullptr;

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
  void *Alloc(size_t size) { return malloc(size); }

  void Free(void *p) {
    PADDLE_ENFORCE(p);
    free(p);
  }

  static NaiveAllocator *Instance() {
    static NaiveAllocator x;
    return &x;
  }

 private:
  std::mutex lock_;
};

template <>
void *Alloc<platform::CPUPlace>(const platform::CPUPlace &place, size_t size) {
  VLOG(1) << "Allocate " << size << " bytes on " << platform::Place(place);
  void *p = GetCPUBuddyAllocator()->Alloc(size);
  if (FLAGS_init_allocated_mem) {
    memset(p, 0xEF, size);
  }
  VLOG(10) << "  pointer=" << p;
  return p;
}

template <>
void Free<platform::CPUPlace>(const platform::CPUPlace &place, void *p) {
  VLOG(1) << "Free pointer=" << p << " on " << platform::Place(place);
  GetCPUBuddyAllocator()->Free(p);
}

template <>
size_t Used<platform::CPUPlace>(const platform::CPUPlace &place) {
  return GetCPUBuddyAllocator()->Used();
}

#ifdef PADDLE_WITH_CUDA
BuddyAllocator *GetGPUBuddyAllocator(int gpu_id) {
  static std::once_flag init_flag;
  static detail::BuddyAllocator **a_arr = nullptr;

  std::call_once(init_flag, [gpu_id]() {
    int gpu_num = platform::GetCUDADeviceCount();
    PADDLE_ENFORCE(gpu_id < gpu_num, "gpu_id:%d should < gpu_num:%d", gpu_id,
                   gpu_num);

    a_arr = new BuddyAllocator *[gpu_num];
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
#endif

template <>
size_t Used<platform::CUDAPlace>(const platform::CUDAPlace &place) {
#ifdef PADDLE_WITH_CUDA
  return GetGPUBuddyAllocator(place.device)->Used();
#else
  PADDLE_THROW("'CUDAPlace' is not supported in CPU only device.");
#endif
}

template <>
void *Alloc<platform::CUDAPlace>(const platform::CUDAPlace &place,
                                 size_t size) {
#ifdef PADDLE_WITH_CUDA
  auto *buddy_allocator = GetGPUBuddyAllocator(place.device);
  auto *ptr = buddy_allocator->Alloc(size);
  if (ptr == nullptr) {
    int cur_dev = platform::GetCurrentDeviceId();
    platform::SetDeviceId(place.device);
    size_t avail, total;
    platform::GpuMemoryUsage(&avail, &total);
    LOG(WARNING) << "Cannot allocate " << string::HumanReadableSize(size)
                 << " in GPU " << place.device << ", available "
                 << string::HumanReadableSize(avail);
    LOG(WARNING) << "total " << total;
    LOG(WARNING) << "GpuMinChunkSize "
                 << string::HumanReadableSize(
                        buddy_allocator->GetMinChunkSize());
    LOG(WARNING) << "GpuMaxChunkSize "
                 << string::HumanReadableSize(
                        buddy_allocator->GetMaxChunkSize());
    LOG(WARNING) << "GPU memory used: "
                 << string::HumanReadableSize(Used<platform::CUDAPlace>(place));
    platform::SetDeviceId(cur_dev);
  }
  if (FLAGS_init_allocated_mem) {
    cudaMemset(ptr, 0xEF, size);
  }
  return ptr;
#else
  PADDLE_THROW("'CUDAPlace' is not supported in CPU only device.");
#endif
}

template <>
void Free<platform::CUDAPlace>(const platform::CUDAPlace &place, void *p) {
#ifdef PADDLE_WITH_CUDA
  GetGPUBuddyAllocator(place.device)->Free(p);
#else
  PADDLE_THROW("'CUDAPlace' is not supported in CPU only device.");
#endif
}

#ifdef PADDLE_WITH_CUDA
BuddyAllocator *GetCUDAPinnedBuddyAllocator() {
  static std::once_flag init_flag;
  static BuddyAllocator *ba = nullptr;

  std::call_once(init_flag, []() {
    ba = new BuddyAllocator(std::unique_ptr<detail::SystemAllocator>(
                                new detail::CUDAPinnedAllocator),
                            platform::CUDAPinnedMinChunkSize(),
                            platform::CUDAPinnedMaxChunkSize());
  });

  return ba;
}
#endif

template <>
size_t Used<platform::CUDAPinnedPlace>(const platform::CUDAPinnedPlace &place) {
#ifdef PADDLE_WITH_CUDA
  return GetCUDAPinnedBuddyAllocator()->Used();
#else
  PADDLE_THROW("'CUDAPinnedPlace' is not supported in CPU only device.");
#endif
}

template <>
void *Alloc<platform::CUDAPinnedPlace>(const platform::CUDAPinnedPlace &place,
                                       size_t size) {
#ifdef PADDLE_WITH_CUDA
  auto *buddy_allocator = GetCUDAPinnedBuddyAllocator();
  void *ptr = buddy_allocator->Alloc(size);

  if (ptr == nullptr) {
    LOG(WARNING) << "cudaMallocHost Cannot allocate " << size
                 << " bytes in CUDAPinnedPlace";
  }
  if (FLAGS_init_allocated_mem) {
    memset(ptr, 0xEF, size);
  }
  return ptr;
#else
  PADDLE_THROW("'CUDAPinnedPlace' is not supported in CPU only device.");
#endif
}

template <>
void Free<platform::CUDAPinnedPlace>(const platform::CUDAPinnedPlace &place,
                                     void *p) {
#ifdef PADDLE_WITH_CUDA
  GetCUDAPinnedBuddyAllocator()->Free(p);
#else
  PADDLE_THROW("'CUDAPinnedPlace' is not supported in CPU only device.");
#endif
}

struct AllocVisitor : public boost::static_visitor<void *> {
  inline explicit AllocVisitor(size_t size) : size_(size) {}

  template <typename Place>
  inline void *operator()(const Place &place) const {
    return Alloc<Place>(place, size_);
  }

 private:
  size_t size_;
};

struct FreeVisitor : public boost::static_visitor<void> {
  inline explicit FreeVisitor(void *ptr) : ptr_(ptr) {}

  template <typename Place>
  inline void operator()(const Place &place) const {
    Free<Place>(place, ptr_);
  }

 private:
  void *ptr_;
};

size_t Usage::operator()(const platform::CPUPlace &cpu) const {
  return Used(cpu);
}

size_t Usage::operator()(const platform::CUDAPlace &gpu) const {
#ifdef PADDLE_WITH_CUDA
  return Used(gpu);
#else
  PADDLE_THROW("'CUDAPlace' is not supported in CPU only device.");
#endif
}

size_t Usage::operator()(const platform::CUDAPinnedPlace &cuda_pinned) const {
#ifdef PADDLE_WITH_CUDA
  return Used(cuda_pinned);
#else
  PADDLE_THROW("'CUDAPinnedPlace' is not supported in CPU only device.");
#endif
}
}  // namespace legacy

namespace allocation {

Allocation *LegacyAllocator::AllocateImpl(size_t size, Allocator::Attr attr) {
  void *ptr = boost::apply_visitor(legacy::AllocVisitor(size), place_);
  return new Allocation(ptr, size, place_);
}

void LegacyAllocator::Free(Allocation *allocation) {
  boost::apply_visitor(legacy::FreeVisitor(allocation->ptr()),
                       allocation->place());
  delete allocation;
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
