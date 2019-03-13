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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef PADDLE_WITH_JEMALLOC
#include <jemalloc/jemalloc.h>
#endif

#include "glog/logging.h"
#include "paddle/fluid/memory/allocation/legacy_allocator.h"
#include "paddle/fluid/memory/detail/buddy_allocator.h"
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/split.h"

DEFINE_bool(init_allocated_mem, false,
            "It is a mistake that the values of the memory allocated by "
            "BuddyAllocator are always zeroed in some op's implementation. "
            "To find this error in time, we use init_allocated_mem to indicate "
            "that initializing the allocated memory with a small value "
            "during unit testing.");
DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_double(initial_gpu_memory_in_mb);
DECLARE_double(reallocate_gpu_memory_in_mb);
DECLARE_bool(benchmark);

namespace paddle {
namespace memory {
namespace legacy {
template <typename Place>
void *Alloc(const Place &place, size_t size);

template <typename Place>
void Free(const Place &place, void *p, size_t size);

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
        platform::CpuMinChunkSize(), platform::CpuMaxChunkSize(),
        platform::CpuMaxChunkSize());
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
  VLOG(10) << "Allocate " << size << " bytes on " << platform::Place(place);
#ifdef PADDLE_WITH_JEMALLOC
  void *p = malloc(size);
#else
  void *p = GetCPUBuddyAllocator()->Alloc(size);
#endif
  if (FLAGS_init_allocated_mem) {
    memset(p, 0xEF, size);
  }
  VLOG(10) << "  pointer=" << p;
  return p;
}

template <>
void Free<platform::CPUPlace>(const platform::CPUPlace &place, void *p,
                              size_t size) {
  VLOG(10) << "Free pointer=" << p << " on " << platform::Place(place);
#ifdef PADDLE_WITH_JEMALLOC
  free(p);
#else
  GetCPUBuddyAllocator()->Free(p);
#endif
}

template <>
size_t Used<platform::CPUPlace>(const platform::CPUPlace &place) {
#ifdef PADDLE_WITH_JEMALLOC
  // fake the result of used memory when PADDLE_WITH_JEMALLOC is ON
  return 0U;
#else
  return GetCPUBuddyAllocator()->Used();
#endif
}

#ifdef PADDLE_WITH_CUDA
class GPUBuddyAllocatorList {
 public:
  GPUBuddyAllocatorList()
      : allocators_(platform::GetCUDADeviceCount()),
        flags_(platform::GetCUDADeviceCount()) {
    allocation::GPUMemMonitor.Initialize(allocators_.size());
  }

  BuddyAllocator *Get(size_t dev_id) {
    PADDLE_ENFORCE(dev_id < flags_.size(), "Invalid device id %s", dev_id);
    std::call_once(flags_[dev_id], [this, dev_id] {
      platform::SetDeviceId(dev_id);
      size_t first_size = platform::GpuFirstAllocateChunkSize();
      size_t re_size = platform::GpuReAllocateChunkSize();
      allocators_[dev_id] =
          new BuddyAllocator(std::unique_ptr<detail::SystemAllocator>(
                                 new detail::GPUAllocator(dev_id)),
                             platform::GpuMinChunkSize(), first_size, re_size);
      VLOG(2) << "\n\nNOTE: each GPU device use "
              << string::HumanReadableSize(first_size) << "(initial chunk) "
              << string::HumanReadableSize(re_size) << "(reallocate chunk) "
              << "% of GPU memory.\n"
              << "You can set GFlags environment variable '"
              << "FLAGS_fraction_of_gpu_memory_to_use"
              << "' or "
                 "'FLAGS_initial_gpu_memory_in_mb/"
                 "FLAGS_reallocate_gpu_memory_in_mb' to change the fraction "
                 "of GPU usage.\n\n";
      VLOG(2) << "Currently, FLAGS_fraction_of_gpu_memory_to_use="
              << FLAGS_fraction_of_gpu_memory_to_use << ", "
              << "FLAGS_initial_gpu_memory_in_mb="
              << FLAGS_initial_gpu_memory_in_mb << ", "
              << "FLAGS_reallocate_gpu_memory_in_mb="
              << FLAGS_reallocate_gpu_memory_in_mb;
    });
    return allocators_[dev_id];
  }

 private:
  std::vector<BuddyAllocator *> allocators_;
  std::vector<std::once_flag> flags_;
};

BuddyAllocator *GetGPUBuddyAllocator(int gpu_id) {
  static GPUBuddyAllocatorList allocators;
  platform::SetDeviceId(gpu_id);
  return allocators.Get(gpu_id);
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
  if (ptr == nullptr && size > 0) {
    int cur_dev = platform::GetCurrentDeviceId();
    platform::SetDeviceId(place.device);
    size_t avail, total;
    platform::GpuMemoryUsage(&avail, &total);
    LOG(FATAL) << "Cannot allocate " << string::HumanReadableSize(size)
               << " in GPU " << place.device << ", available "
               << string::HumanReadableSize(avail) << "total " << total
               << "GpuMinChunkSize "
               << string::HumanReadableSize(buddy_allocator->GetMinChunkSize())
               << "GpuMaxChunkSize "
               << string::HumanReadableSize(buddy_allocator->GetMaxChunkSize())
               << "GPU memory used: "
               << string::HumanReadableSize(Used<platform::CUDAPlace>(place));
    platform::SetDeviceId(cur_dev);
  } else {
    if (FLAGS_benchmark) {
      allocation::GPUMemMonitor.Add(place.device, size);
    }
    if (FLAGS_init_allocated_mem) {
      cudaMemset(ptr, 0xEF, size);
    }
  }
  return ptr;
#else
  PADDLE_THROW("'CUDAPlace' is not supported in CPU only device.");
#endif
}

template <>
void Free<platform::CUDAPlace>(const platform::CUDAPlace &place, void *p,
                               size_t size) {
#ifdef PADDLE_WITH_CUDA
  GetGPUBuddyAllocator(place.device)->Free(p);
  if (FLAGS_benchmark) {
    allocation::GPUMemMonitor.Minus(place.device, size);
  }
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
                            platform::CUDAPinnedMaxChunkSize(),
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
    LOG(WARNING) << "cudaHostAlloc Cannot allocate " << size
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
                                     void *p, size_t size) {
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
  inline explicit FreeVisitor(void *ptr, size_t size)
      : ptr_(ptr), size_(size) {}

  template <typename Place>
  inline void operator()(const Place &place) const {
    Free<Place>(place, ptr_, size_);
  }

 private:
  void *ptr_;
  size_t size_;
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
LegacyMemMonitor GPUMemMonitor;

Allocation *LegacyAllocator::AllocateImpl(size_t size, Allocator::Attr attr) {
  void *ptr = boost::apply_visitor(legacy::AllocVisitor(size), place_);
  auto *tmp_alloc = new Allocation(ptr, size, place_);
  platform::MemEvenRecorder::Instance().PushMemRecord(
      static_cast<void *>(tmp_alloc), place_, size);
  return tmp_alloc;
}

void LegacyAllocator::FreeImpl(Allocation *allocation) {
  boost::apply_visitor(
      legacy::FreeVisitor(allocation->ptr(), allocation->size()),
      allocation->place());
  platform::MemEvenRecorder::Instance().PopMemRecord(
      static_cast<void *>(allocation), place_);
  delete allocation;
}

bool MemInfo::Add(const size_t &size) {
  std::lock_guard<std::mutex> lock(mutex_);
  usage_ += size;
  bool peak_point = usage_ > peak_usage_;
  if (peak_point) peak_usage_ = usage_;
  return peak_point;
}

void MemInfo::Minus(const size_t &size) {
  std::lock_guard<std::mutex> lock(mutex_);
  usage_ -= size;
}

uint64_t MemInfo::GetPeakUsage() const { return peak_usage_; }

LegacyMemMonitor::~LegacyMemMonitor() {
  for (auto &item : gpu_mem_info_) delete item.second;
}

void LegacyMemMonitor::Initialize(const int &device_num) {
  for (auto i = 0; i < device_num; ++i) {
    gpu_mem_info_[i] = new MemInfo();
  }
}

void LegacyMemMonitor::Add(const int &device, const size_t &size) {
  if (gpu_mem_info_[device]->Add(size)) {
    VLOG(3) << "#LegacyMemMonitor# device: " << device
            << " peak memory usage : "
            << (gpu_mem_info_[device]->GetPeakUsage() >> 20) << " MiB";
  }
}

void LegacyMemMonitor::Minus(const int &device, const size_t &size) {
  gpu_mem_info_[device]->Minus(size);
}

uint64_t LegacyMemMonitor::GetMemUsage(const int &device) const {
  return gpu_mem_info_.find(device) == gpu_mem_info_.end()
             ? 0
             : gpu_mem_info_.at(device)->GetPeakUsage();
}

void LegacyMemMonitor::PrintMemUsage() {
  std::vector<int> devices;
  for (const auto &item : gpu_mem_info_) {
    devices.emplace_back(item.first);
  }
  std::sort(devices.begin(), devices.end());
  for (const auto &device : devices) {
    std::cout << "Device : " << device << " Peak Memory Usage : "
              << (gpu_mem_info_[device]->GetPeakUsage() >> 20) << " MiB"
              << std::endl;
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
