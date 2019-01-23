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

#include "paddle/fluid/memory/malloc.h"
#include <gflags/gflags.h>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/memory/allocation/aligned_allocator.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "paddle/fluid/memory/allocation/auto_increment_allocator.h"
#include "paddle/fluid/memory/allocation/best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/conditional_allocator.h"
#include "paddle/fluid/memory/allocation/cpu_allocator.h"
#include "paddle/fluid/memory/allocation/legacy_allocator.h"
#include "paddle/fluid/memory/allocation/locked_allocator.h"
#include "paddle/fluid/memory/allocation/retry_allocator.h"
#include "paddle/fluid/memory/allocation/zero_size_allocator.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/place.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/memory/allocation/cuda_allocator.h"
#include "paddle/fluid/memory/allocation/pinned_allocator.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/gpu_info.h"
#endif

DEFINE_int64(
    gpu_allocator_retry_time, 0,
    "The retry time (milliseconds) when allocator fails "
    "to allocate memory. No retry if this value is not greater than 0");

namespace paddle {
namespace memory {
namespace allocation {

// TODO(yy): Dirty code here. This class should be configurable in runtime.
class CPUManagedAllocator : public Allocator {
 public:
  CPUManagedAllocator() : normal_allocator_(new CPUAllocator()) {}

  bool IsAllocThreadSafe() const override { return true; }

 protected:
  Allocation* AllocateImpl(size_t size, Allocator::Attr attr) override {
    return normal_allocator_->Allocate(size, attr).release();
  }

 private:
  std::shared_ptr<Allocator> normal_allocator_;
};

// TODO(yy): Dirty code here. This class should be configurable in runtime.
class ChunkedAllocator : public Allocator {
 public:
  explicit ChunkedAllocator(std::unique_ptr<Allocator> system_allocator,
                            size_t max_chunk_size, size_t capacity = 1,
                            int64_t retry_time = -1)
      : max_chunk_size_(max_chunk_size), retry_time_(retry_time) {
    raw_allocator_ = std::move(system_allocator);

    if (max_chunk_size_ == 0) {
      default_allocator_ = raw_allocator_;
    } else {
      if (capacity == 1) {
        VLOG(1) << "Create BestFitAllocator with chunk_size "
                << max_chunk_size_;
        default_allocator_ = CreateAllocatorWithChunk();
      } else {
        VLOG(1) << "Create AutoIncrementAllocator with chunk_size "
                << max_chunk_size_ << " and capacity " << capacity;
        default_allocator_ = std::make_shared<AutoIncrementAllocator>(
            [this] { return std::move(CreateAllocatorWithChunk()); }, capacity);
      }
    }

    auto* cond_allocator = new ConditionalAllocator();
    cond_allocator
        ->AddAllocator(
            [this](size_t size, Attr attr) { return size < max_chunk_size_; },
            default_allocator_)
        .AddAllocator(
            [](size_t size, Attr attr) {
              return true;  // default case
            },
            raw_allocator_);
    default_allocator_.reset(cond_allocator);
  }

  ~ChunkedAllocator() override {
    // Specify destruct order.
    default_allocator_.reset();
    chunks_.clear();
    raw_allocator_.reset();
  }

  std::shared_ptr<Allocator> CreateAllocatorWithChunk() {
    chunks_.emplace_back(raw_allocator_->Allocate(max_chunk_size_));
    auto* allocation = chunks_.back().get();
    std::unique_ptr<Allocator> allocator(new LockedAllocator(
        std::unique_ptr<Allocator>(new BestFitAllocator(allocation))));

    if (retry_time_ > 0) {
      auto* retry_allocator =
          new RetryAllocator(std::move(allocator), retry_time_);
      allocator.reset(retry_allocator);
    }

    return std::make_shared<AlignedAllocator<64u>>(std::move(allocator));
  }

  bool IsAllocThreadSafe() const override { return true; }

 protected:
  Allocation* AllocateImpl(size_t size, Allocator::Attr attr) override {
    return default_allocator_->Allocate(size, attr).release();
  }

 protected:
  size_t max_chunk_size_;
  int64_t retry_time_;
  std::vector<AllocationPtr> chunks_;
  std::shared_ptr<Allocator> raw_allocator_;
  std::shared_ptr<Allocator> default_allocator_;
};

#ifdef PADDLE_WITH_CUDA

class CUDAChunkedAllocator : public ChunkedAllocator {
 public:
  explicit CUDAChunkedAllocator(int dev_id)
      : ChunkedAllocator(std::unique_ptr<Allocator>(
                             new CUDAAllocator(platform::CUDAPlace(dev_id))),
                         GetMaxChunkSize(dev_id), GetCapcity(dev_id),
                         GetRetryTime()) {}

 private:
  static size_t GetMaxChunkSize(int dev_id) {
    platform::CUDADeviceGuard guard(dev_id);
    return platform::GpuMaxChunkSize();
  }

  static size_t GetCapcity(int dev_id) {
    platform::CUDADeviceGuard guard(dev_id);
    size_t available, total;
    platform::GpuMemoryUsage(&available, &total);
    size_t max_chunk_size = platform::GpuMaxChunkSize();
    return max_chunk_size == 0 ? 0 : available / max_chunk_size;
  }

  static int64_t GetRetryTime() { return FLAGS_gpu_allocator_retry_time; }
};

class CUDAPinnedChunkedAllocator : public ChunkedAllocator {
 public:
  CUDAPinnedChunkedAllocator()
      : ChunkedAllocator(std::unique_ptr<Allocator>(new CPUPinnedAllocator()),
                         platform::CUDAPinnedMaxChunkSize(), GetCapacity(),
                         -1) {}  // never retry

 private:
  static size_t GetCapacity() {
    size_t total = platform::CpuTotalPhysicalMemory();
    size_t max_chunk_size = platform::CUDAPinnedMaxChunkSize();
    return max_chunk_size == 0 ? 0 : total / max_chunk_size;
  }
};

#endif

template <typename Place, typename AllocatorType>
class AllocatorFacadeBase {
 private:
  AllocatorFacadeBase() : place_() {
    if (GetAllocatorStrategy() == AllocatorStrategy::kLegacy) {
      allocator_.reset(new LegacyAllocator(place_));
    } else {
      allocator_.reset(
          new ZeroSizeAllocator(std::make_shared<AllocatorType>(), place_));
    }
  }

 public:
  static AllocatorFacadeBase& Instance() {
    static AllocatorFacadeBase<Place, AllocatorType> instance;
    return instance;
  }

  AllocationPtr Alloc(size_t size, Allocator::Attr attr) {
    return allocator_->Allocate(size, attr);
  }

 private:
  std::unique_ptr<Allocator> allocator_;
  Place place_;
};

#ifdef PADDLE_WITH_CUDA
class CUDAAllocatorFacade {
 private:
  CUDAAllocatorFacade() {
    dev_cnt_ = platform::GetCUDADeviceCount();
    allocators_.reserve(dev_cnt_);
    for (int i = 0; i < dev_cnt_; ++i) {
      if (GetAllocatorStrategy() == AllocatorStrategy::kLegacy) {
        allocators_.emplace_back(new LegacyAllocator(platform::CUDAPlace(i)));
      } else {
        allocators_.emplace_back(new ZeroSizeAllocator(
            std::shared_ptr<Allocator>(new CUDAChunkedAllocator(i)),
            platform::CUDAPlace(i)));
      }
    }
  }

 public:
  static CUDAAllocatorFacade& Instance() {
    static CUDAAllocatorFacade instance;
    return instance;
  }

  AllocationPtr Alloc(platform::CUDAPlace place, size_t size,
                      Allocator::Attr attr) {
    PADDLE_ENFORCE(place.device >= 0 && place.device < dev_cnt_,
                   "Invalid CUDAPlace %d, total GPU number %d", place.device,
                   dev_cnt_);
    return allocators_[place.device]->Allocate(size, attr);
  }

 private:
  std::vector<std::unique_ptr<Allocator>> allocators_;
  int dev_cnt_;
};
#endif

struct AllocatorVisitor : public boost::static_visitor<Allocation*> {
  inline AllocatorVisitor(size_t size, Allocator::Attr attr)
      : size_(size), attr_(attr) {}

  inline Allocation* operator()(platform::CPUPlace place) const {
    return AllocatorFacadeBase<platform::CPUPlace,
                               CPUManagedAllocator>::Instance()
        .Alloc(size_, attr_)
        .release();
  }

  inline Allocation* operator()(platform::CUDAPlace place) const {
#ifdef PADDLE_WITH_CUDA
    return CUDAAllocatorFacade::Instance().Alloc(place, size_, attr_).release();
#else
    PADDLE_THROW("Unsupported CUDAPlace when use CPU only version.");
#endif
  }

  inline Allocation* operator()(platform::CUDAPinnedPlace place) const {
#ifdef PADDLE_WITH_CUDA
    return AllocatorFacadeBase<platform::CUDAPinnedPlace,
                               CUDAPinnedChunkedAllocator>::Instance()
        .Alloc(size_, attr_)
        .release();
#else
    PADDLE_THROW("Unsupported CUDAPinnedPlace when use CPU only version.");
#endif
  }

 private:
  size_t size_;
  Allocator::Attr attr_;
};

}  // namespace allocation

allocation::AllocationPtr Alloc(const platform::Place& place, size_t size,
                                allocation::Allocator::Attr attr) {
  allocation::AllocatorVisitor visitor(size, attr);
  return AllocationPtr(boost::apply_visitor(visitor, place));
}

std::shared_ptr<allocation::Allocation> AllocShared(
    const platform::Place& place, size_t size,
    allocation::Allocator::Attr attr) {
  return std::shared_ptr<Allocation>(Alloc(place, size, attr));
}

}  // namespace memory
}  // namespace paddle
