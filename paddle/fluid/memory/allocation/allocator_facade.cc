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

#include "paddle/fluid/memory/allocation/allocator.h"
#include <gflags/gflags.h>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/memory/allocation/aligned_allocator.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
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
#include "paddle/fluid/platform/enforce.h"
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

static inline std::shared_ptr<Allocator> WrapRetryAllocator(
    std::shared_ptr<Allocator> allocator, int64_t retry_time) {
  if (retry_time > 0) {
    auto* retry_allocator =
        new RetryAllocator(std::move(allocator), retry_time);
    allocator.reset(retry_allocator);
  }

  return allocator;
}

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
            [this] { return CreateAllocatorWithChunk(); }, capacity);
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
    std::shared_ptr<Allocator> allocator(new LockedAllocator(
        std::shared_ptr<Allocator>(new BestFitAllocator(allocation))));

    allocator = WrapRetryAllocator(allocator, retry_time_);

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

class AllocatorFacadePrivate {
 public:
  std::map<platform::Place, std::shared_ptr<Allocator>> allocators_;

  ~AllocatorFacadePrivate() = default;

  AllocatorFacadePrivate() {
    auto strategy = GetAllocatorStrategy();
    switch (strategy) {
      case AllocatorStrategy::kLegacy: {
        InitLegacyAllocator();
        break;
      }
      case AllocatorStrategy::kNaiveBestFit: {
        InitCPUAllocator();
        InitCUDAAllocator();
        InitCUDAPinnedAllocator();
        WrapZeroSizeAllocator();
        break;
      }
      default: {
        PADDLE_THROW("Unsupported allocator strategy: %d",
                     static_cast<int>(strategy));
      }
    }
  }

 private:
  void InitLegacyAllocator() {
    std::vector<platform::Place> places{platform::CPUPlace()};
#ifdef PADDLE_WITH_CUDA
    for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount(); ++dev_id) {
      places.emplace_back(platform::CUDAPlace(dev_id));
    }
    places.emplace_back(platform::CUDAPinnedPlace());
#endif
    for (auto& p : places) {
      allocators_[p] = std::make_shared<LegacyAllocator>(p);
    }
  }

  void InitCPUAllocator() {
    allocators_[platform::CPUPlace()] = std::make_shared<CPUManagedAllocator>();
  }

  void InitCUDAAllocator() {
#ifdef PADDLE_WITH_CUDA
    int device_count = platform::GetCUDADeviceCount();
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      allocators_[platform::CUDAPlace(dev_id)] =
          std::make_shared<CUDAChunkedAllocator>(dev_id);
    }
#endif
  }

  void InitCUDAPinnedAllocator() {
#ifdef PADDLE_WITH_CUDA
    allocators_[platform::CUDAPinnedPlace()] =
        std::make_shared<CUDAPinnedChunkedAllocator>();
#endif
  }

  void WrapZeroSizeAllocator() {
    for (auto& pair : allocators_) {
      pair.second =
          std::make_shared<ZeroSizeAllocator>(pair.second, pair.first);
    }
  }
};

// Pimpl. Make interface clean.
AllocatorFacade::AllocatorFacade() : m_(new AllocatorFacadePrivate()) {}
AllocatorFacade::~AllocatorFacade() { delete m_; }

AllocatorFacade& AllocatorFacade::Instance() {
  static AllocatorFacade instance;
  return instance;
}

std::shared_ptr<Allocation> AllocatorFacade::AllocShared(
    const platform::Place& place, size_t size, Allocator::Attr attr) {
  return std::shared_ptr<Allocation>(Alloc(place, size, attr));
}

AllocationPtr AllocatorFacade::Alloc(const platform::Place& place, size_t size,
                                     Allocator::Attr attr) {
  auto it = m_->allocators_.find(place);
  if (it == m_->allocators_.end()) {
    throw BadAlloc(
        string::Sprintf("No such allocator for the place, %s", place));
  }
  return m_->allocators_.at(place)->Allocate(size, attr);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
