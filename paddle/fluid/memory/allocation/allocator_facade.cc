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
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "paddle/fluid/memory/allocation/auto_growth_best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/cpu_allocator.h"
#include "paddle/fluid/memory/allocation/locked_allocator.h"
#include "paddle/fluid/memory/allocation/naive_best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/retry_allocator.h"
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
    gpu_allocator_retry_time, 10000,
    "The retry time (milliseconds) when allocator fails "
    "to allocate memory. No retry if this value is not greater than 0");

DEFINE_bool(use_system_allocator, false,
            "Whether to use system allocator to allocate CPU and GPU memory. "
            "Only used for unittests.");

namespace paddle {
namespace memory {
namespace allocation {

class AllocatorFacadePrivate {
 public:
  using AllocatorMap = std::map<platform::Place, std::shared_ptr<Allocator>>;

  AllocatorFacadePrivate() {
    auto strategy = GetAllocatorStrategy();
    switch (strategy) {
      case AllocatorStrategy::kNaiveBestFit: {
        InitNaiveBestFitCPUAllocator();
#ifdef PADDLE_WITH_CUDA
        for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount();
             ++dev_id) {
          InitNaiveBestFitCUDAAllocator(platform::CUDAPlace(dev_id));
        }
        InitNaiveBestFitCUDAPinnedAllocator();
#endif
        break;
      }

      case AllocatorStrategy::kAutoGrowth: {
        InitNaiveBestFitCPUAllocator();
#ifdef PADDLE_WITH_CUDA
        for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount();
             ++dev_id) {
          InitAutoGrowthCUDAAllocator(platform::CUDAPlace(dev_id));
        }
        InitNaiveBestFitCUDAPinnedAllocator();
#endif
        break;
      }

      default: {
        PADDLE_THROW("Unsupported allocator strategy: %d",
                     static_cast<int>(strategy));
      }
    }
    InitZeroSizeAllocators();
    InitSystemAllocators();

    if (FLAGS_gpu_allocator_retry_time > 0) {
      WrapCUDARetryAllocator(FLAGS_gpu_allocator_retry_time);
    }

    CheckAllocThreadSafe();
  }

  inline const std::shared_ptr<Allocator>& GetAllocator(
      const platform::Place& place, size_t size) {
    const auto& allocators =
        (size > 0 ? (UNLIKELY(FLAGS_use_system_allocator) ? system_allocators_
                                                          : allocators_)
                  : zero_size_allocators_);
    auto iter = allocators.find(place);
    PADDLE_ENFORCE(iter != allocators.end(),
                   "No such allocator for the place, %s", place);
    return iter->second;
  }

 private:
  void InitSystemAllocators() {
    system_allocators_[platform::CPUPlace()] = std::make_shared<CPUAllocator>();
#ifdef PADDLE_WITH_CUDA
    system_allocators_[platform::CUDAPinnedPlace()] =
        std::make_shared<CPUPinnedAllocator>();
    int device_count = platform::GetCUDADeviceCount();
    for (int i = 0; i < device_count; ++i) {
      platform::CUDAPlace p(i);
      system_allocators_[p] = std::make_shared<CUDAAllocator>(p);
    }
#endif
  }

  void InitNaiveBestFitCPUAllocator() {
    allocators_[platform::CPUPlace()] =
        std::make_shared<NaiveBestFitAllocator>(platform::CPUPlace());
  }

#ifdef PADDLE_WITH_CUDA
  void InitNaiveBestFitCUDAPinnedAllocator() {
    allocators_[platform::CUDAPinnedPlace()] =
        std::make_shared<NaiveBestFitAllocator>(platform::CUDAPinnedPlace());
  }

  void InitNaiveBestFitCUDAAllocator(platform::CUDAPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }

  void InitAutoGrowthCUDAAllocator(platform::CUDAPlace p) {
    auto cuda_allocator = std::make_shared<CUDAAllocator>(p);
    allocators_[p] = std::make_shared<AutoGrowthBestFitAllocator>(
        cuda_allocator, platform::GpuMinChunkSize());
  }
#endif

  class ZeroSizeAllocator : public Allocator {
   public:
    explicit ZeroSizeAllocator(platform::Place place) : place_(place) {}

    bool IsAllocThreadSafe() const override { return true; }

   protected:
    Allocation* AllocateImpl(size_t size) override {
      return new Allocation(nullptr, 0, place_);
    }

    void FreeImpl(Allocation* allocation) override { delete allocation; }

   private:
    platform::Place place_;
  };

  void InitZeroSizeAllocators() {
    std::vector<platform::Place> places;
    places.emplace_back(platform::CPUPlace());
#ifdef PADDLE_WITH_CUDA
    int device_count = platform::GetCUDADeviceCount();
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      places.emplace_back(platform::CUDAPlace(dev_id));
    }
    places.emplace_back(platform::CUDAPinnedPlace());
#endif

    for (auto& p : places) {
      zero_size_allocators_[p] = std::make_shared<ZeroSizeAllocator>(p);
    }
  }

  static void CheckAllocThreadSafe(const AllocatorMap& allocators) {
    for (auto& pair : allocators) {
      PADDLE_ENFORCE_EQ(pair.second->IsAllocThreadSafe(), true,
                        platform::errors::InvalidArgument(
                            "Public allocators must be thread safe"));
    }
  }

  void CheckAllocThreadSafe() const {
    CheckAllocThreadSafe(allocators_);
    CheckAllocThreadSafe(zero_size_allocators_);
    CheckAllocThreadSafe(system_allocators_);
  }

  void WrapCUDARetryAllocator(size_t retry_time) {
    PADDLE_ENFORCE_GT(retry_time, 0, "Retry time must be larger than 0");
    for (auto& pair : allocators_) {
      if (platform::is_gpu_place(pair.first)) {
        pair.second = std::make_shared<RetryAllocator>(pair.second, retry_time);
      }
    }
  }

 private:
  AllocatorMap allocators_;
  AllocatorMap zero_size_allocators_;
  AllocatorMap system_allocators_;
};

// Pimpl. Make interface clean.
AllocatorFacade::AllocatorFacade() : m_(new AllocatorFacadePrivate()) {}
// delete m_ may cause core dump when the destructor of python in conflict with
// cpp.
AllocatorFacade::~AllocatorFacade() {}

AllocatorFacade& AllocatorFacade::Instance() {
  static AllocatorFacade instance;
  return instance;
}

std::shared_ptr<Allocation> AllocatorFacade::AllocShared(
    const platform::Place& place, size_t size) {
  return std::shared_ptr<Allocation>(Alloc(place, size));
}

AllocationPtr AllocatorFacade::Alloc(const platform::Place& place,
                                     size_t size) {
  return m_->GetAllocator(place, size)->Allocate(size);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
