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

#include "paddle/fluid/memory/allocation/allocator_facade.h"

#include "gflags/gflags.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "paddle/fluid/memory/allocation/auto_growth_best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/cpu_allocator.h"
#include "paddle/fluid/memory/allocation/naive_best_fit_allocator.h"
#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/memory/allocation/npu_pinned_allocator.h"
#endif
#include "paddle/fluid/memory/allocation/aligned_allocator.h"
#include "paddle/fluid/memory/allocation/retry_allocator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/memory/allocation/cuda_allocator.h"
#include "paddle/fluid/memory/allocation/pinned_allocator.h"
#include "paddle/fluid/memory/allocation/thread_local_allocator.h"
#include "paddle/fluid/platform/gpu_info.h"
#endif
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_graph.h"
#endif
#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/xpu/xpu_info.h"
#endif
#include "paddle/fluid/platform/npu_info.h"

PADDLE_DEFINE_EXPORTED_int64(
    gpu_allocator_retry_time, 10000,
    "The retry time (milliseconds) when allocator fails "
    "to allocate memory. No retry if this value is not greater than 0");

PADDLE_DEFINE_EXPORTED_bool(
    use_system_allocator, false,
    "Whether to use system allocator to allocate CPU and GPU memory. "
    "Only used for unittests.");

DECLARE_string(allocator_strategy);

namespace paddle {
namespace memory {
namespace allocation {

#ifdef PADDLE_WITH_CUDA
class CUDAGraphAllocator
    : public Allocator,
      public std::enable_shared_from_this<CUDAGraphAllocator> {
 private:
  class PrivateAllocation : public Allocation {
   public:
    PrivateAllocation(CUDAGraphAllocator* allocator,
                      AllocationPtr underlying_allocation)
        : Allocation(underlying_allocation->ptr(),
                     underlying_allocation->size(),
                     underlying_allocation->place()),
          allocator_(allocator->shared_from_this()),
          underlying_allocation_(std::move(underlying_allocation)) {}

   private:
    std::shared_ptr<Allocator> allocator_;
    AllocationPtr underlying_allocation_;
  };

  explicit CUDAGraphAllocator(const std::shared_ptr<Allocator>& allocator)
      : underlying_allocator_(allocator) {}

 public:
  static std::shared_ptr<Allocator> Create(
      const std::shared_ptr<Allocator>& allocator) {
    return std::shared_ptr<Allocator>(new CUDAGraphAllocator(allocator));
  }

 protected:
  Allocation* AllocateImpl(size_t size) {
    VLOG(10) << "Allocate " << size << " for CUDA Graph";
    return new PrivateAllocation(this, underlying_allocator_->Allocate(size));
  }

  void FreeImpl(Allocation* allocation) {
    VLOG(10) << "delete for CUDA Graph";
    delete allocation;
  }

 private:
  std::shared_ptr<Allocator> underlying_allocator_;
};
#endif

class AllocatorFacadePrivate {
 public:
  using AllocatorMap = std::map<platform::Place, std::shared_ptr<Allocator>>;

  explicit AllocatorFacadePrivate(bool allow_free_idle_chunk = true) {
    strategy_ = GetAllocatorStrategy();
    switch (strategy_) {
      case AllocatorStrategy::kNaiveBestFit: {
        InitNaiveBestFitCPUAllocator();
#ifdef PADDLE_WITH_XPU
        for (int dev_id = 0; dev_id < platform::GetXPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitXPUAllocator(platform::XPUPlace(dev_id));
        }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount();
             ++dev_id) {
          InitNaiveBestFitCUDAAllocator(platform::CUDAPlace(dev_id));
        }
        InitNaiveBestFitCUDAPinnedAllocator();
#endif
#ifdef PADDLE_WITH_ASCEND_CL
        for (int dev_id = 0; dev_id < platform::GetNPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitNPUAllocator(platform::NPUPlace(dev_id));
        }
        InitNaiveBestFitNPUPinnedAllocator();
#endif
        break;
      }

      case AllocatorStrategy::kAutoGrowth: {
        InitNaiveBestFitCPUAllocator();
#ifdef PADDLE_WITH_XPU
        for (int dev_id = 0; dev_id < platform::GetXPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitXPUAllocator(platform::XPUPlace(dev_id));
        }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount();
             ++dev_id) {
          InitAutoGrowthCUDAAllocator(platform::CUDAPlace(dev_id),
                                      allow_free_idle_chunk);
        }
        InitNaiveBestFitCUDAPinnedAllocator();
#endif
        break;
      }

      case AllocatorStrategy::kThreadLocal: {
        InitNaiveBestFitCPUAllocator();
#ifdef PADDLE_WITH_XPU
        for (int dev_id = 0; dev_id < platform::GetXPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitXPUAllocator(platform::XPUPlace(dev_id));
        }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount();
             ++dev_id) {
          InitThreadLocalCUDAAllocator(platform::CUDAPlace(dev_id));
        }
        InitNaiveBestFitCUDAPinnedAllocator();
#endif
        break;
      }

      default: {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unsupported allocator strategy: %d", static_cast<int>(strategy_)));
      }
    }
    InitZeroSizeAllocators();
    InitSystemAllocators();

    if (FLAGS_gpu_allocator_retry_time > 0) {
      WrapCUDARetryAllocator(FLAGS_gpu_allocator_retry_time);
    }

    CheckAllocThreadSafe();
  }

  inline const AllocatorMap& GetAllocatorMap() {
#ifdef PADDLE_WITH_CUDA
    if (UNLIKELY(platform::CUDAGraph::IsCapturing())) {
      auto id = platform::CUDAGraph::CapturingID();
      auto iter = cuda_graph_allocator_map_.find(id);
      PADDLE_ENFORCE_NE(
          iter, cuda_graph_allocator_map_.end(),
          platform::errors::PermissionDenied(
              "No memory pool is prepared for CUDA Graph capturing."));
      return iter->second->allocators_;
    } else {
      return allocators_;
    }
#else
    return allocators_;
#endif
  }

  inline const std::shared_ptr<Allocator>& GetAllocator(
      const platform::Place& place, size_t size) {
    VLOG(4) << "GetAllocator"
            << " " << place << " " << size;
    const auto& allocators =
        (size > 0 ? (UNLIKELY(FLAGS_use_system_allocator) ? system_allocators_
                                                          : GetAllocatorMap())
                  : zero_size_allocators_);
    auto iter = allocators.find(place);
    PADDLE_ENFORCE_NE(iter, allocators.end(),
                      platform::errors::NotFound(
                          "No allocator found for the place, %s", place));
    return iter->second;
  }

 private:
  void InitSystemAllocators() {
    if (!system_allocators_.empty()) return;
    system_allocators_[platform::CPUPlace()] = std::make_shared<CPUAllocator>();
#ifdef PADDLE_WITH_XPU
    int device_count = platform::GetXPUDeviceCount();
    for (int i = 0; i < device_count; ++i) {
      platform::XPUPlace p(i);
      system_allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
    }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void InitNaiveBestFitCUDAPinnedAllocator() {
    allocators_[platform::CUDAPinnedPlace()] =
        std::make_shared<NaiveBestFitAllocator>(platform::CUDAPinnedPlace());
  }

  void InitNaiveBestFitCUDAAllocator(platform::CUDAPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }

  void InitThreadLocalCUDAAllocator(platform::CUDAPlace p) {
    allocators_[p] = std::make_shared<ThreadLocalCUDAAllocator>(p);
  }

  void InitAutoGrowthCUDAAllocator(platform::CUDAPlace p,
                                   bool allow_free_idle_chunk) {
    auto cuda_allocator = std::make_shared<CUDAAllocator>(p);
    auto alignment = platform::GpuMinChunkSize();
    bool need_addr_align = true;
    // NOTE: sometimes, since cuda runtime can not be forked, calling any cuda
    // API in that case may got cuda error(3), i.e.,
    // cudaErrorInitializationError. And, the CUDAAllocator is only initialized
    // but not really used.
    // Here, the try-catch block is added to handle the case that
    // GetDeviceProperties() may failed in the multiple process(for example, in
    // dataloader with num_worker > 0)
    try {
      const auto& prop = platform::GetDeviceProperties(p.GetDeviceId());
      need_addr_align = prop.textureAlignment < alignment;
      VLOG(4) << "GetDeviceProperties ok, textureAlignment: "
              << prop.textureAlignment
              << ", set need_addr_align=" << need_addr_align;
    } catch (...) {
      need_addr_align = true;
      VLOG(4) << "GetDeviceProperties failed, set need_addr_align=true";
    }
    // The address returned is aligned already,
    // ref:
    // https://stackoverflow.com/questions/14082964/cuda-alignment-256bytes-seriously/14083295#14083295
    std::shared_ptr<Allocator> underlying_allocator{nullptr};
    if (need_addr_align) {
      VLOG(10) << "use AlignedAllocator with alignment: " << alignment;
      underlying_allocator =
          std::make_shared<AlignedAllocator>(underlying_allocator, alignment);
    } else {
      VLOG(10) << "not use AlignedAllocator with alignment: " << alignment;
      underlying_allocator = cuda_allocator;
    }
    allocators_[p] = std::make_shared<AutoGrowthBestFitAllocator>(
        underlying_allocator, alignment, 0, allow_free_idle_chunk);
  }
#endif

#ifdef PADDLE_WITH_XPU
  void InitNaiveBestFitXPUAllocator(platform::XPUPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }
#endif

#ifdef PADDLE_WITH_ASCEND_CL
  void InitNaiveBestFitNPUAllocator(platform::NPUPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }

  void InitNaiveBestFitNPUPinnedAllocator() {
    allocators_[platform::NPUPinnedPlace()] =
        std::make_shared<paddle::memory::allocation::NPUPinnedAllocator>();
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
    if (!zero_size_allocators_.empty()) return;
    std::vector<platform::Place> places;
    places.emplace_back(platform::CPUPlace());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    int device_count = platform::GetCUDADeviceCount();
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      places.emplace_back(platform::CUDAPlace(dev_id));
    }
    places.emplace_back(platform::CUDAPinnedPlace());
#endif
#ifdef PADDLE_WITH_XPU
    int device_count = platform::GetXPUDeviceCount();
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      places.emplace_back(platform::XPUPlace(dev_id));
    }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
    int device_count = platform::GetNPUDeviceCount();
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      places.emplace_back(platform::NPUPlace(dev_id));
    }
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
    PADDLE_ENFORCE_GT(
        retry_time, 0,
        platform::errors::InvalidArgument(
            "Retry time should be larger than 0, but got %d", retry_time));
    for (auto& pair : allocators_) {
      if (platform::is_gpu_place(pair.first)) {
        pair.second = std::make_shared<RetryAllocator>(pair.second, retry_time);
      }
    }
  }

#ifdef PADDLE_WITH_CUDA

 public:
  void PrepareMemoryPoolForCUDAGraph(CUDAGraphID id) {
    PADDLE_ENFORCE_EQ(strategy_, AllocatorStrategy::kAutoGrowth,
                      platform::errors::InvalidArgument(
                          "CUDA Graph is only supported when the "
                          "FLAGS_allocator_strategy=\"auto_growth\", but got "
                          "FLAGS_allocator_strategy=\"%s\"",
                          FLAGS_allocator_strategy));
    auto& allocator = cuda_graph_allocator_map_[id];
    PADDLE_ENFORCE_EQ(
        allocator.get(), nullptr,
        platform::errors::InvalidArgument(
            "The memory pool of the CUDA Graph with ID %d have been prepared.",
            id));
    allocator.reset(
        new AllocatorFacadePrivate(/*allow_free_idle_chunk=*/false));
    for (auto& item : allocator->allocators_) {
      auto& old_allocator = item.second;
      old_allocator = CUDAGraphAllocator::Create(old_allocator);
    }
    VLOG(10) << "Prepare memory pool for CUDA Graph with ID " << id;
  }

  void RemoveMemoryPoolOfCUDAGraph(CUDAGraphID id) {
    auto iter = cuda_graph_allocator_map_.find(id);
    PADDLE_ENFORCE_NE(iter, cuda_graph_allocator_map_.end(),
                      platform::errors::InvalidArgument(
                          "Cannot find CUDA Graph with ID = %d", id));
    cuda_graph_allocator_map_.erase(iter);
    VLOG(10) << "Remove memory pool of CUDA Graph with ID " << id;
  }
#endif

 private:
  AllocatorMap allocators_;
#ifdef PADDLE_WITH_CUDA
  std::unordered_map<CUDAGraphID, std::unique_ptr<AllocatorFacadePrivate>>
      cuda_graph_allocator_map_;
#endif
  AllocatorStrategy strategy_;

  static AllocatorMap zero_size_allocators_;
  static AllocatorMap system_allocators_;
};

AllocatorFacadePrivate::AllocatorMap
    AllocatorFacadePrivate::zero_size_allocators_;
AllocatorFacadePrivate::AllocatorMap AllocatorFacadePrivate::system_allocators_;

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

uint64_t AllocatorFacade::Release(const platform::Place& place) {
  return m_->GetAllocator(place, /* A non-zero num to choose allocator_ */ 1)
      ->Release(place);
}

const std::shared_ptr<Allocator>& AllocatorFacade::GetAllocator(
    const platform::Place& place) {
  return m_->GetAllocator(place, /* A non-zero num to choose allocator_ */ 1);
}

#ifdef PADDLE_WITH_CUDA
void AllocatorFacade::PrepareMemoryPoolForCUDAGraph(CUDAGraphID id) {
  return m_->PrepareMemoryPoolForCUDAGraph(id);
}

void AllocatorFacade::RemoveMemoryPoolOfCUDAGraph(CUDAGraphID id) {
  return m_->RemoveMemoryPoolOfCUDAGraph(id);
}
#endif

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
