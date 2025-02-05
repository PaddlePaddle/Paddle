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

#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include <cstdint>

#include "paddle/common/macros.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/aligned_allocator.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/allocator_strategy.h"
#include "paddle/phi/core/memory/allocation/auto_growth_best_fit_allocator.h"
#include "paddle/phi/core/memory/allocation/auto_growth_best_fit_allocator_v2.h"
#include "paddle/phi/core/memory/allocation/cpu_allocator.h"
#include "paddle/phi/core/memory/allocation/naive_best_fit_allocator.h"
#include "paddle/phi/core/memory/allocation/retry_allocator.h"
#include "paddle/phi/core/memory/allocation/stat_allocator.h"
#include "paddle/phi/core/platform/device_context.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <shared_mutex>
#include <utility>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/memory/allocation/cuda_allocator.h"
#include "paddle/phi/core/memory/allocation/cuda_managed_allocator.h"
#include "paddle/phi/core/memory/allocation/pinned_allocator.h"
#include "paddle/phi/core/memory/allocation/stream_safe_cuda_allocator.h"
#include "paddle/phi/core/memory/allocation/thread_local_allocator.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/gpu/cuda/cuda_graph.h"
#elif defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/rocm/hip_graph.h"
#endif

#if CUDA_VERSION >= 10020
#include "paddle/phi/backends/dynload/cuda_driver.h"
#include "paddle/phi/core/memory/allocation/cuda_malloc_async_allocator.h"
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator.h"
#include "paddle/phi/core/memory/allocation/virtual_memory_auto_growth_best_fit_allocator.h"
#endif

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/core/memory/allocation/cuda_malloc_async_allocator.h"  // NOLINT
#endif
#endif

#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/memory/allocation/stream_safe_xpu_allocator.h"
#include "paddle/phi/core/memory/allocation/xpu_allocator.h"
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#endif

#ifdef PADDLE_WITH_IPU
#include "paddle/fluid/platform/device/ipu/ipu_info.h"
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/core/memory/allocation/stream_safe_custom_device_allocator.h"
#endif

#include "paddle/common/flags.h"

PHI_DEFINE_EXPORTED_int64(
    gpu_allocator_retry_time,
    10000,
    "The retry time (milliseconds) when allocator fails "
    "to allocate memory. No retry if this value is not greater than 0");

PHI_DEFINE_EXPORTED_bool(
    use_system_allocator,
    false,
    "Whether to use system allocator to allocate CPU and GPU memory. "
    "Only used for unittests.");

PHI_DEFINE_EXPORTED_bool(use_virtual_memory_auto_growth,
                         false,
                         "Use VirtualMemoryAutoGrowthBestFitAllocator.");

// NOTE(Ruibiao): This FLAGS is just to be compatible with
// the old single-stream CUDA allocator. It will be removed
// after StreamSafeCudaAllocator has been fully tested.
PHI_DEFINE_EXPORTED_bool(use_stream_safe_cuda_allocator,
                         true,
                         "Enable StreamSafeCUDAAllocator");

PHI_DEFINE_EXPORTED_bool(use_cuda_managed_memory,
                         false,
                         "Whether to use CUDAManagedAllocator to allocate "
                         "managed memory, only available for auto_growth "
                         "strategy");

PHI_DEFINE_EXPORTED_bool(
    use_auto_growth_v2,
    false,
    "Whether to use AutoGrowthBestFitAllocatorV2 for auto_growth "
    "strategy");

COMMON_DECLARE_string(allocator_strategy);
COMMON_DECLARE_uint64(auto_growth_chunk_size_in_mb);
COMMON_DECLARE_bool(use_auto_growth_pinned_allocator);
COMMON_DECLARE_bool(use_cuda_malloc_async_allocator);
COMMON_DECLARE_bool(auto_free_cudagraph_allocations_on_launch);

namespace paddle::memory::allocation {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
class CUDAGraphAllocator
    : public Allocator,
      public std::enable_shared_from_this<CUDAGraphAllocator> {
 private:
  class PrivateAllocation : public Allocation {
   public:
    PrivateAllocation(CUDAGraphAllocator* allocator,
                      DecoratedAllocationPtr underlying_allocation)
        : Allocation(underlying_allocation->ptr(),
                     underlying_allocation->base_ptr(),
                     underlying_allocation->size(),
                     underlying_allocation->place()),
          allocator_(allocator->shared_from_this()),
          underlying_allocation_(std::move(underlying_allocation)) {}

   private:
    std::shared_ptr<Allocator> allocator_;
    DecoratedAllocationPtr underlying_allocation_;
  };

  explicit CUDAGraphAllocator(std::shared_ptr<Allocator> allocator)
      : underlying_allocator_(std::move(allocator)) {}

 public:
  ~CUDAGraphAllocator() override = default;

  static std::shared_ptr<Allocator> Create(
      const std::shared_ptr<Allocator>& allocator) {
    return std::shared_ptr<Allocator>(new CUDAGraphAllocator(allocator));
  }

 protected:
  phi::Allocation* AllocateImpl(size_t size) override {
    VLOG(10) << "Allocate " << size << " for CUDA Graph";
    return new PrivateAllocation(this,
                                 static_unique_ptr_cast<Allocation>(
                                     underlying_allocator_->Allocate(size)));
  }

  void FreeImpl(phi::Allocation* allocation) override {
    VLOG(10) << "delete for CUDA Graph";
    delete allocation;
  }

 private:
  std::shared_ptr<Allocator> underlying_allocator_;
};
#endif

static bool IsCUDAGraphCapturing() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  return UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing());
#else
  return false;
#endif
}

class AllocatorFacadePrivate {
 public:
  using AllocatorMap = std::map<phi::Place, std::shared_ptr<Allocator>>;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  using CUDAAllocatorMap =
      std::map<phi::GPUPlace,
               std::map<gpuStream_t, std::shared_ptr<Allocator>>>;
#endif
#ifdef PADDLE_WITH_XPU
  using XPUAllocatorMap =
      std::map<phi::XPUPlace, std::map<XPUStream, std::shared_ptr<Allocator>>>;
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  using CustomDeviceAllocatorMap =
      std::map<phi::CustomPlace,
               std::map<phi::stream::stream_t, std::shared_ptr<Allocator>>>;
#endif

  explicit AllocatorFacadePrivate(bool allow_free_idle_chunk = true)
      :
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        default_stream_safe_cuda_allocators_(),
        cuda_allocators_(),
#endif
#ifdef PADDLE_WITH_CUDA
        default_cuda_malloc_async_allocators_(),
#endif
        allocators_() {
    strategy_ = GetAllocatorStrategy();
    is_stream_safe_cuda_allocator_used_ = false;
    is_cuda_malloc_async_allocator_used_ = false;
    VLOG(2) << "selected allocator strategy:" << int(strategy_) << std::endl;
    switch (strategy_) {
      case AllocatorStrategy::kNaiveBestFit: {
        InitNaiveBestFitCPUAllocator();
#ifdef PADDLE_WITH_IPU
        for (int dev_id = 0; dev_id < platform::GetIPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitIPUAllocator(phi::IPUPlace(dev_id));
        }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        for (int dev_id = 0; dev_id < platform::GetGPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitCUDAAllocator(phi::GPUPlace(dev_id));
        }
        InitNaiveBestFitCUDAPinnedAllocator();
#endif
#ifdef PADDLE_WITH_XPU
        for (int dev_id = 0; dev_id < platform::GetXPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitXPUAllocator(phi::XPUPlace(dev_id));
        }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
        auto device_types = phi::DeviceManager::GetAllCustomDeviceTypes();
        for (const auto& dev_type : device_types) {
          for (auto& dev_id :
               phi::DeviceManager::GetSelectedDeviceList(dev_type)) {
            InitNaiveBestFitCustomDeviceAllocator(
                phi::CustomPlace(dev_type, dev_id));
          }
        }
#endif
        break;
      }

      case AllocatorStrategy::kAutoGrowth: {
        InitNaiveBestFitCPUAllocator();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        allow_free_idle_chunk_ = allow_free_idle_chunk;
        for (int dev_id = 0; dev_id < platform::GetGPUDeviceCount(); ++dev_id) {
          InitAutoGrowthCUDAAllocator(phi::GPUPlace(dev_id),
                                      allow_free_idle_chunk_);
        }
        auto_growth_allocators_ = allocators_;

        // Note(Ruibiao): For GPU multi-stream case without CUDA graph
        // capturing, the 'allocators_' map(place -> Allocator) hold the
        // StreamSafeCUDAAllocator relate to default stream (i.e., the stream
        // directly got from DeviceContext), while the 'cuda_allocators_' map
        // (place -> map(stream -> Allocator)) hold the StreamSafeCUDAAllocator
        // relate to non-default stream (i.e., the stream users pass in). The
        // default stream Allocator is built in the structure of
        // AllocatorFacadePrivate, while the non-default stream is build in a
        // manner in GetAllocator function with 'create_if_not_found = true'.
        // We make special treatment for the default stream for performance
        // reasons. Since most Alloc calls are for default stream in
        // application, treating it separately can avoid lots of overhead of
        // acquiring default stream and applying read-write lock.
        if (FLAGS_use_cuda_malloc_async_allocator) {
          PADDLE_ENFORCE_EQ(FLAGS_use_cuda_managed_memory,
                            false,
                            common::errors::InvalidArgument(
                                "Async allocator cannot be used with CUDA "
                                "managed memory."));
          WrapCUDAMallocAsyncAllocatorForDefault();
          is_cuda_malloc_async_allocator_used_ = true;
        } else {
          if (FLAGS_use_stream_safe_cuda_allocator) {
            if (LIKELY(!IsCUDAGraphCapturing())) {
              WrapStreamSafeCUDAAllocatorForDefault();
            }
            is_stream_safe_cuda_allocator_used_ = true;
          }
        }

        InitNaiveBestFitCUDAPinnedAllocator();
#endif
#ifdef PADDLE_WITH_XPU
        allow_free_idle_chunk_ = allow_free_idle_chunk;
        for (int dev_id = 0; dev_id < platform::GetXPUDeviceCount(); ++dev_id) {
          InitAutoGrowthXPUAllocator(phi::XPUPlace(dev_id),
                                     allow_free_idle_chunk_);
        }
        if (FLAGS_use_stream_safe_cuda_allocator) {
          WrapStreamSafeXPUAllocatorForDefault();
          is_stream_safe_cuda_allocator_used_ = true;
        }

#endif
#ifdef PADDLE_WITH_IPU
        for (int dev_id = 0; dev_id < platform::GetIPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitIPUAllocator(phi::IPUPlace(dev_id));
        }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
        auto device_types = phi::DeviceManager::GetAllCustomDeviceTypes();
        for (const auto& dev_type : device_types) {
          for (auto& dev_id :
               phi::DeviceManager::GetSelectedDeviceList(dev_type)) {
            InitAutoGrowthCustomDeviceAllocator(
                phi::CustomPlace(dev_type, dev_id), allow_free_idle_chunk);
          }
        }
        if (FLAGS_use_stream_safe_cuda_allocator) {
          WrapStreamSafeCustomDeviceAllocatorForDefault();
          is_stream_safe_cuda_allocator_used_ = true;
        }
#endif
        break;
      }

      case AllocatorStrategy::kThreadLocal: {
        InitNaiveBestFitCPUAllocator();
#ifdef PADDLE_WITH_XPU
        for (int dev_id = 0; dev_id < platform::GetXPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitXPUAllocator(phi::XPUPlace(dev_id));
        }
#endif
#ifdef PADDLE_WITH_IPU
        for (int dev_id = 0; dev_id < platform::GetIPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitIPUAllocator(phi::IPUPlace(dev_id));
        }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        for (int dev_id = 0; dev_id < platform::GetGPUDeviceCount(); ++dev_id) {
          InitThreadLocalCUDAAllocator(phi::GPUPlace(dev_id));
        }
        InitNaiveBestFitCUDAPinnedAllocator();
#endif
        break;
      }

      default: {
        PADDLE_THROW(common::errors::InvalidArgument(
            "Unsupported allocator strategy: %d", static_cast<int>(strategy_)));
      }
    }
    InitZeroSizeAllocators();
    InitSystemAllocators();

    if (FLAGS_gpu_allocator_retry_time > 0) {
      WrapCUDARetryAllocator(FLAGS_gpu_allocator_retry_time);
    }

    WrapStatAllocator();

    CheckAllocThreadSafe();

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    // No need to wrap CUDAGraphAllocator for StreamSafeCUDAAllocator
    if (!is_stream_safe_cuda_allocator_used_ &&
        UNLIKELY(IsCUDAGraphCapturing())) {
      WrapCUDAGraphAllocator();
    }
#endif
  }

  inline const AllocatorMap& GetAutoGrowthAllocatorMap() {
    return auto_growth_allocators_;
  }

  inline const std::shared_ptr<Allocator>& GetAllocator(const phi::Place& place,
                                                        size_t size) {
    const auto& allocators =
        (size > 0 ? (UNLIKELY(FLAGS_use_system_allocator) ? system_allocators_
                                                          : GetAllocatorMap())
                  : zero_size_allocators_);
    auto iter = allocators.find(place);
    PADDLE_ENFORCE_NE(iter,
                      allocators.end(),
                      common::errors::NotFound(
                          "No allocator found for the place, %s", place));
    VLOG(6) << "[GetAllocator]"
            << " place = " << place << " size = " << size
            << " Allocator = " << iter->second;
    return iter->second;
  }

  void* GetBasePtr(const std::shared_ptr<phi::Allocation>& allocation) {
    return static_cast<Allocation*>(allocation.get())->base_ptr();
  }

  bool IsStreamSafeCUDAAllocatorUsed() {
    return is_stream_safe_cuda_allocator_used_ &&
           LIKELY(FLAGS_use_system_allocator == false);
  }

  bool IsCUDAMallocAsyncAllocatorUsed() {
    return is_cuda_malloc_async_allocator_used_ &&
           LIKELY(FLAGS_use_system_allocator == false);
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  bool HasCUDAAllocator(const phi::GPUPlace& place, gpuStream_t stream) {
    auto it = cuda_allocators_.find(place);
    if (it == cuda_allocators_.end()) {
      return false;
    }
    const std::map<gpuStream_t, std::shared_ptr<Allocator>>& allocator_map =
        it->second;
    return allocator_map.find(stream) != allocator_map.end();
  }

  const std::shared_ptr<Allocator>& GetAllocator(
      const phi::GPUPlace& place,
      gpuStream_t stream,
      bool create_if_not_found = false) {
    if (LIKELY(!IsCUDAGraphCapturing())) {
      if (stream == GetDefaultStream(place)) {
        VLOG(7) << "Get Allocator by passing in a default stream";
        return GetAllocator(place, /* A non-zero num to choose allocator_ */ 1);
      }
    }

    /* shared_lock_guard */ {
      std::shared_lock<std::shared_timed_mutex> lock_guard(
          cuda_allocator_mutex_);
      if (LIKELY(HasCUDAAllocator(place, stream))) {
        return cuda_allocators_[place][stream];
      } else {
        PADDLE_ENFORCE_NE(create_if_not_found,
                          false,
                          common::errors::NotFound(
                              "No allocator found for stream %s in place %s "
                              "with create_if_not_found = false",
                              stream,
                              place));
      }
    }

    /* unique_lock_guard */ {
      std::unique_lock<std::shared_timed_mutex> lock_guard(
          cuda_allocator_mutex_);
      InitCUDAAllocator(place, stream);
      return cuda_allocators_[place][stream];
    }
  }

  const std::shared_ptr<Allocator> GetDefaultStreamSafeCUDAAllocator(
      const phi::GPUPlace& place) const {
    if (auto iter = default_stream_safe_cuda_allocators_.find(place);
        iter != default_stream_safe_cuda_allocators_.end())
      return iter->second;
#ifdef PADDLE_WITH_CUDA
    if (auto iter = default_cuda_malloc_async_allocators_.find(place);
        iter != default_cuda_malloc_async_allocators_.end())
      return iter->second;
#endif
    PADDLE_THROW(common::errors::NotFound(
        "No StreamSafeCUDAAllocator found for the place, %s", place));
  }

  gpuStream_t GetDefaultStream(const phi::GPUPlace& place) const {
    if (auto allocator = std::dynamic_pointer_cast<StreamSafeCUDAAllocator>(
            GetDefaultStreamSafeCUDAAllocator(place))) {
      return allocator->GetDefaultStream();
#ifdef PADDLE_WITH_CUDA
    } else if (auto allocator =
                   std::dynamic_pointer_cast<CUDAMallocAsyncAllocator>(
                       GetDefaultStreamSafeCUDAAllocator(place))) {
      return allocator->GetDefaultStream();
#endif
    } else {
      PADDLE_THROW(common::errors::NotFound(
          "No StreamSafeCUDAAllocator or CUDAMallocAsyncAllocator found for "
          "the place, %s",
          place));
    }
  }

  void SetDefaultStream(const phi::GPUPlace& place, gpuStream_t stream) {
    if (auto allocator = std::dynamic_pointer_cast<StreamSafeCUDAAllocator>(
            GetDefaultStreamSafeCUDAAllocator(place))) {
      PADDLE_ENFORCE_EQ(allocator->GetDefaultStream(),
                        nullptr,
                        common::errors::Unavailable(
                            "The default stream for "
                            "StreamSafeCUDAAllocator(%p) in %s has been "
                            "set to %p, not allow to change it to %p.",
                            allocator.get(),
                            place,
                            allocator->GetDefaultStream(),
                            stream));

      allocator->SetDefaultStream(stream);
      VLOG(8) << "Set default stream to " << stream
              << " for StreamSafeCUDAAllocator(" << allocator.get() << ") in "
              << place;
#ifdef PADDLE_WITH_CUDA
    } else if (auto allocator =
                   std::dynamic_pointer_cast<CUDAMallocAsyncAllocator>(
                       GetDefaultStreamSafeCUDAAllocator(place))) {
      PADDLE_ENFORCE_EQ(allocator->GetDefaultStream(),
                        nullptr,
                        common::errors::Unavailable(
                            "The default stream for "
                            "StreamSafeCUDAAllocator(%p) in %s has been "
                            "set to %p, not allow to change it to %p.",
                            allocator.get(),
                            place,
                            allocator->GetDefaultStream(),
                            stream));
      allocator->SetDefaultStream(stream);
      VLOG(8) << "Set default stream to " << stream
              << " for CUDAMallocAsyncAllocator(" << allocator.get() << ") in "
              << place;
#endif
    } else {
      PADDLE_THROW(common::errors::NotFound(
          "No StreamSafeCUDAAllocator or CUDAMallocAsyncAllocator found for "
          "the place, %s",
          place));
    }
  }

  bool RecordStream(std::shared_ptr<phi::Allocation> allocation,
                    gpuStream_t stream) {
    if (auto stream_safe_cuda_allocation =
            std::dynamic_pointer_cast<StreamSafeCUDAAllocation>(allocation)) {
      return stream_safe_cuda_allocation->RecordStream(stream);
#ifdef PADDLE_WITH_CUDA
    } else if (auto cuda_malloc_async_allocation =
                   std::dynamic_pointer_cast<CUDAMallocAsyncAllocation>(
                       allocation)) {
      return cuda_malloc_async_allocation->RecordStream(stream);
#endif
    } else {
      VLOG(6) << "RecordStream for a non-StreamSafeCUDAAllocation";
      return false;
    }
  }

  void EraseStream(std::shared_ptr<phi::Allocation> allocation,
                   gpuStream_t stream) {
    if (auto stream_safe_cuda_allocation =
            std::dynamic_pointer_cast<StreamSafeCUDAAllocation>(allocation)) {
      stream_safe_cuda_allocation->EraseStream(stream);
#ifdef PADDLE_WITH_CUDA
    } else if (auto cuda_malloc_async_allocation =
                   std::dynamic_pointer_cast<CUDAMallocAsyncAllocation>(
                       allocation)) {
      cuda_malloc_async_allocation->EraseStream(stream);
#endif
    } else {
      VLOG(6) << "EraseStream for a non-StreamSafeCUDAAllocation";
    }
  }

  gpuStream_t GetStream(
      const std::shared_ptr<phi::Allocation>& allocation) const {
    if (const std::shared_ptr<StreamSafeCUDAAllocation>
            stream_safe_cuda_allocation =
                std::dynamic_pointer_cast<StreamSafeCUDAAllocation>(
                    allocation)) {
      return stream_safe_cuda_allocation->GetOwningStream();
#ifdef PADDLE_WITH_CUDA
    } else if (const std::shared_ptr<CUDAMallocAsyncAllocation>
                   cuda_malloc_async_allocation =
                       std::dynamic_pointer_cast<CUDAMallocAsyncAllocation>(
                           allocation)) {
      return cuda_malloc_async_allocation->GetOwningStream();
#endif
    }

    VLOG(6) << "GetStream for a non-StreamSafeCUDAAllocation";
    return static_cast<phi::GPUContext*>(
               phi::DeviceContextPool::Instance().Get(allocation->place()))
        ->stream();
  }
#endif

#ifdef PADDLE_WITH_XPU
  bool HasXPUAllocator(const phi::XPUPlace& place, XPUStream stream) {
    auto it = xpu_allocators_.find(place);
    if (it == xpu_allocators_.end()) {
      return false;
    }
    const std::map<XPUStream, std::shared_ptr<Allocator>>& allocator_map =
        it->second;
    return allocator_map.find(stream) != allocator_map.end();
  }

  const std::shared_ptr<Allocator>& GetAllocator(
      const phi::XPUPlace& place,
      XPUStream stream,
      bool create_if_not_found = false) {
    if (stream == GetDefaultStream(place)) {
      VLOG(7) << "Get Allocator by passing in a default stream";
      return GetAllocator(place, /* A non-zero num to choose allocator_ */ 1);
    }

    /* shared_lock_guard */ {
      std::shared_lock<std::shared_timed_mutex> lock_guard(
          xpu_allocator_mutex_);
      if (LIKELY(HasXPUAllocator(place, stream))) {
        return xpu_allocators_[place][stream];
      } else {
        PADDLE_ENFORCE_NE(create_if_not_found,
                          false,
                          common::errors::NotFound(
                              "No allocator found for stream %s in place %s "
                              "with create_if_not_found = false",
                              stream,
                              place));
      }
    }

    /* unique_lock_guard */ {
      std::unique_lock<std::shared_timed_mutex> lock_guard(
          xpu_allocator_mutex_);
      InitStreamSafeXPUAllocator(place, stream);
      return xpu_allocators_[place][stream];
    }
  }

  const std::shared_ptr<StreamSafeXPUAllocator>
  GetDefaultStreamSafeXPUAllocator(const phi::XPUPlace& place) const {
    const auto iter = default_stream_safe_xpu_allocators_.find(place);
    PADDLE_ENFORCE_NE(
        iter,
        default_stream_safe_xpu_allocators_.end(),
        common::errors::NotFound(
            "No StreamSafeXPUAllocator found for the place, %s", place));
    return iter->second;
  }

  XPUStream GetDefaultStream(const phi::XPUPlace& place) const {
    const std::shared_ptr<StreamSafeXPUAllocator>& allocator =
        GetDefaultStreamSafeXPUAllocator(place);
    return allocator->GetDefaultStream();
  }

  void SetDefaultStream(const phi::XPUPlace& place, XPUStream stream) {
    const std::shared_ptr<StreamSafeXPUAllocator>& allocator =
        GetDefaultStreamSafeXPUAllocator(place);

    PADDLE_ENFORCE_EQ(
        allocator->GetDefaultStream(),
        nullptr,
        common::errors::Unavailable(
            "The default stream for StreamSafeXPUAllocator(%p) in %s has been "
            "set to %p, not allow to change it to %p.",
            allocator.get(),
            place,
            allocator->GetDefaultStream(),
            stream));

    allocator->SetDefaultStream(stream);
    VLOG(8) << "Set default stream to " << stream
            << " for StreamSafeXPUAllocator(" << allocator.get() << ") in "
            << place;
  }

  bool RecordStream(std::shared_ptr<phi::Allocation> allocation,
                    XPUStream stream) {
    std::shared_ptr<StreamSafeXPUAllocation> stream_safe_xpu_allocation =
        std::dynamic_pointer_cast<StreamSafeXPUAllocation>(allocation);
    if (stream_safe_xpu_allocation != nullptr) {
      return stream_safe_xpu_allocation->RecordStream(stream);
    } else {
      VLOG(6) << "RecordStream for a non-StreamSafeXPUAllocation";
      return false;
    }
  }

  XPUStream GetStream(
      const std::shared_ptr<phi::Allocation>& allocation) const {
    const std::shared_ptr<StreamSafeXPUAllocation> stream_safe_xpu_allocation =
        std::dynamic_pointer_cast<StreamSafeXPUAllocation>(allocation);
    if (stream_safe_xpu_allocation != nullptr) {
      return stream_safe_xpu_allocation->GetOwningStream();
    }

    VLOG(6) << "GetStream for a non-StreamSafeXPUAllocation";
    return static_cast<phi::XPUContext*>(
               phi::DeviceContextPool::Instance().Get(allocation->place()))
        ->stream();
  }
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
  bool HasCustomDeviceAllocator(const phi::CustomPlace& place,
                                phi::stream::stream_t stream) {
    auto it = custom_device_allocators_.find(place);
    if (it == custom_device_allocators_.end()) {
      return false;
    }
    const std::map<phi::stream::stream_t, std::shared_ptr<Allocator>>&
        allocator_map = it->second;
    return allocator_map.find(stream) != allocator_map.end();
  }

  const std::shared_ptr<Allocator>& GetAllocator(
      const phi::CustomPlace& place,
      phi::stream::stream_t stream,
      bool create_if_not_found = false) {
    if (stream == GetDefaultStream(place)) {
      VLOG(7) << "Get Allocator by passing in a default stream";
      return GetAllocator(place, /* A non-zero num to choose allocator_ */ 1);
    }

    /* shared_lock_guard */ {
      std::shared_lock<std::shared_timed_mutex> lock_guard(
          custom_device_allocator_mutex_);
      if (LIKELY(HasCustomDeviceAllocator(place, stream))) {
        return custom_device_allocators_[place][stream];
      } else {
        PADDLE_ENFORCE_NE(create_if_not_found,
                          false,
                          common::errors::NotFound(
                              "No allocator found for stream %s in place %s "
                              "with create_if_not_found = false",
                              stream,
                              place));
      }
    }

    /* unique_lock_guard */ {
      std::unique_lock<std::shared_timed_mutex> lock_guard(
          custom_device_allocator_mutex_);
      InitStreamSafeCustomDeviceAllocator(place, stream);
      return custom_device_allocators_[place][stream];
    }
  }

  const std::shared_ptr<StreamSafeCustomDeviceAllocator>
  GetDefaultStreamSafeCustomDeviceAllocator(
      const phi::CustomPlace& place) const {
    const auto iter = default_stream_safe_custom_device_allocators_.find(place);
    PADDLE_ENFORCE_NE(
        iter,
        default_stream_safe_custom_device_allocators_.end(),
        common::errors::NotFound(
            "No StreamSafeCustomDeviceAllocator found for the place, %s",
            place));
    return iter->second;
  }

  phi::stream::stream_t GetDefaultStream(const phi::CustomPlace& place) const {
    const std::shared_ptr<StreamSafeCustomDeviceAllocator>& allocator =
        GetDefaultStreamSafeCustomDeviceAllocator(place);
    return allocator->GetDefaultStream();
  }

  void SetDefaultStream(const phi::CustomPlace& place,
                        phi::stream::stream_t stream) {
    const std::shared_ptr<StreamSafeCustomDeviceAllocator>& allocator =
        GetDefaultStreamSafeCustomDeviceAllocator(place);

    PADDLE_ENFORCE_EQ(allocator->GetDefaultStream(),
                      nullptr,
                      common::errors::Unavailable(
                          "The default stream for "
                          "StreamSafeCustomDeviceAllocator(%p) in %s has been "
                          "set to %p, not allow to change it to %p.",
                          allocator.get(),
                          place,
                          allocator->GetDefaultStream(),
                          stream));

    allocator->SetDefaultStream(stream);
    VLOG(8) << "Set default stream to " << stream
            << " for StreamSafeCustomDeviceAllocator(" << allocator.get()
            << ") in " << place;
  }

  bool RecordStream(std::shared_ptr<phi::Allocation> allocation,
                    phi::stream::stream_t stream) {
    std::shared_ptr<StreamSafeCustomDeviceAllocation>
        stream_safe_custom_device_allocation =
            std::dynamic_pointer_cast<StreamSafeCustomDeviceAllocation>(
                allocation);
    if (stream_safe_custom_device_allocation != nullptr) {
      return stream_safe_custom_device_allocation->RecordStream(stream);
    } else {
      VLOG(6) << "RecordStream for a non-StreamSafeCustomDeviceAllocation";
      return false;
    }
  }

  phi::stream::stream_t GetStream(
      const std::shared_ptr<phi::Allocation>& allocation) const {
    const std::shared_ptr<StreamSafeCustomDeviceAllocation>
        stream_safe_custom_device_allocation =
            std::dynamic_pointer_cast<StreamSafeCustomDeviceAllocation>(
                allocation);
    if (stream_safe_custom_device_allocation != nullptr) {
      return stream_safe_custom_device_allocation->GetOwningStream();
    }

    VLOG(6) << "GetStream for a non-StreamSafeCustomDeviceAllocation";
    return static_cast<phi::CustomContext*>(
               phi::DeviceContextPool::Instance().Get(allocation->place()))
        ->stream();
  }
#endif

 private:
  class ZeroSizeAllocator : public Allocator {
   public:
    explicit ZeroSizeAllocator(phi::Place place) : place_(place) {}
    bool IsAllocThreadSafe() const override { return true; }

   protected:
    phi::Allocation* AllocateImpl(size_t size) override {
      return new Allocation(nullptr, 0, place_);
    }
    void FreeImpl(phi::Allocation* allocation) override { delete allocation; }

   private:
    phi::Place place_;
  };

  const AllocatorMap& GetAllocatorMap() { return allocators_; }

  void InitNaiveBestFitCPUAllocator() {
#if defined(__APPLE__) && defined(__arm64__)
    // NOTE(wuweilong): It is more efficient to use CPUAllocator directly,
    // but it will cause some problem in Mac OS m1 chip, so we use
    // NaiveBestFitAllocator instead.
    allocators_[phi::CPUPlace()] =
        std::make_shared<NaiveBestFitAllocator>(phi::CPUPlace());
#else
    allocators_[phi::CPUPlace()] = std::make_shared<CPUAllocator>();
#endif
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void InitNaiveBestFitCUDAPinnedAllocator() {
    if (FLAGS_use_auto_growth_pinned_allocator) {
      auto chunk_size = FLAGS_auto_growth_chunk_size_in_mb << 20;
      VLOG(4) << "FLAGS_auto_growth_chunk_size_in_mb is "
              << FLAGS_auto_growth_chunk_size_in_mb;
      auto pinned_allocator = std::make_shared<CPUPinnedAllocator>();
      allocators_[phi::GPUPinnedPlace()] =
          std::make_shared<AutoGrowthBestFitAllocator>(
              pinned_allocator,
              phi::backends::cpu::CUDAPinnedMinChunkSize(),
              chunk_size,
              allow_free_idle_chunk_);
    } else {
      allocators_[phi::GPUPinnedPlace()] =
          std::make_shared<NaiveBestFitAllocator>(phi::GPUPinnedPlace());
    }
  }

  void InitNaiveBestFitCUDAAllocator(phi::GPUPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }

  // Create a new CUDAAllocator or CUDAManagedAllocator for the given device
  std::shared_ptr<Allocator> CreateCUDAAllocator(phi::GPUPlace p) {
    if (FLAGS_use_cuda_managed_memory) {
      PADDLE_ENFORCE_EQ(
          strategy_,
          AllocatorStrategy::kAutoGrowth,
          common::errors::InvalidArgument(
              "CUDA managed memory is only implemented for auto_growth "
              "strategy, not support %s strategy.\n"
              "Please use auto_growth strategy by command `export "
              "FLAGS_allocator_strategy=\"auto_growth\"`, or disable managed "
              "memory by command `export FLAGS_use_cuda_managed_memory=false`",
              FLAGS_allocator_strategy));

      if (!platform::IsGPUManagedMemorySupported(p.device)) {
        PADDLE_THROW(common::errors::Unavailable(
            "Failed to create CUDAManagedAllocator on GPU %d.\n\n"
            "You have enabled CUDA managed memory, but the gpu device does not "
            "support allocating managed memory.\n"
            "If you don't actually need to use managed memory, please disable "
            "it with command `export FLAGS_use_cuda_managed_memory=false`.\n"
            "Or you must use the gpu device that supports managed memory.",
            p.device));
      }
      return std::make_shared<CUDAManagedAllocator>(p);
    }
    return std::make_shared<CUDAAllocator>(p);
  }

  void InitCUDAAllocator(phi::GPUPlace p, gpuStream_t stream) {
    PADDLE_ENFORCE_EQ(
        strategy_,
        AllocatorStrategy::kAutoGrowth,
        common::errors::Unimplemented(
            "Only support auto-growth strategy for StreamSafeCUDAAllocator, "
            "the allocator strategy %d is unsupported for multi-stream",
            static_cast<int>(strategy_)));
    if (FLAGS_use_cuda_malloc_async_allocator) {
      PADDLE_ENFORCE_EQ(
          FLAGS_use_cuda_managed_memory,
          false,
          common::errors::InvalidArgument(
              "Async allocator cannot be used with CUDA managed memory."));
      VLOG(8) << "[CUDAMallocAsyncAllocator] Init CUDA allocator for stream "
              << stream << " in place " << p;
      InitCUDAMallocAsyncAllocator(p, stream);
      WrapCUDARetryAllocator(p, stream, FLAGS_gpu_allocator_retry_time);
      WrapStatAllocator(p, stream);
    } else {
      if (LIKELY(!HasCUDAAllocator(p, stream))) {
        VLOG(8) << "Init CUDA allocator for stream " << stream << " in place "
                << p;
        InitAutoGrowthCUDAAllocator(p, stream);
        WrapStreamSafeCUDAAllocator(p, stream);
        WrapCUDARetryAllocator(p, stream, FLAGS_gpu_allocator_retry_time);
        WrapStatAllocator(p, stream);
      }
    }
  }

  void InitCUDAMallocAsyncAllocator(phi::GPUPlace p, gpuStream_t stream) {
#ifdef PADDLE_WITH_CUDA
    std::shared_ptr<Allocator>& allocator = cuda_allocators_[p][stream];
    cuda_allocators_[p][stream] =
        std::make_shared<CUDAMallocAsyncAllocator>(allocator, p, stream);
#else
    PADDLE_THROW(
        common::errors::Unavailable("CUDAMallocAsyncAllocator is not enabled"));
#endif
  }

  void InitAutoGrowthCUDAAllocator(phi::GPUPlace p, gpuStream_t stream) {
    auto chunk_size = FLAGS_auto_growth_chunk_size_in_mb << 20;
    VLOG(4) << "FLAGS_auto_growth_chunk_size_in_mb is "
            << FLAGS_auto_growth_chunk_size_in_mb;
#if defined(PADDLE_WITH_HIP)
    auto cuda_allocator = CreateCUDAAllocator(p);
    if (FLAGS_use_auto_growth_v2) {
      cuda_allocators_[p][stream] =
          std::make_shared<AutoGrowthBestFitAllocatorV2>(
              cuda_allocator,
              platform::GpuMinChunkSize(),
              p,
              chunk_size,
              allow_free_idle_chunk_);
    } else {
      cuda_allocators_[p][stream] =
          std::make_shared<AutoGrowthBestFitAllocator>(
              cuda_allocator,
              platform::GpuMinChunkSize(),
              chunk_size,
              allow_free_idle_chunk_);
    }
#endif

#if defined(PADDLE_WITH_CUDA)
#if CUDA_VERSION >= 10020
    CUdevice device;
    int val;
    try {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cuDeviceGet(&device, p.GetDeviceId()));

      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuDeviceGetAttribute(
          &val,
          CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
          device));
    } catch (...) {
      val = 0;
    }

    if (val > 0 && FLAGS_use_virtual_memory_auto_growth) {
      auto cuda_allocator = std::make_shared<CUDAVirtualMemAllocator>(p);
      cuda_allocators_[p][stream] =
          std::make_shared<VirtualMemoryAutoGrowthBestFitAllocator>(
              cuda_allocator, platform::GpuMinChunkSize(), p);
    } else {
      auto cuda_allocator = CreateCUDAAllocator(p);
      if (FLAGS_use_auto_growth_v2) {
        cuda_allocators_[p][stream] =
            std::make_shared<AutoGrowthBestFitAllocatorV2>(
                cuda_allocator,
                platform::GpuMinChunkSize(),
                p,
                /*chunk_size=*/chunk_size,
                allow_free_idle_chunk_);
      } else {
        cuda_allocators_[p][stream] =
            std::make_shared<AutoGrowthBestFitAllocator>(
                cuda_allocator,
                platform::GpuMinChunkSize(),
                /*chunk_size=*/chunk_size,
                allow_free_idle_chunk_);
      }
    }
#else
    auto cuda_allocator = CreateCUDAAllocator(p);
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
    if (FLAGS_use_auto_growth_v2) {
      cuda_allocators_[p][stream] =
          std::make_shared<AutoGrowthBestFitAllocatorV2>(
              underlying_allocator,
              alignment,
              p,
              chunk_size,
              allow_free_idle_chunk_);
    } else {
      cuda_allocators_[p][stream] =
          std::make_shared<AutoGrowthBestFitAllocator>(underlying_allocator,
                                                       alignment,
                                                       chunk_size,
                                                       allow_free_idle_chunk_);
    }
#endif
#endif
  }

  // NOTE(Ruibiao): Old single-stream version, will be removed later
  void InitAutoGrowthCUDAAllocator(phi::GPUPlace p,
                                   bool allow_free_idle_chunk) {
    auto chunk_size = FLAGS_auto_growth_chunk_size_in_mb << 20;
    VLOG(4) << "FLAGS_auto_growth_chunk_size_in_mb is "
            << FLAGS_auto_growth_chunk_size_in_mb;
#if defined(PADDLE_WITH_HIP)
    auto cuda_allocator = CreateCUDAAllocator(p);
    if (FLAGS_use_auto_growth_v2) {
      allocators_[p] = std::make_shared<AutoGrowthBestFitAllocatorV2>(
          cuda_allocator,
          platform::GpuMinChunkSize(),
          p,
          /*chunk_size=*/chunk_size,
          allow_free_idle_chunk);
    } else {
      allocators_[p] = std::make_shared<AutoGrowthBestFitAllocator>(
          cuda_allocator,
          platform::GpuMinChunkSize(),
          /*chunk_size=*/chunk_size,
          allow_free_idle_chunk);
    }
#endif

#if defined(PADDLE_WITH_CUDA)
#if CUDA_VERSION >= 10020
    CUdevice device;
    int val;
    try {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cuDeviceGet(&device, p.GetDeviceId()));

      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cuDeviceGetAttribute(
          &val,
          CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
          device));
    } catch (...) {
      val = 0;
    }

    if (val > 0 && FLAGS_use_virtual_memory_auto_growth) {
      auto cuda_allocator = std::make_shared<CUDAVirtualMemAllocator>(p);
      allocators_[p] =
          std::make_shared<VirtualMemoryAutoGrowthBestFitAllocator>(
              cuda_allocator, platform::GpuMinChunkSize(), p);
    } else {
      auto cuda_allocator = CreateCUDAAllocator(p);
      if (FLAGS_use_auto_growth_v2) {
        allocators_[p] = std::make_shared<AutoGrowthBestFitAllocatorV2>(
            cuda_allocator,
            platform::GpuMinChunkSize(),
            p,
            /*chunk_size=*/chunk_size,
            allow_free_idle_chunk);
      } else {
        allocators_[p] = std::make_shared<AutoGrowthBestFitAllocator>(
            cuda_allocator,
            platform::GpuMinChunkSize(),
            /*chunk_size=*/chunk_size,
            allow_free_idle_chunk);
      }
    }

#else
    auto cuda_allocator = CreateCUDAAllocator(p);
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
    if (FLAGS_use_auto_growth_v2) {
      allocators_[p] =
          std::make_shared<AutoGrowthBestFitAllocatorV2>(underlying_allocator,
                                                         alignment,
                                                         p,
                                                         chunk_size,
                                                         allow_free_idle_chunk);
    } else {
      allocators_[p] = std::make_shared<AutoGrowthBestFitAllocator>(
          underlying_allocator, alignment, chunk_size, allow_free_idle_chunk);
    }
#endif
#endif
  }

  void InitThreadLocalCUDAAllocator(phi::GPUPlace p) {
    allocators_[p] = std::make_shared<ThreadLocalCUDAAllocator>(p);
  }

  void WrapStreamSafeCUDAAllocator(phi::GPUPlace p, gpuStream_t stream) {
    VLOG(8) << "[StreamSafeCUDAAllocator] Init CUDA allocator for stream "
            << stream << " in place " << p;
    std::shared_ptr<Allocator>& allocator = cuda_allocators_[p][stream];
    allocator = std::make_shared<StreamSafeCUDAAllocator>(
        allocator,
        p,
        stream,
        /* in_cuda_graph_capturing = */ !allow_free_idle_chunk_);
  }

  void WrapStreamSafeCUDAAllocatorForDefault() {
    for (auto& pair : allocators_) {
      auto& place = pair.first;
      if (phi::is_gpu_place(place)) {
        std::shared_ptr<StreamSafeCUDAAllocator>&& allocator =
            std::make_shared<StreamSafeCUDAAllocator>(
                pair.second,
                place,
                /* default_stream = */ nullptr,
                /* in_cuda_graph_capturing = */ !allow_free_idle_chunk_);
        pair.second = allocator;

        // NOTE(Ruibiao): A tricky implement to give StreamSafeCUDAAllocator an
        // ability to interact with the outside world, i.e., change default
        // stream from outside
        default_stream_safe_cuda_allocators_[place] = allocator;
        VLOG(8) << "WrapStreamSafeCUDAAllocator for " << place
                << ", allocator address = " << pair.second.get();
      }
    }
  }

  void WrapCUDAMallocAsyncAllocatorForDefault() {
#ifdef PADDLE_WITH_CUDA
    for (auto& pair : allocators_) {
      auto& place = pair.first;
      if (phi::is_gpu_place(place)) {
        // we set default stream of the Allocator to 0 (nullptr) here, but we
        // would set it to the compute stream of the device with
        // SetDefaultStream later

        std::shared_ptr<CUDAMallocAsyncAllocator>&& allocator =
            std::make_shared<CUDAMallocAsyncAllocator>(
                pair.second,
                place,
                /* default_stream = */ nullptr);
        pair.second = allocator;

        default_cuda_malloc_async_allocators_[place] = allocator;
        VLOG(8) << "[WrapCUDAMallocAsyncAllocatorForDefault] " << place
                << ", allocator address = " << pair.second.get();
      }
    }
#else
    PADDLE_THROW(
        common::errors::Unavailable("CUDAMallocAsyncAllocator is not enabled"));
#endif
  }

  void WrapCUDARetryAllocator(phi::GPUPlace p,
                              gpuStream_t stream,
                              size_t retry_time) {
    PADDLE_ENFORCE_GT(
        retry_time,
        0,
        common::errors::InvalidArgument(
            "Retry time should be larger than 0, but got %d", retry_time));
    std::shared_ptr<Allocator>& allocator = cuda_allocators_[p][stream];
    allocator = std::make_shared<RetryAllocator>(allocator, retry_time);
  }

  void WrapStatAllocator(phi::GPUPlace p, gpuStream_t stream) {
    std::shared_ptr<Allocator>& allocator = cuda_allocators_[p][stream];
    allocator = std::make_shared<StatAllocator>(allocator);
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void WrapCUDAGraphAllocator() {
    for (auto& item : allocators_) {
      auto& allocator = item.second;
      allocator = CUDAGraphAllocator::Create(allocator);
    }
  }
#endif

  static void CheckCUDAAllocThreadSafe(const CUDAAllocatorMap& allocators) {
    for (auto& place_pair : allocators) {
      for (auto& stream_pair : place_pair.second) {
        PADDLE_ENFORCE_EQ(stream_pair.second->IsAllocThreadSafe(),
                          true,
                          common::errors::InvalidArgument(
                              "Public allocators must be thread safe"));
      }
    }
  }
#endif

#ifdef PADDLE_WITH_XPU
  void InitNaiveBestFitXPUAllocator(phi::XPUPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }

  // Create a new XPUAllocator or XPUManagedAllocator for the given device
  std::shared_ptr<Allocator> CreateXPUAllocator(phi::XPUPlace p) {
    return std::make_shared<XPUAllocator>(p);
  }

  void InitStreamSafeXPUAllocator(phi::XPUPlace p, XPUStream stream) {
    PADDLE_ENFORCE_EQ(
        strategy_,
        AllocatorStrategy::kAutoGrowth,
        common::errors::Unimplemented(
            "Only support auto-growth strategy for StreamSafeXPUAllocator, "
            "the allocator strategy %d is unsupported for multi-stream",
            static_cast<int>(strategy_)));
    if (LIKELY(!HasXPUAllocator(p, stream))) {
      VLOG(8) << "Init XPU allocator for stream " << stream << " in place "
              << p;
      InitAutoGrowthXPUAllocator(p, stream);

      WrapStreamSafeXPUAllocator(p, stream);

      WrapXPURetryAllocator(p, stream, FLAGS_gpu_allocator_retry_time);
      WrapStatAllocator(p, stream);
    }
  }

  void InitAutoGrowthXPUAllocator(phi::XPUPlace p, XPUStream stream) {
    auto chunk_size = FLAGS_auto_growth_chunk_size_in_mb << 6;
    VLOG(4) << "FLAGS_auto_growth_chunk_size_in_mb is "
            << FLAGS_auto_growth_chunk_size_in_mb;
    auto xpu_allocator = CreateXPUAllocator(p);
    auto alignment = platform::XPUMinChunkSize();

    std::shared_ptr<Allocator> underlying_allocator{nullptr};

    VLOG(10) << "not use AlignedAllocator with alignment: " << alignment;
    underlying_allocator = xpu_allocator;

    xpu_allocators_[p][stream] = std::make_shared<AutoGrowthBestFitAllocator>(
        underlying_allocator, alignment, chunk_size, allow_free_idle_chunk_);
  }

  void InitAutoGrowthXPUAllocator(phi::XPUPlace p, bool allow_free_idle_chunk) {
    auto chunk_size = FLAGS_auto_growth_chunk_size_in_mb << 6;
    VLOG(4) << "FLAGS_auto_growth_chunk_size_in_mb is "
            << FLAGS_auto_growth_chunk_size_in_mb;
    auto xpu_allocator = CreateXPUAllocator(p);
    auto alignment = platform::XPUMinChunkSize();

    std::shared_ptr<Allocator> underlying_allocator{nullptr};

    VLOG(10) << "not use AlignedAllocator with alignment: " << alignment;
    underlying_allocator = xpu_allocator;

    allocators_[p] = std::make_shared<AutoGrowthBestFitAllocator>(
        underlying_allocator, alignment, chunk_size, allow_free_idle_chunk);
  }

  void WrapStreamSafeXPUAllocator(phi::XPUPlace p, XPUStream stream) {
    std::shared_ptr<Allocator>& allocator = xpu_allocators_[p][stream];
    allocator = std::make_shared<StreamSafeXPUAllocator>(allocator, p, stream);
  }

  void WrapStreamSafeXPUAllocatorForDefault() {
    for (auto& pair : allocators_) {
      auto& place = pair.first;
      if (phi::is_xpu_place(place)) {
        std::shared_ptr<StreamSafeXPUAllocator>&& allocator =
            std::make_shared<StreamSafeXPUAllocator>(
                pair.second,
                place,
                /* default_stream = */ nullptr);
        pair.second = allocator;
        default_stream_safe_xpu_allocators_[place] = allocator;
        VLOG(8) << "WrapStreamSafeXPUAllocator for " << place
                << ", allocator address = " << pair.second.get();
      }
    }
  }

  void WrapXPURetryAllocator(phi::XPUPlace p,
                             XPUStream stream,
                             size_t retry_time) {
    PADDLE_ENFORCE_GT(
        retry_time,
        0,
        common::errors::InvalidArgument(
            "Retry time should be larger than 0, but got %d", retry_time));
    std::shared_ptr<Allocator>& allocator = xpu_allocators_[p][stream];
    allocator = std::make_shared<RetryAllocator>(allocator, retry_time);
  }

  void WrapStatAllocator(phi::XPUPlace p, XPUStream stream) {
    std::shared_ptr<Allocator>& allocator = xpu_allocators_[p][stream];
    allocator = std::make_shared<StatAllocator>(allocator);
  }

#endif

#ifdef PADDLE_WITH_IPU
  void InitNaiveBestFitIPUAllocator(phi::IPUPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
  void InitNaiveBestFitCustomDeviceAllocator(phi::CustomPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }

  std::shared_ptr<Allocator> CreateCustomDeviceAllocator(phi::CustomPlace p) {
    return std::make_shared<CustomAllocator>(p);
  }

  void InitStreamSafeCustomDeviceAllocator(phi::CustomPlace p,
                                           phi::stream::stream_t stream) {
    PADDLE_ENFORCE_EQ(
        strategy_,
        AllocatorStrategy::kAutoGrowth,
        common::errors::Unimplemented(
            "Only support auto-growth strategy for "
            "StreamSafeCustomDeviceAllocator, "
            "the allocator strategy %d is unsupported for multi-stream",
            static_cast<int>(strategy_)));
    if (LIKELY(!HasCustomDeviceAllocator(p, stream))) {
      VLOG(8) << "Init StreamSafeCustomDeviceAllocator for stream " << stream
              << " in place " << p;
      InitAutoGrowthCustomDeviceAllocator(p, stream);
      WrapStreamSafeCustomDeviceAllocator(p, stream);
    }
  }

  void InitAutoGrowthCustomDeviceAllocator(phi::CustomPlace p,
                                           phi::stream::stream_t stream) {
    auto chunk_size = FLAGS_auto_growth_chunk_size_in_mb << 20;
    VLOG(4) << "FLAGS_auto_growth_chunk_size_in_mb is "
            << FLAGS_auto_growth_chunk_size_in_mb;

    auto custom_allocator =
        std::make_shared<paddle::memory::allocation::CustomAllocator>(p);
    auto alignment = phi::DeviceManager::GetMinChunkSize(p);
    custom_device_allocators_[p][stream] =
        std::make_shared<AutoGrowthBestFitAllocator>(
            custom_allocator,
            alignment,
            chunk_size,
            allow_free_idle_chunk_,
            phi::DeviceManager::GetExtraPaddingSize(p));
  }

  void InitAutoGrowthCustomDeviceAllocator(phi::CustomPlace p,
                                           bool allow_free_idle_chunk) {
    auto chunk_size = FLAGS_auto_growth_chunk_size_in_mb << 20;
    VLOG(4) << "FLAGS_auto_growth_chunk_size_in_mb is "
            << FLAGS_auto_growth_chunk_size_in_mb;
    auto custom_allocator =
        std::make_shared<paddle::memory::allocation::CustomAllocator>(p);
    allocators_[p] = std::make_shared<AutoGrowthBestFitAllocator>(
        custom_allocator,
        phi::DeviceManager::GetMinChunkSize(p),
        /*chunk_size=*/chunk_size,
        allow_free_idle_chunk,
        phi::DeviceManager::GetExtraPaddingSize(p));
  }

  void WrapStreamSafeCustomDeviceAllocatorForDefault() {
    for (auto& pair : allocators_) {
      auto& place = pair.first;
      if (phi::is_custom_place(place)) {
        std::shared_ptr<StreamSafeCustomDeviceAllocator>&& allocator =
            std::make_shared<StreamSafeCustomDeviceAllocator>(
                pair.second,
                place,
                /* default_stream = */
                nullptr);
        pair.second = allocator;
        default_stream_safe_custom_device_allocators_[place] = allocator;
        VLOG(8) << "WrapStreamSafeCustomDeviceAllocatorForDefault for " << place
                << ", allocator address = " << pair.second.get();
      }
    }
  }

  void WrapStreamSafeCustomDeviceAllocator(phi::CustomPlace p,
                                           phi::stream::stream_t stream) {
    std::shared_ptr<Allocator>& allocator =
        custom_device_allocators_[p][stream];
    allocator =
        std::make_shared<StreamSafeCustomDeviceAllocator>(allocator, p, stream);
  }
#endif

  void InitSystemAllocators() {
    if (!system_allocators_.empty()) return;
    system_allocators_[phi::CPUPlace()] = std::make_shared<CPUAllocator>();
#ifdef PADDLE_WITH_XPU
    int device_count = platform::GetXPUDeviceCount();
    for (int i = 0; i < device_count; ++i) {
      phi::XPUPlace p(i);
      system_allocators_[p] = CreateXPUAllocator(p);
    }
#endif
#ifdef PADDLE_WITH_IPU
    int device_count = platform::GetIPUDeviceCount();
    for (int i = 0; i < device_count; ++i) {
      phi::IPUPlace p(i);
      system_allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
    }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    system_allocators_[phi::GPUPinnedPlace()] =
        std::make_shared<CPUPinnedAllocator>();
    int device_count = platform::GetGPUDeviceCount();
    for (int i = 0; i < device_count; ++i) {
      phi::GPUPlace p(i);
      system_allocators_[p] = CreateCUDAAllocator(p);
    }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    auto device_types = phi::DeviceManager::GetAllCustomDeviceTypes();
    for (const auto& dev_type : device_types) {
      for (auto& dev_id : phi::DeviceManager::GetSelectedDeviceList(dev_type)) {
        phi::CustomPlace p(dev_type, dev_id);
        system_allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
      }
    }
#endif
  }

  void InitZeroSizeAllocators() {
    if (!zero_size_allocators_.empty()) return;
    std::vector<phi::Place> places;
    places.emplace_back(phi::CPUPlace());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    int device_count = platform::GetGPUDeviceCount();
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      places.emplace_back(phi::GPUPlace(dev_id));
    }
    places.emplace_back(phi::GPUPinnedPlace());
#endif
#ifdef PADDLE_WITH_XPU
    int device_count = platform::GetXPUDeviceCount();
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      places.emplace_back(phi::XPUPlace(dev_id));
    }
#endif
#ifdef PADDLE_WITH_IPU
    int device_count = platform::GetIPUDeviceCount();
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      places.emplace_back(phi::IPUPlace(dev_id));
    }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    auto device_types = phi::DeviceManager::GetAllCustomDeviceTypes();
    for (const auto& dev_type : device_types) {
      for (auto& dev_id : phi::DeviceManager::GetSelectedDeviceList(dev_type)) {
        places.emplace_back(phi::CustomPlace(dev_type, dev_id));
      }
    }
#endif

    for (auto& p : places) {
      zero_size_allocators_[p] = std::make_shared<ZeroSizeAllocator>(p);
    }
  }

  static void CheckAllocThreadSafe(const AllocatorMap& allocators) {
    for (auto& pair : allocators) {
      PADDLE_ENFORCE_EQ(pair.second->IsAllocThreadSafe(),
                        true,
                        common::errors::InvalidArgument(
                            "Public allocators must be thread safe"));
    }
  }

  void CheckAllocThreadSafe() const {
    CheckAllocThreadSafe(allocators_);
    CheckAllocThreadSafe(zero_size_allocators_);
    CheckAllocThreadSafe(system_allocators_);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (is_stream_safe_cuda_allocator_used_) {
      CheckCUDAAllocThreadSafe(cuda_allocators_);
    }
#endif
  }

  void WrapCUDARetryAllocator(size_t retry_time) {
    PADDLE_ENFORCE_GT(
        retry_time,
        0,
        common::errors::InvalidArgument(
            "Retry time should be larger than 0, but got %d", retry_time));
    for (auto& pair : allocators_) {
      if (phi::is_gpu_place(pair.first) || phi::is_xpu_place(pair.first)) {
        pair.second = std::make_shared<RetryAllocator>(pair.second, retry_time);
      }
    }
  }

  void WrapStatAllocator() {
    for (auto& pair : allocators_) {
      // Now memory stats is only supported for CPU, GPU, XPU and CustomDevice
      const phi::Place& place = pair.first;
      if (phi::is_cpu_place(place) || phi::is_cuda_pinned_place(place) ||
          phi::is_gpu_place(place) || phi::is_custom_place(place) ||
          phi::is_xpu_place(place)) {
        pair.second = std::make_shared<StatAllocator>(pair.second);
      }
    }
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // a standalone CUDA allocator to support multi-stream GC in new executor
  std::map<phi::Place, std::shared_ptr<StreamSafeCUDAAllocator>>
      default_stream_safe_cuda_allocators_;
  CUDAAllocatorMap cuda_allocators_;
  std::shared_timed_mutex cuda_allocator_mutex_;
#endif

#if defined(PADDLE_WITH_CUDA)
  std::map<phi::Place, std::shared_ptr<CUDAMallocAsyncAllocator>>
      default_cuda_malloc_async_allocators_;
#endif

#ifdef PADDLE_WITH_XPU
  // a standalone XPU allocator to support multi-stream GC in new executor
  std::map<phi::Place, std::shared_ptr<StreamSafeXPUAllocator>>
      default_stream_safe_xpu_allocators_;
  XPUAllocatorMap xpu_allocators_;
  std::shared_timed_mutex xpu_allocator_mutex_;
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
  // a standalone custom device allocator to support multi-stream GC in new
  // executor
  std::map<phi::Place, std::shared_ptr<StreamSafeCustomDeviceAllocator>>
      default_stream_safe_custom_device_allocators_;
  CustomDeviceAllocatorMap custom_device_allocators_;
  std::shared_timed_mutex custom_device_allocator_mutex_;
#endif

  AllocatorStrategy strategy_;
  AllocatorMap allocators_;
  AllocatorMap auto_growth_allocators_;
  static AllocatorMap zero_size_allocators_;
  static AllocatorMap system_allocators_;
  bool allow_free_idle_chunk_;
  bool is_stream_safe_cuda_allocator_used_;
  bool is_cuda_malloc_async_allocator_used_;
};
AllocatorFacadePrivate::AllocatorMap
    AllocatorFacadePrivate::zero_size_allocators_;
AllocatorFacadePrivate::AllocatorMap AllocatorFacadePrivate::system_allocators_;

// Pimpl. Make interface clean.
AllocatorFacade::AllocatorFacade() : m_(new AllocatorFacadePrivate()) {}
// delete m_ may cause core dump when the destructor of python in conflict with
// cpp.
AllocatorFacade::~AllocatorFacade() = default;

AllocatorFacade& AllocatorFacade::Instance() {
  static AllocatorFacade* instance = new AllocatorFacade;
  return *instance;
}

AllocatorFacadePrivate* AllocatorFacade::GetPrivate() const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // if we use cuda_malloc_async_allocator, we don't need to open a private pool
  // for each graph
  if (UNLIKELY(IsCUDAGraphCapturing()) &&
      !FLAGS_use_cuda_malloc_async_allocator) {
    auto id = phi::backends::gpu::CUDAGraph::CapturingPoolID();
    auto iter = cuda_graph_map_.find(id);
    PADDLE_ENFORCE_NE(
        iter,
        cuda_graph_map_.end(),
        common::errors::PermissionDenied(
            "No memory pool is prepared for CUDA Graph capturing."));
    VLOG(10) << "Choose CUDA Graph memory pool";
    return iter->second.get();
  }
#endif
  return m_;
}

const std::shared_ptr<Allocator>& AllocatorFacade::GetAllocator(
    const phi::Place& place) {
  return GetPrivate()->GetAllocator(
      place, /* A non-zero num to choose allocator_ */ 1);
}

const std::shared_ptr<Allocator>& AllocatorFacade::GetAutoGrowthAllocator(
    const phi::Place& place) {
  return GetPrivate()->GetAutoGrowthAllocatorMap().at(place);
}

void* AllocatorFacade::GetBasePtr(
    const std::shared_ptr<phi::Allocation>& allocation) {
  PADDLE_ENFORCE_EQ(GetAllocatorStrategy(),
                    AllocatorStrategy::kAutoGrowth,
                    common::errors::Unimplemented(
                        "GetBasePtr() is only implemented for auto_growth "
                        "strategy, not support allocator strategy: %d",
                        static_cast<int>(GetAllocatorStrategy())));
  PADDLE_ENFORCE_EQ(phi::is_gpu_place(allocation->place()),
                    true,
                    common::errors::Unimplemented(
                        "GetBasePtr() is only implemented for CUDAPlace(), not "
                        "support place: %s",
                        allocation->place()));
  return GetPrivate()->GetBasePtr(allocation);
}

const std::shared_ptr<Allocator>& AllocatorFacade::GetZeroAllocator(
    const phi::Place& place) {
  return GetPrivate()->GetAllocator(place, /* zero size */ 0);
}

std::shared_ptr<phi::Allocation> AllocatorFacade::AllocShared(
    const phi::Place& place, size_t size) {
  return std::shared_ptr<phi::Allocation>(Alloc(place, size));
}

AllocationPtr AllocatorFacade::Alloc(const phi::Place& place, size_t size) {
  return GetPrivate()->GetAllocator(place, size)->Allocate(size);
}

uint64_t AllocatorFacade::Release(const phi::Place& place) {
  return GetPrivate()
      ->GetAllocator(place, /* A non-zero num to choose allocator_ */ 1)
      ->Release(place);
}

std::shared_ptr<phi::Allocation> AllocatorFacade::AllocShared(
    const phi::Place& place, size_t size, const phi::Stream& stream) {
  return std::shared_ptr<phi::Allocation>(Alloc(place, size, stream));
}

AllocationPtr AllocatorFacade::Alloc(const phi::Place& place,
                                     size_t size,
                                     const phi::Stream& stream) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  if (phi::is_custom_place(place)) {
    if (!GetPrivate()->IsStreamSafeCUDAAllocatorUsed()) {
      VLOG(6) << "Warning: StreamSafeCustomDeviceAllocator is not used!";
      return Alloc(place, size);
    }
    phi::CustomPlace p(place);
    if (LIKELY(size > 0 && FLAGS_use_system_allocator == false)) {
      phi::stream::stream_t s =
          reinterpret_cast<phi::stream::stream_t>(stream.id());
      return GetPrivate()
          ->GetAllocator(p, s, /* create_if_not_found = */ true)
          ->Allocate(size);
    } else {
      return GetPrivate()->GetAllocator(p, size)->Allocate(size);
    }
  }
#endif
#if defined(PADDLE_WITH_XPU)
  if (phi::is_xpu_place(place)) {
    if (!GetPrivate()->IsStreamSafeCUDAAllocatorUsed()) {
      return Alloc(place, size);
    }
    phi::XPUPlace p(place);
    if (LIKELY(size > 0 && FLAGS_use_system_allocator == false)) {
      XPUStream s = reinterpret_cast<XPUStream>(stream.id());
      return GetPrivate()
          ->GetAllocator(p, s, /* create_if_not_found = */ true)
          ->Allocate(size);
    } else {
      return GetPrivate()->GetAllocator(p, size)->Allocate(size);
    }
  }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  AllocatorFacadePrivate* m = GetPrivate();
  if (!m->IsStreamSafeCUDAAllocatorUsed() &&
      !m->IsCUDAMallocAsyncAllocatorUsed()) {
    VLOG(6) << "Warning: StreamSafeCUDAAllocator and CUDAMallocAsyncAllocator "
               "are not used!";
    return Alloc(place, size);
  }

  phi::GPUPlace p(place.GetDeviceId());
  if (LIKELY(size > 0 && FLAGS_use_system_allocator == false)) {
    gpuStream_t s = reinterpret_cast<gpuStream_t>(stream.id());  // NOLINT
    return m->GetAllocator(p, s, /* create_if_not_found = */ true)
        ->Allocate(size);
  } else {
    return m->GetAllocator(p, size)->Allocate(size);
  }
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "Not compiled with GPU or XPU or CustomDevice."));
#endif
}

bool AllocatorFacade::InSameStream(
    const std::shared_ptr<phi::Allocation>& allocation,
    const phi::Stream& stream) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  gpuStream_t s = reinterpret_cast<gpuStream_t>(stream.id());  // NOLINT
  return s == GetStream(allocation);
#else
  PADDLE_THROW(common::errors::PreconditionNotMet("Not compiled with GPU."));
#endif
}

bool AllocatorFacade::IsStreamSafeCUDAAllocatorUsed() {
  return GetPrivate()->IsStreamSafeCUDAAllocatorUsed();
}

bool AllocatorFacade::IsCUDAMallocAsyncAllocatorUsed() {
  return GetPrivate()->IsCUDAMallocAsyncAllocatorUsed();
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
uint64_t AllocatorFacade::Release(const phi::GPUPlace& place,
                                  gpuStream_t stream) {
  AllocatorFacadePrivate* m = GetPrivate();
  if (!m->IsStreamSafeCUDAAllocatorUsed() &&
      !m->IsCUDAMallocAsyncAllocatorUsed()) {
    VLOG(6) << "Warning: StreamSafeCUDAAllocator and CUDAMallocAsyncAllocator "
               "are not used!";
    return Release(place);
  }

  return m->GetAllocator(place, stream)->Release(place);
}

bool AllocatorFacade::RecordStream(std::shared_ptr<phi::Allocation> allocation,
                                   gpuStream_t stream) {
  return GetPrivate()->RecordStream(allocation, stream);
}

void AllocatorFacade::EraseStream(std::shared_ptr<phi::Allocation> allocation,
                                  gpuStream_t stream) {
  GetPrivate()->EraseStream(allocation, stream);
}

const std::shared_ptr<Allocator>& AllocatorFacade::GetAllocator(
    const phi::Place& place, gpuStream_t stream) {
  AllocatorFacadePrivate* m = GetPrivate();

  if (!m->IsStreamSafeCUDAAllocatorUsed() &&
      !m->IsCUDAMallocAsyncAllocatorUsed()) {
    VLOG(6) << "Warning: StreamSafeCUDAAllocator and CUDAMallocAsyncAllocator "
               "are not used!";
    return GetAllocator(place);
  }

  if (phi::is_gpu_place(place) && FLAGS_use_system_allocator == false) {
    return m->GetAllocator(place,
                           stream,
                           /*create_if_not_found=*/true);
  }
  return m->GetAllocator(place, /* A non-zero num to choose allocator_ */ 1);
}

gpuStream_t AllocatorFacade::GetStream(
    const std::shared_ptr<phi::Allocation>& allocation) const {
  return GetPrivate()->GetStream(allocation);
}

void AllocatorFacade::SetDefaultStream(const phi::GPUPlace& place,
                                       gpuStream_t stream) {
  VLOG(8) << "Set default stream to " << stream << " for AllocatorFacade in "
          << place;
  if (m_->IsStreamSafeCUDAAllocatorUsed() ||
      m_->IsCUDAMallocAsyncAllocatorUsed()) {
    m_->SetDefaultStream(place, stream);
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void AllocatorFacade::PrepareMemoryPoolForCUDAGraph(int64_t id) {
  PADDLE_ENFORCE_EQ(GetAllocatorStrategy(),
                    AllocatorStrategy::kAutoGrowth,
                    common::errors::InvalidArgument(
                        "CUDA Graph is only supported when the "
                        "FLAGS_allocator_strategy=\"auto_growth\", but got "
                        "FLAGS_allocator_strategy=\"%s\"",
                        FLAGS_allocator_strategy));
  auto& allocator = cuda_graph_map_[id];
  auto& ref_cnt = cuda_graph_ref_cnt_[id];
  ++ref_cnt;

  if (FLAGS_use_cuda_malloc_async_allocator) return;
  if (allocator.get() == nullptr) {
    allocator = std::make_unique<AllocatorFacadePrivate>(
        /*allow_free_idle_chunk=*/false);
    VLOG(10) << "Create memory pool for CUDA Graph with memory ID " << id;
  } else {
    VLOG(10) << "Use created memory pool for CUDA Graph with memory ID " << id;
  }
}

void AllocatorFacade::RemoveMemoryPoolOfCUDAGraph(int64_t id) {
  auto ref_cnt_iter = cuda_graph_ref_cnt_.find(id);
  PADDLE_ENFORCE_NE(ref_cnt_iter,
                    cuda_graph_ref_cnt_.end(),
                    common::errors::InvalidArgument(
                        "Cannot find CUDA Graph with memory ID = %d", id));
  auto& ref_cnt = ref_cnt_iter->second;
  --ref_cnt;
  if (ref_cnt == 0) {
    cuda_graph_map_.erase(id);
    cuda_graph_ref_cnt_.erase(ref_cnt_iter);
  } else {
    VLOG(10) << "Decrease memory pool ID " << id << " reference count to be "
             << ref_cnt;
  }
}
#endif
#elif defined(PADDLE_WITH_XPU)
const std::shared_ptr<Allocator>& AllocatorFacade::GetAllocator(
    const phi::Place& place, XPUStream stream) {
  AllocatorFacadePrivate* m = GetPrivate();

  // The XPU currently does not have the concept of MallocAsyncAllocatorUsed
  // and shares the logic of IsStreamSafeCUDAAllocatorUsed.
  if (!m->IsStreamSafeCUDAAllocatorUsed()) {
    VLOG(6) << "Warning: StreamSafeCUDAAllocator "
               "are not used!";
    return GetAllocator(place);
  }

  if (phi::is_xpu_place(place) && FLAGS_use_system_allocator == false) {
    return m->GetAllocator(place,
                           stream,
                           /*create_if_not_found=*/true);
  }
  return m->GetAllocator(place, /* A non-zero num to choose allocator_ */ 1);
}
void AllocatorFacade::SetDefaultStream(const phi::XPUPlace& place,
                                       XPUStream stream) {
  if (m_->IsStreamSafeCUDAAllocatorUsed()) {
    m_->SetDefaultStream(place, stream);
  }
}
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
uint64_t AllocatorFacade::Release(const phi::CustomPlace& place,
                                  phi::stream::stream_t stream) {
  AllocatorFacadePrivate* m = GetPrivate();
  if (!m->IsStreamSafeCUDAAllocatorUsed()) {
    VLOG(6) << "Warning: StreamSafeCustomDeviceAllocator is not used!";
    return Release(place);
  }

  return m->GetAllocator(place, stream)->Release(place);
}

bool AllocatorFacade::RecordStream(std::shared_ptr<phi::Allocation> allocation,
                                   phi::stream::stream_t stream) {
  return GetPrivate()->RecordStream(allocation, stream);
}

const std::shared_ptr<Allocator>& AllocatorFacade::GetAllocator(
    const phi::Place& place, phi::stream::stream_t stream) {
  AllocatorFacadePrivate* m = GetPrivate();

  if (!m->IsStreamSafeCUDAAllocatorUsed()) {
    VLOG(6) << "Warning: StreamSafeCustomDeviceAllocator is not used!";
    return GetAllocator(place);
  }

  if (phi::is_custom_place(place) && FLAGS_use_system_allocator == false) {
    return m->GetAllocator(place,
                           stream,
                           /*create_if_not_found=*/true);
  }
  return m->GetAllocator(place, /* A non-zero num to choose allocator_ */ 1);
}

phi::stream::stream_t AllocatorFacade::GetStream(
    const std::shared_ptr<phi::Allocation>& allocation) const {
  return GetPrivate()->GetStream(allocation);
}

void AllocatorFacade::SetDefaultStream(const phi::CustomPlace& place,
                                       phi::stream::stream_t stream) {
  if (m_->IsStreamSafeCUDAAllocatorUsed()) {
    m_->SetDefaultStream(place, stream);
  }
}

#endif

UNUSED static std::shared_ptr<NaiveBestFitAllocator> unused_obj =
    std::make_shared<NaiveBestFitAllocator>(phi::CPUPlace());

}  // namespace paddle::memory::allocation
