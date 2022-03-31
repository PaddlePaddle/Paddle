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
#include "paddle/fluid/memory/allocation/aligned_allocator.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "paddle/fluid/memory/allocation/auto_growth_best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/cpu_allocator.h"
#include "paddle/fluid/memory/allocation/naive_best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/retry_allocator.h"
#include "paddle/fluid/memory/allocation/stat_allocator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <shared_mutex>
#include "paddle/fluid/memory/allocation/cuda_allocator.h"
#include "paddle/fluid/memory/allocation/cuda_managed_allocator.h"
#include "paddle/fluid/memory/allocation/pinned_allocator.h"
#include "paddle/fluid/memory/allocation/stream_safe_cuda_allocator.h"
#include "paddle/fluid/memory/allocation/thread_local_allocator.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/device/gpu/cuda/cuda_graph.h"
#endif

#if CUDA_VERSION >= 10020
#include "paddle/fluid/memory/allocation/cuda_virtual_mem_allocator.h"
#include "paddle/fluid/memory/allocation/virtual_memory_auto_growth_best_fit_allocator.h"
#include "paddle/fluid/platform/dynload/cuda_driver.h"
#endif
#endif

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#endif

#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/memory/allocation/npu_pinned_allocator.h"
#endif

#ifdef PADDLE_WITH_IPU
#include "paddle/fluid/platform/device/ipu/ipu_info.h"
#endif

#ifdef PADDLE_WITH_MLU
#include "paddle/fluid/platform/device/mlu/mlu_info.h"
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/fluid/memory/allocation/custom_allocator.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#endif

PADDLE_DEFINE_EXPORTED_int64(
    gpu_allocator_retry_time, 10000,
    "The retry time (milliseconds) when allocator fails "
    "to allocate memory. No retry if this value is not greater than 0");

PADDLE_DEFINE_EXPORTED_bool(
    use_system_allocator, false,
    "Whether to use system allocator to allocate CPU and GPU memory. "
    "Only used for unittests.");

PADDLE_DEFINE_EXPORTED_bool(use_virtual_memory_auto_growth, false,
                            "Use VirtualMemoryAutoGrowthBestFitAllocator.");

// NOTE(Ruibiao): This FLAGS is just to be compatibled with
// the old single-stream CUDA allocator. It will be removed
// after StreamSafeCudaAllocator has been fully tested.
PADDLE_DEFINE_EXPORTED_bool(use_stream_safe_cuda_allocator, false,
                            "Enable StreamSafeCUDAAllocator");

PADDLE_DEFINE_EXPORTED_bool(use_cuda_managed_memory, false,
                            "Whether to use CUDAManagedAllocator to allocate "
                            "managed memory, only available for auto_growth "
                            "strategy");

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
                      DecoratedAllocationPtr underlying_allocation)
        : Allocation(
              underlying_allocation->ptr(), underlying_allocation->base_ptr(),
              underlying_allocation->size(), underlying_allocation->place()),
          allocator_(allocator->shared_from_this()),
          underlying_allocation_(std::move(underlying_allocation)) {}

   private:
    std::shared_ptr<Allocator> allocator_;
    DecoratedAllocationPtr underlying_allocation_;
  };

  explicit CUDAGraphAllocator(const std::shared_ptr<Allocator>& allocator)
      : underlying_allocator_(allocator) {}

 public:
  static std::shared_ptr<Allocator> Create(
      const std::shared_ptr<Allocator>& allocator) {
    return std::shared_ptr<Allocator>(new CUDAGraphAllocator(allocator));
  }

 protected:
  phi::Allocation* AllocateImpl(size_t size) {
    VLOG(10) << "Allocate " << size << " for CUDA Graph";
    return new PrivateAllocation(this,
                                 static_unique_ptr_cast<Allocation>(
                                     underlying_allocator_->Allocate(size)));
  }

  void FreeImpl(phi::Allocation* allocation) {
    VLOG(10) << "delete for CUDA Graph";
    delete allocation;
  }

 private:
  std::shared_ptr<Allocator> underlying_allocator_;
};
#endif

static bool IsCUDAGraphCapturing() {
#ifdef PADDLE_WITH_CUDA
  return UNLIKELY(platform::CUDAGraph::IsThisThreadCapturing());
#else
  return false;
#endif
}

class AllocatorFacadePrivate {
 public:
  using AllocatorMap = std::map<platform::Place, std::shared_ptr<Allocator>>;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  using CUDAAllocatorMap =
      std::map<platform::CUDAPlace,
               std::map<gpuStream_t, std::shared_ptr<Allocator>>>;
#endif

  explicit AllocatorFacadePrivate(bool allow_free_idle_chunk = true) {
    strategy_ = GetAllocatorStrategy();
    is_stream_safe_cuda_allocator_used_ = false;

    switch (strategy_) {
      case AllocatorStrategy::kNaiveBestFit: {
        InitNaiveBestFitCPUAllocator();
#ifdef PADDLE_WITH_IPU
        for (int dev_id = 0; dev_id < platform::GetIPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitIPUAllocator(platform::IPUPlace(dev_id));
        }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        for (int dev_id = 0; dev_id < platform::GetGPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitCUDAAllocator(platform::CUDAPlace(dev_id));
        }
        InitNaiveBestFitCUDAPinnedAllocator();
#endif
#ifdef PADDLE_WITH_XPU
        for (int dev_id = 0; dev_id < platform::GetXPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitXPUAllocator(platform::XPUPlace(dev_id));
        }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
        for (int dev_id = 0; dev_id < platform::GetNPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitNPUAllocator(platform::NPUPlace(dev_id));
        }
        InitNaiveBestFitNPUPinnedAllocator();
#endif
#ifdef PADDLE_WITH_MLU
        for (int dev_id = 0; dev_id < platform::GetMLUDeviceCount(); ++dev_id) {
          InitNaiveBestFitMLUAllocator(platform::MLUPlace(dev_id));
        }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
        auto device_types = phi::DeviceManager::GetAllCustomDeviceTypes();
        for (const auto& dev_type : device_types) {
          for (size_t dev_id = 0;
               dev_id < phi::DeviceManager::GetDeviceCount(dev_type);
               ++dev_id) {
            InitNaiveBestFitCustomDeviceAllocator(
                platform::CustomPlace(dev_type, dev_id));
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
          InitAutoGrowthCUDAAllocator(platform::CUDAPlace(dev_id),
                                      allow_free_idle_chunk_);
        }

        // Note(Ruibiao): For GPU multi-stream case without CUDA graph
        // capturing, the 'allocators_' map(place -> Allocator) hold the
        // StreamSafeCUDAAllocator releate to defaultstream (i.e., the stream
        // directly got from DeviceContex), while the 'cuda_allocators_' map
        // (place -> map(stream -> Allocator)) hold the StreamSafeCUDAAllocator
        // releate to non-default stream (i.e., the stream users pass in). The
        // default stream Allocator is built in the structure of
        // AllocatorFacadePrivate, while the non-default stream is build in a
        // manner in GetAllocator function with 'create_if_not_found = ture'.
        // We make special treatment for the default stream for performance
        // reasons. Since most Alloc calls are for default stream in
        // application, treating it separately can avoid lots of overhead of
        // acquiring default stream and applying read-write lock.
        if (FLAGS_use_stream_safe_cuda_allocator) {
          if (LIKELY(!IsCUDAGraphCapturing())) {
            WrapStreamSafeCUDAAllocatorForDefault();
          }
          is_stream_safe_cuda_allocator_used_ = true;
        }

        InitNaiveBestFitCUDAPinnedAllocator();
#endif
#ifdef PADDLE_WITH_ASCEND_CL
        for (int dev_id = 0; dev_id < platform::GetNPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitNPUAllocator(platform::NPUPlace(dev_id));
        }
        InitNaiveBestFitNPUPinnedAllocator();
#endif
#ifdef PADDLE_WITH_XPU
        for (int dev_id = 0; dev_id < platform::GetXPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitXPUAllocator(platform::XPUPlace(dev_id));
        }
#endif
#ifdef PADDLE_WITH_IPU
        for (int dev_id = 0; dev_id < platform::GetIPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitIPUAllocator(platform::IPUPlace(dev_id));
        }
#endif
#ifdef PADDLE_WITH_MLU
        for (int dev_id = 0; dev_id < platform::GetMLUDeviceCount(); ++dev_id) {
          InitNaiveBestFitMLUAllocator(platform::MLUPlace(dev_id));
        }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
        auto device_types = phi::DeviceManager::GetAllCustomDeviceTypes();
        for (const auto& dev_type : device_types) {
          for (size_t dev_id = 0;
               dev_id < phi::DeviceManager::GetDeviceCount(dev_type);
               ++dev_id) {
            InitAutoGrowthCustomDeviceAllocator(
                platform::CustomPlace(dev_type, dev_id), allow_free_idle_chunk);
          }
        }
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
#ifdef PADDLE_WITH_IPU
        for (int dev_id = 0; dev_id < platform::GetIPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitIPUAllocator(platform::IPUPlace(dev_id));
        }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        for (int dev_id = 0; dev_id < platform::GetGPUDeviceCount(); ++dev_id) {
          InitThreadLocalCUDAAllocator(platform::CUDAPlace(dev_id));
        }
        InitNaiveBestFitCUDAPinnedAllocator();
#endif
#ifdef PADDLE_WITH_MLU
        for (int dev_id = 0; dev_id < platform::GetMLUDeviceCount(); ++dev_id) {
          InitNaiveBestFitMLUAllocator(platform::MLUPlace(dev_id));
        }
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

    WrapStatAllocator();

    CheckAllocThreadSafe();

#ifdef PADDLE_WITH_CUDA
    // No need to wrap CUDAGraphAllocator for StreamSafeCUDAAllocator
    if (!is_stream_safe_cuda_allocator_used_ &&
        UNLIKELY(IsCUDAGraphCapturing())) {
      WrapCUDAGraphAllocator();
    }
#endif
  }

  inline const std::shared_ptr<Allocator>& GetAllocator(
      const platform::Place& place, size_t size) {
    VLOG(6) << "GetAllocator"
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

  void* GetBasePtr(const std::shared_ptr<phi::Allocation>& allocation) {
    return static_cast<Allocation*>(allocation.get())->base_ptr();
  }

  bool IsStreamSafeCUDAAllocatorUsed() {
    return is_stream_safe_cuda_allocator_used_ &&
           LIKELY(FLAGS_use_system_allocator == false);
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  bool HasCUDAAllocator(const platform::CUDAPlace& place,
                        const gpuStream_t& stream) {
    auto it = cuda_allocators_.find(place);
    if (it == cuda_allocators_.end()) {
      return false;
    }
    const std::map<gpuStream_t, std::shared_ptr<Allocator>>& allocator_map =
        it->second;
    return allocator_map.find(stream) != allocator_map.end();
  }

  const std::shared_ptr<Allocator>& GetAllocator(
      const platform::CUDAPlace& place, const gpuStream_t& stream,
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
        PADDLE_ENFORCE_NE(create_if_not_found, false,
                          platform::errors::NotFound(
                              "No allocator found for stream %s in place %s "
                              "with create_if_not_found = false",
                              stream, place));
      }
    }

    /* unique_lock_guard */ {
      std::unique_lock<std::shared_timed_mutex> lock_guard(
          cuda_allocator_mutex_);
      InitStreamSafeCUDAAllocator(place, stream);
      return cuda_allocators_[place][stream];
    }
  }

  const std::shared_ptr<StreamSafeCUDAAllocator>
  GetDefaultStreamSafeCUDAAllocator(const platform::CUDAPlace& place) const {
    const auto iter = default_stream_safe_cuda_allocators_.find(place);
    PADDLE_ENFORCE_NE(
        iter, default_stream_safe_cuda_allocators_.end(),
        platform::errors::NotFound(
            "No StreamSafeCUDAAllocator found for the place, %s", place));
    return iter->second;
  }

  const gpuStream_t& GetDefaultStream(const platform::CUDAPlace& place) const {
    const std::shared_ptr<StreamSafeCUDAAllocator>& allocator =
        GetDefaultStreamSafeCUDAAllocator(place);
    return allocator->GetDefaultStream();
  }

  void SetDefaultStream(const platform::CUDAPlace& place,
                        const gpuStream_t& stream) {
    const std::shared_ptr<StreamSafeCUDAAllocator>& allocator =
        GetDefaultStreamSafeCUDAAllocator(place);
    allocator->SetDefaultStream(stream);
    VLOG(8) << "Set default stream to " << stream
            << " for StreamSafeCUDAAllocator(" << allocator.get() << ") in "
            << place;
  }

  void RecordStream(std::shared_ptr<phi::Allocation> allocation,
                    const gpuStream_t& stream) {
    std::shared_ptr<StreamSafeCUDAAllocation> stream_safe_cuda_allocation =
        std::dynamic_pointer_cast<StreamSafeCUDAAllocation>(allocation);
    if (stream_safe_cuda_allocation != nullptr) {
      stream_safe_cuda_allocation->RecordStream(stream);
    } else {
      VLOG(6) << "RecordStream for a non-StreamSafeCUDAAllocation";
    }
  }

  const gpuStream_t GetStream(
      const std::shared_ptr<phi::Allocation>& allocation) const {
    const std::shared_ptr<StreamSafeCUDAAllocation>
        stream_safe_cuda_allocation =
            std::dynamic_pointer_cast<StreamSafeCUDAAllocation>(allocation);
    if (stream_safe_cuda_allocation != nullptr) {
      return stream_safe_cuda_allocation->GetOwningStream();
    }

    VLOG(6) << "GetStream for a non-StreamSafeCUDAAllocation";
    return static_cast<phi::GPUContext*>(
               platform::DeviceContextPool::Instance().Get(allocation->place()))
        ->stream();
  }
#endif

 private:
  class ZeroSizeAllocator : public Allocator {
   public:
    explicit ZeroSizeAllocator(platform::Place place) : place_(place) {}
    bool IsAllocThreadSafe() const override { return true; }

   protected:
    phi::Allocation* AllocateImpl(size_t size) override {
      return new Allocation(nullptr, 0, place_);
    }
    void FreeImpl(phi::Allocation* allocation) override { delete allocation; }

   private:
    platform::Place place_;
  };

  const AllocatorMap& GetAllocatorMap() { return allocators_; }

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

  // Create a new CUDAAllocator or CUDAManagedAllocator for the given device
  std::shared_ptr<Allocator> CreateCUDAAllocator(platform::CUDAPlace p) {
    if (FLAGS_use_cuda_managed_memory) {
      PADDLE_ENFORCE_EQ(
          strategy_, AllocatorStrategy::kAutoGrowth,
          platform::errors::InvalidArgument(
              "CUDA managed memory is only implemented for auto_growth "
              "strategy, not support %s strategy.\n"
              "Please use auto_growth strategy by command `export "
              "FLAGS_allocator_strategy=\"auto_growth\"`, or disable managed "
              "memory by command `export FLAGS_use_cuda_managed_memory=false`",
              FLAGS_allocator_strategy));

      if (!platform::IsGPUManagedMemorySupported(p.device)) {
        PADDLE_THROW(platform::errors::Unavailable(
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

  void InitStreamSafeCUDAAllocator(platform::CUDAPlace p, gpuStream_t stream) {
    PADDLE_ENFORCE_EQ(
        strategy_, AllocatorStrategy::kAutoGrowth,
        platform::errors::Unimplemented(
            "Only support auto-growth strategey for StreamSafeCUDAAllocator, "
            "the allocator strategy %d is unsupported for multi-stream",
            static_cast<int>(strategy_)));
    if (LIKELY(!HasCUDAAllocator(p, stream))) {
      VLOG(8) << "Init CUDA allocator for stream " << stream << " in place "
              << p;
      InitAutoGrowthCUDAAllocator(p, stream);
      WrapStreamSafeCUDAAllocator(p, stream);
      WrapCUDARetryAllocator(p, stream, FLAGS_gpu_allocator_retry_time);
      WrapStatAllocator(p, stream);
    }
  }

  void InitAutoGrowthCUDAAllocator(platform::CUDAPlace p, gpuStream_t stream) {
#if defined(PADDLE_WITH_HIP)
    auto cuda_allocator = CreateCUDAAllocator(p);
    cuda_allocators_[p][stream] = std::make_shared<AutoGrowthBestFitAllocator>(
        cuda_allocator, platform::GpuMinChunkSize(), 0, allow_free_idle_chunk_);
#endif

#if defined(PADDLE_WITH_CUDA)
#if CUDA_VERSION >= 10020
    CUdevice device;
    int val;
    try {
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::cuDeviceGet(&device, p.GetDeviceId()));

      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::cuDeviceGetAttribute(
              &val, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
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
      cuda_allocators_[p][stream] =
          std::make_shared<AutoGrowthBestFitAllocator>(
              cuda_allocator, platform::GpuMinChunkSize(),
              allow_free_idle_chunk_);
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

    cuda_allocators_[p][stream] = std::make_shared<AutoGrowthBestFitAllocator>(
        underlying_allocator, alignment, 0, allow_free_idle_chunk_);
#endif
#endif
  }

  // NOTE(Ruibiao): Old single-stream version, will be removed later
  void InitAutoGrowthCUDAAllocator(platform::CUDAPlace p,
                                   bool allow_free_idle_chunk) {
#if defined(PADDLE_WITH_HIP)
    auto cuda_allocator = CreateCUDAAllocator(p);
    allocators_[p] = std::make_shared<AutoGrowthBestFitAllocator>(
        cuda_allocator, platform::GpuMinChunkSize(), allow_free_idle_chunk);
#endif

#if defined(PADDLE_WITH_CUDA)
#if CUDA_VERSION >= 10020
    CUdevice device;
    int val;
    try {
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::cuDeviceGet(&device, p.GetDeviceId()));

      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::cuDeviceGetAttribute(
              &val, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
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
      allocators_[p] = std::make_shared<AutoGrowthBestFitAllocator>(
          cuda_allocator, platform::GpuMinChunkSize(), allow_free_idle_chunk);
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
    allocators_[p] = std::make_shared<AutoGrowthBestFitAllocator>(
        underlying_allocator, alignment, 0, allow_free_idle_chunk);
#endif
#endif
  }

  void InitThreadLocalCUDAAllocator(platform::CUDAPlace p) {
    allocators_[p] = std::make_shared<ThreadLocalCUDAAllocator>(p);
  }

  void WrapStreamSafeCUDAAllocator(platform::CUDAPlace p, gpuStream_t stream) {
    std::shared_ptr<Allocator>& allocator = cuda_allocators_[p][stream];
    allocator = std::make_shared<StreamSafeCUDAAllocator>(
        allocator, p, stream,
        /* in_cuda_graph_capturing = */ !allow_free_idle_chunk_);
  }

  void WrapStreamSafeCUDAAllocatorForDefault() {
    for (auto& pair : allocators_) {
      auto& place = pair.first;
      if (platform::is_gpu_place(place)) {
        std::shared_ptr<StreamSafeCUDAAllocator>&& allocator =
            std::make_shared<StreamSafeCUDAAllocator>(
                pair.second, place, /* default_stream = */ nullptr,
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

  void WrapCUDARetryAllocator(platform::CUDAPlace p, gpuStream_t stream,
                              size_t retry_time) {
    PADDLE_ENFORCE_GT(
        retry_time, 0,
        platform::errors::InvalidArgument(
            "Retry time should be larger than 0, but got %d", retry_time));
    std::shared_ptr<Allocator>& allocator = cuda_allocators_[p][stream];
    allocator = std::make_shared<RetryAllocator>(allocator, retry_time);
  }

  void WrapStatAllocator(platform::CUDAPlace p, gpuStream_t stream) {
    std::shared_ptr<Allocator>& allocator = cuda_allocators_[p][stream];
    allocator = std::make_shared<StatAllocator>(allocator);
  }

#ifdef PADDLE_WITH_CUDA
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
        PADDLE_ENFORCE_EQ(stream_pair.second->IsAllocThreadSafe(), true,
                          platform::errors::InvalidArgument(
                              "Public allocators must be thread safe"));
      }
    }
  }
#endif

#ifdef PADDLE_WITH_XPU
  void InitNaiveBestFitXPUAllocator(platform::XPUPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }
#endif

#ifdef PADDLE_WITH_IPU
  void InitNaiveBestFitIPUAllocator(platform::IPUPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }
#endif

#ifdef PADDLE_WITH_MLU
  void InitNaiveBestFitMLUAllocator(platform::MLUPlace p) {
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

#ifdef PADDLE_WITH_CUSTOM_DEVICE
  void InitNaiveBestFitCustomDeviceAllocator(platform::CustomPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }

  void InitAutoGrowthCustomDeviceAllocator(platform::CustomPlace p,
                                           bool allow_free_idle_chunk) {
    auto custom_allocator =
        std::make_shared<paddle::memory::allocation::CustomAllocator>(p);
    allocators_[p] = std::make_shared<AutoGrowthBestFitAllocator>(
        custom_allocator, phi::DeviceManager::GetMinChunkSize(p),
        allow_free_idle_chunk);
  }
#endif

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
#ifdef PADDLE_WITH_IPU
    int device_count = platform::GetIPUDeviceCount();
    for (int i = 0; i < device_count; ++i) {
      platform::IPUPlace p(i);
      system_allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
    }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    system_allocators_[platform::CUDAPinnedPlace()] =
        std::make_shared<CPUPinnedAllocator>();
    int device_count = platform::GetGPUDeviceCount();
    for (int i = 0; i < device_count; ++i) {
      platform::CUDAPlace p(i);
      system_allocators_[p] = CreateCUDAAllocator(p);
    }
#endif
#ifdef PADDLE_WITH_MLU
    int device_count = platform::GetMLUDeviceCount();
    for (int i = 0; i < device_count; ++i) {
      platform::MLUPlace p(i);
      system_allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
    }
#endif
  }

  void InitZeroSizeAllocators() {
    if (!zero_size_allocators_.empty()) return;
    std::vector<platform::Place> places;
    places.emplace_back(platform::CPUPlace());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    int device_count = platform::GetGPUDeviceCount();
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
#ifdef PADDLE_WITH_IPU
    int device_count = platform::GetIPUDeviceCount();
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      places.emplace_back(platform::IPUPlace(dev_id));
    }
#endif
#ifdef PADDLE_WITH_MLU
    int device_count = platform::GetMLUDeviceCount();
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      places.emplace_back(platform::MLUPlace(dev_id));
    }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    auto device_types = phi::DeviceManager::GetAllCustomDeviceTypes();
    for (const auto& dev_type : device_types) {
      for (size_t dev_id = 0;
           dev_id < phi::DeviceManager::GetDeviceCount(dev_type); dev_id++) {
        places.emplace_back(platform::CustomPlace(dev_type, dev_id));
      }
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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (is_stream_safe_cuda_allocator_used_) {
      CheckCUDAAllocThreadSafe(cuda_allocators_);
    }
#endif
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

  void WrapStatAllocator() {
    for (auto& pair : allocators_) {
      // Now memory stats is only supported for GPU
      if (platform::is_gpu_place(pair.first)) {
        pair.second = std::make_shared<StatAllocator>(pair.second);
      }
    }
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // a standalone CUDA allocator to support multi-stream GC in new executor
  std::map<platform::Place, std::shared_ptr<StreamSafeCUDAAllocator>>
      default_stream_safe_cuda_allocators_;
  CUDAAllocatorMap cuda_allocators_;
  std::shared_timed_mutex cuda_allocator_mutex_;
#endif
  AllocatorStrategy strategy_;
  AllocatorMap allocators_;
  static AllocatorMap zero_size_allocators_;
  static AllocatorMap system_allocators_;
  bool allow_free_idle_chunk_;
  bool is_stream_safe_cuda_allocator_used_;
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
  static AllocatorFacade* instance = new AllocatorFacade;
  return *instance;
}

AllocatorFacadePrivate* AllocatorFacade::GetPrivate() const {
#ifdef PADDLE_WITH_CUDA
  if (UNLIKELY(IsCUDAGraphCapturing())) {
    auto id = platform::CUDAGraph::CapturingID();
    auto iter = cuda_graph_map_.find(id);
    PADDLE_ENFORCE_NE(
        iter, cuda_graph_map_.end(),
        platform::errors::PermissionDenied(
            "No memory pool is prepared for CUDA Graph capturing."));
    VLOG(10) << "Choose CUDA Graph memory pool";
    return iter->second.get();
  }
#endif
  return m_;
}

const std::shared_ptr<Allocator>& AllocatorFacade::GetAllocator(
    const platform::Place& place) {
  return GetPrivate()->GetAllocator(
      place, /* A non-zero num to choose allocator_ */ 1);
}

void* AllocatorFacade::GetBasePtr(
    const std::shared_ptr<phi::Allocation>& allocation) {
  PADDLE_ENFORCE_EQ(GetAllocatorStrategy(), AllocatorStrategy::kAutoGrowth,
                    paddle::platform::errors::Unimplemented(
                        "GetBasePtr() is only implemented for auto_growth "
                        "strategy, not support allocator strategy: %d",
                        static_cast<int>(GetAllocatorStrategy())));
  PADDLE_ENFORCE_EQ(platform::is_gpu_place(allocation->place()), true,
                    paddle::platform::errors::Unimplemented(
                        "GetBasePtr() is only implemented for CUDAPlace(), not "
                        "suppot place: %s",
                        allocation->place()));
  return GetPrivate()->GetBasePtr(allocation);
}

const std::shared_ptr<Allocator>& AllocatorFacade::GetZeroAllocator(
    const platform::Place& place) {
  return GetPrivate()->GetAllocator(place, /* zero size */ 0);
}

std::shared_ptr<phi::Allocation> AllocatorFacade::AllocShared(
    const platform::Place& place, size_t size) {
  return std::shared_ptr<phi::Allocation>(Alloc(place, size));
}

AllocationPtr AllocatorFacade::Alloc(const platform::Place& place,
                                     size_t size) {
  return GetPrivate()->GetAllocator(place, size)->Allocate(size);
}

uint64_t AllocatorFacade::Release(const platform::Place& place) {
  return GetPrivate()
      ->GetAllocator(place, /* A non-zero num to choose allocator_ */ 1)
      ->Release(place);
}

std::shared_ptr<phi::Allocation> AllocatorFacade::AllocShared(
    const platform::Place& place, size_t size, const phi::Stream& stream) {
  return std::shared_ptr<phi::Allocation>(Alloc(place, size, stream));
}

AllocationPtr AllocatorFacade::Alloc(const platform::Place& place, size_t size,
                                     const phi::Stream& stream) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  AllocatorFacadePrivate* m = GetPrivate();
  if (!m->IsStreamSafeCUDAAllocatorUsed()) {
    VLOG(6) << "Warning: StreamSafeCUDAAllocator is not used!";
    return Alloc(place, size);
  }

  platform::CUDAPlace p(place.GetDeviceId());
  if (LIKELY(size > 0 && FLAGS_use_system_allocator == false)) {
    gpuStream_t s = reinterpret_cast<gpuStream_t>(stream.id());
    return m->GetAllocator(p, s, /* create_if_not_found = */ true)
        ->Allocate(size);
  } else {
    return m->GetAllocator(p, size)->Allocate(size);
  }
#else
  PADDLE_THROW(platform::errors::PreconditionNotMet("Not compiled with GPU."));
#endif
}

bool AllocatorFacade::InSameStream(
    const std::shared_ptr<phi::Allocation>& allocation,
    const phi::Stream& stream) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  gpuStream_t s = reinterpret_cast<gpuStream_t>(stream.id());
  return s == GetStream(allocation);
#else
  PADDLE_THROW(platform::errors::PreconditionNotMet("Not compiled with GPU."));
#endif
}

bool AllocatorFacade::IsStreamSafeCUDAAllocatorUsed() {
  return GetPrivate()->IsStreamSafeCUDAAllocatorUsed();
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
uint64_t AllocatorFacade::Release(const platform::CUDAPlace& place,
                                  const gpuStream_t& stream) {
  AllocatorFacadePrivate* m = GetPrivate();
  if (!m->IsStreamSafeCUDAAllocatorUsed()) {
    VLOG(6) << "Warning: StreamSafeCUDAAllocator is not used!";
    return Release(place);
  }

  return m->GetAllocator(place, stream)->Release(place);
}

void AllocatorFacade::RecordStream(std::shared_ptr<phi::Allocation> allocation,
                                   const gpuStream_t& stream) {
  GetPrivate()->RecordStream(allocation, stream);
}

const std::shared_ptr<Allocator>& AllocatorFacade::GetAllocator(
    const platform::Place& place, const gpuStream_t& stream) {
  AllocatorFacadePrivate* m = GetPrivate();

  if (!m->IsStreamSafeCUDAAllocatorUsed()) {
    VLOG(6) << "Warning: StreamSafeCUDAAllocator is not used!";
    return GetAllocator(place);
  }

  if (platform::is_gpu_place(place) && FLAGS_use_system_allocator == false) {
    return m->GetAllocator(place, stream,
                           /*create_if_not_found=*/true);
  }
  return m->GetAllocator(place, /* A non-zero num to choose allocator_ */ 1);
}

const gpuStream_t AllocatorFacade::GetStream(
    const std::shared_ptr<phi::Allocation>& allocation) const {
  return GetPrivate()->GetStream(allocation);
}

void AllocatorFacade::SetDefaultStream(const platform::CUDAPlace& place,
                                       const gpuStream_t& stream) {
  if (m_->IsStreamSafeCUDAAllocatorUsed()) {
    m_->SetDefaultStream(place, stream);
  }
}

#ifdef PADDLE_WITH_CUDA
void AllocatorFacade::PrepareMemoryPoolForCUDAGraph(CUDAGraphID id) {
  PADDLE_ENFORCE_EQ(GetAllocatorStrategy(), AllocatorStrategy::kAutoGrowth,
                    platform::errors::InvalidArgument(
                        "CUDA Graph is only supported when the "
                        "FLAGS_allocator_strategy=\"auto_growth\", but got "
                        "FLAGS_allocator_strategy=\"%s\"",
                        FLAGS_allocator_strategy));
  auto& allocator = cuda_graph_map_[id];
  PADDLE_ENFORCE_EQ(
      allocator.get(), nullptr,
      platform::errors::InvalidArgument(
          "The memory pool of the CUDA Graph with ID %d have been prepared.",
          id));
  allocator.reset(new AllocatorFacadePrivate(/*allow_free_idle_chunk=*/false));

  VLOG(10) << "Prepare memory pool for CUDA Graph with ID " << id;
}

void AllocatorFacade::RemoveMemoryPoolOfCUDAGraph(CUDAGraphID id) {
  auto iter = cuda_graph_map_.find(id);
  PADDLE_ENFORCE_NE(iter, cuda_graph_map_.end(),
                    platform::errors::InvalidArgument(
                        "Cannot find CUDA Graph with ID = %d", id));
  cuda_graph_map_.erase(iter);
  VLOG(10) << "Remove memory pool of CUDA Graph with ID " << id;
}
#endif
#endif
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
