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
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/memory/allocation/cuda_allocator.h"
#include "paddle/fluid/memory/allocation/pinned_allocator.h"
#include "paddle/fluid/memory/allocation/stream_safe_cuda_allocator.h"
#include "paddle/fluid/memory/allocation/thread_local_allocator.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

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
PADDLE_DEFINE_EXPORTED_bool(use_stream_safe_cuda_allocator, true,
                            "Enable StreamSafeCUDAAllocator");

DECLARE_string(allocator_strategy);

#ifdef PADDLE_WITH_MLU
PADDLE_DEFINE_EXPORTED_int64(
    mlu_allocator_retry_time, 10000,
    "The retry time (milliseconds) when allocator fails "
    "to allocate memory. No retry if this value is not greater than 0");
#endif

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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  using CUDAAllocatorMap =
      std::map<platform::CUDAPlace,
               std::map<gpuStream_t, std::shared_ptr<Allocator>>>;
#endif

  explicit AllocatorFacadePrivate(bool allow_free_idle_chunk = true) {
    strategy_ = GetAllocatorStrategy();
    switch (strategy_) {
      case AllocatorStrategy::kNaiveBestFit: {
        InitNaiveBestFitCPUAllocator();
#ifdef PADDLE_WITH_IPU
        for (int dev_id = 0; dev_id < platform::GetIPUDeviceCount(); ++dev_id) {
          InitNaiveBestFitIPUAllocator(platform::IPUPlace(dev_id));
        }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        if (FLAGS_use_stream_safe_cuda_allocator) {
          LOG(WARNING) << "FLAGS_use_stream_safe_cuda_allocator is invalid for "
                          "naive_best_fit strategy";
          FLAGS_use_stream_safe_cuda_allocator = false;
        }
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
        break;
      }

      case AllocatorStrategy::kAutoGrowth: {
        InitNaiveBestFitCPUAllocator();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        allow_free_idle_chunk_ = allow_free_idle_chunk;
        if (FLAGS_use_stream_safe_cuda_allocator) {
          // TODO(Ruibiao): Support multi-stream allocator for other strategies
          default_stream_ = nullptr;
          for (int dev_id = 0; dev_id < platform::GetGPUDeviceCount();
               ++dev_id) {
            InitStreamSafeCUDAAllocator(platform::CUDAPlace(dev_id),
                                        default_stream_);
          }
        } else {
          for (int dev_id = 0; dev_id < platform::GetGPUDeviceCount();
               ++dev_id) {
            InitAutoGrowthCUDAAllocator(platform::CUDAPlace(dev_id),
                                        allow_free_idle_chunk_);
          }
        }
        InitNaiveBestFitCUDAPinnedAllocator();
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
        if (FLAGS_use_stream_safe_cuda_allocator) {
          LOG(WARNING) << "FLAGS_use_stream_safe_cuda_allocator is invalid for "
                          "thread_local strategy";
          FLAGS_use_stream_safe_cuda_allocator = false;
        }

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

    CheckAllocThreadSafe();
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  const std::shared_ptr<Allocator>& GetAllocator(
      const platform::CUDAPlace& place, const gpuStream_t& stream,
      bool create_if_not_found = false) {
    auto place_it = cuda_allocators_.find(place);
    PADDLE_ENFORCE_NE(place_it, cuda_allocators_.end(),
                      platform::errors::NotFound(
                          "No allocator found for the place %s", place));

    const std::map<gpuStream_t, std::shared_ptr<Allocator>>& allocator_map =
        place_it->second;
    auto stream_it = allocator_map.find(stream);
    if (stream_it == allocator_map.end()) {
      if (create_if_not_found) {
        InitStreamSafeCUDAAllocator(place, stream);
        return cuda_allocators_[place][stream];
      } else {
        PADDLE_THROW(platform::errors::NotFound(
            "No allocator found for stream %s in place %s", stream, place));
      }
    }
    return stream_it->second;
  }

  gpuStream_t GetDefaultStream() { return default_stream_; }

  void RecordStream(Allocation* allocation, const gpuStream_t& stream) {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(allocation->place()), true,
        platform::errors::InvalidArgument(
            "Not allow to record stream for an allocation with place %s",
            allocation->place()));
    dynamic_cast<StreamSafeCUDAAllocation*>(allocation)->RecordStream(stream);
  }

#ifdef PADDLE_WITH_CUDA
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
#endif

 private:
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

  const AllocatorMap& GetAllocatorMap() {
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

  void InitNaiveBestFitCPUAllocator() {
    allocators_[platform::CPUPlace()] =
        std::make_shared<NaiveBestFitAllocator>(platform::CPUPlace());
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void InitNaiveBestFitCUDAPinnedAllocator() {
    allocators_[platform::CUDAPinnedPlace()] =
        std::make_shared<NaiveBestFitAllocator>(platform::CUDAPinnedPlace());
  }

  void InitStreamSafeCUDAAllocator(platform::CUDAPlace p, gpuStream_t stream) {
    PADDLE_ENFORCE_EQ(
        strategy_, AllocatorStrategy::kAutoGrowth,
        platform::errors::Unimplemented(
            "Only support auto-growth strategey for StreamSafeCUDAAllocator, "
            "the allocator strategy %d is unsupported for multi-stream",
            static_cast<int>(strategy_)));
    VLOG(9) << "Init CUDA allocator for stream " << stream << " in place " << p;
    std::lock_guard<SpinLock> lock_guard(cuda_allocators_lock_);
    try {
      GetAllocator(p, stream);
      VLOG(9) << "Other thread had build a allocator for stream " << stream
              << " in place " << p;
    } catch (platform::EnforceNotMet&) {
      InitAutoGrowthCUDAAllocator(p, stream);
      WrapStreamSafeCUDAAllocator(p, stream);
      WrapCUDARetryAllocator(p, stream, FLAGS_gpu_allocator_retry_time);
    } catch (...) {
      throw;
    }
  }

  void InitNaiveBestFitCUDAAllocator(platform::CUDAPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }

  void InitAutoGrowthCUDAAllocator(platform::CUDAPlace p, gpuStream_t stream) {
#if defined(PADDLE_WITH_HIP)
    auto cuda_allocator = std::make_shared<CUDAAllocator>(p);
    cuda_allocators_[p][stream] = std::make_shared<AutoGrowthBestFitAllocator>(
        cuda_allocator, platform::GpuMinChunkSize(), allow_free_idle_chunk_);
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
      auto cuda_allocator = std::make_shared<CUDAAllocator>(p);
      cuda_allocators_[p][stream] =
          std::make_shared<AutoGrowthBestFitAllocator>(
              cuda_allocator, platform::GpuMinChunkSize(),
              allow_free_idle_chunk_);
    }
#else
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

    cuda_allocators_[p][stream] = std::make_shared<AutoGrowthBestFitAllocator>(
        underlying_allocator, alignment, 0, allow_free_idle_chunk_);
#endif
#endif
  }

  // NOTE(Ruibiao): Old single-stream version, will be removed later
  void InitAutoGrowthCUDAAllocator(platform::CUDAPlace p,
                                   bool allow_free_idle_chunk) {
#if defined(PADDLE_WITH_HIP)
    auto cuda_allocator = std::make_shared<CUDAAllocator>(p);
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
      auto cuda_allocator = std::make_shared<CUDAAllocator>(p);
      allocators_[p] = std::make_shared<AutoGrowthBestFitAllocator>(
          cuda_allocator, platform::GpuMinChunkSize(), allow_free_idle_chunk);
    }

#else
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
#endif
#endif
  }

  void InitThreadLocalCUDAAllocator(platform::CUDAPlace p) {
    allocators_[p] = std::make_shared<ThreadLocalCUDAAllocator>(p);
  }

  void WrapStreamSafeCUDAAllocator(platform::CUDAPlace p, gpuStream_t stream) {
    const std::shared_ptr<Allocator>& underlying_allocator =
        GetAllocator(p, stream);
    cuda_allocators_[p][stream] = std::make_shared<StreamSafeCUDAAllocator>(
        underlying_allocator, p, stream);
  }

  void WrapCUDARetryAllocator(platform::CUDAPlace p, gpuStream_t stream,
                              size_t retry_time) {
    PADDLE_ENFORCE_GT(
        retry_time, 0,
        platform::errors::InvalidArgument(
            "Retry time should be larger than 0, but got %d", retry_time));
    std::shared_ptr<Allocator> allocator = GetAllocator(p, stream);
    allocator = std::make_shared<RetryAllocator>(allocator, retry_time);
  }

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

<<<<<<< HEAD
#ifdef PADDLE_WITH_IPU
  void InitNaiveBestFitIPUAllocator(platform::IPUPlace p) {
=======
#ifdef PADDLE_WITH_MLU
  void InitNaiveBestFitMLUAllocator(platform::MLUPlace p) {
>>>>>>> gitlab_mlu_dev
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
      system_allocators_[p] = std::make_shared<CUDAAllocator>(p);
    }
#endif
#ifdef PADDLE_WITH_MLU
    int device_count = platform::GetMLUDeviceCount();
    for (int i = 0; i < device_count; ++i) {
      platform::XPUPlace p(i);
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
<<<<<<< HEAD
#ifdef PADDLE_WITH_IPU
    int device_count = platform::GetIPUDeviceCount();
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      places.emplace_back(platform::IPUPlace(dev_id));
=======
#ifdef PADDLE_WITH_MLU
    int device_count = platform::GetMLUDeviceCount();
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      places.emplace_back(platform::MLUPlace(dev_id));
>>>>>>> gitlab_mlu_dev
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
    if (FLAGS_use_stream_safe_cuda_allocator) {
      CheckCUDAAllocThreadSafe(cuda_allocators_);
    }
#endif
  }

  // NOTE(Ruibiao): Old single-stream version, will be removed later
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // a standalone CUDA allocator to support multi-stream GC in new executor
  CUDAAllocatorMap cuda_allocators_;
  gpuStream_t default_stream_;
  SpinLock cuda_allocators_lock_;
#ifdef PADDLE_WITH_CUDA
  std::unordered_map<CUDAGraphID, std::unique_ptr<AllocatorFacadePrivate>>
      cuda_graph_allocator_map_;
#endif
#endif
  AllocatorStrategy strategy_;
  AllocatorMap allocators_;
  static AllocatorMap zero_size_allocators_;
  static AllocatorMap system_allocators_;
  bool allow_free_idle_chunk_;
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

const std::shared_ptr<Allocator>& AllocatorFacade::GetAllocator(
    const platform::Place& place) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (FLAGS_use_stream_safe_cuda_allocator && platform::is_gpu_place(place) &&
      FLAGS_use_system_allocator == false) {
    return m_->GetAllocator(BOOST_GET_CONST(platform::CUDAPlace, place),
                            m_->GetDefaultStream());
  }
#endif
  return m_->GetAllocator(place, /* A non-zero num to choose allocator_ */ 1);
}

std::shared_ptr<Allocation> AllocatorFacade::AllocShared(
    const platform::Place& place, size_t size) {
  return std::shared_ptr<Allocation>(Alloc(place, size));
}

AllocationPtr AllocatorFacade::Alloc(const platform::Place& place,
                                     size_t size) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (FLAGS_use_stream_safe_cuda_allocator && platform::is_gpu_place(place) &&
      size > 0 && FLAGS_use_system_allocator == false) {
    return Alloc(BOOST_GET_CONST(platform::CUDAPlace, place), size,
                 m_->GetDefaultStream());
  }
#endif
  return m_->GetAllocator(place, size)->Allocate(size);
}

uint64_t AllocatorFacade::Release(const platform::Place& place) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (FLAGS_use_stream_safe_cuda_allocator && platform::is_gpu_place(place) &&
      FLAGS_use_system_allocator == false) {
    return Release(BOOST_GET_CONST(platform::CUDAPlace, place),
                   m_->GetDefaultStream());
  }
#endif
  return m_->GetAllocator(place, /* A non-zero num to choose allocator_ */ 1)
      ->Release(place);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
std::shared_ptr<Allocation> AllocatorFacade::AllocShared(
    const platform::CUDAPlace& place, size_t size, const gpuStream_t& stream) {
  PADDLE_ENFORCE_EQ(
      FLAGS_use_stream_safe_cuda_allocator, true,
      platform::errors::Unimplemented(
          "StreamSafeCUDAAllocator is disabled, you should not call this "
          "multi-stream 'AllocaShared' function. "
          "To enable it, you can enter 'export "
          "FLAGS_use_stream_safe_cuda_allocator=true' in the terminal."));
  return std::shared_ptr<Allocation>(Alloc(place, size, stream));
}

AllocationPtr AllocatorFacade::Alloc(const platform::CUDAPlace& place,
                                     size_t size, const gpuStream_t& stream) {
  PADDLE_ENFORCE_EQ(
      FLAGS_use_stream_safe_cuda_allocator, true,
      platform::errors::Unimplemented(
          "StreamSafeCUDAAllocator is disabled, you should not call this "
          "multi-stream 'Alloca' function. "
          "To enable it, you can enter 'export "
          "FLAGS_use_stream_safe_cuda_allocator=true' in the terminal."));
  if (LIKELY(size > 0 && FLAGS_use_system_allocator == false)) {
    return m_->GetAllocator(place, stream, /* creat_if_not_found = */ true)
        ->Allocate(size);
  } else {
    return m_->GetAllocator(place, size)->Allocate(size);
  }
}

uint64_t AllocatorFacade::Release(const platform::CUDAPlace& place,
                                  const gpuStream_t& stream) {
  PADDLE_ENFORCE_EQ(
      FLAGS_use_stream_safe_cuda_allocator, true,
      platform::errors::Unimplemented(
          "StreamSafeCUDAAllocator is disabled, you should not call this "
          "multi-stream 'Release' function. "
          "To enable it, you can enter 'export "
          "FLAGS_use_stream_safe_cuda_allocator=true' in the terminal."));
  return m_->GetAllocator(place, stream)->Release(place);
}

void AllocatorFacade::RecordStream(Allocation* allocation,
                                   const gpuStream_t& stream) {
  PADDLE_ENFORCE_EQ(
      FLAGS_use_stream_safe_cuda_allocator, true,
      platform::errors::Unimplemented(
          "StreamSafeCUDAAllocator is disabled, you should not call this "
          "'RecordStream' function. "
          "To enable it, you can enter 'export "
          "FLAGS_use_stream_safe_cuda_allocator=true' in the terminal."));
  m_->RecordStream(allocation, stream);
}

#ifdef PADDLE_WITH_CUDA
void AllocatorFacade::PrepareMemoryPoolForCUDAGraph(CUDAGraphID id) {
  return m_->PrepareMemoryPoolForCUDAGraph(id);
}

void AllocatorFacade::RemoveMemoryPoolOfCUDAGraph(CUDAGraphID id) {
  return m_->RemoveMemoryPoolOfCUDAGraph(id);
}
#endif
#endif
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
