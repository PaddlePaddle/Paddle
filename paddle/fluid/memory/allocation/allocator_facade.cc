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

#include <atomic>
#include <chrono>
#include <condition_variable>

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <cuda_runtime.h>
#endif
#include "gflags/gflags.h"

#include "paddle/fluid/memory/allocation/aligned_allocator.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "paddle/fluid/memory/allocation/auto_growth_best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/cpu_allocator.h"
#include "paddle/fluid/memory/allocation/naive_best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/retry_allocator.h"
#include "paddle/fluid/memory/allocation/stream_safe_cuda_allocator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/npu_info.h"
#include "paddle/fluid/platform/place.h"
#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/memory/allocation/npu_pinned_allocator.h"
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/memory/allocation/cuda_allocator.h"
#include "paddle/fluid/memory/allocation/pinned_allocator.h"
#include "paddle/fluid/memory/allocation/thread_local_allocator.h"
#include "paddle/fluid/platform/gpu_info.h"
#endif
#if CUDA_VERSION >= 10020
#include "paddle/fluid/memory/allocation/cuda_virtual_mem_allocator.h"
#include "paddle/fluid/memory/allocation/virtual_memory_auto_growth_best_fit_allocator.h"
#include "paddle/fluid/platform/dynload/cuda_driver.h"
#endif
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_graph.h"
#endif
#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/xpu/xpu_info.h"
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  using CUDAAllocatorMap =
      std::map<platform::CUDAPlace,
               std::map<cudaStream_t, std::shared_ptr<Allocator>>>;
#endif

  explicit AllocatorFacadePrivate(bool allow_free_idle_chunk = true) {
    strategy_ = GetAllocatorStrategy();
    CheckStrategy(strategy_);

    InitNaiveBestFitCPUAllocator();

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    allow_free_idle_chunk_ = allow_free_idle_chunk;
    default_cuda_stream_ = nullptr;
    for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount(); ++dev_id) {
      InitCUDAAllocator(platform::CUDAPlace(dev_id), default_cuda_stream_);
    }
    InitNaiveBestFitCUDAPinnedAllocator();
#endif

#ifdef PADDLE_WITH_XPU
    for (int dev_id = 0; dev_id < platform::GetXPUDeviceCount(); ++dev_id) {
      InitNaiveBestFitXPUAllocator(platform::XPUPlace(dev_id));
    }
#endif

#ifdef PADDLE_WITH_ASCEND_CL
    if (strategy_ == AllocatorStrategy::kNaiveBestFit) {
      for (int dev_id = 0; dev_id < platform::GetNPUDeviceCount(); ++dev_id) {
        InitNaiveBestFitNPUAllocator(platform::NPUPlace(dev_id));
      }
      InitNaiveBestFitNPUPinnedAllocator();
    }
#endif

    InitZeroSizeAllocators();
    InitSystemAllocators();
    CheckAllocThreadSafe();
  }

  const std::shared_ptr<Allocator>& GetAllocator(const platform::Place& place,
                                                 size_t size) {
    VLOG(6) << "GetAllocator"
            << " " << place << " " << size;

    if (platform::is_gpu_place(place) && size > 0) {
      return GetCUDAAllocator(boost::get<platform::CUDAPlace>(place),
                              default_cuda_stream_);
    }

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
  const std::shared_ptr<Allocator>& GetCUDAAllocator(
      const platform::CUDAPlace& place, const cudaStream_t& stream) {
    auto place_it = cuda_allocators_.find(place);
    PADDLE_ENFORCE_NE(place_it, cuda_allocators_.end(),
                      platform::errors::NotFound(
                          "No allocator found for the place %s", place));

    const std::map<cudaStream_t, std::shared_ptr<Allocator>>& allocator_map =
        place_it->second;
    auto stream_it = allocator_map.find(stream);
    PADDLE_ENFORCE_NE(
        stream_it, allocator_map.end(),
        platform::errors::NotFound(
            "No allocator found for stream %s in place %s", stream, place));

    return stream_it->second;
  }

  cudaStream_t GetDefaultCudaStream() { return default_cuda_stream_; }

  void NotifyGPURetryThreads() { cuda_retry_cv_.notify_all(); }

  void RecordStream(Allocation* allocation, const cudaStream_t& stream) {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(allocation->place()), true,
        platform::errors::InvalidArgument(
            "Not allow to record stream for an allocation with place %s",
            allocation->place()));
    dynamic_cast<StreamSafeCUDAAllocation*>(allocation)->RecordStream(stream);
  }

  AllocationPtr CUDAAlloc(const platform::CUDAPlace& place,
                          const cudaStream_t& stream, size_t size) {
    std::shared_ptr<Allocator> cuda_allocator;
    /* NOTE(Ruibiao): This code does not lead to lock competition
     * for seraching initialized CUDA allocator in multithreaded scenario.
     * However, when the corresponding CUDA allocator is not initialized,
     * it may result in large lookup overhead,
     * which call GetCUDAAAllocator 3 times in the worst case.
    **/
    try {
      cuda_allocator = GetCUDAAllocator(place, stream);
    } catch (platform::EnforceNotMet& err) {
      VLOG(9) << "No allocator found for stream " << stream << "in place "
              << place << " , build a new one";
      std::unique_lock<std::mutex> lock(cuda_retry_mutex_);
      try {
        cuda_allocator = GetCUDAAllocator(place, stream);
      } catch (platform::EnforceNotMet& err) {
        InitCUDAAllocator(place, stream);
        cuda_allocator = GetCUDAAllocator(place, stream);
      } catch (...) {
        throw;
      }
    } catch (...) {
      throw;
    }

    if (FLAGS_gpu_allocator_retry_time <= 0) {
      return cuda_allocator->Allocate(size);
    }

    // In fact, we can unify the code of allocation success and failure
    // But it would add lock even when allocation success at the first time
    try {
      return cuda_allocator->Allocate(size);
    } catch (BadAlloc&) {
      VLOG(9) << "Allocation failed when allocating " << size
              << " bytes for stream " << stream;
      for (auto pair : cuda_allocators_[place]) {
        std::shared_ptr<Allocator> cuda_allocator = pair.second;
        std::dynamic_pointer_cast<StreamSafeCUDAAllocator>(cuda_allocator)
            ->ProcessEventsAndFree();
      }
      try {
        return cuda_allocator->Allocate(size);
      } catch (BadAlloc&) {
        {
          WaitedAllocateSizeGuard guard(&cuda_waited_allocate_size_, size);
          VLOG(10)
              << "Still allocation failed after calling ProcessEventAndFree, "
              << " cuda_waited_allocate_size_ = " << cuda_waited_allocate_size_;
          // We can just write allocation retry inside the predicate function of
          // wait_until. But it needs to acquire the lock when executing
          // predicate
          // function. For better performance, we use loop here
          auto end_time =
              std::chrono::high_resolution_clock::now() +
              std::chrono::milliseconds(FLAGS_gpu_allocator_retry_time);
          auto wait_until = [&end_time, this] {
            std::unique_lock<std::mutex> lock(cuda_retry_mutex_);
            return cuda_retry_cv_.wait_until(lock, end_time);
          };

          size_t retry_times = 0;
          while (wait_until() != std::cv_status::timeout) {
            try {
              return cuda_allocator->Allocate(size);
            } catch (BadAlloc&) {
              ++retry_times;
              VLOG(10) << "Allocation failed when retrying " << retry_times
                       << " times when allocating " << size
                       << " bytes. Wait still.";
            } catch (...) {
              throw;
            }
          }
        }
        VLOG(10) << "Allocation failed because of timeout when allocating "
                 << size << " bytes.";
        return cuda_allocator->Allocate(
            size);  // If timeout, try last allocation request
      } catch (...) {
        throw;
      }
    } catch (...) {
      throw;
    }
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

  void CheckStrategy(AllocatorStrategy strategy) {
    if (strategy != AllocatorStrategy::kNaiveBestFit &&
        strategy != AllocatorStrategy::kAutoGrowth &&
        strategy != AllocatorStrategy::kThreadLocal) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported allocator strategy: %d", static_cast<int>(strategy_)));
    }
  }

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

  void InitCUDAAllocator(platform::CUDAPlace p, cudaStream_t stream) {
    switch (strategy_) {
      case AllocatorStrategy::kNaiveBestFit: {
        InitNaiveBestFitCUDAAllocator(p, stream);
        break;
      }
      case AllocatorStrategy::kAutoGrowth: {
        InitAutoGrowthCUDAAllocator(p, stream);
        break;
      }
      case AllocatorStrategy::kThreadLocal: {
        InitThreadLocalCUDAAllocator(p, stream);
        break;
      }
      default: {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unsupported allocator strategy: %d", static_cast<int>(strategy_)));
      }
    }
    WrapStreamSafeCUDAAllocator(p, stream);
  }

  void InitNaiveBestFitCUDAAllocator(platform::CUDAPlace p,
                                     cudaStream_t stream) {
    cuda_allocators_[p][stream] = std::make_shared<NaiveBestFitAllocator>(p);
  }

  void InitAutoGrowthCUDAAllocator(platform::CUDAPlace p, cudaStream_t stream) {
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
      PADDLE_ENFORCE_CUDA_SUCCESS(
          paddle::platform::dynload::cuDeviceGet(&device, p.GetDeviceId()));

      PADDLE_ENFORCE_CUDA_SUCCESS(
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

  void InitThreadLocalCUDAAllocator(platform::CUDAPlace p,
                                    cudaStream_t stream) {
    cuda_allocators_[p][stream] = std::make_shared<ThreadLocalCUDAAllocator>(p);
  }

  void WrapStreamSafeCUDAAllocator(platform::CUDAPlace p, cudaStream_t stream) {
    const std::shared_ptr<Allocator>& underlying_allocator =
        GetCUDAAllocator(p, stream);
    cuda_allocators_[p][stream] =
        std::make_shared<StreamSafeCUDAAllocator>(underlying_allocator, stream);
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

#ifdef PADDLE_WITH_ASCEND_CL
  void InitNaiveBestFitNPUAllocator(platform::NPUPlace p) {
    allocators_[p] = std::make_shared<NaiveBestFitAllocator>(p);
  }

  void InitNaiveBestFitNPUPinnedAllocator() {
    allocators_[platform::NPUPinnedPlace()] =
        std::make_shared<paddle::memory::allocation::NPUPinnedAllocator>();
  }
#endif

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

  void CheckAllocThreadSafe() const {
    CheckAllocThreadSafe(allocators_);
    CheckAllocThreadSafe(zero_size_allocators_);
    CheckAllocThreadSafe(system_allocators_);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    CheckCUDAAllocThreadSafe(cuda_allocators_);
#endif
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // a standalone CUDA allocator to support multi-stream GC in new executor
  CUDAAllocatorMap cuda_allocators_;
  cudaStream_t default_cuda_stream_;
  static std::condition_variable cuda_retry_cv_;
  std::mutex cuda_retry_mutex_;
  std::mutex cuda_init_mutex_;
  std::atomic<size_t> cuda_waited_allocate_size_{0};
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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
std::condition_variable AllocatorFacadePrivate::cuda_retry_cv_;
#endif

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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (platform::is_gpu_place(place)) {
    return Alloc(boost::get<platform::CUDAPlace>(place),
                 m_->GetDefaultCudaStream(), size);
  }
#endif
  return m_->GetAllocator(place, size)->Allocate(size);
}

uint64_t AllocatorFacade::Release(const platform::Place& place) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (platform::is_gpu_place(place)) {
    return Release(boost::get<platform::CUDAPlace>(place),
                   m_->GetDefaultCudaStream());
  }
#endif
  return m_->GetAllocator(place, /* A non-zero num to choose allocator_ */ 1)
      ->Release(place);
}

const std::shared_ptr<Allocator>& AllocatorFacade::GetAllocator(
    const platform::Place& place) {
  return m_->GetAllocator(place, /* A non-zero num to choose allocator_ */ 1);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
std::shared_ptr<Allocation> AllocatorFacade::AllocShared(
    const platform::CUDAPlace& place, const cudaStream_t& stream, size_t size) {
  return std::shared_ptr<Allocation>(Alloc(place, stream, size));
}

AllocationPtr AllocatorFacade::Alloc(const platform::CUDAPlace& place,
                                     const cudaStream_t& stream, size_t size) {
  if (size > 0) {
    return m_->CUDAAlloc(place, stream, size);
  } else {
    return m_->GetAllocator(place, size)->Allocate(size);
  }
}

uint64_t AllocatorFacade::Release(const platform::CUDAPlace& place,
                                  const cudaStream_t& stream) {
  return m_->GetCUDAAllocator(place, stream)->Release(place);
}

void AllocatorFacade::NotifyGPURetryThreads() { m_->NotifyGPURetryThreads(); }

void AllocatorFacade::RecordStream(Allocation* allocation,
                                   const cudaStream_t& stream) {
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

void NotifyGPURetryThreads() {
  allocation::AllocatorFacade::Instance().NotifyGPURetryThreads();
}
#endif
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
