// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/stream_safe_cuda_allocator.h"

#include <algorithm>
#include <thread>

#include "glog/logging.h"

#include "paddle/common/flags.h"
#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/memory/allocation/retry_allocator.h"
#include "paddle/phi/core/memory/allocation/stat_allocator.h"
#include "paddle/phi/core/memory/allocation/vmm_allocator_v2_types.h"
#include "paddle/phi/core/memory/stats.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/utils/string/printf.h"

COMMON_DECLARE_bool(vmm_v2_remap_on_oom);

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/gpu/cuda/cuda_graph.h"
#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_multi_pool_allocator_v2.h"
#elif defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/rocm/hip_graph.h"
#endif

namespace paddle::memory::allocation {

namespace {

#if defined(PADDLE_WITH_CUDA)
VMMAutoGrowthBestFitMultiPoolAllocatorV2* GetVMMV2MultiPoolAllocator(
    const std::shared_ptr<Allocator>& allocator) {
  if (allocator == nullptr) {
    return nullptr;
  }
  if (auto* vmm = dynamic_cast<VMMAutoGrowthBestFitMultiPoolAllocatorV2*>(
          allocator.get())) {
    return vmm;
  }
  if (auto* retry = dynamic_cast<RetryAllocator*>(allocator.get())) {
    return GetVMMV2MultiPoolAllocator(retry->GetUnderLyingAllocator());
  }
  if (auto* stat = dynamic_cast<StatAllocator*>(allocator.get())) {
    return GetVMMV2MultiPoolAllocator(stat->GetUnderLyingAllocator());
  }
  return nullptr;
}

enum class VMMV2RemapOutcome {
  kDisabled,
  kNoProgress,
  kRetryFailed,
};

std::string ExtractAllocationFailureSummary(const BadAlloc& error) {
  std::string message = error.what();
  constexpr char kSummaryMarker[] = "Error Message Summary:";
  const auto marker = message.rfind(kSummaryMarker);
  if (marker != std::string::npos) {
    auto summary_start = message.find('\n', marker);
    if (summary_start != std::string::npos) {
      summary_start = message.find('\n', summary_start + 1);
    }
    if (summary_start != std::string::npos) {
      message.erase(0, summary_start + 1);
    }
  }

  const auto first = message.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "unknown allocation failure";
  }
  const auto last = message.find_last_not_of(" \t\r\n");
  return message.substr(first, last - first + 1);
}

std::string BuildVMMV2OOMSummary(
    VMMAutoGrowthBestFitMultiPoolAllocatorV2* allocator,
    const GPUPlace& place,
    size_t requested_size,
    VMMV2RemapOutcome remap_outcome,
    size_t remapped_bytes = 0) {
  size_t total_free = 0;
  size_t largest_free_block = 0;
  allocator->GetFreeBlockStats(
      &total_free, &largest_free_block, requested_size);

  size_t available = 0;
  size_t total = 0;
  size_t driver_available = 0;
  size_t driver_total = 0;
  const bool memory_limited = platform::RecordedGpuMemGetInfo(
      &available, &total, &driver_available, &driver_total, place.device);
  const int64_t paddle_allocated =
      paddle::memory::DeviceMemoryStatCurrentValue("Allocated", place.device);
  const int64_t paddle_reserved =
      paddle::memory::DeviceMemoryStatCurrentValue("Reserved", place.device);

  std::string remap_result;
  switch (remap_outcome) {
    case VMMV2RemapOutcome::kDisabled:
      remap_result = "disabled";
      break;
    case VMMV2RemapOutcome::kNoProgress:
      remap_result = "no releasable free memory was found";
      break;
    case VMMV2RemapOutcome::kRetryFailed:
      remap_result =
          string::Sprintf("reclaimed %s, but the allocation retry still failed",
                          string::HumanReadableSize(remapped_bytes));
      break;
  }

  std::string configured_limit;
  if (memory_limited) {
    configured_limit = string::Sprintf(
        "\nConfigured Paddle memory limit: available=%s, limit=%s.",
        string::HumanReadableSize(available),
        string::HumanReadableSize(total));
  }

  return string::Sprintf(
      "\n\nOut of memory error on GPU %d. Cannot allocate %s memory.\n"
      "Memory pool state: total free=%s, largest free block=%s.\n"
      "Paddle memory state: allocated=%s, reserved=%s.\n"
      "GPU memory state: available=%s, total=%s.%s\n"
      "Memory defragmentation: %s.\n"
      "Please stop other processes using GPU %d or use another GPU; "
      "otherwise, decrease the model batch size.\n",
      place.device,
      string::HumanReadableSize(requested_size),
      string::HumanReadableSize(total_free),
      string::HumanReadableSize(largest_free_block),
      string::HumanReadableSize(std::max<int64_t>(paddle_allocated, 0)),
      string::HumanReadableSize(std::max<int64_t>(paddle_reserved, 0)),
      string::HumanReadableSize(driver_available),
      string::HumanReadableSize(driver_total),
      configured_limit,
      remap_result,
      place.device);
}

void MarkVMMV2RemapPendingStream(StreamSafeCUDAAllocator* allocator,
                                 StreamSafeCUDAAllocation* allocation) {
  if (!FLAGS_vmm_v2_remap_on_oom) {
    return;
  }
  if (allocator->GetVMMV2Allocator() == nullptr) {
    return;
  }
  PADDLE_ENFORCE_EQ(
      allocation->SetVMMV2RemapEvent(),
      true,
      common::errors::PreconditionNotMet(
          "VMM V2 allocation %p cannot record its remap-safety stream.",
          allocation->ptr()));
}

#else
VMMAutoGrowthBestFitMultiPoolAllocatorV2* GetVMMV2MultiPoolAllocator(
    const std::shared_ptr<Allocator>& allocator) {
  (void)allocator;
  return nullptr;
}

void MarkVMMV2RemapPendingStream(StreamSafeCUDAAllocator* allocator,
                                 StreamSafeCUDAAllocation* allocation) {
  (void)allocator;
  (void)allocation;
}
#endif

void ClearGpuLastError() {
#ifdef PADDLE_WITH_CUDA
  cudaGetLastError();
#elif defined(PADDLE_WITH_HIP)
  hipGetLastError();
#endif
}

}  // namespace

StreamSafeCUDAAllocation::StreamSafeCUDAAllocation(
    DecoratedAllocationPtr underlying_allocation,
    gpuStream_t owning_stream,
    StreamSafeCUDAAllocator* allocator)
    : Allocation(underlying_allocation->ptr(),
                 underlying_allocation->base_ptr(),
                 underlying_allocation->size(),
                 underlying_allocation->place()),
      underlying_allocation_(std::move(underlying_allocation)),
#if defined(PADDLE_WITH_CUDA)
      vmm_v2_remap_allocation_(
          dynamic_cast<VMMRemapEventAllocation*>(underlying_allocation_.get())),
#else
      vmm_v2_remap_allocation_(nullptr),
#endif
      owning_stream_(owning_stream),
      allocator_(allocator->shared_from_this()) {
}

bool StreamSafeCUDAAllocation::RecordStream(gpuStream_t stream) {
  VLOG(8) << "Try record stream " << stream << " for address " << ptr();
  if (stream == owning_stream_) {
    return false;
  }

  std::call_once(once_flag_,
                 [this] { phi::backends::gpu::SetDeviceId(place_.device); });

  std::lock_guard<SpinLock> lock_guard(outstanding_event_map_lock_);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
    graph_capturing_stream_set_.insert(stream);
    return true;
  }
#endif

  RecordStreamWithNoGraphCapturing(stream);
  RecordGraphCapturingStreams();
  return true;
}

void StreamSafeCUDAAllocation::EraseStream(gpuStream_t stream) {
  VLOG(8) << "Try remove stream " << stream << " for address " << ptr();
  std::lock_guard<SpinLock> lock_guard(outstanding_event_map_lock_);
  auto it = outstanding_event_map_.find(stream);
  if (it == outstanding_event_map_.end()) {
    return;
  }

#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(it->second));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipEventDestroy(it->second));
#endif
  outstanding_event_map_.erase(it);
}

bool StreamSafeCUDAAllocation::CanBeFreed() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
    return graph_capturing_stream_set_.empty() &&
           outstanding_event_map_.empty();
  }
#endif

  std::call_once(once_flag_,
                 [this] { phi::backends::gpu::SetDeviceId(place_.device); });

  RecordGraphCapturingStreams();

  for (auto it = outstanding_event_map_.begin();
       it != outstanding_event_map_.end();
       ++it) {
    gpuEvent_t& event = it->second;
#ifdef PADDLE_WITH_CUDA
    gpuError_t err = cudaEventQuery(event);
    if (err == cudaErrorNotReady) {
      VLOG(9) << "Event " << event << " for " << ptr() << " is not completed";
      // Erase the completed event before "it"
      outstanding_event_map_.erase(outstanding_event_map_.begin(), it);
      return false;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(err);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(event));
#else
    gpuError_t err = hipEventQuery(event);
    if (err == hipErrorNotReady) {
      VLOG(9) << "Event " << event << " for " << ptr() << " is not completed";
      // Erase the completed event before "it"
      outstanding_event_map_.erase(outstanding_event_map_.begin(), it);
      return false;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(err);
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventDestroy(event));
#endif
    VLOG(8) << "Destroy event " << event;
  }
  return true;
}

gpuStream_t StreamSafeCUDAAllocation::GetOwningStream() const {
  return owning_stream_;
}

void StreamSafeCUDAAllocation::RecordGraphCapturingStreams() {
  for (gpuStream_t stream : graph_capturing_stream_set_) {
    RecordStreamWithNoGraphCapturing(stream);
  }
  graph_capturing_stream_set_.clear();
}

bool StreamSafeCUDAAllocation::SetVMMV2RemapEvent() {
#if defined(PADDLE_WITH_CUDA)
  if (vmm_v2_remap_allocation_ == nullptr) {
    return false;
  }
  // Keep the free path cheap: store the owning stream here and let the remap
  // scanner lazily create/query an event only if this block becomes a move
  // candidate. Creating one CUDA event for every free is measurable in steady
  // training even when compaction never runs.
  return vmm_v2_remap_allocation_->SetVMMRemapEvent(owning_stream_, nullptr);
#else
  return false;
#endif
}

void StreamSafeCUDAAllocation::RecordStreamWithNoGraphCapturing(
    gpuStream_t stream) {
  gpuEvent_t record_event;
  auto it = outstanding_event_map_.find(stream);
  if (it == outstanding_event_map_.end()) {
    gpuEvent_t new_event;
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventCreateWithFlags(&new_event, cudaEventDisableTiming));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        hipEventCreateWithFlags(&new_event, hipEventDisableTiming));
#endif
    outstanding_event_map_[stream] = new_event;
    record_event = new_event;
    VLOG(9) << "Create a new event " << new_event;
  } else {
    record_event = it->second;
    VLOG(9) << "Reuse event " << record_event;
  }

#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(record_event, stream));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(record_event, stream));
#endif
  VLOG(8) << "Record event " << record_event << " to stream " << stream;
}

StreamSafeCUDAAllocator::StreamSafeCUDAAllocator(
    std::shared_ptr<Allocator> underlying_allocator,
    GPUPlace place,
    gpuStream_t default_stream,
    bool in_cuda_graph_capturing)
    : underlying_allocator_(std::move(underlying_allocator)),
      vmm_v2_allocator_(GetVMMV2MultiPoolAllocator(underlying_allocator_)),
      place_(place),
      default_stream_(default_stream),
      in_cuda_graph_capturing_(in_cuda_graph_capturing) {
  if (LIKELY(!in_cuda_graph_capturing)) {
    std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
    allocator_map_[place].emplace_back(this);
  }
}

StreamSafeCUDAAllocator::~StreamSafeCUDAAllocator() {
  if (LIKELY(!in_cuda_graph_capturing_)) {
    std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
    std::vector<StreamSafeCUDAAllocator*>& allocators = allocator_map_[place_];
    allocators.erase(std::remove(allocators.begin(), allocators.end(), this),
                     allocators.end());
  }
}

bool StreamSafeCUDAAllocator::IsAllocThreadSafe() const { return true; }

gpuStream_t StreamSafeCUDAAllocator::GetDefaultStream() const {
  return default_stream_;
}

void StreamSafeCUDAAllocator::SetDefaultStream(gpuStream_t stream) {
  default_stream_ = stream;
}

phi::Allocation* StreamSafeCUDAAllocator::AllocateImpl(size_t size) {
  phi::RecordEvent record("StreamSafeCUDAAllocator::Allocate",
                          phi::TracerEventType::UserDefined,
                          9 /*level*/);
  ProcessUnfreedAllocations();
  VLOG(8) << "Try allocate " << size << " bytes";
  AllocationPtr underlying_allocation;
  try {
    underlying_allocation = underlying_allocator_->Allocate(size);
  } catch (const BadAlloc& first_bad_alloc) {
    const std::string first_failure = first_bad_alloc.what();
    VLOG(4) << "Allocation failed when allocating " << size << " bytes";
    auto* vmm = GetVMMV2MultiPoolAllocator(underlying_allocator_);
    if (vmm == nullptr) {
      ReleaseImpl(place_);
      try {
        underlying_allocation = underlying_allocator_->Allocate(size);
      } catch (const BadAlloc& second_bad_alloc) {
        ClearGpuLastError();
        PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
            "Allocation of %zu bytes failed after releasing memory from all "
            "streams.\n"
            "Initial allocation failure:\n%s\n"
            "Retry allocation failure:\n%s",
            size,
            first_failure.c_str(),
            second_bad_alloc.what()));
      }
    }
#if defined(PADDLE_WITH_CUDA)
    if (vmm != nullptr) {
      std::unique_lock<SpinLock> allocator_map_guard(allocator_map_lock_);
      for (auto* alloc : allocator_map_[place_]) {
        alloc->ProcessUnfreedAllocations();
      }
      allocator_map_guard.unlock();
      try {
        underlying_allocation = underlying_allocator_->Allocate(size);
      } catch (const BadAlloc& second_bad_alloc) {
        if (FLAGS_vmm_v2_remap_on_oom) {
          const auto* grow_oom =
              dynamic_cast<const VMMGrowOOM*>(&second_bad_alloc);
          const size_t remapped_bytes = vmm->RemapForAllocation(
              place_, size, grow_oom == nullptr ? nullptr : &grow_oom->info());
          if (remapped_bytes > 0) {
            try {
              underlying_allocation = underlying_allocator_->Allocate(size);
            } catch (const BadAlloc& final_bad_alloc) {
              const std::string oom_summary =
                  BuildVMMV2OOMSummary(vmm,
                                       place_,
                                       size,
                                       VMMV2RemapOutcome::kRetryFailed,
                                       remapped_bytes);
              const std::string initial_summary =
                  ExtractAllocationFailureSummary(first_bad_alloc);
              const std::string final_summary =
                  ExtractAllocationFailureSummary(final_bad_alloc);
              ClearGpuLastError();
              PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
                  "%s"
                  "Allocation attempts:\n"
                  "1. Initial attempt: %s\n"
                  "2. Retry after memory defragmentation: %s",
                  oom_summary,
                  initial_summary.c_str(),
                  final_summary.c_str()));
            }
          } else {
            const std::string oom_summary = BuildVMMV2OOMSummary(
                vmm, place_, size, VMMV2RemapOutcome::kNoProgress);
            const std::string initial_summary =
                ExtractAllocationFailureSummary(first_bad_alloc);
            const std::string retry_summary =
                ExtractAllocationFailureSummary(second_bad_alloc);
            ClearGpuLastError();
            PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
                "%s"
                "Allocation attempts:\n"
                "1. Initial attempt: %s\n"
                "2. Retry after reclaiming pending frees: %s",
                oom_summary,
                initial_summary.c_str(),
                retry_summary.c_str()));
          }
        } else {
          const std::string oom_summary = BuildVMMV2OOMSummary(
              vmm, place_, size, VMMV2RemapOutcome::kDisabled);
          const std::string initial_summary =
              ExtractAllocationFailureSummary(first_bad_alloc);
          const std::string retry_summary =
              ExtractAllocationFailureSummary(second_bad_alloc);
          ClearGpuLastError();
          PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
              "%s"
              "Allocation attempts:\n"
              "1. Initial attempt: %s\n"
              "2. Retry after reclaiming pending frees: %s",
              oom_summary,
              initial_summary.c_str(),
              retry_summary.c_str()));
        }
      }
    }
#endif
  }
  StreamSafeCUDAAllocation* allocation = new StreamSafeCUDAAllocation(
      static_unique_ptr_cast<Allocation>(std::move(underlying_allocation)),
      default_stream_,
      this);
  VLOG(8) << "Thread " << std::this_thread::get_id() << " Allocate "
          << allocation->size() << " bytes at address " << allocation->ptr()
          << "  , stream: " << default_stream_;
  return allocation;
}

void StreamSafeCUDAAllocator::FreeImpl(phi::Allocation* allocation) {
  phi::RecordEvent record("StreamSafeCUDAAllocator::Free",
                          phi::TracerEventType::UserDefined,
                          9 /*level*/);
  StreamSafeCUDAAllocation* stream_safe_cuda_allocation =
      static_cast<StreamSafeCUDAAllocation*>(allocation);

  VLOG(8) << "Try free allocation " << stream_safe_cuda_allocation->ptr();
  if (stream_safe_cuda_allocation->CanBeFreed()) {
    VLOG(9) << "Directly delete allocation";
    MarkVMMV2RemapPendingStream(this, stream_safe_cuda_allocation);
    delete stream_safe_cuda_allocation;
  } else {
    VLOG(9) << "Put into unfreed_allocation list";
    std::lock_guard<SpinLock> lock_guard(unfreed_allocation_lock_);
    unfreed_allocations_.emplace_back(stream_safe_cuda_allocation);
  }
}

uint64_t StreamSafeCUDAAllocator::ReleaseImpl(const Place& place) {
  if (UNLIKELY(in_cuda_graph_capturing_)) {
    VLOG(7) << "Memory release forbidden in CUDA Graph Capturing";
    return 0;
  }

  std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
  std::vector<StreamSafeCUDAAllocator*>& allocators = allocator_map_[place];
  uint64_t released_size = 0;
  for (StreamSafeCUDAAllocator* allocator : allocators) {
    released_size += allocator->ProcessUnfreedAllocationsAndRelease();
  }
  VLOG(8) << "Release " << released_size << " bytes memory from all streams";
  return released_size;
}

size_t StreamSafeCUDAAllocator::CompactImpl(const Place& place) {
  std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
  std::vector<StreamSafeCUDAAllocator*>& allocators = allocator_map_[place];

  // Execution layer for compact(remap): first reclaim cross-stream pending
  // frees. Only the current stream allocator can satisfy this allocation
  // retry, so remap compaction must stay local to the allocator that observed
  // OOM. Compacting other stream allocators cannot provide a block to this
  // retry and may unnecessarily remap communication-stream memory.
  for (StreamSafeCUDAAllocator* allocator : allocators) {
    allocator->ProcessUnfreedAllocations();
  }

  return underlying_allocator_->Compact(place_);
}

void StreamSafeCUDAAllocator::ProcessUnfreedAllocations() {
  // NOTE(Ruibiao): This condition is to reduce lock completion. It does not
  // need to be thread-safe since here occasional misjudgments are permissible.
  if (unfreed_allocations_.empty()) {
    return;
  }

  std::lock_guard<SpinLock> lock_guard(unfreed_allocation_lock_);
  for (auto it = unfreed_allocations_.begin();
       it != unfreed_allocations_.end();) {
    if ((*it)->CanBeFreed()) {
      MarkVMMV2RemapPendingStream(this, *it);
      delete *it;
      it = unfreed_allocations_.erase(it);
    } else {
      ++it;
    }
  }
}

uint64_t StreamSafeCUDAAllocator::ProcessUnfreedAllocationsAndRelease() {
  ProcessUnfreedAllocations();
  return underlying_allocator_->Release(place_);
}

thread_local std::once_flag StreamSafeCUDAAllocation::once_flag_;

std::map<Place, std::vector<StreamSafeCUDAAllocator*>>
    StreamSafeCUDAAllocator::allocator_map_;
SpinLock StreamSafeCUDAAllocator::allocator_map_lock_;

}  // namespace paddle::memory::allocation
