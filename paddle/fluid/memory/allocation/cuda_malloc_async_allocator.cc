// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/cuda_malloc_async_allocator.h"
#include <cstdint>
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/allocation/stream_safe_cuda_allocator.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <string>

#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/enforce.h"
#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/gpu/cuda/cuda_graph.h"
#elif defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/rocm/hip_graph.h"
#endif

namespace paddle::memory::allocation {

thread_local std::once_flag CUDAMallocAsyncAllocation::once_flag_;

void CUDAMallocAsyncAllocation::RecordGraphCapturingStreams() {
  for (gpuStream_t stream : graph_capturing_stream_set_) {
    RecordStreamWithNoGraphCapturing(stream);
  }
  graph_capturing_stream_set_.clear();
}

void CUDAMallocAsyncAllocation::RecordStreamWithNoGraphCapturing(
    gpuStream_t stream) {
  if (event_map_.find(stream) == event_map_.end()) {
    gpuEvent_t event;
    PADDLE_ENFORCE_GPU_SUCCESS(
        gpuEventCreateWithFlags(&event, gpuEventDisableTiming));
    PADDLE_ENFORCE_GPU_SUCCESS(gpuEventRecord(event, stream));
    event_map_[stream] = event;
  } else {
    PADDLE_ENFORCE_GPU_SUCCESS(gpuEventRecord(event_map_[stream], stream));
  }
}

void CUDAMallocAsyncAllocation::RecordStream(gpuStream_t stream) {
  std::call_once(once_flag_,
                 [this] { phi::backends::gpu::SetDeviceId(place_.device); });
  if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
    // Disallow recording when graph is capturing
    graph_capturing_stream_set_.insert(stream);
    return;
  } else {
    RecordStreamWithNoGraphCapturing(stream);
    // Record the stream after graph is captured
    RecordGraphCapturingStreams();
  }
}

void CUDAMallocAsyncAllocation::EraseStream(gpuStream_t stream) {
  std::lock_guard<SpinLock> lock_guard(event_map_lock_);
  event_map_.erase(stream);
}

void CUDAMallocAsyncAllocation::Free(int dev_id) {
  platform::RecordedGpuFreeAsync(ptr(), size(), place_.device, malloc_stream_);
}

// if synchronize, we sync the event so the block could be fully released.
bool CUDAMallocAsyncAllocation::CanBeFreed(bool synchronize) {
  if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
    return graph_capturing_stream_set_.empty() && event_map_.empty();
  }
  // When try to free a block, we record the stream that should be record during
  // capturing.
  RecordGraphCapturingStreams();

  std::call_once(once_flag_,
                 [this] { phi::backends::gpu::SetDeviceId(place_.device); });

  for (auto it = event_map_.begin(); it != event_map_.end();) {
    gpuEvent_t& event = it->second;
    if (synchronize) {
      PADDLE_ENFORCE_GPU_SUCCESS(gpuEventSynchronize(event));
    } else {
      gpuError_t err = gpuEventQuery(event);
      if (err == gpuErrorNotReady) {
        VLOG(9) << "Event " << event << " for " << ptr() << " is not completed";
        return false;
      }
      PADDLE_ENFORCE_GPU_SUCCESS(err);
    }
    PADDLE_ENFORCE_GPU_SUCCESS(gpuEventDestroy(event));
    VLOG(8) << "Destroy event " << event;
    it = event_map_.erase(it);
  }
  return true;
}

CUDAMallocAsyncAllocator::CUDAMallocAsyncAllocator(
    std::shared_ptr<Allocator> underlying_allocator,
    const platform::CUDAPlace& place,
    gpuStream_t default_stream)
    : underlying_allocator_(std::move(underlying_allocator)),
      place_(place),
      default_stream_(default_stream),
      memory_stream_(nullptr) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      gpuStreamCreateWithPriority(&memory_stream_, gpuStreamNonBlocking, 0));
}

bool CUDAMallocAsyncAllocator::IsAllocThreadSafe() const { return true; }

void CUDAMallocAsyncAllocator::ProcessUnfreedAllocations(bool synchronize) {
  if (unfreed_allocations_.empty()) {
    return;
  }

  std::lock_guard<SpinLock> lock_guard(unfreed_allocation_lock_);
  for (auto it = unfreed_allocations_.begin();
       it != unfreed_allocations_.end();) {
    CUDAMallocAsyncAllocation* allocation = (*it);
    if (allocation->CanBeFreed(synchronize)) {
      allocation->Free(place_.device);
      delete allocation;
      it = unfreed_allocations_.erase(it);
    } else {
      ++it;
    }
  }
}

void CUDAMallocAsyncAllocator::TryFree(CUDAMallocAsyncAllocation* allocation) {
  if (allocation->CanBeFreed()) {
    allocation->Free(place_.device);
    delete allocation;
  } else {
    std::lock_guard<SpinLock> lock_guard(unfreed_allocation_lock_);
    unfreed_allocations_.emplace_back(allocation);
  }
}

uint64_t CUDAMallocAsyncAllocator::ReleaseImpl(const platform::Place& place) {
  if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
    VLOG(7) << "Memory release forbidden in CUDA Graph Captruing";
    return 0;
  }

  uint64_t released_size = 0;
  // we synchronize the event so all the block could be release.
  ProcessUnfreedAllocations(true);
  if (underlying_allocator_)
    released_size += underlying_allocator_->Release(place_);
  VLOG(8) << "Release " << released_size << " bytes memory from all streams";
  return released_size;
}

void CUDAMallocAsyncAllocator::FreeImpl(phi::Allocation* phi_allocation) {
  auto* allocation = dynamic_cast<CUDAMallocAsyncAllocation*>(phi_allocation);

  // VLOG(0) << "Free " << allocation->ptr();
  // During graph capturing, only free the memory blocks owned by the graph;
  // others are cached.
  if (phi::backends::gpu::CUDAGraph::IsThisThreadCapturing()) {
    if (graph_owned_allocations_.find(allocation) ==
        graph_owned_allocations_.end()) {
      // If the block is not owned by the graph, cache it for release after
      // capturing.
      phi::backends::gpu::CUDAGraph::AddPostCaptureCallbackDuringCapturing(
          [=]() {
            // Release this block after capturing
            VLOG(0) << "[PostCaptureCallback] Releasing ptr = "
                    << allocation->ptr() << " size = " << allocation->size();
            TryFree(allocation);
          });

      return;
    }
  }

  // If not capturing or if the block is graph-owned, free it immediately.
  if (phi::backends::gpu::CUDAGraph::IsThisThreadCapturing()) {
    graph_owned_allocations_.erase(allocation);
  }
  TryFree(allocation);
}

phi::Allocation* CUDAMallocAsyncAllocator::AllocateImpl(size_t size) {
  std::call_once(once_flag_, [this] { platform::SetDeviceId(place_.device); });
  ProcessUnfreedAllocations();

  void* ptr;
  auto result = platform::RecordedGpuMallocAsync(
      &ptr, size, place_.device, default_stream_);
  if (LIKELY(result == gpuSuccess)) {
    auto* allocation = new CUDAMallocAsyncAllocation(
        ptr, size, platform::Place(place_), default_stream_);

    // If capturing, associate allocation with the current graph.
    if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
      // auto graph_id = phi::backends::gpu::CUDAGraph::CapturingPoolID();
      graph_owned_allocations_.insert(allocation);
    }
    return allocation;
  }

  size_t avail, total, actual_avail, actual_total;
  bool is_limited = platform::RecordedGpuMemGetInfo(
      &avail, &total, &actual_avail, &actual_total, place_.device);
  size_t allocated = total - avail;

  std::string err_msg;
  if (is_limited) {
    auto limit_size = (total >> 20);
    err_msg = string::Sprintf(
        "Or set environment variable `FLAGS_gpu_memory_limit_mb` to a larger "
        "value. Currently `FLAGS_gpu_memory_limit_mb` is %d, so the maximum "
        "GPU memory usage is limited to %d MB.\n"
        "   The command is `export FLAGS_gpu_memory_limit_mb=xxx`.",
        limit_size,
        limit_size);
  }

  PADDLE_THROW_BAD_ALLOC(platform::errors::ResourceExhausted(
      "\n\nOut of memory error on GPU %d. "
      "Cannot allocate %s memory on GPU %d, %s memory has been allocated and "
      "available memory is only %s.\n\n"
      "Please check whether there is any other process using GPU %d.\n"
      "1. If yes, please stop them, or start PaddlePaddle on another GPU.\n"
      "2. If no, please decrease the batch size of your model. %s\n",
      place_.device,
      string::HumanReadableSize(size),
      place_.device,
      string::HumanReadableSize(allocated),
      string::HumanReadableSize(avail),
      place_.device,
      err_msg));
}

gpuStream_t CUDAMallocAsyncAllocator::GetDefaultStream() const {
  return default_stream_;
}

void CUDAMallocAsyncAllocator::SetDefaultStream(gpuStream_t stream) {
  default_stream_ = stream;
}

}  // namespace paddle::memory::allocation
