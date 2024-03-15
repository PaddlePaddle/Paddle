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

#include <cstddef>
#include <cstdint>
#include <mutex>
#include "paddle/common/macros.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/allocation/cuda_malloc_async_allocator.h"
#include "paddle/fluid/memory/allocation/stream_safe_cuda_allocator.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <string>

#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/backends/gpu/cuda/cuda_graph.h"

/*
 * Note: [cuda_malloc_async_pool_memory_throttle_ratio]
 * The primary purpose of the memory_throttle_ratio is to provide a
 * threshold that determines when to initiate synchronization operations to
 * deallocate memory. This mechanism helps in ensuring that the system does
 * not exceed its memory capacity while also attempting to minimize performance
 * degradation caused by frequent memory synchronization.
 *
 * ```
 *   utilization = (allocated_size + pending_release_size) / total_memory_size
 *   if(utilization > memory_throttle_ratio)
 *      sync(free_stream, malloc_stream)
 * ```
 *
 * When the utilization exceeds the memory_throttle_ratio, we
 * initiate a stream synchronization operation before malloc.
 *
 * During synchronization, all memory deallocation requests in the free queue
 * are processed, effectively lowering the memory utilization before
 * any new memory allocation operations are going to proceed.
 *
 * [Impact on Performance and Memory Usage]
 *
 * - Lower memory_throttle_ratio Values
 * the synchronization operation will be triggered more frequently.
 * This can lead to better memory utilization but might result in decreased
 * performance due to the increased number of synchronization operations.
 *
 * - Higher memory_throttle_ratio Values
 * Conversely, setting a higher value allows for more memory to be allocated
 * before triggering synchronization, which can enhance performance by reducing
 * the number of sync operations. However, this increases the risk of reaching
 * an OOM condition since more memory can be allocated without
 * immediate deallocation.
 */
PHI_DECLARE_double(cuda_malloc_async_pool_memory_throttle_ratio);

namespace paddle {
namespace memory {
namespace allocation {

thread_local std::once_flag CUDAMallocAsyncAllocation::once_flag_;

inline void sync_streams(gpuStream_t to_record, gpuStream_t to_wait) {
  cudaEvent_t event = nullptr;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, to_record));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(to_wait, event));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(event));
}

// CUDAMallocAsyncAllocation

void CUDAMallocAsyncAllocation::RecordStream(gpuStream_t stream) {
  std::call_once(once_flag_,
                 [this] { phi::backends::gpu::SetDeviceId(place_.device); });
  std::lock_guard<SpinLock> lock_guard(recorded_streams_lock_);
  if (malloc_stream_ == stream) {
    // Called record_stream on tensor whose original malloc_stream matches the
    // recorded stream. This should have no effect.
    return;
  }
  recorded_streams_.insert(stream);
}

void CUDAMallocAsyncAllocation::EraseStream(gpuStream_t stream) {
  std::lock_guard<SpinLock> lock_guard(recorded_streams_lock_);
  recorded_streams_.erase(stream);
}

size_t CUDAMallocAsyncAllocation::Free() {
  if (recorded_streams_.empty()) {
    platform::RecordedGpuFreeAsync(
        ptr(), size(), place_.device, malloc_stream_);

    if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
      phi::backends::gpu::CUDAGraph::AddJoiningStreamDuringCapturing(
          malloc_stream_);
    }
    return size();
  } else {
    sync_streams(malloc_stream_, free_stream_);

    for (const auto& recorded_stream : recorded_streams_) {
      sync_streams(recorded_stream, free_stream_);
    }

    platform::RecordedGpuFreeAsync(ptr(), size(), place_.device, free_stream_);

    if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
      phi::backends::gpu::CUDAGraph::AddJoiningStreamDuringCapturing(
          free_stream_);
    }
    return 0;
  }
}

// CUDAMallocAsyncAllocator

CUDAMallocAsyncAllocator::CUDAMallocAsyncAllocator(
    std::shared_ptr<Allocator> underlying_allocator,
    const platform::CUDAPlace& place,
    gpuStream_t default_stream)
    : underlying_allocator_(std::move(underlying_allocator)),
      place_(place),
      default_stream_(default_stream),
      current_allocated_size_(0),
      pending_release_size_(0),
      memory_throttle_ratio_(
          FLAGS_cuda_malloc_async_pool_memory_throttle_ratio) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamCreateWithPriority(&free_stream_, cudaStreamNonBlocking, 0));
  cudaDeviceGetDefaultMemPool(&mempool_, place.device);

  size_t avail, total, actual_avail, actual_total;
  platform::RecordedGpuMemGetInfo(
      &avail, &total, &actual_avail, &actual_total, place_.device);
  max_size_ = total;

  phi::backends::gpu::CUDAGraph::AddPreCaptureCallback(
      [&]() { this->ClearFreeStream(true); });

  VLOG(0) << "[CUDAMallocAsyncAllocator] " << (this) << " place " << place
          << " max_size " << string::HumanReadableSize(max_size_)
          << " memory_throttle_ratio " << memory_throttle_ratio_;
}

uint64_t CUDAMallocAsyncAllocator::ReleaseImpl(const platform::Place& place) {
  if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
    VLOG(7) << "Memory release forbidden in CUDA Graph Captruing";
    return 0;
  }

  uint64_t released_size = 0;
  // we synchronize the event so all the block could be release.
  if (underlying_allocator_)
    released_size += underlying_allocator_->Release(place_);
  VLOG(8) << "Release " << released_size << " bytes memory from all streams";
  return released_size;
}

void CUDAMallocAsyncAllocator::ClearFreeStream(bool sync) {
  if (sync) {
    VLOG(0)<< "[CUDAMallocAsyncAllocator] " << (this)  << " synchronize the free stream to ensure all unrelesed blocks are freed";
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(free_stream_));
  } else {
    sync_streams(free_stream_, default_stream_);
  }
  current_allocated_size_ -= pending_release_size_;
  pending_release_size_ = 0;
}

void CUDAMallocAsyncAllocator::MallocThrottling() {
  if(UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())){
    // we disable MallocThrottling when capturing
    return;
  }
  double allocated =
      static_cast<double>(current_allocated_size_ + pending_release_size_);
  double utilization = allocated / static_cast<double>(max_size_);

  if (utilization > memory_throttle_ratio_) {
    VLOG(10) << "utilization_ratio " << utilization
             << " current_allocated_size "
             << string::HumanReadableSize(current_allocated_size_)
             << " pending_release_size "
             << string::HumanReadableSize(pending_release_size_);
    CUDAMallocAsyncAllocator::ClearFreeStream();
  }
}

void CUDAMallocAsyncAllocator::FreeAllocation(
    CUDAMallocAsyncAllocation* allocation) {
  auto current_released_size = allocation->Free();
  current_allocated_size_ -= current_released_size;
  // The amount of pending release size (the space that has been queued to
  // free_stream, that are going to be freed in the future)
  pending_release_size_ += (allocation->size() - current_released_size);
}

void CUDAMallocAsyncAllocator::FreeImpl(phi::Allocation* phi_allocation) {
  auto* allocation = dynamic_cast<CUDAMallocAsyncAllocation*>(phi_allocation);

  // During graph capturing, only free the memory blocks owned by the graph;
  // others are cached.
  if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
    if (graph_owned_allocations_.find(allocation) ==
        graph_owned_allocations_.end()) {
      // If the block is not owned by the graph, cache it for release after
      // capturing.
      phi::backends::gpu::CUDAGraph::AddPostCaptureCallbackDuringCapturing(
          [=]() {
            // Release this block after capturing
            VLOG(0) << "[PostCaptureCallback] Releasing ptr = "
                    << allocation->ptr() << " size = "
                    << string::HumanReadableSize(allocation->size());
            FreeAllocation(allocation);
          });

      return;
    }
    // the block is graph-owned, free it immediately.
    graph_owned_allocations_.erase(allocation);
  }

  // If not capturing or if the block is graph-owned, free it immediately.
  FreeAllocation(allocation);
}

phi::Allocation* CUDAMallocAsyncAllocator::AllocateImpl(size_t size) {
  std::call_once(once_flag_, [this] { platform::SetDeviceId(place_.device); });

  MallocThrottling();

  void* ptr;
  auto result = platform::RecordedGpuMallocAsync(
      &ptr, size, place_.device, default_stream_);
  if (LIKELY(result == gpuSuccess)) {
    auto* allocation = new CUDAMallocAsyncAllocation(
        ptr, size, platform::Place(place_), default_stream_, free_stream_);

    // If capturing, associate allocation with the current graph.
    if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
      graph_owned_allocations_.insert(allocation);
    }
    current_allocated_size_ += size;
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

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
