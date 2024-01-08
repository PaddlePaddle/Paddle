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
#include "paddle/phi/backends/gpu/cuda/cuda_graph.h"

namespace paddle {
namespace memory {
namespace allocation {
bool CUDAMallocAsyncAllocator::IsAllocThreadSafe() const { return true; }
void CUDAMallocAsyncAllocator::FreeImpl(phi::Allocation* allocation) {
  auto* casted_allocation =
      dynamic_cast<CUDAMallocAsyncAllocation*>(allocation);

  // During graph capturing, only free the memory blocks owned by the graph;
  // others are cached.
  if (phi::backends::gpu::CUDAGraph::IsThisThreadCapturing()) {
    if (graph_owned_allocations_.find(casted_allocation) ==
        graph_owned_allocations_.end()) {
      // If the block is not owned by the graph, cache it for release after
      // capturing.
      phi::backends::gpu::CUDAGraph::AddPostCaptureCallbackDuringCapturing(
          [=]() {
            // Release this block after capturing
            VLOG(0) << "[PostCaptureCallback] Releasing ptr = "
                    << allocation->ptr() << " size = " << allocation->size();
            platform::RecordedGpuFreeAsync(
                allocation->ptr(),
                allocation->size(),
                place_.device,
                casted_allocation->GetOwningStream());
            delete allocation;
          });

      return;
    }
  }

  // If not capturing or if the block is graph-owned, free it immediately.
  platform::RecordedGpuFreeAsync(allocation->ptr(),
                                 allocation->size(),
                                 place_.device,
                                 casted_allocation->GetOwningStream());
  if (phi::backends::gpu::CUDAGraph::IsThisThreadCapturing()) {
    graph_owned_allocations_.erase(casted_allocation);
  }
  delete allocation;
}

phi::Allocation* CUDAMallocAsyncAllocator::AllocateImpl(size_t size) {
  std::call_once(once_flag_, [this] { platform::SetDeviceId(place_.device); });

  void* ptr;
  auto result =
      platform::RecordedGpuMallocAsync(&ptr, size, place_.device, stream_);
  if (LIKELY(result == gpuSuccess)) {
    auto* allocation = new CUDAMallocAsyncAllocation(
        ptr, size, platform::Place(place_), stream_);

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
  return stream_;
}

void CUDAMallocAsyncAllocator::SetDefaultStream(gpuStream_t stream) {
  stream_ = stream;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
