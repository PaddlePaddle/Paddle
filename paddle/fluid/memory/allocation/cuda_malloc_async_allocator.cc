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

#include <cstdint>
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

namespace paddle {
namespace memory {
namespace allocation {
bool CUDAMallocAsyncAllocator::IsAllocThreadSafe() const { return true; }
void CUDAMallocAsyncAllocator::FreeImpl(phi::Allocation* allocation) {
  CUDAMallocAsyncAllocation* allocation_ =
      dynamic_cast<CUDAMallocAsyncAllocation*>(allocation);

  if ((UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing()))) {
    // If the graph is in capturing mode, only the memory blocks owned by the
    // graph should be freed. Meanwhile, the blocks that are not owned by
    // the graph are retained and will be released once the capturing is
    // complete.
    auto id = phi::backends::gpu::CUDAGraph::CapturingPoolID();
    if (graph_owned_allocation[id].find(allocation_) ==
        graph_owned_allocation[id].end()) {
      VLOG(0) << "Cache block ptr = " << allocation->ptr()
              << " to release after capturing.";
      cached_blocks_during_capturing.insert(allocation_);
      return;
    }
  }

  PADDLE_ENFORCE_EQ(
      allocation->place(),
      place_,
      platform::errors::PermissionDenied(
          "GPU memory is freed in incorrect device. This may be a bug"));
  platform::RecordedGpuFreeAsync(allocation->ptr(),
                                 allocation->size(),
                                 place_.device,
                                 allocation_->GetOwningStream());
  delete allocation;
}

void CUDAMallocAsyncAllocator::FlushCachedBlockDuringCapturing(
    int64_t graph_id) {
  if ((UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing()))) {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "This API should be called after the graph is captured."));
  }
  for (auto allocation : graph_owned_allocation[graph_id]) {
    FreeImpl(allocation);
  }
}

phi::Allocation* CUDAMallocAsyncAllocator::AllocateImpl(size_t size) {
  std::call_once(once_flag_, [this] { platform::SetDeviceId(place_.device); });

  void* ptr;
  auto result =
      platform::RecordedGpuMallocAsync(&ptr, size, place_.device, stream_);
  if (LIKELY(result == gpuSuccess)) {
    auto allocation = new CUDAMallocAsyncAllocation(
        ptr, size, platform::Place(place_), stream_);

    if ((UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing()))) {
      auto id = phi::backends::gpu::CUDAGraph::CapturingPoolID();
      graph_owned_allocation[id].insert(allocation);
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
