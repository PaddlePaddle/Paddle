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

#include "paddle/phi/core/memory/allocation/xpu_pinned_allocator.h"

#include "paddle/phi/core/memory/stats.h"
#include "paddle/phi/core/platform/profiler/mem_tracing.h"

#if defined(PADDLE_WITH_XPU)
#include <cuda.h>
#include <cuda_runtime.h>
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#endif

namespace paddle::memory::allocation {

// Define the destructor so the vtable gets emitted.
XPUPinnedAllocator::~XPUPinnedAllocator() = default;

bool XPUPinnedAllocator::IsAllocThreadSafe() const { return true; }

void XPUPinnedAllocator::FreeImpl(phi::Allocation* allocation) {
#if defined(PADDLE_WITH_XPU)
  PADDLE_ENFORCE_XPU_SUCCESS(cudaFreeHost(allocation->ptr()));
#else
  PADDLE_THROW(common::errors::PermissionDenied(
      "'XPUPinnedPlace' is not supported. Please re-compile with WITH_XPU."));
#endif
  VLOG(10) << "cudaFreeHost " << allocation->ptr();
  HOST_MEMORY_STAT_UPDATE(Reserved, 0, -allocation->size());
  platform::RecordMemEvent(allocation->ptr(),
                           allocation->place(),
                           allocation->size(),
                           phi::TracerMemEventType::ReservedFree);
  delete allocation;
}

phi::Allocation* XPUPinnedAllocator::AllocateImpl(size_t size) {
  void* ptr;

#if defined(PADDLE_WITH_XPU)
  PADDLE_ENFORCE_XPU_SUCCESS(cudaHostAlloc(&ptr, size, cudaHostAllocPortable));
#else
  PADDLE_THROW(common::errors::PermissionDenied(
      "'XPUPinnedPlace' is not supported. Please re-compile with WITH_XPU."));
#endif

  VLOG(10) << "cudaHostAlloc " << size << " " << ptr;
  HOST_MEMORY_STAT_UPDATE(Reserved, 0, size);
  platform::RecordMemEvent(ptr,
                           phi::XPUPinnedPlace(),
                           size,
                           phi::TracerMemEventType::ReservedAllocate);
  return new Allocation(ptr, size, phi::XPUPinnedPlace());
}

}  // namespace paddle::memory::allocation
