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

#include "paddle/fluid/memory/allocation/pinned_allocator.h"

#include "paddle/fluid/memory/stats.h"
#include "paddle/fluid/platform/profiler/mem_tracing.h"

#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#endif
namespace paddle {
namespace memory {
namespace allocation {
bool CPUPinnedAllocator::IsAllocThreadSafe() const { return true; }
void CPUPinnedAllocator::FreeImpl(phi::Allocation *allocation) {
#if defined(PADDLE_WITH_HIP)
  PADDLE_ENFORCE_GPU_SUCCESS(hipHostFree(allocation->ptr()));
#elif defined(PADDLE_WITH_XPU)
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_host_free(allocation->ptr()));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaFreeHost(allocation->ptr()));
#endif
  VLOG(10) << "cudaFreeHost " << allocation->ptr();
  HOST_MEMORY_STAT_UPDATE(Reserved, 0, -allocation->size());
  platform::RecordMemEvent(allocation->ptr(),
                           allocation->place(),
                           allocation->size(),
                           platform::TracerMemEventType::ReservedFree);
  delete allocation;
}
phi::Allocation *CPUPinnedAllocator::AllocateImpl(size_t size) {
  void *ptr;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#if defined(PADDLE_WITH_HIP)
  PADDLE_ENFORCE_GPU_SUCCESS(hipHostMalloc(&ptr, size, hipHostMallocPortable));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaHostAlloc(&ptr, size, cudaHostAllocPortable));
#endif
  VLOG(10) << "cudaHostAlloc " << size << " " << ptr;
  HOST_MEMORY_STAT_UPDATE(Reserved, 0, size);
  platform::RecordMemEvent(ptr,
                           platform::CUDAPinnedPlace(),
                           size,
                           platform::TracerMemEventType::ReservedAllocate);
  return new Allocation(ptr, size, platform::CUDAPinnedPlace());
#endif
#ifdef PADDLE_WITH_XPU
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_host_alloc(&ptr, size, 0));
  VLOG(10) << "xpu_host_alloc" << size << " " << ptr;
  return new Allocation(ptr, size, platform::XPUPinnedPlace());
#endif
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
