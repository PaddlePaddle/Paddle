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

#include "paddle/fluid/memory/allocation/custom_pinned_allocator.h"

#include "paddle/fluid/memory/stats.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler/mem_tracing.h"

namespace paddle {
namespace memory {
namespace allocation {
bool CustomCPUPinnedAllocator::IsAllocThreadSafe() const { return true; }
void CustomCPUPinnedAllocator::FreeImpl(phi::Allocation* allocation) {
  phi::DeviceManager::GetDeviceWithPlace(place_)->MemoryDeallocateHost(
      allocation->ptr(), allocation->size());
  VLOG(10) << "CustomPinnedFree " << allocation->ptr();
  HOST_MEMORY_STAT_UPDATE(Reserved, 0, -allocation->size());
  platform::RecordMemEvent(allocation->ptr(),
                           allocation->place(),
                           allocation->size(),
                           platform::TracerMemEventType::ReservedFree);
  delete allocation;
}
phi::Allocation* CustomCPUPinnedAllocator::AllocateImpl(size_t size) {
  void* ptr =
      phi::DeviceManager::GetDeviceWithPlace(place_)->MemoryAllocateHost(size);
  if (LIKELY(ptr)) {
    VLOG(10) << "CustomPinnedAlloc " << size << " " << ptr;
    HOST_MEMORY_STAT_UPDATE(Reserved, 0, size);
    platform::RecordMemEvent(ptr,
                             platform::CustomPinnedPlace(),
                             size,
                             platform::TracerMemEventType::ReservedAllocate);
    return new Allocation(ptr, size, platform::CustomPinnedPlace());
  }
  PADDLE_THROW_BAD_ALLOC(platform::errors::ResourceExhausted(
      "\n\nOut of memory error on Pinned Memory for %s\n\n", place_));
  return nullptr;
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
