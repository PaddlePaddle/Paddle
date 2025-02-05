// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/custom_allocator.h"

#include "paddle/phi/api/profiler/trace_event.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/platform/device/device_wrapper.h"
#include "paddle/phi/core/platform/profiler.h"

namespace paddle {
namespace memory {
namespace allocation {

bool CustomAllocator::IsAllocThreadSafe() const { return true; }
void CustomAllocator::FreeImpl(phi::Allocation* allocation) {
  PADDLE_ENFORCE_EQ(
      allocation->place(),
      place_,
      common::errors::PermissionDenied("CustomDevice memory is "
                                       "freed in incorrect device. "
                                       "This may be a bug"));
  if (phi::DeviceManager::HasDeviceType(place_.GetDeviceType())) {
    phi::DeviceManager::GetDeviceWithPlace(place_)->MemoryDeallocate(
        allocation->ptr(), allocation->size());
  }
  DEVICE_MEMORY_STAT_UPDATE(
      Reserved, place_.GetDeviceId(), -allocation->size());
  platform::RecordMemEvent(allocation->ptr(),
                           place_,
                           allocation->size(),
                           phi::TracerMemEventType::ReservedFree);
  delete allocation;
}

phi::Allocation* CustomAllocator::AllocateImpl(size_t size) {
  std::call_once(once_flag_, [this] { phi::DeviceManager::SetDevice(place_); });

  void* ptr =
      phi::DeviceManager::GetDeviceWithPlace(place_)->MemoryAllocate(size);
  if (LIKELY(ptr)) {
    DEVICE_MEMORY_STAT_UPDATE(Reserved, place_.GetDeviceId(), size);
    platform::RecordMemEvent(
        ptr, place_, size, phi::TracerMemEventType::ReservedAllocate);
    return new Allocation(ptr, size, place_);
  }

  size_t avail, total;
  phi::DeviceManager::MemoryStats(place_, &total, &avail);

  auto dev_type = phi::PlaceHelper::GetDeviceType(place_);
  auto dev_id = phi::PlaceHelper::GetDeviceId(place_);

  PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
      "\n\nOut of memory error on %s:%d. "
      "Cannot allocate %s memory on %s:%d, "
      "available memory is only %s.\n\n"
      "Please check whether there is any other process using %s:%d.\n"
      "1. If yes, please stop them, or start PaddlePaddle on another %s.\n"
      "2. If no, please decrease the batch size of your model.\n\n",
      dev_type,
      dev_id,
      string::HumanReadableSize(size),
      dev_type,
      dev_id,
      string::HumanReadableSize(avail),
      dev_type,
      dev_id,
      dev_type));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
