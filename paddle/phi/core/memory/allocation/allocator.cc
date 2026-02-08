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

#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/memory/stats.h"

namespace paddle::memory::allocation {

void Allocator::FreeImpl(phi::Allocation* allocation) {
  static_cast<Allocation*>(allocation)
      ->TopDecoratedAllocator()
      ->Free(allocation);
}

void MultiScalePoolAllocator::RecordAlloc(uintptr_t allocator,
                                          uint64_t id,
                                          size_t size) {
#if defined(PADDLE_WITH_CUDA)
  std::lock_guard<SpinLock> lock(spinlock_);
  const auto current_device_id = phi::backends::gpu::GetCurrentDeviceId();
  const auto max_reserved =
      paddle::memory::DeviceMemoryStatPeakValue("Reserved", current_device_id);
  const auto cur_allocated = paddle::memory::DeviceMemoryStatCurrentValue(
      "Allocated", current_device_id);
  allocation_records_.emplace_back(
      allocator, true, id, size, cur_allocated, max_reserved);
#endif
}

void MultiScalePoolAllocator::RecordFree(uintptr_t allocator,
                                         uint64_t id,
                                         size_t size) {
#if defined(PADDLE_WITH_CUDA)
  std::lock_guard<SpinLock> lock(spinlock_);
  const auto current_device_id = phi::backends::gpu::GetCurrentDeviceId();
  const auto max_reserved =
      paddle::memory::DeviceMemoryStatPeakValue("Reserved", current_device_id);
  const auto cur_allocated = paddle::memory::DeviceMemoryStatCurrentValue(
      "Allocated", current_device_id);
  allocation_records_.emplace_back(
      allocator, false, id, size, cur_allocated, max_reserved);
#endif
}

}  // namespace paddle::memory::allocation
