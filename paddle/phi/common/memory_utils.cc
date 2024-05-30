// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/memory_utils.h"

namespace phi::memory_utils {

Allocator::AllocationPtr Alloc(const phi::Place& place,
                               size_t size,
                               const phi::Stream& stream) {
  return MemoryUtils::Instance().Alloc(place, size, stream);
}

Allocator::AllocationPtr Alloc(const phi::Place& place, size_t size) {
  return MemoryUtils::Instance().Alloc(place, size);
}

std::shared_ptr<Allocation> AllocShared(const phi::Place& place,
                                        size_t size,
                                        const phi::Stream& stream) {
  return MemoryUtils::Instance().AllocShared(place, size, stream);
}

std::shared_ptr<Allocation> AllocShared(const phi::Place& place, size_t size) {
  return MemoryUtils::Instance().AllocShared(place, size);
}

bool InSameStream(const std::shared_ptr<Allocation>& allocation,
                  const phi::Stream& stream) {
  return MemoryUtils::Instance().InSameStream(allocation, stream);
}

void AllocationDeleter(Allocation* allocation) {
  MemoryUtils::Instance().AllocationDeleter(allocation);
}

void Copy(const Place& dst_place,
          void* dst,
          const Place& src_place,
          const void* src,
          size_t num,
          void* stream) {
  MemoryUtils::Instance().Copy(dst_place, dst, src_place, src, num, stream);
}

void Copy(const Place& dst_place,
          void* dst,
          const Place& src_place,
          const void* src,
          size_t num) {
  MemoryUtils::Instance().Copy(dst_place, dst, src_place, src, num);
}

int64_t DeviceMemoryStatCurrentValue(const std::string& stat_type, int dev_id) {
  return MemoryUtils::Instance().DeviceMemoryStatCurrentValue(stat_type,
                                                              dev_id);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void GpuMemoryUsage(size_t* available, size_t* total) {
  return MemoryUtils::Instance().GpuMemoryUsage(available, total);
}
#endif

void InitDevices() { MemoryUtils::Instance().InitDevices(); }

void EmplaceDeviceContexts(
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        place_to_device_context,
    const std::vector<phi::Place>& places,
    bool disable_setting_default_stream_for_allocator,
    int stream_priority) {
  MemoryUtils::Instance().EmplaceDeviceContexts(
      place_to_device_context,
      places,
      disable_setting_default_stream_for_allocator,
      stream_priority);
}

#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && \
    (defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL))
const phi::Allocator* GetAllocator(int device_id, phi::gpuStream_t stream) {
  return MemoryUtils::Instance().GetAllocator(device_id, stream);
}

const phi::Allocator* GetHostAllocator() {
  return MemoryUtils::Instance().GetHostAllocator();
}

const phi::Allocator* GetZeroAllocator(int device_id) {
  return MemoryUtils::Instance().GetZeroAllocator(device_id);
}

const phi::Allocator* GetHostZeroAllocator() {
  return MemoryUtils::Instance().GetHostZeroAllocator();
}

const phi::Allocator* GetPinnedAllocator() {
  return MemoryUtils::Instance().GetPinnedAllocator();
}

std::shared_ptr<std::remove_pointer<phi::gpuEvent_t>::type> GetCudaEvent(
    int device_id) {
  return MemoryUtils::Instance().GetCudaEvent(device_id);
}
#elif (defined(PADDLE_WITH_XPU) && defined(PADDLE_WITH_XPU_BKCL))
const phi::Allocator* GetAllocator(int device_id, XPUStream stream) {
  return MemoryUtils::Instance().GetAllocator(device_id, stream);
}

const phi::Allocator* GetHostAllocator() {
  return MemoryUtils::Instance().GetHostAllocator();
}

const phi::Allocator* GetZeroAllocator(int device_id) {
  return MemoryUtils::Instance().GetZeroAllocator(device_id);
}

const phi::Allocator* GetHostZeroAllocator() {
  return MemoryUtils::Instance().GetHostZeroAllocator();
}

// XPUs do not have the concept of pinned memory,
// so the get_pinned_allocator function is not set.
std::shared_ptr<std::remove_pointer<XPUEvent>::type> GetXpuEvent(
    int device_id) {
  return MemoryUtils::Instance().GetXpuEvent(device_id);
}
#endif

}  // namespace phi::memory_utils
