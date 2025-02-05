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

#include "paddle/phi/core/memory/allocation/stream_safe_custom_device_allocator.h"
#include <thread>

#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/backends/context_pool.h"

namespace paddle {
namespace memory {
namespace allocation {

StreamSafeCustomDeviceAllocation::StreamSafeCustomDeviceAllocation(
    DecoratedAllocationPtr underlying_allocation,
    phi::stream::stream_t owning_stream,
    StreamSafeCustomDeviceAllocator* allocator)
    : Allocation(underlying_allocation->ptr(),
                 underlying_allocation->base_ptr(),
                 underlying_allocation->size(),
                 underlying_allocation->place()),
      underlying_allocation_(std::move(underlying_allocation)),
      owning_stream_(std::move(owning_stream)),
      allocator_(allocator->shared_from_this()) {}

bool StreamSafeCustomDeviceAllocation::RecordStream(
    phi::stream::stream_t stream) {
  VLOG(8) << "Try record stream " << stream << " for address " << ptr();
  if (stream == owning_stream_) {
    return false;
  }
  std::call_once(once_flag_, [this] { phi::DeviceManager::SetDevice(place_); });
  std::lock_guard<SpinLock> lock_guard(outstanding_event_map_lock_);

  auto it = outstanding_event_map_.find(stream);
  if (it == outstanding_event_map_.end()) {
    outstanding_event_map_.insert(
        {stream, std::make_shared<phi::event::Event>()});
    outstanding_event_map_[stream]->Init(place());
    VLOG(9) << "Create a new event "
            << outstanding_event_map_[stream]->raw_event();
  }
  auto stream_wrapper = phi::stream::Stream(place(), stream);
  VLOG(8) << "Record event " << outstanding_event_map_[stream]->raw_event()
          << " to stream " << stream;
  outstanding_event_map_[stream]->Record(&stream_wrapper);
  return true;
}

bool StreamSafeCustomDeviceAllocation::CanBeFreed() {
  std::lock_guard<SpinLock> lock_guard(outstanding_event_map_lock_);
  if (!phi::DeviceManager::HasDeviceType(place_.GetDeviceType())) {
    return true;
  }
  std::call_once(once_flag_, [this] { phi::DeviceManager::SetDevice(place_); });
  for (auto it = outstanding_event_map_.begin();
       it != outstanding_event_map_.end();) {
    auto& event = it->second;
    if (!event->Query()) {
      VLOG(9) << "Event " << event->raw_event() << " for " << ptr()
              << " is not completed";
      return false;
    }
    VLOG(8) << "Destroy event " << event->raw_event();
    event->Destroy();
    it = outstanding_event_map_.erase(it);
  }
  outstanding_event_map_.clear();
  return true;
}

phi::stream::stream_t StreamSafeCustomDeviceAllocation::GetOwningStream()
    const {
  return owning_stream_;
}

void StreamSafeCustomDeviceAllocation::SetOwningStream(
    phi::stream::stream_t s) {
  owning_stream_ = s;
}

StreamSafeCustomDeviceAllocator::StreamSafeCustomDeviceAllocator(
    std::shared_ptr<Allocator> underlying_allocator,
    phi::CustomPlace place,
    phi::stream::stream_t default_stream)
    : underlying_allocator_(std::move(underlying_allocator)),
      place_(std::move(place)),
      default_stream_(std::move(default_stream)) {
  std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
  allocator_map_[place_].emplace_back(this);
}

StreamSafeCustomDeviceAllocator::~StreamSafeCustomDeviceAllocator() {
  std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
  std::vector<StreamSafeCustomDeviceAllocator*>& allocators =
      allocator_map_[place_];
  allocators.erase(std::remove(allocators.begin(), allocators.end(), this),
                   allocators.end());
}

phi::stream::stream_t StreamSafeCustomDeviceAllocator::GetDefaultStream()
    const {
  return default_stream_;
}

void StreamSafeCustomDeviceAllocator::SetDefaultStream(
    phi::stream::stream_t stream) {
  default_stream_ = stream;
}

phi::Allocation* StreamSafeCustomDeviceAllocator::AllocateImpl(size_t size) {
  phi::RecordEvent record("StreamSafeCustomDeviceAllocator::Allocate",
                          phi::TracerEventType::UserDefined,
                          9 /*level*/);
  ProcessUnfreedAllocations();
  VLOG(8) << "Try allocate " << size << " bytes";
  AllocationPtr underlying_allocation;
  try {
    underlying_allocation = underlying_allocator_->Allocate(size);
  } catch (BadAlloc&) {
    VLOG(4) << "Allocation failed when allocating " << size << " bytes";
    ReleaseImpl(place_);
    try {
      underlying_allocation = underlying_allocator_->Allocate(size);
    } catch (...) {
      VLOG(3)
          << "Still allocation failed after release memory from all streams";
      throw;
    }
  } catch (...) {
    throw;
  }
  StreamSafeCustomDeviceAllocation* allocation =
      new StreamSafeCustomDeviceAllocation(
          static_unique_ptr_cast<Allocation>(std::move(underlying_allocation)),
          default_stream_,
          this);
  VLOG(8) << "Thread " << std::this_thread::get_id() << " Allocate "
          << allocation->size() << " bytes at address " << allocation->ptr()
          << "  , stream: " << default_stream_;
  return allocation;
}

void StreamSafeCustomDeviceAllocator::FreeImpl(phi::Allocation* allocation) {
  phi::RecordEvent record("StreamSafeCustomDeviceAllocator::Free",
                          phi::TracerEventType::UserDefined,
                          9 /*level*/);
  StreamSafeCustomDeviceAllocation* stream_safe_cuda_allocation =
      static_cast<StreamSafeCustomDeviceAllocation*>(allocation);

  VLOG(8) << "Try free allocation " << stream_safe_cuda_allocation->ptr();
  if (!stream_safe_cuda_allocation->GetOwningStream()) {
    stream_safe_cuda_allocation->SetOwningStream(
        default_stream_ ? default_stream_
                        : reinterpret_cast<phi::CustomContext*>(
                              phi::DeviceContextPool::Instance().Get(place_))
                              ->stream());
  }
  if (stream_safe_cuda_allocation->CanBeFreed()) {
    VLOG(9) << "Directly delete allocation";
    delete stream_safe_cuda_allocation;
  } else {
    VLOG(9) << "Put into unfreed_allocation list";
    std::lock_guard<SpinLock> lock_guard(unfreed_allocation_lock_);
    unfreed_allocations_.emplace_back(stream_safe_cuda_allocation);
  }
}

uint64_t StreamSafeCustomDeviceAllocator::ReleaseImpl(const phi::Place& place) {
  std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
  std::vector<StreamSafeCustomDeviceAllocator*>& allocators =
      allocator_map_[place];
  uint64_t released_size = 0;
  for (StreamSafeCustomDeviceAllocator* allocator : allocators) {
    released_size += allocator->ProcessUnfreedAllocationsAndRelease();
  }
  VLOG(8) << "Release " << released_size << " bytes memory from all streams";
  return released_size;
}

void StreamSafeCustomDeviceAllocator::ProcessUnfreedAllocations() {
  // NOTE(Ruibiao): This condition is to reduce lock completion. It does not
  // need to be thread-safe since here occasional misjudgments are permissible.
  if (unfreed_allocations_.empty()) {
    return;
  }

  std::lock_guard<SpinLock> lock_guard(unfreed_allocation_lock_);
  for (auto it = unfreed_allocations_.begin();
       it != unfreed_allocations_.end();) {
    if ((*it)->CanBeFreed()) {
      delete *it;
      it = unfreed_allocations_.erase(it);
    } else {
      ++it;
    }
  }
}

uint64_t
StreamSafeCustomDeviceAllocator::ProcessUnfreedAllocationsAndRelease() {
  ProcessUnfreedAllocations();
  return underlying_allocator_->Release(place_);
}

thread_local std::once_flag StreamSafeCustomDeviceAllocation::once_flag_;

std::map<phi::Place, std::vector<StreamSafeCustomDeviceAllocator*>>
    StreamSafeCustomDeviceAllocator::allocator_map_;
SpinLock StreamSafeCustomDeviceAllocator::allocator_map_lock_;

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
