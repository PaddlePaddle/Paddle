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

#include "paddle/phi/core/memory/allocation/stream_safe_xpu_allocator.h"
#include <thread>

#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {

StreamSafeXPUAllocation::StreamSafeXPUAllocation(
    DecoratedAllocationPtr underlying_allocation,
    XPUStream owning_stream,
    StreamSafeXPUAllocator* allocator)
    : Allocation(underlying_allocation->ptr(),
                 underlying_allocation->base_ptr(),
                 underlying_allocation->size(),
                 underlying_allocation->place()),
      underlying_allocation_(std::move(underlying_allocation)),
      owning_stream_(std::move(owning_stream)),
      allocator_(allocator->shared_from_this()) {}

bool StreamSafeXPUAllocation::RecordStream(XPUStream stream) {
  VLOG(8) << "Try record stream " << stream << " for address " << ptr();
  if (stream == owning_stream_) {
    return false;
  }

  std::call_once(once_flag_,
                 [this] { phi::backends::xpu::SetXPUDeviceId(place_.device); });

  std::lock_guard<SpinLock> lock_guard(outstanding_event_map_lock_);

  RecordStreamPrivate(stream);
  return true;
}

bool StreamSafeXPUAllocation::CanBeFreed() {
  std::call_once(once_flag_,
                 [this] { phi::backends::xpu::SetXPUDeviceId(place_.device); });
  for (auto it = outstanding_event_map_.begin();
       it != outstanding_event_map_.end();
       ++it) {
    XPUEvent& event = it->second;

    PADDLE_ENFORCE_XRE_SUCCESS(xpu_event_destroy(event));
    VLOG(8) << "Destroy event " << event;
  }
  return true;
}

XPUStream StreamSafeXPUAllocation::GetOwningStream() const {
  return owning_stream_;
}

void StreamSafeXPUAllocation::RecordStreamPrivate(XPUStream stream) {
  XPUEvent record_event;
  auto it = outstanding_event_map_.find(stream);
  if (it == outstanding_event_map_.end()) {
    XPUEvent new_event;
    PADDLE_ENFORCE_XRE_SUCCESS(xpu_event_create(&new_event));
    outstanding_event_map_[stream] = new_event;
    record_event = new_event;
    VLOG(9) << "Create a new event " << new_event;
  } else {
    record_event = it->second;
    VLOG(9) << "Reuse event " << record_event;
  }

  PADDLE_ENFORCE_XRE_SUCCESS(xpu_event_record(record_event, stream));
  VLOG(8) << "Record event " << record_event << " to stream " << stream;
}

StreamSafeXPUAllocator::StreamSafeXPUAllocator(
    std::shared_ptr<Allocator> underlying_allocator,
    phi::XPUPlace place,
    XPUStream default_stream)
    : underlying_allocator_(std::move(underlying_allocator)),
      place_(std::move(place)),
      default_stream_(std::move(default_stream)) {
  std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
  allocator_map_[place].emplace_back(this);
}

StreamSafeXPUAllocator::~StreamSafeXPUAllocator() {
  std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
  std::vector<StreamSafeXPUAllocator*>& allocators = allocator_map_[place_];
  allocators.erase(std::remove(allocators.begin(), allocators.end(), this),
                   allocators.end());
}

bool StreamSafeXPUAllocator::IsAllocThreadSafe() const { return true; }

XPUStream StreamSafeXPUAllocator::GetDefaultStream() const {
  return default_stream_;
}

void StreamSafeXPUAllocator::SetDefaultStream(XPUStream stream) {
  default_stream_ = stream;
}

phi::Allocation* StreamSafeXPUAllocator::AllocateImpl(size_t size) {
  phi::RecordEvent record("StreamSafeXPUAllocator::Allocate",
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
  StreamSafeXPUAllocation* allocation = new StreamSafeXPUAllocation(
      static_unique_ptr_cast<Allocation>(std::move(underlying_allocation)),
      default_stream_,
      this);
  VLOG(8) << "Thread " << std::this_thread::get_id() << " Allocate "
          << allocation->size() << " bytes at address " << allocation->ptr()
          << "  , stream: " << default_stream_;
  return allocation;
}

void StreamSafeXPUAllocator::FreeImpl(phi::Allocation* allocation) {
  phi::RecordEvent record("StreamSafeXPUAllocator::Free",
                          phi::TracerEventType::UserDefined,
                          9 /*level*/);
  StreamSafeXPUAllocation* stream_safe_xpu_allocation =
      static_cast<StreamSafeXPUAllocation*>(allocation);

  VLOG(8) << "Try free allocation " << stream_safe_xpu_allocation->ptr();
  if (stream_safe_xpu_allocation->CanBeFreed()) {
    VLOG(9) << "Directly delete allocation";
    delete stream_safe_xpu_allocation;
  } else {
    VLOG(9) << "Put into unfreed_allocation list";
    std::lock_guard<SpinLock> lock_guard(unfreed_allocation_lock_);
    unfreed_allocations_.emplace_back(stream_safe_xpu_allocation);
  }
}

uint64_t StreamSafeXPUAllocator::ReleaseImpl(const phi::Place& place) {
  std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
  std::vector<StreamSafeXPUAllocator*>& allocators = allocator_map_[place];
  uint64_t released_size = 0;
  for (StreamSafeXPUAllocator* allocator : allocators) {
    released_size += allocator->ProcessUnfreedAllocationsAndRelease();
  }
  VLOG(8) << "Release " << released_size << " bytes memory from all streams";
  return released_size;
}

void StreamSafeXPUAllocator::ProcessUnfreedAllocations() {
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

uint64_t StreamSafeXPUAllocator::ProcessUnfreedAllocationsAndRelease() {
  ProcessUnfreedAllocations();
  return underlying_allocator_->Release(place_);
}

thread_local std::once_flag StreamSafeXPUAllocation::once_flag_;

std::map<phi::Place, std::vector<StreamSafeXPUAllocator*>>
    StreamSafeXPUAllocator::allocator_map_;
SpinLock StreamSafeXPUAllocator::allocator_map_lock_;

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
