// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/stream_safe_cuda_allocator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

StreamSafeCUDAAllocation::StreamSafeCUDAAllocation(
    AllocationPtr underlying_allocation, gpuStream_t owning_stream)
    : Allocation(underlying_allocation->ptr(), underlying_allocation->size(),
                 underlying_allocation->place()),
      underlying_allocation_(std::move(underlying_allocation)),
      owning_stream_(owning_stream) {}

void StreamSafeCUDAAllocation::RecordStream(gpuStream_t stream) {
  if (stream == owning_stream_) {
    return;
  }

  std::lock_guard<SpinLock> lock_guard(outstanding_event_map_lock_);
  gpuEvent_t recored_event;
  auto it = outstanding_event_map_.find(stream);
  if (it == outstanding_event_map_.end()) {
    gpuEvent_t new_event;
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaEventCreateWithFlags(&new_event, cudaEventDisableTiming));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(
        hipEventCreateWithFlags(&new_event, hipEventDisableTiming));
#endif
    outstanding_event_map_[stream] = new_event;
    recored_event = new_event;
  } else {
    recored_event = it->second;
  }

#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(recored_event, stream));
#else
  PADDLE_ENFORCE_CUDA_SUCCESS(hipEventRecord(recored_event, stream));
#endif
}

bool StreamSafeCUDAAllocation::CanBeFreed() {
  // NOTE(Ruibiao): This function will not execute concurrently, so
  // outstanding_event_lock_ is not required here
  for (auto it = outstanding_event_map_.begin();
       it != outstanding_event_map_.end(); ++it) {
    gpuEvent_t& event = it->second;
#ifdef PADDLE_WITH_CUDA
    gpuError_t err = cudaEventQuery(event);
    if (err == cudaErrorNotReady) {
      VLOG(9) << "Event " << event << " for " << ptr() << " is not completed";
      // Erase the completded event before "it"
      outstanding_event_map_.erase(outstanding_event_map_.begin(), it);
      return false;
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(err);
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventDestroy(event));
#else
    gpuError_t err = hipEventQuery(*deque_it);
    if (err == hipErrorNotReady) {
      VLOG(9) << "Event " << event << " for " << ptr() << " is not completed";
      // Erase the completded event before "it"
      outstanding_event_map_.erase(outstanding_event_map_.begin(), it);
      return false;
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(err);
    PADDLE_ENFORCE_CUDA_SUCCESS(hipEventDestroy(event));
#endif
  }
  return true;
}

StreamSafeCUDAAllocator::StreamSafeCUDAAllocator(
    const std::shared_ptr<Allocator>& underlying_allocator,
    const platform::CUDAPlace& place, const gpuStream_t default_stream)
    : underlying_allocator_(underlying_allocator),
      place_(place),
      default_stream_(default_stream) {
  std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
  allocator_map_[place].emplace_back(this);
}

StreamSafeCUDAAllocator::~StreamSafeCUDAAllocator() {
  std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
  std::vector<StreamSafeCUDAAllocator*>& allocators = allocator_map_[place_];
  allocators.erase(std::remove(allocators.begin(), allocators.end(), this),
                   allocators.end());
}

bool StreamSafeCUDAAllocator::IsAllocThreadSafe() const { return true; }

Allocation* StreamSafeCUDAAllocator::AllocateImpl(size_t size) {
  ProcessUnfreedAllocations();
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
  StreamSafeCUDAAllocation* allocation = new StreamSafeCUDAAllocation(
      std::move(underlying_allocation), default_stream_);
  VLOG(8) << "Allocate " << allocation->size() << " bytes at address "
          << allocation->ptr();
  return allocation;
}

void StreamSafeCUDAAllocator::FreeImpl(Allocation* allocation) {
  VLOG(8) << "Try free allocation " << allocation->ptr();
  if (dynamic_cast<StreamSafeCUDAAllocation*>(allocation)->CanBeFreed()) {
    delete allocation;
  } else {
    std::lock_guard<SpinLock> lock_guard(unfreed_allocation_lock_);
    unfreed_allocations_.emplace_back(allocation);
  }
}

uint64_t StreamSafeCUDAAllocator::ReleaseImpl(const platform::Place& place) {
  std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
  std::vector<StreamSafeCUDAAllocator*>& allocators =
      allocator_map_[BOOST_GET_CONST(platform::CUDAPlace, place)];
  uint64_t released_size = 0;
  for (StreamSafeCUDAAllocator* allocator : allocators) {
    released_size += allocator->ProcessUnfreedAllocationsWithRelease();
  }
  VLOG(8) << "Release " << released_size << " bytes memory from all streams";
  return released_size;
}

void StreamSafeCUDAAllocator::ProcessUnfreedAllocations() {
  std::lock_guard<SpinLock> lock_guard(unfreed_allocation_lock_);
  for (auto it = unfreed_allocations_.begin();
       it != unfreed_allocations_.end();) {
    if (dynamic_cast<StreamSafeCUDAAllocation*>(*it)->CanBeFreed()) {
      delete *it;
      it = unfreed_allocations_.erase(it);
    } else {
      ++it;
    }
  }
}

uint64_t StreamSafeCUDAAllocator::ProcessUnfreedAllocationsWithRelease() {
  ProcessUnfreedAllocations();
  return underlying_allocator_->Release(place_);
}

std::map<platform::CUDAPlace, std::vector<StreamSafeCUDAAllocator*>>
    StreamSafeCUDAAllocator::allocator_map_;
SpinLock StreamSafeCUDAAllocator::allocator_map_lock_;

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
