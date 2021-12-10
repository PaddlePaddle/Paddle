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
      owning_stream_(owning_stream),
      recorded_streams_(std::make_shared<std::set<gpuStream_t>>()) {}

void StreamSafeCUDAAllocation::RecordStream(gpuStream_t stream) {
  VLOG(8) << "Record stream " << stream << " to " << ptr();
  if (stream == owning_stream_) {
    return;
  }
  std::lock_guard<SpinLock> lock_guard(spin_lock_);
  recorded_streams_->insert(stream);
}

std::shared_ptr<std::set<gpuStream_t>>
StreamSafeCUDAAllocation::GetRecordedStreams() {
  return recorded_streams_;
}

StreamSafeCUDAAllocator::StreamSafeCUDAAllocator(
    const std::shared_ptr<Allocator>& underlying_allocator,
    const platform::CUDAPlace& place, const gpuStream_t default_stream)
    : underlying_allocator_(underlying_allocator),
      place_(place),
      default_stream_(default_stream) {
  std::lock_guard<SpinLock> lock_guard(allocators_map_lock_);
  allocators_map_[place].emplace_back(this);
}

StreamSafeCUDAAllocator::~StreamSafeCUDAAllocator() {
  std::lock_guard<SpinLock> lock_guard(allocators_map_lock_);
  std::vector<StreamSafeCUDAAllocator*>& allocators = allocators_map_[place_];
  allocators.erase(std::remove(allocators.begin(), allocators.end(), this),
                   allocators.end());
}

bool StreamSafeCUDAAllocator::IsAllocThreadSafe() const { return true; }

Allocation* StreamSafeCUDAAllocator::AllocateImpl(size_t size) {
  ProcessEventsAndFree();
  AllocationPtr underlying_allocation;
  try {
    underlying_allocation = underlying_allocator_->Allocate(size);
  } catch (BadAlloc&) {
    VLOG(9) << "Allocation failed when allocating " << size << " bytes";
    uint64_t release_size = ReleaseImpl(place_);
    VLOG(9) << "Release " << release_size << " bytes memory from all streams";
    try {
      underlying_allocation = underlying_allocator_->Allocate(size);
    } catch (...) {
      VLOG(9) << "Still allocation failed after release memory";
      throw;
    }
  } catch (...) {
    throw;
  }

  StreamSafeCUDAAllocation* allocation = new StreamSafeCUDAAllocation(
      std::move(underlying_allocation), default_stream_);
  return allocation;
}

void StreamSafeCUDAAllocator::FreeImpl(Allocation* allocation) {
  if (dynamic_cast<StreamSafeCUDAAllocation*>(allocation)
          ->GetRecordedStreams()
          ->empty()) {
    delete allocation;
  } else {
    std::lock_guard<SpinLock> lock_guard(outstanding_events_map_lock_);
    FreeStreamSafeCUDAAllocation(allocation);
  }
}

uint64_t StreamSafeCUDAAllocator::ReleaseImpl(const platform::Place& place) {
  std::lock_guard<SpinLock> lock_guard(allocators_map_lock_);
  std::vector<StreamSafeCUDAAllocator*>& allocators =
      allocators_map_[BOOST_GET_CONST(platform::CUDAPlace, place)];
  uint64_t release_size = 0;
  for (StreamSafeCUDAAllocator* allocator : allocators) {
    release_size += allocator->ProcessEventsAndFreeWithRelease();
  }
  VLOG(8) << "Release " << release_size
          << " bytes memory from all stream for place " << place;
  return release_size;
}

void StreamSafeCUDAAllocator::CreateEventForAllRecordedStream(
    std::set<gpuStream_t>* recorded_streams,
    std::deque<gpuEvent_t>* outstanding_events) {
  for (gpuStream_t stream : *recorded_streams) {
    gpuEvent_t event;
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, stream));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        hipEventCreateWithFlags(&event, hipEventDisableTiming));
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(event, stream));
#endif
    outstanding_events->emplace_back(event);
    VLOG(9) << "Record event " << event << " in stream " << stream;
  }
  recorded_streams->clear();
}

void StreamSafeCUDAAllocator::FreeStreamSafeCUDAAllocation(
    Allocation* allocation) {
  std::deque<gpuEvent_t>& outstanding_events =
      outstanding_events_map_[allocation];
  CreateEventForAllRecordedStream(
      dynamic_cast<StreamSafeCUDAAllocation*>(allocation)
          ->GetRecordedStreams()
          .get(),
      &outstanding_events);
  if (!outstanding_events.empty()) {
    VLOG(8) << allocation->ptr() << " is not ready to free";
    return;
  }

  VLOG(8) << "Free " << allocation->ptr();
  outstanding_events_map_.erase(allocation);
  delete allocation;
}

void StreamSafeCUDAAllocator::ProcessEventsAndFree() {
  std::lock_guard<SpinLock> lock_guard(outstanding_events_map_lock_);
  for (auto map_it = outstanding_events_map_.begin();
       map_it != outstanding_events_map_.end();) {
    std::deque<gpuEvent_t>& outstanding_events = map_it->second;
    VLOG(10) << "Check " << outstanding_events.size()
             << " outstanding events for " << map_it->first->ptr();
    auto deque_it = outstanding_events.begin();
    while (deque_it != outstanding_events.end()) {
#ifdef PADDLE_WITH_CUDA
      gpuError_t err = cudaEventQuery(*deque_it);
      if (err == cudaErrorNotReady) {
        VLOG(10) << "Event " << *deque_it << " for " << map_it->first->ptr()
                 << " is not completed";
        outstanding_events.erase(outstanding_events.begin(), deque_it);
        break;
      }
      PADDLE_ENFORCE_GPU_SUCCESS(err);
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(*deque_it));
#else
      gpuError_t err = hipEventQuery(*deque_it);
      if (err == hipErrorNotReady) {
        VLOG(10) << "Event " << *deque_it << " for " << map_it->first->ptr()
                 << " is not completed";
        // Erase the completded event before "deque_it"
        outstanding_events.erase(outstanding_events.begin(), deque_it);
        break;
      }
      PADDLE_ENFORCE_GPU_SUCCESS(err);
      PADDLE_ENFORCE_GPU_SUCCESS(hipEventDestroy(*deque_it));
#endif
      ++deque_it;
    }

    if (deque_it == outstanding_events.end()) {
      outstanding_events.clear();
      Allocation* allocation = map_it->first;
      // "map_it" may be invalid after calling FreeStreamSafeCUDAAllocation
      auto next_it = ++map_it;
      FreeStreamSafeCUDAAllocation(allocation);
      map_it = next_it;
    } else {
      ++map_it;
    }
  }
}

uint64_t StreamSafeCUDAAllocator::ProcessEventsAndFreeWithRelease() {
  ProcessEventsAndFree();
  return underlying_allocator_->Release(place_);
}

std::map<platform::CUDAPlace, std::vector<StreamSafeCUDAAllocator*>>
    StreamSafeCUDAAllocator::allocators_map_;
SpinLock StreamSafeCUDAAllocator::allocators_map_lock_;

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
