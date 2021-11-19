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
  std::lock_guard<std::mutex> lock(mutex_);
  recorded_streams_->insert(stream);
}

std::shared_ptr<std::set<gpuStream_t>>
StreamSafeCUDAAllocation::GetRecordedStreams() {
  return recorded_streams_;
}

StreamSafeCUDAAllocator::StreamSafeCUDAAllocator(
    const std::shared_ptr<Allocator>& underlying_allocator,
    const gpuStream_t default_stream)
    : underlying_allocator_(underlying_allocator),
      default_stream_(default_stream) {}

bool StreamSafeCUDAAllocator::IsAllocThreadSafe() const { return true; }

Allocation* StreamSafeCUDAAllocator::AllocateImpl(size_t size) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  ProcessEventsAndFree();
  AllocationPtr underlying_allocation = underlying_allocator_->Allocate(size);
  StreamSafeCUDAAllocation* allocation = new StreamSafeCUDAAllocation(
      std::move(underlying_allocation), default_stream_);
  allocation_info_map_[allocation] = std::make_shared<AllocationInfo>();
  return allocation;
}

void StreamSafeCUDAAllocator::FreeImpl(Allocation* allocation) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GetAllocationInfo(allocation)->can_be_freed = true;
  FreeStreamSafeCUDAAllocation(allocation);
}

uint64_t StreamSafeCUDAAllocator::ReleaseImpl(const platform::Place& place) {
  /*lock_guard*/ {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ProcessEventsAndFree();
  }
  return underlying_allocator_->Release(place);
}

void StreamSafeCUDAAllocator::CreateEventForAllRecordedStream(
    std::set<gpuStream_t>* recorded_streams,
    std::deque<gpuEvent_t>* outstanding_events) {
  for (gpuStream_t stream : *recorded_streams) {
    gpuEvent_t event;
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(event, stream));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(
        hipEventCreateWithFlags(&event, hipEventDisableTiming));
    PADDLE_ENFORCE_CUDA_SUCCESS(hipEventRecord(event, stream));
#endif
    outstanding_events->emplace_back(event);
    VLOG(9) << "Record event " << event << " in stream " << stream;
  }
  recorded_streams->clear();
}

void StreamSafeCUDAAllocator::FreeStreamSafeCUDAAllocation(
    Allocation* allocation) {
  std::shared_ptr<AllocationInfo> allocation_info =
      GetAllocationInfo(allocation);
  if (!allocation_info->can_be_freed) {
    return;
  }

  std::deque<gpuEvent_t>& outstanding_events =
      allocation_info->outstanding_events;
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
  allocation_info_map_.erase(allocation);
  delete allocation;
}

std::shared_ptr<StreamSafeCUDAAllocator::AllocationInfo>
StreamSafeCUDAAllocator::GetAllocationInfo(Allocation* allocation) {
  auto it = allocation_info_map_.find(allocation);
  PADDLE_ENFORCE_NE(
      it, allocation_info_map_.end(),
      platform::errors::NotFound(
          "The recorded allocation is not malloced by this allocator"));
  return it->second;
}

void StreamSafeCUDAAllocator::ProcessEventsAndFree() {
  for (auto map_it = allocation_info_map_.begin();
       map_it != allocation_info_map_.end();) {
    std::deque<gpuEvent_t>& outstanding_events =
        map_it->second->outstanding_events;
    VLOG(10) << "Check " << outstanding_events.size()
             << " outstanding events for " << map_it->first->ptr();
    auto deque_it = outstanding_events.begin();
    while (deque_it != outstanding_events.end()) {
#ifdef PADDLE_WITH_CUDA
      gpuError_t err = cudaEventQuery(*deque_it);
      if (err == cudaErrorNotReady) {
        VLOG(10) << "Event " << *deque_it << " for " << map_it->first->ptr()
                 << " is not complete";
        outstanding_events.erase(outstanding_events.begin(), deque_it);
        break;
      }
      PADDLE_ENFORCE_CUDA_SUCCESS(err);
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventDestroy(*deque_it));
      ++deque_it;
#else
      gpuError_t err = hipEventQuery(*deque_it);
      if (err == hipErrorNotReady) {
        VLOG(10) << "Event " << *deque_it << " for " << map_it->first->ptr()
                 << " is not complete";
        outstanding_events.erase(outstanding_events.begin(), deque_it);
        break;
      }
      PADDLE_ENFORCE_CUDA_SUCCESS(err);
      PADDLE_ENFORCE_CUDA_SUCCESS(hipEventDestroy(*deque_it));
      ++deque_it;
#endif
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

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
