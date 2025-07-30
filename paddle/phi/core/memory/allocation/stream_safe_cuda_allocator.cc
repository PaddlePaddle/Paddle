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

#include "paddle/phi/core/memory/allocation/stream_safe_cuda_allocator.h"
#include <thread>

#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/backends/gpu/gpu_info.h"

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/gpu/cuda/cuda_graph.h"
#elif defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/rocm/hip_graph.h"
#endif

namespace paddle::memory::allocation {

StreamSafeCUDAAllocation::StreamSafeCUDAAllocation(
    DecoratedAllocationPtr underlying_allocation,
    gpuStream_t owning_stream,
    StreamSafeCUDAAllocator* allocator)
    : Allocation(underlying_allocation->ptr(),
                 underlying_allocation->base_ptr(),
                 underlying_allocation->size(),
                 underlying_allocation->place()),
      underlying_allocation_(std::move(underlying_allocation)),
      owning_stream_(owning_stream),
      allocator_(allocator->shared_from_this()) {}

bool StreamSafeCUDAAllocation::RecordStream(gpuStream_t stream) {
  VLOG(8) << "Try record stream " << stream << " for address " << ptr();
  if (stream == owning_stream_) {
    return false;
  }

  std::call_once(once_flag_,
                 [this] { phi::backends::gpu::SetDeviceId(place_.device); });

  std::lock_guard<SpinLock> lock_guard(outstanding_event_map_lock_);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
    graph_capturing_stream_set_.insert(stream);
    return true;
  }
#endif

  RecordStreamWithNoGraphCapturing(stream);
  RecordGraphCapturingStreams();
  return true;
}

void StreamSafeCUDAAllocation::EraseStream(gpuStream_t stream) {
  VLOG(8) << "Try remove stream " << stream << " for address " << ptr();
  std::lock_guard<SpinLock> lock_guard(outstanding_event_map_lock_);
  auto it = outstanding_event_map_.find(stream);
  if (it == outstanding_event_map_.end()) {
    return;
  }

#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(it->second));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipEventDestroy(it->second));
#endif
  outstanding_event_map_.erase(it);
}

bool StreamSafeCUDAAllocation::CanBeFreed() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
    return graph_capturing_stream_set_.empty() &&
           outstanding_event_map_.empty();
  }
#endif

  std::call_once(once_flag_,
                 [this] { phi::backends::gpu::SetDeviceId(place_.device); });

  RecordGraphCapturingStreams();

  for (auto it = outstanding_event_map_.begin();
       it != outstanding_event_map_.end();
       ++it) {
    gpuEvent_t& event = it->second;
#ifdef PADDLE_WITH_CUDA
    gpuError_t err = cudaEventQuery(event);
    if (err == cudaErrorNotReady) {
      VLOG(9) << "Event " << event << " for " << ptr() << " is not completed";
      // Erase the completed event before "it"
      outstanding_event_map_.erase(outstanding_event_map_.begin(), it);
      return false;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(err);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(event));
#else
    gpuError_t err = hipEventQuery(event);
    if (err == hipErrorNotReady) {
      VLOG(9) << "Event " << event << " for " << ptr() << " is not completed";
      // Erase the completed event before "it"
      outstanding_event_map_.erase(outstanding_event_map_.begin(), it);
      return false;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(err);
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventDestroy(event));
#endif
    VLOG(8) << "Destroy event " << event;
  }
  return true;
}

gpuStream_t StreamSafeCUDAAllocation::GetOwningStream() const {
  return owning_stream_;
}

void StreamSafeCUDAAllocation::RecordGraphCapturingStreams() {
  for (gpuStream_t stream : graph_capturing_stream_set_) {
    RecordStreamWithNoGraphCapturing(stream);
  }
  graph_capturing_stream_set_.clear();
}

void StreamSafeCUDAAllocation::RecordStreamWithNoGraphCapturing(
    gpuStream_t stream) {
  gpuEvent_t record_event;
  auto it = outstanding_event_map_.find(stream);
  if (it == outstanding_event_map_.end()) {
    gpuEvent_t new_event;
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventCreateWithFlags(&new_event, cudaEventDisableTiming));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        hipEventCreateWithFlags(&new_event, hipEventDisableTiming));
#endif
    outstanding_event_map_[stream] = new_event;
    record_event = new_event;
    VLOG(9) << "Create a new event " << new_event;
  } else {
    record_event = it->second;
    VLOG(9) << "Reuse event " << record_event;
  }

#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(record_event, stream));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(record_event, stream));
#endif
  VLOG(8) << "Record event " << record_event << " to stream " << stream;
}

StreamSafeCUDAAllocator::StreamSafeCUDAAllocator(
    std::shared_ptr<Allocator> underlying_allocator,
    phi::GPUPlace place,
    gpuStream_t default_stream,
    bool in_cuda_graph_capturing)
    : underlying_allocator_(std::move(underlying_allocator)),
      place_(place),
      default_stream_(default_stream),
      in_cuda_graph_capturing_(in_cuda_graph_capturing) {
  if (LIKELY(!in_cuda_graph_capturing)) {
    std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
    allocator_map_[place].emplace_back(this);
  }
}

StreamSafeCUDAAllocator::~StreamSafeCUDAAllocator() {
  if (LIKELY(!in_cuda_graph_capturing_)) {
    std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
    std::vector<StreamSafeCUDAAllocator*>& allocators = allocator_map_[place_];
    allocators.erase(std::remove(allocators.begin(), allocators.end(), this),
                     allocators.end());
  }
}

bool StreamSafeCUDAAllocator::IsAllocThreadSafe() const { return true; }

gpuStream_t StreamSafeCUDAAllocator::GetDefaultStream() const {
  return default_stream_;
}

void StreamSafeCUDAAllocator::SetDefaultStream(gpuStream_t stream) {
  default_stream_ = stream;
}

phi::Allocation* StreamSafeCUDAAllocator::AllocateImpl(size_t size) {
  phi::RecordEvent record("StreamSafeCUDAAllocator::Allocate",
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
  StreamSafeCUDAAllocation* allocation = new StreamSafeCUDAAllocation(
      static_unique_ptr_cast<Allocation>(std::move(underlying_allocation)),
      default_stream_,
      this);
  VLOG(8) << "Thread " << std::this_thread::get_id() << " Allocate "
          << allocation->size() << " bytes at address " << allocation->ptr()
          << "  , stream: " << default_stream_;
  return allocation;
}

void StreamSafeCUDAAllocator::FreeImpl(phi::Allocation* allocation) {
  phi::RecordEvent record("StreamSafeCUDAAllocator::Free",
                          phi::TracerEventType::UserDefined,
                          9 /*level*/);
  StreamSafeCUDAAllocation* stream_safe_cuda_allocation =
      static_cast<StreamSafeCUDAAllocation*>(allocation);

  VLOG(8) << "Try free allocation " << stream_safe_cuda_allocation->ptr();
  if (stream_safe_cuda_allocation->CanBeFreed()) {
    VLOG(9) << "Directly delete allocation";
    delete stream_safe_cuda_allocation;
  } else {
    VLOG(9) << "Put into unfreed_allocation list";
    std::lock_guard<SpinLock> lock_guard(unfreed_allocation_lock_);
    unfreed_allocations_.emplace_back(stream_safe_cuda_allocation);
  }
}

uint64_t StreamSafeCUDAAllocator::ReleaseImpl(const phi::Place& place) {
  if (UNLIKELY(in_cuda_graph_capturing_)) {
    VLOG(7) << "Memory release forbidden in CUDA Graph Capturing";
    return 0;
  }

  std::lock_guard<SpinLock> lock_guard(allocator_map_lock_);
  std::vector<StreamSafeCUDAAllocator*>& allocators = allocator_map_[place];
  uint64_t released_size = 0;
  for (StreamSafeCUDAAllocator* allocator : allocators) {
    released_size += allocator->ProcessUnfreedAllocationsAndRelease();
  }
  VLOG(8) << "Release " << released_size << " bytes memory from all streams";
  return released_size;
}

void StreamSafeCUDAAllocator::ProcessUnfreedAllocations() {
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

uint64_t StreamSafeCUDAAllocator::ProcessUnfreedAllocationsAndRelease() {
  ProcessUnfreedAllocations();
  return underlying_allocator_->Release(place_);
}

thread_local std::once_flag StreamSafeCUDAAllocation::once_flag_;

std::map<phi::Place, std::vector<StreamSafeCUDAAllocator*>>
    StreamSafeCUDAAllocator::allocator_map_;
SpinLock StreamSafeCUDAAllocator::allocator_map_lock_;

}  // namespace paddle::memory::allocation
