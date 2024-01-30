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

#pragma once
#include <mutex>  // NOLINT
#include <unordered_set>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/allocation/spin_lock.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {

class CUDAMallocAsyncAllocation : public Allocation {
 public:
  CUDAMallocAsyncAllocation(void* ptr,
                            size_t size,
                            platform::Place place,
                            gpuStream_t stream)
      : Allocation(ptr, size, place),
        malloc_stream_(stream),
        used_in_another_stream(false) {}

  gpuStream_t GetOwningStream() const { return malloc_stream_; }
  // Ensure that the block is released after the recorded stream event
  void RecordStream(gpuStream_t stream);
  void RecordGraphCapturingStreams();
  void RecordStreamWithNoGraphCapturing(gpuStream_t stream);

  void EraseStream(gpuStream_t stream);

  bool CanBeFreed(bool synchronize = false);
  void Free(int dev_id, gpuStream_t free_stream);

 private:
  thread_local static std::once_flag once_flag_;

  gpuStream_t malloc_stream_;
  bool used_in_another_stream;
  std::set<gpuStream_t> graph_capturing_stream_set_;

  SpinLock event_map_lock_;
  std::map<gpuStream_t, gpuEvent_t> event_map_;
};

class CUDAMallocAsyncAllocator : public Allocator {
 public:
  explicit CUDAMallocAsyncAllocator(
      std::shared_ptr<Allocator> underlying_allocator,
      const platform::CUDAPlace& place,
      gpuStream_t default_stream);

  bool IsAllocThreadSafe() const override;
  gpuStream_t GetDefaultStream() const;
  void SetDefaultStream(gpuStream_t stream);

 protected:
  // Implementation of freeing an allocation.
  void FreeImpl(phi::Allocation* allocation) override;
  // Implementation of allocating memory of a certain size.
  phi::Allocation* AllocateImpl(size_t size) override;
  uint64_t ReleaseImpl(const platform::Place& place) override;

 private:
  void ProcessUnfreedAllocations(bool synchronize = false);
  void TryFree(CUDAMallocAsyncAllocation* allocation);

  std::shared_ptr<Allocator> underlying_allocator_;
  platform::CUDAPlace place_;  // The CUDA place (device context)
  gpuStream_t stream_;         // Default stream associated with this allocator
  gpuStream_t free_stream_;
  std::once_flag once_flag_;  // Flag to ensure some actions are done only once

  // Map from graph ID to the set of allocations it owns.
  std::unordered_set<CUDAMallocAsyncAllocation*> graph_owned_allocations_;

  std::list<CUDAMallocAsyncAllocation*> unfreed_allocations_;
  SpinLock unfreed_allocation_lock_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
