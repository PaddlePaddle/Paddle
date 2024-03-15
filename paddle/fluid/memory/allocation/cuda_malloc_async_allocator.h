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
#include "paddle/fluid/platform/device/gpu/gpu_types.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {

// TODO(eee4017): It may be beneficial to introduce an abstract class named
// `StreamAllocator` in future developments. This class would serve as a central
// entity for methods specifically related to stream management, such as
// `RecordStream` and `EraseStream`. The introduction of `StreamAllocator` would
// enable both `StreamSafeCUDAAllocator` and `CUDAMallocAsyncAllocator` to
// inherit directly from it,

// The `CUDAMallocAsyncAllocation` class extends `Allocation` and is used for
// managing memory allocations with CUDA async malloc. It includes methods to
// handle stream associations and to query the owning stream of the allocation.
class CUDAMallocAsyncAllocation : public Allocation {
 public:
  CUDAMallocAsyncAllocation(void* ptr,
                            size_t size,
                            platform::Place place,
                            gpuStream_t malloc_stream,
                            gpuStream_t free_stream)
      : Allocation(ptr, size, place),
        malloc_stream_(malloc_stream),
        free_stream_(free_stream),
        used_in_another_stream(false) {}

  gpuStream_t GetOwningStream() const { return malloc_stream_; }

  void RecordStream(gpuStream_t stream);
  void EraseStream(gpuStream_t stream);
  size_t Free();

 private:
  static thread_local std::once_flag once_flag_;
  gpuStream_t malloc_stream_;
  gpuStream_t free_stream_;

  bool used_in_another_stream;
  std::unordered_set<gpuStream_t> graph_capturing_stream_set_;

  SpinLock recorded_streams_lock_;
  std::unordered_set<gpuStream_t> recorded_streams_;
};

// The `CUDAMallocAsyncAllocator` class extends `Allocator` and is specialized
// for asynchronous memory allocation in CUDA. It offers thread-safe allocation
// and incorporates a default stream for memory operations.
class CUDAMallocAsyncAllocator : public Allocator {
 public:
  explicit CUDAMallocAsyncAllocator(
      std::shared_ptr<Allocator> underlying_allocator,
      const platform::CUDAPlace& place,
      gpuStream_t default_stream);

  bool IsAllocThreadSafe() const override { return true; }
  gpuStream_t GetDefaultStream() const;
  void SetDefaultStream(gpuStream_t stream);
  void ClearFreeStream(bool sync = false);

 protected:
  void FreeImpl(phi::Allocation* allocation) override;
  phi::Allocation* AllocateImpl(size_t size) override;
  uint64_t ReleaseImpl(const platform::Place& place) override;

 private:
  void MallocThrottling();
  void FreeAllocation(CUDAMallocAsyncAllocation* allocation);

  std::shared_ptr<Allocator> underlying_allocator_;
  platform::CUDAPlace place_;  // Specifies the CUDA device context.

  cudaMemPool_t mempool_;
  gpuStream_t default_stream_;  // Default stream for memory operations.

  // we create a `free stream` for each allocator (each device should have a
  // unique allocator) if an allocation is recorded on other stream than default
  // stream, we release the allocation on `free stream`
  gpuStream_t free_stream_;

  size_t current_allocated_size_;
  size_t pending_release_size_;
  size_t max_size_;

  double memory_throttle_ratio_;

  std::once_flag once_flag_;

  std::unordered_set<CUDAMallocAsyncAllocation*> graph_owned_allocations_;

  std::list<CUDAMallocAsyncAllocation*> unfreed_allocations_;
  SpinLock unfreed_allocation_lock_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
