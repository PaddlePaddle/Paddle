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

#include "paddle/phi/backends/gpu/cuda/cuda_graph.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"
#include "paddle/phi/core/platform/device/gpu/gpu_types.h"

namespace paddle {
namespace memory {
namespace allocation {

class CUDAMallocAsyncAllocation;
class CUDAMallocAsyncAllocator;

#ifdef PADDLE_WITH_CUDA

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
                            phi::Place place,
                            gpuStream_t malloc_stream,
                            gpuStream_t free_stream)
      : Allocation(ptr, size, place),
        malloc_stream_(malloc_stream),
        free_stream_(free_stream) {}

  gpuStream_t GetOwningStream() const { return malloc_stream_; }

  bool RecordStream(gpuStream_t stream);
  void EraseStream(gpuStream_t stream);
  size_t Free();

 private:
  static thread_local std::once_flag once_flag_;
  gpuStream_t malloc_stream_;
  gpuStream_t free_stream_;

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
      const phi::GPUPlace& place,
      gpuStream_t default_stream);

  bool IsAllocThreadSafe() const override { return true; }
  gpuStream_t GetDefaultStream() const;
  void SetDefaultStream(gpuStream_t stream);
  void ClearFreeStream(bool sync = false);

 protected:
  void FreeImpl(phi::Allocation* allocation) override;
  phi::Allocation* AllocateImpl(size_t size) override;
  uint64_t ReleaseImpl(const phi::Place& place) override;

 private:
  void LazyInitializeCudaFreeStream();
  void MallocThrottling();
  void FreeAllocation(CUDAMallocAsyncAllocation* allocation);

  std::shared_ptr<Allocator> underlying_allocator_;
  phi::GPUPlace place_;  // Specifies the CUDA device context.

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

  /*
   * Life cycle management of graph_owned_allocations_:
   *
   * Each element within `graph_owned_allocations_` is initialized at
   * `AllocateImpl`. However, there are two distinct ways of deconstruction.
   *
   * (A.) Deallocating occurs within `FreeImpl`.
   * This implies that the allocation is initialized and disposed of during a
   * graph capture, as in scenario (1.)
   *
   * (B.) Deallocation takes place in the callback after the graph is
   * destructed. Meaning, the allocation is initialized during a graph capture
   * but disposed of outside that context, as in scenario (2.)
   */
  std::unordered_map<CUDAMallocAsyncAllocation*, CUDAGraphID>
      graph_owned_allocations_;
  SpinLock graph_owned_allocations_lock_;
};

#endif

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
