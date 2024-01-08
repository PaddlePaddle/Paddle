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
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {

class CUDAMallocAsyncAllocation : public Allocation {
 public:
  CUDAMallocAsyncAllocation(void* ptr,
                            size_t size,
                            platform::Place place,
                            gpuStream_t owning_stream)
      : Allocation(ptr, size, place), owning_stream_(owning_stream) {}
  gpuStream_t GetOwningStream() const { return owning_stream_; }

 private:
  gpuStream_t owning_stream_;
};

class CUDAMallocAsyncAllocator : public Allocator {
 public:
  explicit CUDAMallocAsyncAllocator(const platform::CUDAPlace& place,
                                    gpuStream_t default_stream)
      : place_(place), stream_(default_stream) {}

  bool IsAllocThreadSafe() const override;
  gpuStream_t GetDefaultStream() const;
  void SetDefaultStream(gpuStream_t stream);

 protected:
  // Implementation of freeing an allocation.
  void FreeImpl(phi::Allocation* allocation) override;
  // Implementation of allocating memory of a certain size.
  phi::Allocation* AllocateImpl(size_t size) override;

 private:
  platform::CUDAPlace place_;  // The CUDA place (device context)
  gpuStream_t stream_;         // Default stream associated with this allocator
  std::once_flag once_flag_;   // Flag to ensure some actions are done only once

  // Map from graph ID to the set of allocations it owns.
  std::unordered_set<CUDAMallocAsyncAllocation*> graph_owned_allocations_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
