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

#pragma once

#include <deque>
#include <list>
#include <map>
#include <mutex>
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/allocation/spin_lock.h"
#include "paddle/fluid/platform/place.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif

namespace paddle {
namespace memory {
namespace allocation {

class StreamSafeCUDAAllocation : public Allocation {
 public:
  StreamSafeCUDAAllocation(AllocationPtr underlying_allocation,
                           gpuStream_t owning_stream);
  void RecordStream(const gpuStream_t &stream);
  bool CanBeFreed();

  const gpuStream_t &GetOwningStream() const;

 private:
  AllocationPtr underlying_allocation_;
  std::map<gpuStream_t, gpuEvent_t> outstanding_event_map_;
  gpuStream_t owning_stream_;
  SpinLock outstanding_event_map_lock_;
};

class StreamSafeCUDAAllocator : public Allocator {
 public:
  StreamSafeCUDAAllocator(std::shared_ptr<Allocator> underlying_allocator,
                          platform::CUDAPlace place,
                          gpuStream_t default_stream);
  ~StreamSafeCUDAAllocator();
  bool IsAllocThreadSafe() const override;

 protected:
  Allocation *AllocateImpl(size_t size) override;
  void FreeImpl(Allocation *allocation) override;
  uint64_t ReleaseImpl(const platform::Place &place) override;

 private:
  void ProcessUnfreedAllocations();
  uint64_t ProcessUnfreedAllocationsWithRelease();

  static std::map<platform::CUDAPlace, std::vector<StreamSafeCUDAAllocator *>>
      allocator_map_;
  static SpinLock allocator_map_lock_;

  std::shared_ptr<Allocator> underlying_allocator_;
  platform::CUDAPlace place_;
  gpuStream_t default_stream_;
  std::list<StreamSafeCUDAAllocation *> unfreed_allocations_;
  SpinLock unfreed_allocation_lock_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
