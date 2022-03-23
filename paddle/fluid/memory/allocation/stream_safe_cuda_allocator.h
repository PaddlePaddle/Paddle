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

#include <list>
#include <map>
#include <set>
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

class StreamSafeCUDAAllocator;

class StreamSafeCUDAAllocation : public Allocation {
 public:
  StreamSafeCUDAAllocation(DecoratedAllocationPtr underlying_allocation,
                           gpuStream_t owning_stream,
                           StreamSafeCUDAAllocator *allocator);

  void RecordStream(const gpuStream_t &stream);
  bool CanBeFreed();
  const gpuStream_t &GetOwningStream() const;

 private:
  void RecordGraphCapturingStreams();
  void RecordStreamWithNoGraphCapturing(const gpuStream_t &stream);
  DecoratedAllocationPtr underlying_allocation_;
  std::set<gpuStream_t> graph_capturing_stream_set_;
  std::map<gpuStream_t, gpuEvent_t> outstanding_event_map_;
  gpuStream_t owning_stream_;
  SpinLock outstanding_event_map_lock_;
  // To compatiable with CUDA Graph, hold the allocator shared_ptr so that
  // Allocator will not deconstruct before Allocation
  std::shared_ptr<Allocator> allocator_;
};

class StreamSafeCUDAAllocator
    : public Allocator,
      public std::enable_shared_from_this<StreamSafeCUDAAllocator> {
 public:
  StreamSafeCUDAAllocator(std::shared_ptr<Allocator> underlying_allocator,
                          platform::CUDAPlace place, gpuStream_t default_stream,
                          bool in_cuda_graph_capturing = false);
  ~StreamSafeCUDAAllocator();

  bool IsAllocThreadSafe() const override;
  const gpuStream_t &GetDefaultStream() const;
  void SetDefaultStream(const gpuStream_t &stream);

 protected:
  phi::Allocation *AllocateImpl(size_t size) override;
  void FreeImpl(phi::Allocation *allocation) override;
  uint64_t ReleaseImpl(const platform::Place &place) override;

 private:
  void ProcessUnfreedAllocations();
  uint64_t ProcessUnfreedAllocationsAndRelease();

  static std::map<platform::Place, std::vector<StreamSafeCUDAAllocator *>>
      allocator_map_;
  static SpinLock allocator_map_lock_;

  std::shared_ptr<Allocator> underlying_allocator_;
  platform::CUDAPlace place_;
  gpuStream_t default_stream_;
  std::list<StreamSafeCUDAAllocation *> unfreed_allocations_;
  SpinLock unfreed_allocation_lock_;

  bool in_cuda_graph_capturing_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
