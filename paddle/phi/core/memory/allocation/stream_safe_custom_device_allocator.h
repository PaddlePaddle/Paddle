// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"

namespace paddle {
namespace memory {
namespace allocation {

class StreamSafeCustomDeviceAllocator;

class StreamSafeCustomDeviceAllocation : public Allocation {
 public:
  StreamSafeCustomDeviceAllocation(DecoratedAllocationPtr underlying_allocation,
                                   phi::stream::stream_t owning_stream,
                                   StreamSafeCustomDeviceAllocator *allocator);

  bool RecordStream(phi::stream::stream_t stream);
  bool CanBeFreed();
  phi::stream::stream_t GetOwningStream() const;
  void SetOwningStream(phi::stream::stream_t s);

 private:
  thread_local static std::once_flag once_flag_;
  DecoratedAllocationPtr underlying_allocation_;
  std::map<phi::stream::stream_t, std::shared_ptr<phi::event::Event>>
      outstanding_event_map_;
  phi::stream::stream_t owning_stream_;
  SpinLock outstanding_event_map_lock_;
  std::shared_ptr<Allocator> allocator_;
  bool will_be_freed_{false};
};

class StreamSafeCustomDeviceAllocator
    : public Allocator,
      public std::enable_shared_from_this<StreamSafeCustomDeviceAllocator> {
 public:
  StreamSafeCustomDeviceAllocator(
      std::shared_ptr<Allocator> underlying_allocator,
      phi::CustomPlace place,
      phi::stream::stream_t default_stream);
  ~StreamSafeCustomDeviceAllocator();

  bool IsAllocThreadSafe() const override { return true; }
  phi::stream::stream_t GetDefaultStream() const;
  void SetDefaultStream(phi::stream::stream_t stream);

 protected:
  phi::Allocation *AllocateImpl(size_t size) override;
  void FreeImpl(phi::Allocation *allocation) override;
  uint64_t ReleaseImpl(const phi::Place &place) override;

 private:
  void ProcessUnfreedAllocations();
  uint64_t ProcessUnfreedAllocationsAndRelease();

  static std::map<phi::Place, std::vector<StreamSafeCustomDeviceAllocator *>>
      allocator_map_;
  static SpinLock allocator_map_lock_;

  std::shared_ptr<Allocator> underlying_allocator_;
  phi::CustomPlace place_;
  phi::stream::stream_t default_stream_;
  std::list<StreamSafeCustomDeviceAllocation *> unfreed_allocations_;
  SpinLock unfreed_allocation_lock_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
