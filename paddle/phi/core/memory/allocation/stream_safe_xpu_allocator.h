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

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {

class StreamSafeXPUAllocator;

class StreamSafeXPUAllocation : public Allocation {
 public:
  StreamSafeXPUAllocation(DecoratedAllocationPtr underlying_allocation,
                          XPUStream owning_stream,
                          StreamSafeXPUAllocator *allocator);

  bool RecordStream(XPUStream stream);
  bool CanBeFreed();
  XPUStream GetOwningStream() const;

 private:
  thread_local static std::once_flag once_flag_;
  void RecordStreamPrivate(XPUStream stream);
  DecoratedAllocationPtr underlying_allocation_;

  std::map<XPUStream, XPUEvent> outstanding_event_map_;
  XPUStream owning_stream_;
  SpinLock outstanding_event_map_lock_;
  std::shared_ptr<Allocator> allocator_;
};

class StreamSafeXPUAllocator
    : public Allocator,
      public std::enable_shared_from_this<StreamSafeXPUAllocator> {
 public:
  StreamSafeXPUAllocator(std::shared_ptr<Allocator> underlying_allocator,
                         phi::XPUPlace place,
                         XPUStream default_stream);
  ~StreamSafeXPUAllocator();

  bool IsAllocThreadSafe() const override;
  XPUStream GetDefaultStream() const;
  void SetDefaultStream(XPUStream stream);

 protected:
  phi::Allocation *AllocateImpl(size_t size) override;
  void FreeImpl(phi::Allocation *allocation) override;
  uint64_t ReleaseImpl(const phi::Place &place) override;

 private:
  void ProcessUnfreedAllocations();
  uint64_t ProcessUnfreedAllocationsAndRelease();

  static std::map<phi::Place, std::vector<StreamSafeXPUAllocator *>>
      allocator_map_;
  static SpinLock allocator_map_lock_;

  std::shared_ptr<Allocator> underlying_allocator_;
  phi::XPUPlace place_;
  XPUStream default_stream_;
  std::list<StreamSafeXPUAllocation *> unfreed_allocations_;
  SpinLock unfreed_allocation_lock_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
