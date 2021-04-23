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

#include "paddle/fluid/memory/allocation/npu_pinned_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

void NPUPinnedAllocator::ProcessEventsAndFree() {
  for (auto it = npu_events_.begin(); it != npu_events_.end(); ++it) {
    aclrtEvent event = it->second;
    aclrtEventStatus status = ACL_EVENT_STATUS_COMPLETE;
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtQueryEvent(event, &status));

    if (status == ACL_EVENT_STATUS_NOT_READY) {
      Allocation *allocation = it->first;
      void *ptr = allocation->ptr();
      VLOG(1) << "ProcessEventsAndFree, Free ptr : " << ptr;
      PADDLE_ENFORCE_NPU_SUCCESS(aclrtFreeHost(ptr));
      delete allocation;
      npu_events_.erase(allocation);
    }
  }
}

void NPUPinnedAllocator::FreeImpl(Allocation *allocation) {
  void *ptr = allocation->ptr();
  auto event = npu_events_.at(allocation);
  aclrtEventStatus status = ACL_EVENT_STATUS_COMPLETE;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtQueryEvent(event, &status));

  if (status == ACL_EVENT_STATUS_COMPLETE) {
    VLOG(4) << "Free ptr : " << ptr;
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtFreeHost(ptr));
    delete allocation;
    npu_events_.erase(allocation);

  } else {
    VLOG(4) << "Not Free ptr : " << ptr;
  }
  return;
}

uint64_t NPUPinnedAllocator::ReleaseImpl(const platform::Place &place) {
  return static_cast<uint64_t>(0);
}

Allocation *NPUPinnedAllocator::AllocateImpl(size_t size) {
  ProcessEventsAndFree();
  void *ptr;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtMallocHost(&ptr, size));
  VLOG(4) << "Malloc ptr : " << ptr;
  return new Allocation(ptr, size, platform::NPUPinnedPlace());
}

void NPUPinnedAllocator::RecordEvent(Allocation *allocation,
                                     aclrtStream stream) {
  VLOG(4) << " NPUPinnedAllocator::RecordEvent for allocation " << allocation;

  aclrtEvent event = nullptr;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtCreateEvent(&event));
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtRecordEvent(event, stream));
  npu_events_.insert({allocation, event});
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
