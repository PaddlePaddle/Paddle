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

#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/memory/allocation/npu_pinned_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

void NPUPinnedAllocator::ProcessEventsAndFree() {
  for (auto it = npu_events_.begin(); it != npu_events_.end();) {
    aclrtEvent event = it->second;
    aclrtEventStatus status = ACL_EVENT_STATUS_COMPLETE;
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtQueryEvent(event, &status));

    if (status == ACL_EVENT_STATUS_COMPLETE) {
      Allocation *allocation = it->first;
      void *ptr = allocation->ptr();
      free(ptr);
      npu_events_.erase(it++);
      delete allocation;
      PADDLE_ENFORCE_NPU_SUCCESS(aclrtDestroyEvent(event));
    } else {
      ++it;
    }
  }
}

Allocation *NPUPinnedAllocator::AllocateImpl(size_t size) {
  std::lock_guard<std::mutex> lock(mtx_);
  ProcessEventsAndFree();
  void *ptr;
  int error = posix_memalign(&ptr, kAlignment, size);
  PADDLE_ENFORCE_EQ(
      error, 0,
      platform::errors::ResourceExhausted(
          "Fail to alloc memory of %ld size, error code is %d.", size, error));
  return new Allocation(ptr, size, platform::NPUPinnedPlace());
}

void NPUPinnedAllocator::FreeImpl(Allocation *allocation) {
  std::lock_guard<std::mutex> lock(mtx_);
  void *ptr = allocation->ptr();
  auto iter = npu_events_.find(allocation);

  // Managed by GC if not called RecordEvent.
  if (iter == npu_events_.end()) {
    // double free? No such problem has been found so far.
    // Or maybe we need a set<Allocation*> to record which
    // Allocation managed by GC.
    free(ptr);
    delete allocation;
    return;
  }

  aclrtEvent event = iter->second;
  aclrtEventStatus status = ACL_EVENT_STATUS_COMPLETE;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtQueryEvent(event, &status));
  if (status == ACL_EVENT_STATUS_COMPLETE) {
    free(ptr);
    npu_events_.erase(allocation);
    delete allocation;
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtDestroyEvent(event));
  }
  return;
}

uint64_t NPUPinnedAllocator::ReleaseImpl(const platform::Place &place) {
  std::lock_guard<std::mutex> lock(mtx_);
  // Empty implementation
  return static_cast<uint64_t>(0);
}

void NPUPinnedAllocator::RecordEvent(Allocation *allocation,
                                     aclrtStream stream) {
  std::lock_guard<std::mutex> lock(mtx_);
  aclrtEvent event = nullptr;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtCreateEvent(&event));
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtRecordEvent(event, stream));
  npu_events_.insert({allocation, event});
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
#endif
