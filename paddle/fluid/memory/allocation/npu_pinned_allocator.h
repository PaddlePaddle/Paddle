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
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/npu_info.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {

class NPUPinnedAllocator : public Allocator {
 public:
  bool IsAllocThreadSafe() const override { return true; }
  void ProcessEventsAndFree();
  void RecordEvent(Allocation *allocation, aclrtStream stream);
  ~NPUPinnedAllocator() override {
    std::cout << " ~NPUPinnedAllocator()";

    ProcessEventsAndFree();
    PADDLE_ENFORCE_EQ(
        npu_events_.size(), 0,
        platform::errors::Fatal("When free NPUPinnedAllocator, all allocation "
                                "must be freed and the status of all npu "
                                "events must be ACL_EVENT_STATUS_COMPLETE."));
  }

  // protected:
  Allocation *AllocateImpl(size_t size) override;
  void FreeImpl(Allocation *allocation) override;
  uint64_t ReleaseImpl(const platform::Place &place) override;

 private:
  std::unordered_map<Allocation *, aclrtEvent> npu_events_;
  //    platform::Place place_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
