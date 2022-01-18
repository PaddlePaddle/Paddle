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

#include "paddle/fluid/memory/allocation/npu_allocator.h"
#include <string>
#include "paddle/fluid/platform/device/npu/npu_info.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

bool NPUAllocator::IsAllocThreadSafe() const { return true; }
void NPUAllocator::FreeImpl(pten::Allocation* allocation) {
  PADDLE_ENFORCE_EQ(
      allocation->place(), place_,
      platform::errors::PermissionDenied(
          "NPU memory is freed in incorrect device. This may be a bug"));
  platform::RecordedNPUFree(allocation->ptr(), allocation->size(),
                            place_.device);
  delete allocation;
}

pten::Allocation* NPUAllocator::AllocateImpl(size_t size) {
  std::call_once(once_flag_,
                 [this] { platform::SetNPUDeviceId(place_.device); });

  void* ptr;
  auto result = platform::RecordedNPUMalloc(&ptr, size, place_.device);
  if (LIKELY(result == ACL_ERROR_NONE)) {
    return new Allocation(ptr, size, platform::Place(place_));
  }

  size_t avail, total, actual_avail, actual_total;
  bool is_limited = platform::RecordedNPUMemGetInfo(
      &avail, &total, &actual_avail, &actual_total, place_.device);

  std::string err_msg;
  if (is_limited) {
    auto limit_size = (total >> 20);
    err_msg = string::Sprintf(
        "Or set environment variable `FLAGS_gpu_memory_limit_mb` to a larger "
        "value. Currently `FLAGS_gpu_memory_limit_mb` is %d, so the maximum "
        "GPU memory usage is limited to %d MB.\n"
        "   The command is `export FLAGS_gpu_memory_limit_mb=xxx`.",
        limit_size, limit_size);
  }

  PADDLE_THROW_BAD_ALLOC(platform::errors::ResourceExhausted(
      "\n\nOut of memory error on NPU %d. "
      "Cannot allocate %s memory on NPU %d, "
      "available memory is only %s.\n\n"
      "Please check whether there is any other process using NPU %d.\n"
      "1. If yes, please stop them, or start PaddlePaddle on another NPU.\n"
      "2. If no, please decrease the batch size of your model. %s\n\n",
      place_.device, string::HumanReadableSize(size), place_.device,
      string::HumanReadableSize(avail), place_.device, err_msg));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
