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

#include "paddle/phi/core/memory/allocation/xpu_allocator.h"

#include <string>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {
bool XPUAllocator::IsAllocThreadSafe() const { return true; }
void XPUAllocator::FreeImpl(phi::Allocation* allocation) {
  PADDLE_ENFORCE_EQ(
      allocation->place(),
      place_,
      common::errors::PermissionDenied(
          "XPU memory is freed in incorrect device. This may be a bug"));
  platform::RecordedXPUFree(
      allocation->ptr(), allocation->size(), place_.device);
  delete allocation;
}

phi::Allocation* XPUAllocator::AllocateImpl(size_t size) {
  std::call_once(once_flag_,
                 [this] { platform::SetXPUDeviceId(place_.device); });

  void* ptr;
  auto result = platform::RecordedXPUMalloc(&ptr, size, place_.device);
  if (LIKELY(result == XPU_SUCCESS)) {
    return new Allocation(ptr, size, phi::Place(place_));
  }

  PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
      "\n\nOut of memory error on XPU %d. "
      "Cannot allocate %s memory on XPU %d.\n\n",
      place_.device,
      string::HumanReadableSize(size),
      place_.device));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
