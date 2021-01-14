// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/cuda_allocator.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace memory {
namespace allocation {
bool CUDAAllocator::IsAllocThreadSafe() const { return true; }
void CUDAAllocator::FreeImpl(Allocation* allocation) {
  PADDLE_ENFORCE_EQ(
      BOOST_GET_CONST(platform::CUDAPlace, allocation->place()), place_,
      platform::errors::PermissionDenied(
          "GPU memory is freed in incorrect device. This may be a bug"));
  platform::RecordedCudaFree(allocation->ptr(), allocation->size(),
                             place_.device);
  delete allocation;
}

Allocation* CUDAAllocator::AllocateImpl(size_t size) {
  std::call_once(once_flag_, [this] { platform::SetDeviceId(place_.device); });

  void* ptr;
  auto result = platform::RecordedCudaMalloc(&ptr, size, place_.device);
  if (LIKELY(result == cudaSuccess)) {
    return new Allocation(ptr, size, platform::Place(place_));
  }

  size_t avail, total, actual_avail, actual_total;
  bool is_limited = platform::RecordedCudaMemGetInfo(
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
      "\n\nOut of memory error on GPU %d. "
      "Cannot allocate %s memory on GPU %d, "
      "available memory is only %s.\n\n"
      "Please check whether there is any other process using GPU %d.\n"
      "1. If yes, please stop them, or start PaddlePaddle on another GPU.\n"
      "2. If no, please decrease the batch size of your model. %s\n\n",
      place_.device, string::HumanReadableSize(size), place_.device,
      string::HumanReadableSize(avail), place_.device, err_msg));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
