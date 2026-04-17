// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/cuda.h>

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/device_event_base.h"

namespace torch::cuda {

c10::DeviceIndex device_count() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  return phi::backends::gpu::GetGPUDeviceCount();
#else
  PADDLE_THROW(common::errors::Unavailable(
      "Paddle is not compiled with CUDA. Cannot visit device count."));
#endif
}

bool is_available() { return cuda::device_count() > 0; }

void synchronize(int64_t device_index) {
  TORCH_CHECK(is_available(), "No CUDA GPUs are available");
  auto num_gpus = cuda::device_count();
  TORCH_CHECK(device_index < 0 || device_index < num_gpus,
              "Device index out of range: ",
              device_index);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // Match PyTorch semantics:
  // 1. `device_index == -1` means "current CUDA device".
  // 2. Explicit device synchronization must not leak a changed current device
  //    to the caller after returning.
  const c10::cuda::CUDAGuard device_guard(c10::Device(
      c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device_index)));
  c10::cuda::device_synchronize();
#else
  PADDLE_THROW(common::errors::Unavailable(
      "Paddle is not compiled with CUDA. Cannot visit device synchronize."));
#endif
}

}  // namespace torch::cuda
