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

#pragma once

#include <c10/core/Device.h>
#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
using gpuStream_t = cudaStream_t;
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
using gpuStream_t = hipStream_t;
#endif

#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/device_event_base.h"

namespace c10::cuda {

void device_synchronize() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  int curr_device_id = paddle::platform::GetCurrentDeviceId();
  paddle::platform::SetDeviceId(curr_device_id);
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#endif
#else
  PADDLE_THROW(common::errors::Unavailable(
      "Paddle is not compiled with CUDA. Cannot visit device synchronize."));
#endif
}

void __inline__ stream_synchronize(gpuStream_t stream) {
  phi::backends::gpu::GpuStreamSync(stream);
}
}  // namespace c10::cuda

namespace at::cuda {
using c10::cuda::device_synchronize;
using c10::cuda::stream_synchronize;
}  // namespace at::cuda
