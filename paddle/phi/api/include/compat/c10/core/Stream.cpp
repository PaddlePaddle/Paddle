// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <c10/core/Stream.h>

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include "paddle/common/enforce.h"

namespace c10 {

// id_ encodes the raw platform stream handle via reinterpret_cast<StreamId>.
// A zero id_ corresponds to the null (default) stream on any backend.
// native_handle() reverses that cast to expose the underlying platform handle.
void* Stream::native_handle() const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (device_type() == DeviceType::CUDA) {
    return reinterpret_cast<void*>(static_cast<intptr_t>(id_));
  }
#endif
#if defined(PADDLE_WITH_XPU)
  if (device_type() == DeviceType::XPU) {
    return reinterpret_cast<void*>(static_cast<intptr_t>(id_));
  }
#endif
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
  if (device_type() == DeviceType::CUSTOM) {
    return reinterpret_cast<void*>(static_cast<intptr_t>(id_));
  }
#endif
  // Match PyTorch error message format for unsupported device types
  PD_CHECK(false,
           "native_handle() is not supported for this device type (",
           static_cast<int>(device_type()),
           ")");
}

bool Stream::query() const {
#if defined(PADDLE_WITH_CUDA)
  if (device_type() == DeviceType::CUDA) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<intptr_t>(id_));
    cudaError_t err = cudaStreamQuery(s);
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    PADDLE_ENFORCE_GPU_SUCCESS(err);
  }
#elif defined(PADDLE_WITH_HIP)
  if (device_type() == DeviceType::CUDA) {
    hipStream_t s = reinterpret_cast<hipStream_t>(static_cast<intptr_t>(id_));
    hipError_t err = hipStreamQuery(s);
    if (err == hipSuccess) return true;
    if (err == hipErrorNotReady) return false;
    PADDLE_ENFORCE_GPU_SUCCESS(err);
  }
#endif
  // CPU streams are always ready.
  return true;
}

void Stream::synchronize() const {
#if defined(PADDLE_WITH_CUDA)
  if (device_type() == DeviceType::CUDA) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<intptr_t>(id_));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(s));
    return;
  }
#elif defined(PADDLE_WITH_HIP)
  if (device_type() == DeviceType::CUDA) {
    hipStream_t s = reinterpret_cast<hipStream_t>(static_cast<intptr_t>(id_));
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamSynchronize(s));
    return;
  }
#endif
  // CPU streams: nothing to synchronize.
}

}  // namespace c10
