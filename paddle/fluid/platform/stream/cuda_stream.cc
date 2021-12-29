/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/stream/cuda_stream.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace stream {

bool CUDAStream::Init(const Place& place, const Priority& priority,
                      const StreamFlag& flag) {
  PADDLE_ENFORCE_EQ(is_gpu_place(place), true,
                    platform::errors::InvalidArgument(
                        "Cuda stream must be created using cuda place."));
  place_ = place;
  CUDADeviceGuard guard(BOOST_GET_CONST(CUDAPlace, place_).device);
  if (priority == Priority::kHigh) {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreateWithPriority(
        &stream_, static_cast<unsigned int>(flag), -1));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreateWithPriority(
        &stream_, static_cast<unsigned int>(flag), -1));
#endif
  } else if (priority == Priority::kNormal) {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamCreateWithPriority(
        &stream_, static_cast<unsigned int>(flag), 0));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreateWithPriority(
        &stream_, static_cast<unsigned int>(flag), 0));
#endif
  }
  callback_manager_.reset(new StreamCallbackManager<gpuStream_t>(stream_));
  VLOG(3) << "GPUStream Init stream: " << stream_
          << ", priority: " << static_cast<int>(priority)
          << ", flag:" << static_cast<int>(flag);
  return true;
}

void CUDAStream::Destroy() {
  CUDADeviceGuard guard(BOOST_GET_CONST(CUDAPlace, place_).device);
  Wait();
  WaitCallback();
  if (stream_ && owned_stream_) {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(stream_));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(stream_));
#endif
  }
  stream_ = nullptr;
}

void CUDAStream::Wait() const {
#ifdef PADDLE_WITH_HIP
  hipError_t e_sync = hipSuccess;
#if !defined(_WIN32)
  e_sync = hipStreamSynchronize(stream_);
#else
  while (e_sync = hipStreamQuery(stream_)) {
    if (e_sync == hipErrorNotReady) continue;
    break;
  }
#endif
#else
  cudaError_t e_sync = cudaSuccess;
#if !defined(_WIN32)
  e_sync = cudaStreamSynchronize(stream_);
#else
  while (e_sync = cudaStreamQuery(stream_)) {
    if (e_sync == cudaErrorNotReady) continue;
    break;
  }
#endif
#endif  // PADDLE_WITH_HIP

  PADDLE_ENFORCE_GPU_SUCCESS(e_sync);
}

// Note: Can only be used under thread_local semantics.
void CUDAStream::SetStream(gpuStream_t stream) {
  if (owned_stream_ && stream_) {
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(stream_));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(stream_));
#endif
  }
  owned_stream_ = false;
  stream_ = stream;
  callback_manager_.reset(new StreamCallbackManager<gpuStream_t>(stream_));
}

CUDAStream* get_current_stream(int deviceId) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (deviceId == -1) {
    deviceId = platform::GetCurrentDeviceId();
  }

  auto& pool = platform::DeviceContextPool::Instance();

  platform::Place device = CUDAPlace(deviceId);

  auto stream = static_cast<platform::CUDADeviceContext*>(pool.Get(device))
                    ->context()
                    ->Stream()
                    .get();
  return stream;
#else
  PADDLE_THROW(platform::errors::Unavailable(
      "Paddle is not compiled with CUDA. Cannot visit cuda current stream."));
  return nullptr;
#endif
}

CUDAStream* set_current_stream(CUDAStream* stream) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto& device = stream->GetPlace();
  auto& pool = platform::DeviceContextPool::Instance();
  return static_cast<platform::CUDADeviceContext*>(pool.Get(device))
      ->context()
      ->SetStream(stream);
#else
  PADDLE_THROW(platform::errors::Unavailable(
      "Paddle is not compiled with CUDA. Cannot visit cuda current stream."));
  return nullptr;
#endif
}
}  // namespace stream
}  // namespace platform
}  // namespace paddle
