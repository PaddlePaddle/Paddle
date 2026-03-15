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
#include <c10/core/Stream.h>
#include <c10/cuda/CUDAException.h>
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/cuda_stream.h"

namespace c10::cuda {

using StreamId = int64_t;

class CUDAStream {
 public:
  CUDAStream() = delete;

  explicit CUDAStream(Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::CUDA);
  }

  StreamId id() const { return stream_.id(); }

  operator cudaStream_t() const { return stream(); }

  operator Stream() const { return unwrap(); }

  cudaStream_t stream() const {
    return reinterpret_cast<cudaStream_t>(stream_.id());
  }

  Stream unwrap() const { return stream_; }

  DeviceType device_type() const { return DeviceType::CUDA; }

 private:
  Stream stream_;
};

/**
 * Get the current CUDA stream, for the passed CUDA device, or for the
 * current device if no device index is passed.  The current CUDA stream
 * will usually be the default CUDA stream for the device, but it may
 * be different if someone called 'setCurrentCUDAStream' or used 'StreamGuard'
 * or 'CUDAStreamGuard'.
 */
inline CUDAStream getCurrentCUDAStream(c10::DeviceIndex device_index = -1) {
  if (device_index == -1) {
    device_index = phi::backends::gpu::GetCurrentDeviceId();
  }

  // Encode the raw cudaStream_t handle as a c10::StreamId (int64_t) using the
  // same reinterpret_cast convention as phi::Stream::id_ / raw_stream().
  auto* phi_stream = paddle::GetCurrentCUDAStream(phi::GPUPlace(device_index));
  c10::StreamId sid = static_cast<c10::StreamId>(
      reinterpret_cast<intptr_t>(phi_stream->raw_stream()));
  return CUDAStream(
      c10::Stream(c10::Stream::UNSAFE,
                  c10::Device(c10::DeviceType::CUDA, device_index),
                  sid));
}

/**
 * Get a stream from the pool in round-robin fashion.
 * Returns a high priority stream if isHighPriority is true.
 */
inline CUDAStream getStreamFromPool(const bool isHighPriority = false,
                                    c10::DeviceIndex device_index = -1) {
  if (device_index == -1) {
    device_index = phi::backends::gpu::GetCurrentDeviceId();
  }
  // Get the raw cudaStream_t from paddle's stream pool
  auto* phi_stream = paddle::GetCurrentCUDAStream(phi::GPUPlace(device_index));
  // TODO(youge325): Implement proper stream pool with priority support
  // For now, just return the current stream
  c10::StreamId sid = static_cast<c10::StreamId>(
      reinterpret_cast<intptr_t>(phi_stream->raw_stream()));
  return CUDAStream(
      c10::Stream(c10::Stream::UNSAFE,
                  c10::Device(c10::DeviceType::CUDA, device_index),
                  sid));
}

/**
 * Set the current CUDA stream for the current device.
 * This affects all future CUDA operations on the current thread.
 *
 * In Paddle, the "current stream" is a field inside GPUContext rather than a
 * per-thread TLS value.  We therefore obtain the mutable GPUContext for the
 * target device from DeviceContextPool and call SetStream() on it, which is
 * the canonical Paddle API for injecting an external stream.
 */
inline void setCurrentCUDAStream(CUDAStream stream) {
  c10::DeviceIndex device_index = stream.unwrap().device_index();
  cudaStream_t cuda_stream = stream.stream();

  // Switch the active CUDA device so that subsequent CUDA runtime calls land
  // on the correct device.
  cudaSetDevice(device_index);

  // Update the stream stored inside Paddle's GPUContext for this device.
  auto& pool = paddle::experimental::DeviceContextPool::Instance();
  auto* dev_ctx = static_cast<phi::GPUContext*>(
      pool.GetMutable(phi::GPUPlace(device_index)));
  PADDLE_ENFORCE_NOT_NULL(
      dev_ctx,
      phi::errors::NotFound("GPUContext not found for device %d.",
                            device_index));
  dev_ctx->SetStream(cuda_stream);
}

#define getDefaultCUDAStream getCurrentCUDAStream;

}  // namespace c10::cuda

namespace at::cuda {
using c10::cuda::CUDAStream;
using c10::cuda::getCurrentCUDAStream;
using c10::cuda::getStreamFromPool;
using c10::cuda::setCurrentCUDAStream;
}  // namespace at::cuda
