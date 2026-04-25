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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/cuda/CUDAException.h>

#include <ostream>

#include "paddle/common/macros.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/place.h"

namespace c10::cuda {

using StreamId = int64_t;

static constexpr int max_compile_time_stream_priorities = 4;

class CUDAStream {
 public:
  enum Unchecked { UNCHECKED };

  CUDAStream() = delete;

  explicit CUDAStream(Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::CUDA);
  }

  explicit CUDAStream(Unchecked /*unused*/, Stream stream) : stream_(stream) {}

  bool operator==(const CUDAStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const CUDAStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  StreamId id() const { return stream_.id(); }

#ifdef PADDLE_WITH_HIP
  operator hipStream_t() const { return stream(); }
#else
  operator cudaStream_t() const { return stream(); }
#endif

  operator Stream() const { return unwrap(); }

  bool query() const { return unwrap().query(); }

  void synchronize() const { unwrap().synchronize(); }

  int priority() const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    phi::backends::gpu::GPUDeviceGuard guard(device_index());
    int priority = 0;
#ifdef PADDLE_WITH_HIP
    C10_CUDA_CHECK(hipStreamGetPriority(stream(), &priority));
#else
    C10_CUDA_CHECK(cudaStreamGetPriority(stream(), &priority));
#endif
    return priority;
#else
    return 0;
#endif
  }

#ifdef PADDLE_WITH_HIP
  hipStream_t stream() const {
    return reinterpret_cast<hipStream_t>(stream_.id());
  }
#else
  cudaStream_t stream() const {
    return reinterpret_cast<cudaStream_t>(stream_.id());
  }
#endif

  Stream unwrap() const { return stream_; }

  DeviceType device_type() const { return DeviceType::CUDA; }

  DeviceIndex device_index() const { return stream_.device_index(); }

  Device device() const { return Device(DeviceType::CUDA, device_index()); }

#ifdef PADDLE_WITH_HIP
  hipStream_t raw_stream() const { return stream(); }
#else
  cudaStream_t raw_stream() const { return stream(); }
#endif

  struct c10::StreamData3 pack3() const {
    return stream_.pack3();
  }

  static CUDAStream unpack3(StreamId stream_id,
                            DeviceIndex device_index,
                            DeviceType device_type) {
    return CUDAStream(Stream::unpack3(stream_id, device_index, device_type));
  }

  static std::tuple<int, int> priority_range() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    int least_priority = 0;
    int greatest_priority = 0;
#ifdef PADDLE_WITH_HIP
    C10_CUDA_CHECK(
        hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
#else
    C10_CUDA_CHECK(
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
#endif
    greatest_priority =
        std::max(-max_compile_time_stream_priorities + 1, greatest_priority);
    return std::make_tuple(least_priority, greatest_priority);
#else
    return std::make_tuple(0, 0);
#endif
  }

 private:
  Stream stream_;
};

#ifdef PADDLE_WITH_HIP
inline CUDAStream make_cuda_stream(hipStream_t raw,
                                   c10::DeviceIndex device_index) {
  c10::StreamId sid =
      static_cast<c10::StreamId>(reinterpret_cast<intptr_t>(raw));
  return CUDAStream(
      c10::Stream(c10::Stream::UNSAFE,
                  c10::Device(c10::DeviceType::CUDA, device_index),
                  sid));
}
#else
inline CUDAStream make_cuda_stream(cudaStream_t raw,
                                   c10::DeviceIndex device_index) {
  c10::StreamId sid =
      static_cast<c10::StreamId>(reinterpret_cast<intptr_t>(raw));
  return CUDAStream(
      c10::Stream(c10::Stream::UNSAFE,
                  c10::Device(c10::DeviceType::CUDA, device_index),
                  sid));
}
#endif

/**
 * Get the current CUDA stream for the passed CUDA device, or for the
 * current device if no device index is passed.
 */
PADDLE_API CUDAStream getCurrentCUDAStream(c10::DeviceIndex device_index = -1);

/**
 * Get a new stream from the CUDA stream pool.
 * Priority -1 is high priority, 0 is default/low priority.
 * Matches PyTorch behavior where negative priority = high priority.
 */
PADDLE_API CUDAStream getStreamFromPool(const int priority = 0,
                                        c10::DeviceIndex device_index = -1);

/**
 * Get a new stream from the CUDA stream pool.
 * Bool overload: true = high priority (-1), false = default priority (0).
 */
PADDLE_API CUDAStream getStreamFromPool(const bool isHighPriority,
                                        c10::DeviceIndex device_index = -1);

#ifdef PADDLE_WITH_HIP
PADDLE_API CUDAStream getStreamFromExternal(hipStream_t ext_stream,
                                            c10::DeviceIndex device_index);
#else
PADDLE_API CUDAStream getStreamFromExternal(cudaStream_t ext_stream,
                                            c10::DeviceIndex device_index);
#endif

/**
 * Set the current CUDA stream for the device of the given stream.
 *
 * Keeps the compat c10 stream state aligned with Paddle's GPUContext so
 * Paddle stream guards and c10 callers observe the same current stream.
 */
PADDLE_API void setCurrentCUDAStream(CUDAStream stream);

PADDLE_API CUDAStream getDefaultCUDAStream(c10::DeviceIndex device_index = -1);

inline std::ostream& operator<<(std::ostream& stream, const CUDAStream& s) {
  return stream << s.unwrap();
}

}  // namespace c10::cuda

namespace std {
template <>
struct hash<c10::cuda::CUDAStream> {
  size_t operator()(c10::cuda::CUDAStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
}  // namespace std

namespace at::cuda {
using c10::cuda::CUDAStream;
using c10::cuda::getCurrentCUDAStream;
using c10::cuda::getDefaultCUDAStream;
using c10::cuda::getStreamFromExternal;
using c10::cuda::getStreamFromPool;
using c10::cuda::setCurrentCUDAStream;
}  // namespace at::cuda
