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

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <mutex>
#include <ostream>
#include <tuple>

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/cuda_stream.h"

namespace c10::cuda {

using StreamId = int64_t;

static constexpr int max_compile_time_stream_priorities = 4;

namespace detail {

constexpr int kStreamsPerPool = 32;
constexpr int kMaxDevices = 64;

inline int gpu_device_count() {
  static const int count = phi::backends::gpu::GetGPUDeviceCount();
  return count;
}

inline void check_device_index(int device_index) {
  const int limit = std::min(gpu_device_count(), kMaxDevices);
  TORCH_CHECK(device_index >= 0 && device_index < limit,
              "CUDA device index out of range: ",
              device_index,
              " (available devices: ",
              limit,
              ", max supported by this build: ",
              kMaxDevices,
              ")");
}

struct StreamPoolState {
  cudaStream_t low_priority[kStreamsPerPool]{};
  cudaStream_t high_priority[kStreamsPerPool]{};
  std::atomic<uint32_t> lp_counter{0};
  std::atomic<uint32_t> hp_counter{0};
  std::once_flag init_flag;
};

inline StreamPoolState& get_pool(int device_index) {
  check_device_index(device_index);
  static StreamPoolState states[kMaxDevices];
  return states[device_index];
}

inline void init_pool(int device_index, StreamPoolState* state) {
  phi::backends::gpu::GPUDeviceGuard guard(device_index);
  int lo_pri = 0, hi_pri = 0;
  C10_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo_pri, &hi_pri));
  for (int i = 0; i < kStreamsPerPool; ++i) {
    C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &state->low_priority[i], cudaStreamNonBlocking, lo_pri));
    C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &state->high_priority[i], cudaStreamNonBlocking, hi_pri));
  }
}

struct TLSStreamState {
  cudaStream_t streams[kMaxDevices]{};
  bool has_stream[kMaxDevices]{};
};

inline TLSStreamState& get_tls() {
  thread_local TLSStreamState s;
  return s;
}

}  // namespace detail

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

  operator cudaStream_t() const { return stream(); }

  operator Stream() const { return unwrap(); }

  bool query() const { return unwrap().query(); }

  void synchronize() const { unwrap().synchronize(); }

  int priority() const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    phi::backends::gpu::GPUDeviceGuard guard(device_index());
    int priority = 0;
    C10_CUDA_CHECK(cudaStreamGetPriority(stream(), &priority));
    return priority;
#else
    return 0;
#endif
  }

  cudaStream_t stream() const {
    return reinterpret_cast<cudaStream_t>(stream_.id());
  }

  Stream unwrap() const { return stream_; }

  DeviceType device_type() const { return DeviceType::CUDA; }

  DeviceIndex device_index() const { return stream_.device_index(); }

  Device device() const { return Device(DeviceType::CUDA, device_index()); }

  cudaStream_t raw_stream() const { return stream(); }

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
    C10_CUDA_CHECK(
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
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

inline CUDAStream make_cuda_stream(cudaStream_t raw,
                                   c10::DeviceIndex device_index) {
  c10::StreamId sid =
      static_cast<c10::StreamId>(reinterpret_cast<intptr_t>(raw));
  return CUDAStream(
      c10::Stream(c10::Stream::UNSAFE,
                  c10::Device(c10::DeviceType::CUDA, device_index),
                  sid));
}

inline CUDAStream getCurrentCUDAStream(c10::DeviceIndex device_index = -1) {
  if (device_index == -1) {
    device_index = phi::backends::gpu::GetCurrentDeviceId();
  }
  detail::check_device_index(device_index);

  auto& tls = detail::get_tls();
  cudaStream_t raw;
  if (tls.has_stream[device_index]) {
    raw = tls.streams[device_index];
  } else {
    auto* phi_stream =
        paddle::GetCurrentCUDAStream(phi::GPUPlace(device_index));
    raw = phi_stream->raw_stream();
  }
  return make_cuda_stream(raw, device_index);
}

inline CUDAStream getStreamFromPool(const int priority,
                                    c10::DeviceIndex device_index = -1) {
  if (device_index == -1) {
    device_index = phi::backends::gpu::GetCurrentDeviceId();
  }
  auto& state = detail::get_pool(device_index);
  std::call_once(state.init_flag, [device_index, &state]() {
    detail::init_pool(device_index, &state);
  });

  cudaStream_t raw;

  // Keep parity with PyTorch API shape: negative priority selects the
  // high-priority pool, non-negative selects the low-priority pool.
  if (priority < 0) {
    raw = state.high_priority[state.hp_counter.fetch_add(1) %
                              detail::kStreamsPerPool];
  } else {
    raw = state.low_priority[state.lp_counter.fetch_add(1) %
                             detail::kStreamsPerPool];
  }
  return make_cuda_stream(raw, device_index);
}

/**
 * Get a new stream from the CUDA stream pool.
 *
 * This overload matches PyTorch's bool-based entry point and preserves the
 * single-argument form `getStreamFromPool(true)` for high-priority requests.
 */
inline CUDAStream getStreamFromPool(const bool isHighPriority = false,
                                    c10::DeviceIndex device_index = -1) {
  return getStreamFromPool(isHighPriority ? -1 : 0, device_index);
}

inline CUDAStream getStreamFromExternal(cudaStream_t ext_stream,
                                        c10::DeviceIndex device_index) {
  detail::check_device_index(device_index);
  return make_cuda_stream(ext_stream, device_index);
}

/**
 * Set the current CUDA stream for the device of the given stream in the
 * calling thread.
 *
 * Implements per-thread, per-device current stream semantics: the change is
 * local to the calling OS thread and does not affect any shared state such as
 * Paddle's GPUContext.  Other threads continue to see their own current stream.
 */
inline void setCurrentCUDAStream(CUDAStream stream) {
  c10::DeviceIndex idx = stream.unwrap().device_index();
  detail::check_device_index(idx);
  auto& tls = detail::get_tls();
  tls.streams[idx] = stream.stream();
  tls.has_stream[idx] = true;
}

inline CUDAStream getDefaultCUDAStream(c10::DeviceIndex device_index = -1) {
  if (device_index == -1) {
    device_index = phi::backends::gpu::GetCurrentDeviceId();
  }
  detail::check_device_index(device_index);
  return CUDAStream(c10::Stream(
      c10::Stream::DEFAULT, c10::Device(c10::DeviceType::CUDA, device_index)));
}

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
