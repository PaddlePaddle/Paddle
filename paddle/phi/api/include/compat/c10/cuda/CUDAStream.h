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

#include <array>
#include <atomic>
#include <mutex>
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/cuda_stream.h"

namespace c10::cuda {

using StreamId = int64_t;

// ── Per-device stream pool and per-thread current stream ─────────────────────

namespace detail {

constexpr int kStreamsPerPool = 32;
// Upper bound for static pool/TLS arrays. 64 covers all current CUDA hardware.
constexpr int kMaxDevices = 64;

// Device count is invariant after CUDA initialization; cache it to avoid
// repeated driver calls on the hot path.
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
  cudaDeviceGetStreamPriorityRange(&lo_pri, &hi_pri);
  for (int i = 0; i < kStreamsPerPool; ++i) {
    C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &state->low_priority[i], cudaStreamNonBlocking, lo_pri));
    C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &state->high_priority[i], cudaStreamNonBlocking, hi_pri));
  }
}

// Per-thread, per-device current stream state.
// thread_local inside an inline function is ODR-safe across translation units
// (C++11 §3.2): all TUs share the same thread-local instance per thread.
struct TLSStreamState {
  cudaStream_t streams[kMaxDevices]{};
  bool has_stream[kMaxDevices]{};
};

inline TLSStreamState& get_tls() {
  thread_local TLSStreamState s;
  return s;
}

}  // namespace detail

// ── CUDAStream ───────────────────────────────────────────────────────────────

class CUDAStream {
 public:
  CUDAStream() = delete;

  explicit CUDAStream(Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::CUDA);
  }

  bool operator==(const CUDAStream& other) const noexcept {
    return stream_ == other.stream_;
  }

  bool operator!=(const CUDAStream& other) const noexcept {
    return stream_ != other.stream_;
  }

  StreamId id() const { return stream_.id(); }

  operator cudaStream_t() const { return stream(); }

  operator Stream() const { return unwrap(); }

  cudaStream_t stream() const {
    return reinterpret_cast<cudaStream_t>(stream_.id());
  }

  Stream unwrap() const { return stream_; }

  DeviceType device_type() const { return DeviceType::CUDA; }

  DeviceIndex device_index() const { return stream_.device_index(); }

  Device device() const { return Device(DeviceType::CUDA, device_index()); }

  // TODO(youge325): Remove after DeepEP paddle branch is updated to use
  // stream()
  cudaStream_t raw_stream() const { return stream(); }

 private:
  Stream stream_;
};

// Build a CUDAStream from a raw platform stream handle and a device index.
// The handle is encoded as a StreamId via reinterpret_cast, matching Paddle's
// phi::Stream / phi::CUDAStream convention.
inline CUDAStream make_cuda_stream(cudaStream_t raw,
                                   c10::DeviceIndex device_index) {
  c10::StreamId sid =
      static_cast<c10::StreamId>(reinterpret_cast<intptr_t>(raw));
  return CUDAStream(
      c10::Stream(c10::Stream::UNSAFE,
                  c10::Device(c10::DeviceType::CUDA, device_index),
                  sid));
}

/**
 * Get the current CUDA stream for the given device (or the current device if
 * device_index == -1).
 *
 * Returns the per-thread current stream if one has been set via
 * setCurrentCUDAStream() for this thread and device; otherwise falls back to
 * Paddle's default stream for the device.
 */
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

/**
 * Get a stream from the per-device pool in round-robin fashion.
 * Returns a high priority stream if isHighPriority is true.
 *
 * The pool is lazily initialized on first use for each device.  Each device
 * has kStreamsPerPool low-priority and kStreamsPerPool high-priority streams
 * that are reused round-robin.  Pool streams are always distinct from the
 * current stream, enabling cross-stream dependency management and correct
 * record_stream lifetime semantics.
 */
inline CUDAStream getStreamFromPool(const bool isHighPriority = false,
                                    c10::DeviceIndex device_index = -1) {
  if (device_index == -1) {
    device_index = phi::backends::gpu::GetCurrentDeviceId();
  }
  // get_pool also performs bounds-checking on device_index.
  auto& state = detail::get_pool(device_index);
  std::call_once(state.init_flag, [device_index, &state]() {
    detail::init_pool(device_index, &state);
  });

  cudaStream_t raw;
  if (isHighPriority) {
    raw = state.high_priority[state.hp_counter.fetch_add(1) %
                              detail::kStreamsPerPool];
  } else {
    raw = state.low_priority[state.lp_counter.fetch_add(1) %
                             detail::kStreamsPerPool];
  }
  return make_cuda_stream(raw, device_index);
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
  // The default CUDA stream is always the null stream (cudaStreamDefault,
  // handle == 0), regardless of any per-thread current stream override.
  // This matches PyTorch semantics where getDefaultCUDAStream() returns the
  // fixed device-level default stream, while getCurrentCUDAStream() returns
  // the per-thread current stream (which may differ after
  // setCurrentCUDAStream).
  return CUDAStream(c10::Stream(
      c10::Stream::DEFAULT, c10::Device(c10::DeviceType::CUDA, device_index)));
}

}  // namespace c10::cuda

namespace at::cuda {
using c10::cuda::CUDAStream;
using c10::cuda::getCurrentCUDAStream;
using c10::cuda::getDefaultCUDAStream;
using c10::cuda::getStreamFromPool;
using c10::cuda::setCurrentCUDAStream;
}  // namespace at::cuda
