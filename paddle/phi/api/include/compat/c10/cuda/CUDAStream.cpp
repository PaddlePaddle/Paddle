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

#include <c10/cuda/CUDAStream.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/cuda_stream.h"
#endif

namespace c10::cuda {

namespace {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

constexpr int kStreamsPerPool = 32;

std::once_flag g_init_once;
c10::DeviceIndex g_num_gpus = -1;

struct DevicePools {
#ifdef PADDLE_WITH_HIP
  std::vector<hipStream_t> low_priority;
  std::vector<hipStream_t> high_priority;
#else
  std::vector<cudaStream_t> low_priority;
  std::vector<cudaStream_t> high_priority;
#endif
  std::atomic<uint32_t> lp_counter{0};
  std::atomic<uint32_t> hp_counter{0};
  std::once_flag init_flag;
};

std::vector<std::unique_ptr<DevicePools>> g_pools;

#ifdef PADDLE_WITH_HIP
thread_local std::vector<std::optional<hipStream_t>>
    g_thread_local_current_streams;
#else
thread_local std::vector<std::optional<cudaStream_t>>
    g_thread_local_current_streams;
#endif

std::vector<std::unique_ptr<phi::CUDAStream>> g_compat_phi_streams;
std::mutex g_compat_phi_streams_mutex;

void initGlobalState() {
  std::call_once(g_init_once, []() {
    g_num_gpus =
        static_cast<c10::DeviceIndex>(phi::backends::gpu::GetGPUDeviceCount());
    g_pools.resize(g_num_gpus);
    for (auto& ptr : g_pools) {
      ptr = std::make_unique<DevicePools>();
    }
    g_compat_phi_streams.resize(g_num_gpus);
  });
}

void initDevicePools(c10::DeviceIndex device_index) {
  phi::backends::gpu::GPUDeviceGuard guard(device_index);
  int lo_pri = 0, hi_pri = 0;
#ifdef PADDLE_WITH_HIP
  C10_CUDA_CHECK(hipDeviceGetStreamPriorityRange(&lo_pri, &hi_pri));
#else
  C10_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo_pri, &hi_pri));
#endif

  auto& pool = *g_pools[device_index];
  pool.low_priority.resize(kStreamsPerPool);
  pool.high_priority.resize(kStreamsPerPool);

  for (int i = 0; i < kStreamsPerPool; ++i) {
#ifdef PADDLE_WITH_HIP
    C10_CUDA_CHECK(hipStreamCreateWithPriority(
        &pool.low_priority[i], hipStreamNonBlocking, lo_pri));
    C10_CUDA_CHECK(hipStreamCreateWithPriority(
        &pool.high_priority[i], hipStreamNonBlocking, hi_pri));
#else
    C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &pool.low_priority[i], cudaStreamNonBlocking, lo_pri));
    C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &pool.high_priority[i], cudaStreamNonBlocking, hi_pri));
#endif
  }
}

inline void check_gpu(c10::DeviceIndex device_index) {
  TORCH_CHECK(device_index >= 0 && device_index < g_num_gpus,
              "Device index value ",
              static_cast<int>(device_index),
              " is out of index range [0, ",
              static_cast<int>(g_num_gpus),
              ")");
}

inline phi::GPUContext* getMutableGPUContext(c10::DeviceIndex device_index) {
  return static_cast<phi::GPUContext*>(
      paddle::experimental::DeviceContextPool::Instance().GetMutable(
          phi::GPUPlace(device_index)));
}

#endif  // defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

}  // namespace

#ifdef PADDLE_WITH_HIP
inline CUDAStream make_cuda_stream(hipStream_t raw,
                                   c10::DeviceIndex device_index) {
#else
inline CUDAStream make_cuda_stream(cudaStream_t raw,
                                   c10::DeviceIndex device_index) {
#endif
  c10::StreamId sid =
      static_cast<c10::StreamId>(reinterpret_cast<intptr_t>(raw));
  return CUDAStream(
      c10::Stream(c10::Stream::UNSAFE,
                  c10::Device(c10::DeviceType::CUDA, device_index),
                  sid));
}

CUDAStream getStreamFromPool(const int priority,
                             c10::DeviceIndex device_index) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  initGlobalState();
  if (device_index == -1) {
    device_index =
        static_cast<c10::DeviceIndex>(phi::backends::gpu::GetCurrentDeviceId());
  }
  check_gpu(device_index);

  std::call_once(
      g_pools[device_index]->init_flag, initDevicePools, device_index);

  const uint32_t idx = (priority < 0 ? g_pools[device_index]->hp_counter++
                                     : g_pools[device_index]->lp_counter++) %
                       kStreamsPerPool;
#ifdef PADDLE_WITH_HIP
  hipStream_t raw = (priority < 0 ? g_pools[device_index]->high_priority[idx]
                                  : g_pools[device_index]->low_priority[idx]);
#else
  cudaStream_t raw = (priority < 0 ? g_pools[device_index]->high_priority[idx]
                                   : g_pools[device_index]->low_priority[idx]);
#endif

  return make_cuda_stream(raw, device_index);
#else
  TORCH_CHECK(false, "getStreamFromPool is not supported without CUDA/HIP");
  return getDefaultCUDAStream(device_index);
#endif
}

CUDAStream getStreamFromPool(const bool isHighPriority,
                             c10::DeviceIndex device_index) {
  return getStreamFromPool(isHighPriority ? -1 : 0, device_index);
}

#ifdef PADDLE_WITH_HIP
CUDAStream getStreamFromExternal(hipStream_t ext_stream,
                                 c10::DeviceIndex device_index) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  initGlobalState();
  check_gpu(device_index);
#endif
  return make_cuda_stream(ext_stream, device_index);
}
#else
CUDAStream getStreamFromExternal(cudaStream_t ext_stream,
                                 c10::DeviceIndex device_index) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  initGlobalState();
  check_gpu(device_index);
#endif
  return make_cuda_stream(ext_stream, device_index);
}
#endif

CUDAStream getDefaultCUDAStream(c10::DeviceIndex device_index) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  initGlobalState();
  if (device_index == -1) {
    device_index =
        static_cast<c10::DeviceIndex>(phi::backends::gpu::GetCurrentDeviceId());
  }
  check_gpu(device_index);
#endif
  return CUDAStream(c10::Stream(
      c10::Stream::DEFAULT, c10::Device(c10::DeviceType::CUDA, device_index)));
}

CUDAStream getCurrentCUDAStream(c10::DeviceIndex device_index) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  initGlobalState();
  if (device_index == -1) {
    device_index =
        static_cast<c10::DeviceIndex>(phi::backends::gpu::GetCurrentDeviceId());
  }
  check_gpu(device_index);
  // PyTorch-compatible thread-local semantics: if this thread has explicitly
  // set a current stream for this device, return it. Otherwise return the
  // default stream instead of reading Paddle's shared GPUContext stream, so
  // worker threads do not inherit another thread's blocked current stream.
  if (device_index < static_cast<c10::DeviceIndex>(
                         g_thread_local_current_streams.size()) &&
      g_thread_local_current_streams[device_index].has_value()) {
    auto raw = *g_thread_local_current_streams[device_index];
    if (raw == nullptr) {
      return getDefaultCUDAStream(device_index);
    }
    return make_cuda_stream(raw, device_index);
  }
  return getDefaultCUDAStream(device_index);
#else
  return getDefaultCUDAStream(device_index);
#endif
}

void setCurrentCUDAStream(CUDAStream stream) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  initGlobalState();
  c10::DeviceIndex idx = stream.unwrap().device_index();
  check_gpu(idx);
  // Update thread-local current stream state first (PyTorch semantics)
  if (idx >=
      static_cast<c10::DeviceIndex>(g_thread_local_current_streams.size())) {
    g_thread_local_current_streams.resize(idx + 1);
  }
  g_thread_local_current_streams[idx] = stream.stream();
  // Also update Paddle's global device context stream for backward
  // compatibility, so that Paddle kernel launches (which read from
  // GPUContext) still use the correct stream.
  // Use SetCUDAStream instead of SetStream to avoid destroying
  // external stream handles (e.g., pool streams from getStreamFromPool).
  auto* ctx = getMutableGPUContext(idx);
  {
    std::lock_guard<std::mutex> lock(g_compat_phi_streams_mutex);
    if (!g_compat_phi_streams[idx]) {
#ifdef PADDLE_WITH_HIP
      g_compat_phi_streams[idx] = std::make_unique<phi::CUDAStream>(
          phi::GPUPlace(idx), static_cast<hipStream_t>(0));
#else
      g_compat_phi_streams[idx] = std::make_unique<phi::CUDAStream>(
          phi::GPUPlace(idx), static_cast<cudaStream_t>(0));
#endif
    }
    g_compat_phi_streams[idx]->set_raw_stream(stream.stream());
    ctx->SetCUDAStream(g_compat_phi_streams[idx].get(), true);
  }
#else
  (void)stream;
#endif
}

}  // namespace c10::cuda
