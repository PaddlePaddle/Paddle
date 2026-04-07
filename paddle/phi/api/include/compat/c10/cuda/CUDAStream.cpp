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
#include <vector>

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/gpu_info.h"
#endif

namespace c10::cuda {

namespace {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

constexpr int kStreamsPerPool = 32;

std::once_flag g_init_once;
c10::DeviceIndex g_num_gpus = -1;

struct DevicePools {
  std::vector<cudaStream_t> low_priority;
  std::vector<cudaStream_t> high_priority;
  std::atomic<uint32_t> lp_counter{0};
  std::atomic<uint32_t> hp_counter{0};
  std::once_flag init_flag;
};

std::vector<std::unique_ptr<DevicePools>> g_pools;

thread_local std::vector<cudaStream_t> tls_current_streams;
thread_local bool tls_streams_initialized = false;

void initGlobalState() {
  std::call_once(g_init_once, []() {
    g_num_gpus =
        static_cast<c10::DeviceIndex>(phi::backends::gpu::GetGPUDeviceCount());
    g_pools.resize(g_num_gpus);
    for (auto& ptr : g_pools) {
      ptr = std::make_unique<DevicePools>();
    }
  });
}

void initDevicePools(c10::DeviceIndex device_index) {
  phi::backends::gpu::GPUDeviceGuard guard(device_index);
  int lo_pri = 0, hi_pri = 0;
  C10_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo_pri, &hi_pri));

  auto& pool = *g_pools[device_index];
  pool.low_priority.resize(kStreamsPerPool);
  pool.high_priority.resize(kStreamsPerPool);

  for (int i = 0; i < kStreamsPerPool; ++i) {
    C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &pool.low_priority[i], cudaStreamNonBlocking, lo_pri));
    C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &pool.high_priority[i], cudaStreamNonBlocking, hi_pri));
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

inline void initTLSCurrentStreams() {
  if (!tls_streams_initialized) {
    tls_current_streams.resize(g_num_gpus, nullptr);
    tls_streams_initialized = true;
  }
}

#endif  // defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

}  // namespace

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
  cudaStream_t raw = (priority < 0 ? g_pools[device_index]->high_priority[idx]
                                   : g_pools[device_index]->low_priority[idx]);

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

CUDAStream getStreamFromExternal(cudaStream_t ext_stream,
                                 c10::DeviceIndex device_index) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  initGlobalState();
  check_gpu(device_index);
#endif
  return make_cuda_stream(ext_stream, device_index);
}

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
  initTLSCurrentStreams();
  cudaStream_t raw = tls_current_streams[device_index];
  if (raw == nullptr) {
    return getDefaultCUDAStream(device_index);
  }
  return make_cuda_stream(raw, device_index);
#else
  return getDefaultCUDAStream(device_index);
#endif
}

void setCurrentCUDAStream(CUDAStream stream) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  initGlobalState();
  c10::DeviceIndex idx = stream.unwrap().device_index();
  check_gpu(idx);
  initTLSCurrentStreams();
  tls_current_streams[idx] = stream.stream();
#else
  (void)stream;
#endif
}

}  // namespace c10::cuda
