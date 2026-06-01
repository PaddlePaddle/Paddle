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

#include <functional>
#include <utility>

#ifdef __NVCC__
#include <cuda_profiler_api.h>

#define GPUEvent_t cudaEvent_t
#define GPUStream_t cudaStream_t

#define GPUEventCreate(e) cudaEventCreate(e)
#define GPUEventDestroy(e) cudaEventDestroy(e)
#define GPUEventRecord(e, s) cudaEventRecord(e, s)
#define GPUEventSynchronize(e) cudaEventSynchronize(e)
#define GPUEventElapsedTime(ms, s, e) cudaEventElapsedTime(ms, s, e)
#define GPUProfilerStart() cudaProfilerStart()
#define GPUProfilerStop() cudaProfilerStop()
#define GPUStreamSynchronize(s) cudaStreamSynchronize(s)
#define CHECK_GPU CHECK_CUDA
#endif

namespace ap {

class GpuTimer {
 public:
  explicit GpuTimer(bool profile) : profile_(profile) {
    CHECK_GPU(GPUEventCreate(&start_));
    CHECK_GPU(GPUEventCreate(&stop_));
  }

  ~GpuTimer() {
    CHECK_GPU(GPUEventDestroy(start_));
    CHECK_GPU(GPUEventDestroy(stop_));
  }

  void Start(GPUStream_t stream) {
    CHECK_GPU(GPUEventRecord(start_, stream));
    if (profile_) {
      CHECK_GPU(GPUProfilerStart());
    }
  }

  void Stop(GPUStream_t stream) {
    CHECK_GPU(GPUEventRecord(stop_, stream));
    if (profile_) {
      CHECK_GPU(GPUProfilerStop());
    }
  }

  float ElapsedTime() {
    float milliseconds = 0;
    CHECK_GPU(GPUEventSynchronize(stop_));
    CHECK_GPU(GPUEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds;
  }

 private:
  bool profile_{false};
  GPUEvent_t start_{nullptr};
  GPUEvent_t stop_{nullptr};
};

template <typename FuncType, typename... Args>
int ProfileBestConfig(const std::vector<FuncType> &funcs,
                      void *stream_ptr,
                      Args &&...args) {
  std::cout
      << "=================================================================="
      << std::endl;

  constexpr int kWarmupIters = 1;
  constexpr int kRepeatIters = 100;

  GpuTimer gpu_timer(false);
  float min_time_ms = 100000.f;
  int min_time_idx = -1;

  GPUStream_t stream = *reinterpret_cast<GPUStream_t *>(stream_ptr);

  for (int idx = 0; idx < funcs.size(); ++idx) {
    auto func = funcs[idx];
    for (int i = 0; i < kWarmupIters; i++) {
      func(std::forward<Args>(args)...);
    }
    if (stream) {
      CHECK_GPU(GPUStreamSynchronize(stream));
    }

    gpu_timer.Start(stream);
    for (int i = 0; i < kRepeatIters; i++) {
      func(std::forward<Args>(args)...);
    }
    gpu_timer.Stop(stream);

    float elapsed_time_ms = gpu_timer.ElapsedTime();
    std::cout << "-- [ProfileBestConfig] No " << idx
              << ", elapsed_time: " << elapsed_time_ms << " ms" << std::endl;
    if (elapsed_time_ms < min_time_ms) {
      min_time_ms = elapsed_time_ms;
      min_time_idx = idx;
    }
  }

  std::cout << "-- [ProfileBestConfig] best config idx: " << min_time_idx
            << std::endl;
  std::cout
      << "=================================================================="
      << std::endl;
  return min_time_idx;
}

}  // namespace ap
