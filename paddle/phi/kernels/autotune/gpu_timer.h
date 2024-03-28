// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/common/errors.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/port.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/enforce.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

namespace phi {

#ifdef PADDLE_WITH_HIP
static void RecordEventTimerCallback(hipStream_t stream,
                                     hipError_t status,
                                     void *user_data) {
  struct timeval time_now {};
  gettimeofday(&time_now, nullptr);
  double *cpu_time = static_cast<double *>(user_data);
  *cpu_time = (time_now.tv_sec * 1000) + (time_now.tv_usec / 1000.0);
  VLOG(3) << "RecordEventCallback: " << std::to_string(*cpu_time);
}
#else
static void CUDART_CB RecordEventTimerCallback(cudaStream_t stream,
                                               cudaError_t status,
                                               void *user_data) {
  struct timeval time_now {};
  gettimeofday(&time_now, nullptr);
  double *cpu_time = static_cast<double *>(user_data);
  *cpu_time = (time_now.tv_sec * 1000) + (time_now.tv_usec / 1000.0);
  VLOG(3) << "RecordEventCallback: " << std::to_string(*cpu_time);
}
#endif

class GpuTimer {
 public:
  GpuTimer() {
#ifdef PADDLE_WITH_HIP
    hipEventCreate(&start_);
    hipEventCreate(&stop_);
#else
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
#endif
    PADDLE_ENFORCE_NOT_NULL(
        start_, phi::errors::PreconditionNotMet("Start Event is not ready."));
    PADDLE_ENFORCE_NOT_NULL(
        stop_, phi::errors::PreconditionNotMet("Stop Event is not ready."));
  }

  ~GpuTimer() {
#ifdef PADDLE_WITH_HIP
    hipEventDestroy(start_);
    hipEventDestroy(stop_);
#else
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
#endif
  }

  void Start(gpuStream_t stream) {
#ifdef PADDLE_WITH_HIP
    hipEventRecord(start_, stream);
#else
    cudaEventRecord(start_, stream);
#endif
  }

  void Stop(gpuStream_t stream) {
#ifdef PADDLE_WITH_HIP
    hipEventRecord(stop_, stream);
#else
    cudaEventRecord(stop_, stream);
#endif
  }

  float ElapsedTime() {
    float milliseconds = 0;
#ifdef PADDLE_WITH_HIP
    hipEventSynchronize(stop_);
    hipEventElapsedTime(&milliseconds, start_, stop_);
#else
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&milliseconds, start_, stop_);
#endif
    return milliseconds;
  }

 private:
  gpuEvent_t start_;
  gpuEvent_t stop_;
};

class CalculateStreamTimer {
 public:
  CalculateStreamTimer()
      : calculated_stream_(nullptr),
        start_time_(0),
        end_time_(0),
        is_started_(false) {}

  explicit CalculateStreamTimer(const phi::Place &place)
      : calculated_stream_(nullptr),
        start_time_(0),
        end_time_(0),
        is_started_(false),
        place_(place) {}

  void Start() {
    // Note(sonder): Since it is not possible to directly obtain the start time
    // of the event, "gettimeofday" is used here to retrieve it. The callback is
    // used to record the start time of the event.
    if (!is_started_) {
      calculated_stream_ = dynamic_cast<phi::GPUContext *>(
                               phi::DeviceContextPool::Instance().Get(place_))
                               ->stream();
    }
    if (calculated_stream_ != nullptr) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipStreamAddCallback(calculated_stream_,
                               RecordEventTimerCallback,
                               reinterpret_cast<void *>(&start_time_),
                               0));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaStreamAddCallback(calculated_stream_,
                                RecordEventTimerCallback,
                                reinterpret_cast<void *>(&start_time_),
                                0));
#endif
      is_started_ = true;
    }
  }

  void Stop() {
    if (calculated_stream_ != nullptr) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipStreamAddCallback(calculated_stream_,
                               RecordEventTimerCallback,
                               reinterpret_cast<void *>(&end_time_),
                               0));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaStreamAddCallback(calculated_stream_,
                                RecordEventTimerCallback,
                                reinterpret_cast<void *>(&end_time_),
                                0));
#endif
      is_started_ = false;
    }
  }

  double StartTime() {
    if (calculated_stream_ != nullptr) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipStreamSynchronize(calculated_stream_));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(calculated_stream_));
#endif
    }
    return start_time_;
  }

  double EndTime() {
    if (calculated_stream_ != nullptr) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipStreamSynchronize(calculated_stream_));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(calculated_stream_));
#endif
    }
    return end_time_;
  }

  bool IsStarted() { return is_started_; }

  void SetStream(gpuStream_t stream) { calculated_stream_ = stream; }

 private:
  gpuStream_t calculated_stream_;
  double start_time_;
  double end_time_;
  bool is_started_;
  const phi::Place place_;
};

}  // namespace phi
