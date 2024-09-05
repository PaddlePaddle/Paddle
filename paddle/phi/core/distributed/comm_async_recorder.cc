// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>

#include "paddle/common/macros.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/distributed/comm_async_recorder.h"
#include "paddle/phi/core/distributed/nccl_tools.h"

namespace phi {
namespace distributed {

CommAsyncRecorder::CommAsyncRecorder(const phi::Place& place,
                                     int gid,
                                     gpuStream_t stream)
    : place_(place), gid_(gid), nccl_stream_(stream), is_start_(false) {
#ifdef PADDLE_WITH_CUDA
  CUDA_CHECK(cudaEventCreate(&nccl_start_event_));
  CUDA_CHECK(cudaEventCreate(&nccl_end_event_));
#else  // PADDLE_WITH_HIP
  HIP_CHECK(hipEventCreate(&nccl_start_event_));
  HIP_CHECK(hipEventCreate(&nccl_end_event_));
#endif
}

void CommAsyncRecorder::EventDestroy() {
  backends::gpu::GPUDeviceGuard guard(place_.device);
#ifdef PADDLE_WITH_CUDA
  CUDA_CHECK(cudaEventDestroy(nccl_start_event_));
  CUDA_CHECK(cudaEventDestroy(nccl_end_event_));
#else  // PADDLE_WITH_HIP
  HIP_CHECK(hipEventDestroy(nccl_start_event_));
  HIP_CHECK(hipEventDestroy(nccl_end_event_));
#endif
}

void CommAsyncRecorder::StartRecord() {
  backends::gpu::GPUDeviceGuard guard(place_.device);
#ifdef PADDLE_WITH_CUDA
  CUDA_CHECK(cudaEventRecord(nccl_start_event_, nccl_stream_));
#else  // PADDLE_WITH_HIP
  HIP_CHECK(hipEventRecord(nccl_start_event_, nccl_stream_));
#endif
}

void CommAsyncRecorder::EndRecord() {
  backends::gpu::GPUDeviceGuard guard(place_.device);
#ifdef PADDLE_WITH_CUDA
  CUDA_CHECK(cudaEventRecord(nccl_end_event_, nccl_stream_));
#else  // PADDLE_WITH_HIP
  HIP_CHECK(hipEventRecord(nccl_end_event_, nccl_stream_));
#endif
}

bool CommAsyncRecorder::QueryEnd() const { return EventQuery(nccl_end_event_); }

bool CommAsyncRecorder::QueryStart() const {
  return EventQuery(nccl_start_event_);
}

void CommAsyncRecorder::Start() {
  if (IsStart()) {
    return;
  }
  is_start_ = true;
}

float CommAsyncRecorder::RecordTime() const {
  float elapsedTime = 0.f;
#ifdef PADDLE_WITH_CUDA
  CUDA_CHECK(
      cudaEventElapsedTime(&elapsedTime, nccl_start_event_, nccl_end_event_));
#else  // PADDLE_WITH_HIP
  HIP_CHECK(
      hipEventElapsedTime(&elapsedTime, nccl_start_event_, nccl_end_event_));
#endif
  return elapsedTime;
}

bool CommAsyncRecorder::EventQuery(gpuEvent_t event) const {
#ifdef PADDLE_WITH_CUDA
  CUDA_CHECK(cudaGetLastError());
  cudaError_t ret = cudaEventQuery(event);
  if (ret == cudaSuccess) {
    return true;
  } else if (ret != cudaErrorNotReady) {
    CUDA_CHECK(ret);
  } else {
    // ignore and clear the error if not ready
    CUDA_CHECK(cudaGetLastError());
  }
#else  // PADDLE_WITH_HIP
  HIP_CHECK(hipGetLastError());
  hipError_t ret = hipEventQuery(event);
  if (ret == hipSuccess) {
    return true;
  } else if (ret != hipErrorNotReady) {
    HIP_CHECK(ret);
  } else {
    // ignore and clear the error if not ready
    HIP_CHECK(hipGetLastError());
  }
#endif
  return false;
}

void CommAsyncRecorder::SynchronizeAllRecorders() {
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else  // PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif
}
}  // namespace distributed
}  // namespace phi
