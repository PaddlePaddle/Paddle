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

#include "paddle/phi/core/distributed/nccl_async_recorder.h"
#include "paddle/common/macros.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/distributed/nccl_tools.h"

namespace phi {
namespace distributed {

NCCLAsyncRecorder::NCCLAsyncRecorder(const phi::Place& place,
                                     int rank,
                                     int gid,
                                     gpuStream_t stream,
                                     CommType comm_type)
    : place_(place),
      rank_(rank),
      gid_(gid),
      nccl_stream_(stream),
      comm_type_(comm_type),
      is_start_(false) {
#ifdef PADDLE_WITH_CUDA
  CUDA_CHECK(cudaEventCreate(&nccl_start_event_));
  CUDA_CHECK(cudaEventCreate(&nccl_end_event_));
#else  // PADDLE_WITH_HIP
  HIP_CHECK(hipEventCreate(&nccl_start_event_));
  HIP_CHECK(hipEventCreate(&nccl_end_event_));
#endif
}

void NCCLAsyncRecorder::EventDestroy() {
  backends::gpu::GPUDeviceGuard guard(place_.device);
#ifdef PADDLE_WITH_CUDA
  CUDA_CHECK(cudaEventDestroy(nccl_start_event_));
  CUDA_CHECK(cudaEventDestroy(nccl_end_event_));
#else  // PADDLE_WITH_HIP
  HIP_CHECK(hipEventDestroy(nccl_start_event_));
  HIP_CHECK(hipEventDestroy(nccl_end_event_));
#endif
}

void NCCLAsyncRecorder::StartRecord() {
  backends::gpu::GPUDeviceGuard guard(place_.device);
#ifdef PADDLE_WITH_CUDA
  CUDA_CHECK(cudaEventRecord(nccl_start_event_, nccl_stream_));
#else  // PADDLE_WITH_HIP
  HIP_CHECK(hipEventRecord(nccl_start_event_, nccl_stream_));
#endif
}

void NCCLAsyncRecorder::EndRecord() {
  backends::gpu::GPUDeviceGuard guard(place_.device);
#ifdef PADDLE_WITH_CUDA
  CUDA_CHECK(cudaEventRecord(nccl_end_event_, nccl_stream_));
#else  // PADDLE_WITH_HIP
  HIP_CHECK(hipEventRecord(nccl_end_event_, nccl_stream_));
#endif
}

bool NCCLAsyncRecorder::QueryEnd() const { return EventQuery(nccl_end_event_); }

bool NCCLAsyncRecorder::QueryStart() const {
  return EventQuery(nccl_start_event_);
}

void NCCLAsyncRecorder::Start() {
  if (IsStart()) {
    return;
  }
  // start_time_ = std::chrono::high_resolution_clock::now();
  is_start_ = true;
}

float NCCLAsyncRecorder::RecordTime() const {
  // auto end_time = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<float, std::milli> duration = end_time - start_time_;
  // return duration.count();

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

bool NCCLAsyncRecorder::EventQuery(gpuEvent_t event) const {
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

}  // namespace distributed
}  // namespace phi
