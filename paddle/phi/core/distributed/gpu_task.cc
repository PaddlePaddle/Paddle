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

#include "paddle/phi/core/distributed/gpu_task.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_tools.h"
#include "paddle/phi/core/utils/data_type.h"

COMMON_DECLARE_bool(enable_time_compare);

namespace phi::distributed {

void CUDART_CB GetCPUTime(cudaStream_t stream, cudaError_t status, void* data) {
  auto* time_ptr =
      static_cast<std::chrono::time_point<std::chrono::steady_clock>*>(data);
  *time_ptr = std::chrono::steady_clock::now();
}

GPUTask::GPUTask(const phi::Place& place,
                 const std::string& group_key,
                 uint64_t seq,
                 int64_t numel,
                 bool sync_op,
                 bool use_calc_stream,
                 gpuStream_t stream,
                 CommType comm_type)
    : place_(place),
      group_key_(group_key),
      seq_(seq),
      numel_(numel),
      nccl_stream_(stream),
      comm_type_(comm_type),
      sync_op_(sync_op),
      use_calc_stream_(use_calc_stream),
      start_event_(nullptr),
      end_event_(nullptr),
      start_event_created_(false),
      end_event_created_(false),
      completed_(false),
      has_printed_(false),
      skip_(false) {
  if (numel <= 1024 * 1024) {
    skip_ = true;
  }
}

void GPUTask::StartRecord() {
  if (skip_) {
    return;
  }
  backends::gpu::GPUDeviceGuard guard(place_.device);
  if (!start_event_created_) {
#ifdef PADDLE_WITH_CUDA
    CUDA_CHECK(cudaEventCreateWithFlags(&start_event_, 0));
    CUDA_CHECK(
        cudaStreamAddCallback(nccl_stream_, GetCPUTime, &start_time_, 0));
    CUDA_CHECK(cudaEventRecord(start_event_, nccl_stream_));
#else  // PADDLE_WITH_HIP
    HIP_CHECK(hipEventCreateWithFlags(&start_event_, 0));
    HIP_CHECK(hipEventRecord(start_event_, nccl_stream_));
#endif
    start_event_created_ = true;
  }
}
void GPUTask::EndRecord() {
  if (skip_) {
    return;
  }
  backends::gpu::GPUDeviceGuard guard(place_.device);
  if (!end_event_created_) {
#ifdef PADDLE_WITH_CUDA
    CUDA_CHECK(cudaEventCreateWithFlags(&end_event_, 0));
    CUDA_CHECK(cudaStreamAddCallback(nccl_stream_, GetCPUTime, &end_time_, 0));
    CUDA_CHECK(cudaEventRecord(end_event_, nccl_stream_));
#else  // PADDLE_WITH_HIP
    HIP_CHECK(hipEventCreateWithFlags(&end_event_, 0));
    HIP_CHECK(hipEventRecord(end_event_, nccl_stream_));
#endif
    end_event_created_ = true;
  }
}

#ifdef PADDLE_WITH_CUDA
void GPUTask::ClearRecord() {
  if (skip_) {
    return;
  }
  if (start_event_created_) {
    backends::gpu::GPUDeviceGuard guard(place_.device);
    CUDA_CHECK(cudaEventDestroy(start_event_));
    start_event_created_ = false;
  }
  if (end_event_created_) {
    backends::gpu::GPUDeviceGuard guard(place_.device);
    CUDA_CHECK(cudaEventDestroy(end_event_));
    end_event_created_ = false;
  }
}
#else  // PADDLE_WITH_HIP
void GPUTask::ClearRecord() {
  if (skip_) {
    return;
  }
  if (start_event_created_) {
    backends::gpu::GPUDeviceGuard guard(place_.device);
    HIP_CHECK(hipEventDestroy(start_event_));
    start_event_created_ = false;
  }
  if (end_event_created_) {
    backends::gpu::GPUDeviceGuard guard(place_.device);
    HIP_CHECK(hipEventDestroy(end_event_));
    end_event_created_ = false;
  }
}
#endif

bool GPUTask::CudaEventQuery(gpuEvent_t event) {
  if (skip_) {
    return true;
  }
#ifdef PADDLE_WITH_CUDA
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

bool GPUTask::IsCompleted() {
  if (skip_ || completed_) {
    return true;
  }
  if (end_event_created_ && CudaEventQuery(end_event_)) {
    completed_ = true;
  }
  return completed_;
}

std::string GPUTask::GetTraceMsg() {
  if (skip_) {
    return "";
  }
  double start_time =
      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
          start_time_.time_since_epoch())
          .count();
  double end_time =
      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
          end_time_.time_since_epoch())
          .count();
  if (FLAGS_enable_time_compare) {
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start_event_, end_event_);
    return group_key_ + "," + std::to_string(seq_) + "," +
           std::to_string(static_cast<std::uint8_t>(comm_type_)) + "," +
           std::to_string(numel_) + "," + std::to_string(start_time) + "," +
           std::to_string(end_time) + "," + std::to_string(use_calc_stream_) +
           "," + std::to_string(end_time - start_time) + "," +
           std::to_string(elapsed_ms * 1000) + "," +
           std::to_string((end_time - start_time) / (elapsed_ms * 1000));
  } else {
    return group_key_ + "," + std::to_string(seq_) + "," +
           std::to_string(static_cast<std::uint8_t>(comm_type_)) + "," +
           std::to_string(numel_) + "," + std::to_string(start_time) + "," +
           std::to_string(end_time);
  }
}

bool GPUTask::HasPrinted() {
  if (skip_) {
    return true;
  }
  return has_printed_;
}

void GPUTask::SetPrint() { has_printed_ = true; }

bool GPUTask::Skip() { return skip_; }
}  // namespace phi::distributed
