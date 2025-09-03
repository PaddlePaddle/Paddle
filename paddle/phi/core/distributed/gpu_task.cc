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
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_tools.h"
#include "paddle/phi/core/utils/data_type.h"
#if defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/dynload/rccl.h"
#else
#include "paddle/phi/backends/dynload/nccl.h"
#endif

COMMON_DECLARE_bool(enable_time_compare);

namespace phi::distributed {

GPUTask::GPUTask(const phi::Place& place,
                 gpuStream_t stream,
                 const std::string& label)
    : place_(place),
      stream_(stream),
      label_(label),
      start_event_(nullptr),
      end_event_(nullptr),
      start_event_created_(false),
      end_event_created_(false),
      completed_(false),
      has_printed_(false) {
  if (stream_ == nullptr) {
    stream_ = static_cast<phi::GPUContext*>(
                  phi::DeviceContextPool::Instance().Get(place))
                  ->stream();
  }
}

void GPUTask::StartRecord() {
  backends::gpu::GPUDeviceGuard guard(place_.device);
#ifdef PADDLE_WITH_CUDA
  if (!start_event_created_) {
    CUDA_CHECK(cudaEventCreateWithFlags(&start_event_, 0));
    start_event_created_ = true;
  }
  CUDA_CHECK(cudaEventRecord(start_event_, stream_));
#else  // PADDLE_WITH_HIP
  if (!start_event_created_) {
    HIP_CHECK(hipEventCreateWithFlags(&start_event_, 0));
    start_event_created_ = true;
  }
  HIP_CHECK(hipEventRecord(start_event_, stream_));
#endif
  //   if (!start_event_created_) {
  //     start_event_ =
  //     CudaEventResourcePool::Instance().New(place_.device);//std::shared_ptr<gpuEvent_t>
  //     start_event_created_ = true;
  //   }
  // #ifdef PADDLE_WITH_CUDA
  //   CUDA_CHECK(cudaEventRecord(*start_event_, stream_));
  // #else  // PADDLE_WITH_HIP
  //   HIP_CHECK(hipEventRecord(*start_event_, stream_));
  // #endif
}

void GPUTask::EndRecord() {
  backends::gpu::GPUDeviceGuard guard(place_.device);
#ifdef PADDLE_WITH_CUDA
  if (!end_event_created_) {
    CUDA_CHECK(cudaEventCreateWithFlags(&end_event_, 0));
    end_event_created_ = true;
  }
  CUDA_CHECK(cudaEventRecord(end_event_, stream_));
#else  // PADDLE_WITH_HIP
  if (!end_event_created_) {
    HIP_CHECK(hipEventCreateWithFlags(&end_event_, 0));
    end_event_created_ = true;
  }
  HIP_CHECK(hipEventRecord(end_event_, stream_));
#endif
  //   if (!end_event_created_) {
  //     end_event_ =
  //     CudaEventResourcePool::Instance().New(place_.device);//std::shared_ptr<gpuEvent_t>
  //     end_event_created_ = true;
  //   }
  // #ifdef PADDLE_WITH_CUDA
  //   CUDA_CHECK(cudaEventRecord(*end_event_, stream_));
  // #else  // PADDLE_WITH_HIP
  //   HIP_CHECK(hipEventRecord(*end_event_, stream_));
  // #endif
}

#ifdef PADDLE_WITH_CUDA
void GPUTask::ClearRecord() {
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
  if (completed_) {
    return true;
  }
  if (end_event_created_ && CudaEventQuery(end_event_)) {
    // if (end_event_created_ && CudaEventQuery(*end_event_)) {
    completed_ = true;
  }
  return completed_;
}

std::string GPUTask::GetTraceMsg(gpuEvent_t zero_event) {
  float start_ms, end_ms;
#ifdef PADDLE_WITH_CUDA
  cudaEventElapsedTime(&start_ms, zero_event, start_event_);
  cudaEventElapsedTime(&end_ms, zero_event, end_event_);
#else  // PADDLE_WITH_HIP
  hipEventElapsedTime(&start_ms, zero_event, start_event_);
  hipEventElapsedTime(&end_ms, zero_event, end_event_);
#endif
  // #ifdef PADDLE_WITH_CUDA
  //   cudaEventElapsedTime(&start_ms, zero_event, *start_event_);
  //   cudaEventElapsedTime(&end_ms, zero_event, *end_event_);
  // #else // PADDLE_WITH_HIP
  //   hipEventElapsedTime(&start_ms, zero_event, *start_event_);
  //   hipEventElapsedTime(&end_ms, zero_event, *end_event_);
  // #endif
  return std::to_string(start_ms) + "," + std::to_string(end_ms) + "," + label_;
}

bool GPUTask::HasPrinted() { return has_printed_; }

void GPUTask::SetPrint() { has_printed_ = true; }

CudaEventResourcePool& CudaEventResourcePool::Instance() {
  static CudaEventResourcePool pool;
  return pool;
}

CudaEventResourcePool::CudaEventResourcePool() {
  int dev_cnt = phi::backends::gpu::GetGPUDeviceCount();
  pool_.reserve(dev_cnt);
  for (int dev_idx = 0; dev_idx < dev_cnt; ++dev_idx) {
    auto creator = [dev_idx] {
      phi::backends::gpu::SetDeviceId(dev_idx);
      gpuEvent_t* event = new gpuEvent_t;
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipEventCreateWithFlags(event, 0));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreateWithFlags(event, 0));
#endif
      return event;
    };

    auto deleter = [dev_idx](gpuEvent_t* event) {
      phi::backends::gpu::SetDeviceId(dev_idx);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipEventDestroy(*event));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(*event));
      delete event;
#endif
    };

    pool_.emplace_back(
        paddle::platform::ResourcePool<gpuEvent_t>::Create(creator, deleter));
  }
}

std::shared_ptr<gpuEvent_t> CudaEventResourcePool::New(int dev_idx) {
  PADDLE_ENFORCE_GE(
      dev_idx,
      0,
      common::errors::InvalidArgument(
          "The dev_idx should be not less than 0, but got %d.", dev_idx));
  PADDLE_ENFORCE_LT(
      dev_idx,
      pool_.size(),
      common::errors::OutOfRange(
          "The dev_idx should be less than device count %d, but got %d.",
          pool_.size(),
          dev_idx));
  return pool_[dev_idx]->New();
}

}  // namespace phi::distributed
