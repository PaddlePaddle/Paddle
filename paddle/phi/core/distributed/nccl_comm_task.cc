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

#include "paddle/phi/core/distributed/nccl_comm_task.h"

#include "gflags/gflags.h"

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_tools.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi::distributed {

NCCLCommTask::NCCLCommTask(const phi::Place& place,
                           const std::string& group_key,
                           int rank,
                           int size,
                           int gid,
                           uint64_t seq,
                           int64_t numel,
                           bool sync_op,
                           bool use_calc_stream,
                           ncclComm_t nccl_comm,
                           gpuStream_t stream,
                           CommType comm_type,
                           int64_t timeout)
    : CommTask("NCCL",
               place,
               group_key,
               rank,
               size,
               gid,
               seq,
               numel,
               nccl_comm,
               stream,
               comm_type),
      timeout_(std::chrono::milliseconds(timeout)),
      sync_op_(sync_op),
      use_calc_stream_(use_calc_stream),
      nccl_start_event_(nullptr),
      nccl_end_event_(nullptr) {
  start_trace_updated_ = false;
  start_event_created_ = false;
  end_event_created_ = false;
  start_time_ = std::chrono::steady_clock::now();
}

void NCCLCommTask::StartRecord() {
  backends::gpu::GPUDeviceGuard guard(place_.device);
  if (!start_event_created_) {
#ifdef PADDLE_WITH_CUDA
    CUDA_CHECK(cudaEventCreateWithFlags(&nccl_start_event_, cuda_event_flags_));
#else  // PADDLE_WITH_HIP
    HIP_CHECK(hipEventCreateWithFlags(&nccl_start_event_, hip_event_flags_));
#endif
    start_event_created_ = true;
  }
#ifdef PADDLE_WITH_CUDA
  CUDA_CHECK(cudaEventRecord(nccl_start_event_, nccl_stream_));
#else  // PADDLE_WITH_HIP
  HIP_CHECK(hipEventRecord(nccl_start_event_, nccl_stream_));
#endif
}
void NCCLCommTask::EndRecord() {
  backends::gpu::GPUDeviceGuard guard(place_.device);
  if (!end_event_created_) {
#ifdef PADDLE_WITH_CUDA
    CUDA_CHECK(cudaEventCreateWithFlags(&nccl_end_event_, cuda_event_flags_));
#else  // PADDLE_WITH_HIP
    HIP_CHECK(hipEventCreateWithFlags(&nccl_end_event_, hip_event_flags_));
#endif
    end_event_created_ = true;
  }
#ifdef PADDLE_WITH_CUDA
  CUDA_CHECK(cudaEventRecord(nccl_end_event_, nccl_stream_));
#else  // PADDLE_WITH_HIP
  HIP_CHECK(hipEventRecord(nccl_end_event_, nccl_stream_));
#endif
}

#ifdef PADDLE_WITH_CUDA
void NCCLCommTask::ClearRecord() {
  if (start_event_created_) {
    backends::gpu::GPUDeviceGuard guard(place_.device);
    CUDA_CHECK(cudaEventDestroy(nccl_start_event_));
    start_event_created_ = false;
  }
  if (end_event_created_) {
    backends::gpu::GPUDeviceGuard guard(place_.device);
    CUDA_CHECK(cudaEventDestroy(nccl_end_event_));
    end_event_created_ = false;
  }
}
#else  // PADDLE_WITH_HIP
void NCCLCommTask::ClearRecord() {
  if (start_event_created_) {
    backends::gpu::GPUDeviceGuard guard(place_.device);
    HIP_CHECK(hipEventDestroy(nccl_start_event_));
    start_event_created_ = false;
  }
  if (end_event_created_) {
    backends::gpu::GPUDeviceGuard guard(place_.device);
    HIP_CHECK(hipEventDestroy(nccl_end_event_));
    end_event_created_ = false;
  }
}
#endif

bool NCCLCommTask::CudaEventQuery(gpuEvent_t event) {
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

std::string GetNCCLErrorDetail(ncclResult_t result) {
  std::string detail;
  std::string last_error;
#ifdef ENABLE_NCCL_GET_LAST_ERROR
  last_error =
      ", Last error: " + std::string(phi::dynload::ncclGetLastError(NULL));
#endif
  switch (result) {
    case ncclUnhandledCudaError:
      detail = "ncclUnhandledCudaError: Call to CUDA function failed.";
      break;
    case ncclSystemError:
      detail =
          "ncclSystemError: System call (e.g. socket, malloc) or external "
          "library call failed or device error. ";
#ifndef NCCL_REMOTE_ERROR
      // Before ncclRemoteError was created, unexpected remote disconnect was
      // categorized as ncclSystemError
      detail += "It can be also caused by unexpected exit of a remote peer.";
#endif
      break;
    case ncclInternalError:
      detail = "ncclInternalError: Internal check failed.";
      break;
    case ncclInvalidArgument:
      detail = "ncclInvalidArgument: Invalid value for an argument.";
      break;
    case ncclInvalidUsage:
      detail =
          "ncclInvalidUsage: This usually reflects invalid usage of NCCL "
          "library.";
      break;
#ifdef NCCL_REMOTE_ERROR
    case ncclRemoteError:
      detail =
          "ncclRemoteError: A call failed possibly due to a network error or a "
          "remote process exiting prematurely.";
      break;
#endif
    default:
      detail = "Unknown NCCL error!";
  }
  return detail + last_error;
}

std::string NCCLCommTask::GetCommErrors() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!comm_error_.empty()) {
    return comm_error_;
  }

  ncclResult_t nccl_async_error;
  NCCL_CHECK(
      phi::dynload::ncclCommGetAsyncError(nccl_comm_, &nccl_async_error));
  if (nccl_async_error != ncclSuccess) {
    comm_error_ =
        "\n\t Find nccl comm error: " + GetNCCLErrorDetail(nccl_async_error);
  }
  return comm_error_;
}

bool NCCLCommTask::IsStarted() {
  if (started_) {
    return true;
  }
  if (start_event_created_ && CudaEventQuery(nccl_start_event_)) {
    started_ = true;
    updated_ = true;
  }
  return started_;
}

bool NCCLCommTask::IsCompleted() {
  if (completed_) {
    return true;
  }
  if (end_event_created_ && CudaEventQuery(nccl_end_event_)) {
    completed_ = true;
    updated_ = true;
  }
  return completed_;
}

void NCCLCommTask::SetUpdated(bool updated) { updated_ = updated; }

bool NCCLCommTask::IsUpdated() { return updated_; }

bool NCCLCommTask::IsTimeout() {
  auto current_timepoint = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             current_timepoint - start_time_) >= timeout_;
}

void NCCLCommTask::AbortComm() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (aborted_) {
    return;
  }
  NCCL_CHECK(phi::dynload::ncclCommAbort(nccl_comm_));

  aborted_ = true;
  nccl_comm_ = nullptr;
  return;
}

std::string NCCLCommTask::GetTraceMsg() {
  auto global_ranks =
      phi::distributed::CommContextManager::GetInstance().GetGroupRanks(
          group_key_);
  return "group_key:" + group_key_ +
         ",group_ranks:" + VectorToString(global_ranks) +
         ",global_rank:" + std::to_string(global_rank_) +
         ",local_rank:" + std::to_string(rank_) +
         ",comm_count:" + std::to_string(seq_) +
         ",op:" + CommTypeToString(comm_type_) +
         ",started:" + std::to_string(started_) +
         ",completed:" + std::to_string(completed_) +
         ",numel:" + std::to_string(numel_) +
         ",nranks:" + std::to_string(size_);
}

}  // namespace phi::distributed
