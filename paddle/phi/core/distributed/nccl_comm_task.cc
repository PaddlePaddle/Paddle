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
#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/distributed/nccl_tools.h"
#include "paddle/phi/core/distributed/trace_utils.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {
namespace distributed {

NCCLCommTask::NCCLCommTask(const phi::Place& place,
                           int rank,
                           int size,
                           uint64_t seq,
                           int64_t numel,
                           bool sync_op,
                           bool use_calc_stream,
                           ncclComm_t nccl_comm,
                           gpuStream_t stream,
                           CommType comm_type)
    : CommTask("NCCL", place, rank, size, numel, comm_type),
      seq_(seq),
      numel_(numel),
      sync_op_(sync_op),
      use_calc_stream_(use_calc_stream),
      nccl_comm_(nccl_comm),
      nccl_stream_(stream) {
  VLOG(0) << "debug NCCLCommTask construct before";
  start_trace_updated_ = false;
  start_event_created_ = false;
  end_event_created_ = false;
  start_time_ = std::chrono::steady_clock::now();
  VLOG(0) << "debug NCCLCommTask construct after";
}

void NCCLCommTask::StartRecord() {
  VLOG(0) << "debug NCCLCommTask StartRecord rank " << GetRank();
  backends::gpu::GPUDeviceGuard guard(place_.device);
  if (!start_event_created_) {
    CUDA_CHECK(cudaEventCreateWithFlags(&nccl_start_event_, cuda_event_flags_));
    start_event_created_ = true;
  }
  VLOG(0) << "debug NCCLCommTask StartRecord event: " << nccl_start_event_
          << ", stream: " << nccl_stream_
          << ", device: " << static_cast<int>(place_.device);
  cudaError_t r = cudaEventRecord(nccl_start_event_, nccl_stream_);
  if (r != cudaSuccess) {
    VLOG(0) << "Failed, cuda error file: " << __FILE__ << ", line: " << __LINE__
            << ", err: " << cudaGetErrorString(r);
  }
  VLOG(0) << "debug NCCLCommTask StartRecord ";
}
void NCCLCommTask::EndRecord(phi::gpuStream_t stream) {
  backends::gpu::GPUDeviceGuard guard(place_.device);
  if (!end_event_created_) {
    CUDA_CHECK(cudaEventCreateWithFlags(&nccl_end_event_, cuda_event_flags_));
    end_event_created_ = true;
  }
  CUDA_CHECK(cudaEventRecord(nccl_end_event_, stream));
}

bool NCCLCommTask::CudaEventQuery(cudaEvent_t event) {
  cudaError_t ret = cudaEventQuery(event);
  if (ret == cudaSuccess) {
    return true;
  } else if (ret != cudaErrorNotReady) {
    CUDA_CHECK(ret);
  } else {
    // ignore and clear the error if not ready
    CUDA_CHECK(cudaGetLastError());
  }
  return false;
}

void NCCLCommTask::CheckAndSetException() {
  if (GetException()) {
    return;
  }
  auto exception_ptr = CheckCommErrors();
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
  if (exception_) {
    LOG(INFO) << "Rank " << rank_
              << ", found async exception when checking for nccl errors: " +
                     GetExceptionMsgFromExceptionPtr(exception_);
  }
}

std::string GetNCCLErrorDetail(ncclResult_t result) {
  std::string detail;
  std::string last_error;
#ifdef ENABLE_NCCL_GET_LAST_ERROR
  last_error =
      "\nLast error:\n" + std::string(phi::dynload::ncclGetLastError(NULL));
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

std::exception_ptr NCCLCommTask::CheckCommErrors() {
  std::unique_lock<std::mutex> lock(mutex_);
  NCCL_CHECK(
      phi::dynload::ncclCommGetAsyncError(nccl_comm_, &nccl_async_error_));
  if (nccl_async_error_ != ncclSuccess) {
    return std::make_exception_ptr(
        std::runtime_error("NCCL communicator has error: " +
                           GetNCCLErrorDetail(nccl_async_error_)));
  }
  return nullptr;
}

bool NCCLCommTask::IsStarted() {
  CheckAndSetException();
  return GetException() || CudaEventQuery(nccl_start_event_);
}

bool NCCLCommTask::IsCompleted() {
  CheckAndSetException();
  return GetException() || CudaEventQuery(nccl_end_event_);
}

bool NCCLCommTask::IsSuccess() {
  if (GetException()) {
    return false;
  }

  return CheckCommErrors() && CudaEventQuery(nccl_end_event_);
}

bool NCCLCommTask::IsTimeout() {
  auto current_timepoint = std::chrono::steady_clock::now();
  return (std::chrono::duration_cast<std::chrono::milliseconds>(
              current_timepoint - start_time_) >= timeout_);
}

void NCCLCommTask::SetException(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception;
}

std::exception_ptr NCCLCommTask::GetException() {
  std::unique_lock<std::mutex> lock(mutex_);
  return exception_;
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

}  // namespace distributed
}  // namespace phi
