// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include "glog/logging.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/macros.h"

namespace phi {
namespace distributed {

class CommTask {
 public:
  CommTask(const std::string& backend = "",
           const phi::Place& place = phi::Place(),
           int rank = -1,
           int size = 0,
           int64_t numel = 0,
           CommType comm_type = CommType::UNKNOWN)
      : backend_(backend),
        place_(place),
        rank_(rank),
        size_(size),
        numel_(numel),
        comm_type_(comm_type) {
    VLOG(0) << "debug CommTask construct";
  }
  virtual ~CommTask() = default;

  std::string GetBackend() { return backend_; }
  phi::Place GetPlace() { return place_; }
  int GetRank() { return rank_; }
  int GetSize() { return size_; }
  int64_t GetNumel() { return numel_; }
  uint64_t GetSeq() { return seq_; }
  CommType GetCommType() { return comm_type_; }
  bool GetTraceUpdated() { return start_trace_updated_; }
  void SetTraceUpdated(bool updated) { start_trace_updated_ = updated; }
  std::chrono::time_point<std::chrono::steady_clock> GetStartTime() {
    return start_time_;
  }

  virtual void StartRecord() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return;
  }
  virtual void EndRecord(phi::gpuStream_t stream) {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return;
  }
  virtual bool CudaEventQuery(cudaEvent_t event) {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return false;
  }

  virtual void SetException(std::exception_ptr exception) {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return;
  }
  virtual void CheckAndSetException() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return;
  }
  virtual std::exception_ptr CheckCommErrors() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return nullptr;
  }
  virtual bool IsStarted() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return false;
  }
  virtual bool IsTimeout() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return false;
  }
  virtual bool IsCompleted() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return false;
  }
  virtual bool IsSuccess() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return false;
  }
  virtual std::exception_ptr GetException() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return nullptr;
  }
  virtual void AbortComm() {
    PADDLE_THROW(
        phi::errors::Unimplemented("%s is not implemented.", __func__));
    return;
  }

 protected:
  std::string backend_;
  phi::Place place_;
  int rank_;
  int size_;
  int64_t numel_;
  uint64_t seq_{0};
  CommType comm_type_;
  bool start_trace_updated_{false};

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool completed_ = false;
  bool aborted_{false};
  std::exception_ptr exception_;
  std::chrono::time_point<std::chrono::steady_clock> start_time_;

 private:
  DISABLE_COPY_AND_ASSIGN(CommTask);
};

}  // namespace distributed
}  // namespace phi
