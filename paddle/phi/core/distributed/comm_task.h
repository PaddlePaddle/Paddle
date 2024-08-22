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
#include "paddle/common/macros.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"

#if defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/dynload/rccl.h"
#endif
#if defined(PADDLE_WITH_NCCL)
#include "paddle/phi/backends/dynload/nccl.h"
#endif

namespace phi {
namespace distributed {

class Store;
class CommTask {
 public:
  CommTask(const std::string& backend = "",
           const phi::Place& place = phi::Place(),
           const std::string& group_key = "",
           int rank = -1,
           int size = 0,
           int gid = 0,
           uint64_t seq = 0,
           int64_t numel = 0,
           ncclComm_t nccl_comm = nullptr,
           gpuStream_t nccl_stream = nullptr,
           CommType comm_type = CommType::UNKNOWN)
      : backend_(backend),
        place_(place),
        group_key_(group_key),
        rank_(rank),
        size_(size),
        gid_(gid),
        seq_(seq),
        numel_(numel),
        nccl_comm_(nccl_comm),
        nccl_stream_(nccl_stream),
        comm_type_(comm_type) {
    const char* global_rank = std::getenv("PADDLE_TRAINER_ID");
    PADDLE_ENFORCE_NOT_NULL(
        global_rank,
        common::errors::NotFound(
            "The environment variable 'PADDLE_TRAINER_ID' cannot be found."));
    global_rank_ = std::atoi(global_rank);
  }
  virtual ~CommTask() = default;

  std::string UniqueKey() {
    return "group_key:" + group_key_ + ",op:" + CommTypeToString(comm_type_) +
           ",gid:" + std::to_string(gid_) + ",seq:" + std::to_string(seq_);
  }

  std::string GroupKey() { return group_key_; }
  std::string GetBackend() { return backend_; }
  phi::Place GetPlace() { return place_; }
  int GetGlobalRank() { return global_rank_; }
  int GetRank() { return rank_; }
  int GetSize() { return size_; }
  int GetGid() { return gid_; }
  int64_t GetNumel() { return numel_; }
  uint64_t GetSeq() { return seq_; }
  CommType GetCommType() { return comm_type_; }
  bool GetTraceUpdated() { return start_trace_updated_; }
  void SetTraceUpdated() { start_trace_updated_ = true; }
  std::chrono::time_point<std::chrono::steady_clock> GetStartTime() {
    return start_time_;
  }
  std::shared_ptr<Store> GetStore() { return store_; }
  void SetStore(std::shared_ptr<Store> store) { store_ = store; }

  ncclComm_t nccl_comm() { return nccl_comm_; }
  gpuStream_t nccl_stream() { return nccl_stream_; }

  virtual std::string GetTraceMsg() {
    PADDLE_THROW(
        common::errors::Unimplemented("%s is not implemented.", __func__));
    return "";
  }
  virtual void StartRecord() {
    PADDLE_THROW(
        common::errors::Unimplemented("%s is not implemented.", __func__));
    return;
  }
  virtual void EndRecord() {
    PADDLE_THROW(
        common::errors::Unimplemented("%s is not implemented.", __func__));
    return;
  }

  virtual void ClearRecord() {
    PADDLE_THROW(
        common::errors::Unimplemented("%s is not implemented.", __func__));
    return;
  }

  virtual std::string GetCommErrors() {
    PADDLE_THROW(
        common::errors::Unimplemented("%s is not implemented.", __func__));
    return "";
  }
  virtual bool IsStarted() {
    PADDLE_THROW(
        common::errors::Unimplemented("%s is not implemented.", __func__));
    return false;
  }
  virtual bool IsTimeout() {
    PADDLE_THROW(
        common::errors::Unimplemented("%s is not implemented.", __func__));
    return false;
  }
  virtual bool IsCompleted() {
    PADDLE_THROW(
        common::errors::Unimplemented("%s is not implemented.", __func__));
    return false;
  }
  virtual void SetUpdated(bool updated) {
    PADDLE_THROW(
        common::errors::Unimplemented("%s is not implemented.", __func__));
    return;
  }
  virtual bool IsUpdated() {
    PADDLE_THROW(
        common::errors::Unimplemented("%s is not implemented.", __func__));
    return false;
  }
  virtual void AbortComm() {
    PADDLE_THROW(
        common::errors::Unimplemented("%s is not implemented.", __func__));
    return;
  }

 protected:
  std::string backend_;
  phi::Place place_;
  std::string group_key_;
  int global_rank_;
  int rank_;
  int size_;
  int gid_;
  uint64_t seq_{0};
  int64_t numel_;
  ncclComm_t nccl_comm_;
  gpuStream_t nccl_stream_;
  CommType comm_type_;
  bool start_trace_updated_{false};

  // task status
  bool started_ = false;
  bool completed_ = false;
  // task status changed
  bool updated_ = true;
  bool aborted_{false};
  std::chrono::time_point<std::chrono::steady_clock> start_time_;
  std::shared_ptr<Store> store_;

 private:
  DISABLE_COPY_AND_ASSIGN(CommTask);
};

}  // namespace distributed
}  // namespace phi
