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

#if defined(PADDLE_WITH_GLOO)
#include <gloo/rendezvous/prefix_store.h>

#include "paddle/phi/core/distributed/gloo_comm_context.h"
#include "paddle/phi/core/distributed/gloo_utils.h"
#include "paddle/phi/core/distributed/store/gloo_store.h"
#endif

#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/comm_task_manager.h"

#include <memory>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/distributed/store/store.h"
#include "paddle/phi/core/distributed/trace_utils.h"
#include "paddle/phi/core/enforce.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {
namespace distributed {

std::thread CommTaskManager::comm_task_loop_thread_;
const int64_t CommTaskManager::loop_thread_sleep_millis = 10000;

std::atomic<bool> CommTaskManager::terminated_;
std::mutex CommTaskManager::comm_task_list_mutex_;
std::condition_variable CommTaskManager::comm_task_list_cv_;
std::list<std::unique_ptr<CommTask>> CommTaskManager::comm_task_list_;

CommTaskManager::CommTaskManager() {
  terminated_.store(false);
  comm_task_loop_thread_ = std::thread(&CommTaskManager::CommTaskLoop, this);
  LOG(INFO) << "CommTaskManager init success.";
}
CommTaskManager::~CommTaskManager() {
  terminated_.store(true);

  if (comm_task_loop_thread_.joinable()) {
    comm_task_loop_thread_.join();
    comm_task_list_cv_.notify_one();
  }
  LOG(INFO) << "CommTaskManager destruct success.";
}

void CommTaskManager::CommTaskEnqueue(std::unique_ptr<CommTask> comm_task) {
  if (!terminated_.load()) {
    std::lock_guard<std::mutex> lock(comm_task_list_mutex_);
    comm_task_list_.emplace_back(std::move(comm_task));
  }
}

void CommTaskManager::CommTaskLoop() {
  bool done = false;
  while (!terminated_.load() || !done) {
    std::unique_lock<std::mutex> lock(comm_task_list_mutex_);
    comm_task_list_cv_.wait_for(
        lock,
        std::chrono::milliseconds(loop_thread_sleep_millis),
        [&]() -> bool { return terminated_.load(); });
    for (auto task = comm_task_list_.begin(); task != comm_task_list_.end();) {
      (*task)->CheckAndSetException();
      if ((*task)->IsTimeout()) {
        std::string exception_msg = (*task)->GetTraceMsg();
        exception_msg += GenerateTraceMsg((*task)->GetStore(),
                                          (*task)->GetBackend(),
                                          (*task)->GetRank(),
                                          (*task)->GetGid(),
                                          (*task)->GetSize());
        LOG(ERROR) << exception_msg;
        std::exception_ptr exception_ptr =
            std::make_exception_ptr(std::runtime_error(exception_msg));
        (*task)->SetException(exception_ptr);
        (*task)->AbortComm();
      }

      if (!(*task)->GetTraceUpdated() && (*task)->IsStarted() &&
          !terminated_.load() && !store_error_) {
        std::string trace_key = GetTraceStartKey(
            (*task)->GetBackend(), (*task)->GetRank(), (*task)->GetGid());
        store_error_ =
            !UpdateTraceMsg((*task)->GetStore(),
                            trace_key,
                            (*task)->GetSeq(),
                            CommTypeToString((*task)->GetCommType()));
        (*task)->SetTraceUpdated();
      }

      if ((*task)->IsCompleted()) {
        if (!(*task)->GetTraceUpdated() && !terminated_.load() &&
            !store_error_) {
          std::string trace_key = GetTraceStartKey(
              (*task)->GetBackend(), (*task)->GetRank(), (*task)->GetGid());
          store_error_ =
              !UpdateTraceMsg((*task)->GetStore(),
                              trace_key,
                              (*task)->GetSeq(),
                              CommTypeToString((*task)->GetCommType()));
          (*task)->SetTraceUpdated();
        }
        if (!terminated_.load() && !store_error_) {
          std::string trace_key = GetTraceEndKey(
              (*task)->GetBackend(), (*task)->GetRank(), (*task)->GetGid());
          store_error_ =
              !UpdateTraceMsg((*task)->GetStore(),
                              trace_key,
                              (*task)->GetSeq(),
                              CommTypeToString((*task)->GetCommType()));
        }
        task = comm_task_list_.erase(task);
      } else {
        ++task;
      }
    }
  }
  done = comm_task_list_.empty();
}

}  // namespace distributed
}  // namespace phi
