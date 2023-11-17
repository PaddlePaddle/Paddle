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
std::list<std::shared_ptr<CommTask>> CommTaskManager::comm_task_list_;
std::unordered_map<std::string, std::shared_ptr<CommTask>>
    CommTaskManager::init_comm_task_map_;
std::unordered_map<std::string, std::shared_ptr<CommTask>>
    CommTaskManager::start_comm_task_map_;
std::unordered_map<std::string, std::shared_ptr<CommTask>>
    CommTaskManager::group_last_comm_task_;
std::chrono::time_point<std::chrono::steady_clock>
    CommTaskManager::last_update_time_ = std::chrono::steady_clock::now();

CommTaskManager::CommTaskManager() {
  terminated_.store(false);
  comm_task_loop_thread_ = std::thread(&CommTaskManager::CommTaskLoop, this);
  LOG(INFO) << "CommTaskManager init success.";
}
CommTaskManager::~CommTaskManager() {
  terminated_.store(true);

  if (comm_task_loop_thread_.joinable()) {
    comm_task_list_cv_.notify_one();
    comm_task_loop_thread_.join();
  }
  LOG(INFO) << "CommTaskManager destruct success.";
}

void CommTaskManager::CommTaskEnqueue(std::shared_ptr<CommTask> comm_task) {
  if (!terminated_.load()) {
    std::lock_guard<std::mutex> lock(comm_task_list_mutex_);
    comm_task_list_.emplace_back(std::move(comm_task));
  }
}

void CommTaskManager::Stop() {
  terminated_.store(true);

  LOG(INFO) << "CommTaskManager stopped begin.";
  if (comm_task_loop_thread_.joinable()) {
    comm_task_list_cv_.notify_one();
    comm_task_loop_thread_.join();
  }
  LOG(INFO) << "CommTaskManager stopped.";
}

void CommTaskManager::CommTaskLoop() {
  bool done = false;
  while (!terminated_.load() || !done) {
    std::unique_lock<std::mutex> lock(comm_task_list_mutex_);
    comm_task_list_cv_.wait_for(
        lock,
        std::chrono::milliseconds(loop_thread_sleep_millis),
        [&]() -> bool { return terminated_.load(); });

    if (IsTimeout() && !logged_) {
      // case 1: all group is empty, has no task
      // report error immediately
      if (group_last_comm_task_.empty()) {
        LOG(ERROR) << "Find no task started in all group";
      } else {
        // case 2: all group is not empty, but all last task is completed
        // when some group is empty ?
        bool all_completed = true;
        for (auto iter : group_last_comm_task_) {
          if (!iter.second->IsCompleted()) {
            all_completed = false;
            break;
          }
        }
        if (all_completed) {
          // report error immediately
          LOG(ERROR)
              << "Find no task started with prev task completed in all group";
        } else {
          // case 3: all group is not empty, some group task started but not
          // finished analyze by automated script
          for (auto iter : group_last_comm_task_) {
            LogLongStr("Find last group comm task:",
                       iter.second->GetTraceMsg());
          }
        }
      }
      logged_ = true;
    }
    for (auto iter = comm_task_list_.begin(); iter != comm_task_list_.end();) {
      auto task = *iter;
      if (task->IsTimeout()) {
        if (!task->IsStarted()) {
          LOG(WARNING) << "Find timeout init but not start task:"
                       << task->GetTraceMsg();
          std::string task_key = task->UniqueKey();
          init_comm_task_map_[task_key] = task;
        } else if (!task->IsCompleted()) {
          LOG(WARNING) << "Find timeout start but not finish task:"
                       << task->GetTraceMsg();
          std::string task_key = task->UniqueKey();
          start_comm_task_map_[task_key] = task;
        }
        iter = comm_task_list_.erase(iter);
      } else {
        if (task->IsStarted()) {
          if (task->IsCompleted()) {
            task->ClearRecord();
            iter = comm_task_list_.erase(iter);
          }
          UpdateLastCommTask(task);
        } else {
          ++iter;
        }
      }
    }

    for (auto iter = init_comm_task_map_.begin();
         iter != init_comm_task_map_.end();) {
      auto task = iter->second;
      if (task->IsStarted()) {
        std::string task_key = task->UniqueKey();
        start_comm_task_map_[task_key] = task;
        iter = init_comm_task_map_.erase(iter);
        LOG(INFO) << "Start timeout task: " << task->GetTraceMsg();
      } else {
        ++iter;
      }
    }

    for (auto iter = start_comm_task_map_.begin();
         iter != start_comm_task_map_.end();) {
      auto task = iter->second;
      if (task->IsCompleted()) {
        task->ClearRecord();
        UpdateLastCommTask(task);
        iter = start_comm_task_map_.erase(iter);
        LOG(INFO) << "Finish timeout task: " << task->GetTraceMsg();
      } else {
        ++iter;
      }
    }

    if (comm_task_list_.empty() && init_comm_task_map_.empty() &&
        start_comm_task_map_.empty()) {
      done = true;
    } else {
      done = false;
    }
  }
}

void CommTaskManager::UpdateLastCommTask(std::shared_ptr<CommTask> task) {
  if (!task->IsUpdated()) {
    return;
  }
  group_last_comm_task_[task->GroupKey()] = task;
  last_update_time_ = std::chrono::steady_clock::now();
  task->SetUpdated(false);
}

void CommTaskManager::SetTimeout(int64_t timeout) {
  timeout_ = std::chrono::milliseconds(timeout);
}

bool CommTaskManager::IsTimeout() {
  auto current_timepoint = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             current_timepoint - last_update_time_) >= timeout_;
}
}  // namespace distributed
}  // namespace phi
