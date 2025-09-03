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

#include <fstream>
#include <future>
#include "glog/logging.h"

#include "paddle/phi/core/distributed/gpu_task_manager.h"
#include "paddle/phi/core/distributed/nccl_tools.h"
#if defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/dynload/rccl.h"
#else
#include "paddle/phi/backends/dynload/nccl.h"
#endif

namespace phi::distributed {

std::thread GPUTaskManager::gpu_task_loop_thread_;
std::thread GPUTaskManager::gpu_task_clear_loop_thread_;
const int64_t GPUTaskManager::loop_thread_sleep_millis = 10000;

std::atomic<bool> GPUTaskManager::terminated_;

std::mutex GPUTaskManager::gpu_task_list_mutex_;
std::condition_variable GPUTaskManager::gpu_task_list_cv_;
std::list<std::shared_ptr<GPUTask>> GPUTaskManager::gpu_task_list_;

std::mutex GPUTaskManager::gpu_task_clear_list_mutex_;
std::condition_variable GPUTaskManager::gpu_task_clear_list_cv_;
std::list<std::shared_ptr<GPUTask>> GPUTaskManager::gpu_task_clear_list_;

gpuEvent_t GPUTaskManager::start_event_;

GPUTaskManager::GPUTaskManager() {
  terminated_.store(false);
  gpu_task_loop_thread_ = std::thread(&GPUTaskManager::GPUTaskLoop, this);
  gpu_task_clear_loop_thread_ =
      std::thread(&GPUTaskManager::GPUTaskClearLoop, this);
#ifdef PADDLE_WITH_CUDA
  CUDA_CHECK(cudaEventCreateWithFlags(&start_event_, 0));
#else  // PADDLE_WITH_HIP
  HIP_CHECK(hipEventCreateWithFlags(&start_event_, 0));
#endif
}

GPUTaskManager::~GPUTaskManager() {
  terminated_.store(true);

  if (gpu_task_loop_thread_.joinable()) {
    gpu_task_list_cv_.notify_one();
    gpu_task_loop_thread_.join();
  }

  if (gpu_task_clear_loop_thread_.joinable()) {
    gpu_task_clear_list_cv_.notify_one();
    gpu_task_clear_loop_thread_.join();
  }

  LOG(INFO) << "GPUTaskManager destruct success.";
}

void GPUTaskManager::Stop() {
  terminated_.store(true);

  if (gpu_task_loop_thread_.joinable()) {
    gpu_task_list_cv_.notify_one();
    gpu_task_loop_thread_.join();
  }

  if (gpu_task_clear_loop_thread_.joinable()) {
    gpu_task_clear_list_cv_.notify_one();
    gpu_task_clear_loop_thread_.join();
  }

  LOG(INFO) << "GPUTaskManager stopped.";
}

void GPUTaskManager::SetStartTime() {
#ifdef PADDLE_WITH_CUDA
  CUDA_CHECK(cudaEventRecord(start_event_));
#else  // PADDLE_WITH_HIP
  HIP_CHECK(hipEventRecord(start_event_));
#endif
  LOG(INFO) << "GPUTaskManager init";
}

void GPUTaskManager::GPUTaskEnqueue(std::shared_ptr<GPUTask> gpu_task) {
  if (!terminated_.load()) {
    std::lock_guard<std::mutex> lock(gpu_task_list_mutex_);
    gpu_task_list_.emplace_back(std::move(gpu_task));
  }
}

void GPUTaskManager::GPUTaskClearEnqueue(std::shared_ptr<GPUTask> gpu_task) {
  if (!terminated_.load()) {
    std::lock_guard<std::mutex> lock(gpu_task_clear_list_mutex_);
    gpu_task_clear_list_.emplace_back(gpu_task);
  }
}

void GPUTaskManager::GPUTaskLoop() {
  bool done = false;
  while (!terminated_.load() || !done) {
    std::unique_lock<std::mutex> lock(gpu_task_list_mutex_);

    gpu_task_list_cv_.wait_for(
        lock,
        std::chrono::milliseconds(loop_thread_sleep_millis),
        [&]() -> bool { return terminated_.load(); });

    for (auto iter = gpu_task_list_.begin(); iter != gpu_task_list_.end();) {
      auto task = *iter;
      if (task->IsCompleted()) {
        GPUTaskClearEnqueue(task);
        iter = gpu_task_list_.erase(iter);
      } else {
        ++iter;
      }
    }

    if (gpu_task_list_.empty()) {
      done = true;
    } else {
      done = false;
    }
  }
}

void GPUTaskManager::GPUTaskClearLoop() {
  std::future<void> future;
  while (!terminated_.load()) {
    if (future.valid()) {
      future.wait();
    }
    {
      std::unique_lock<std::mutex> lock(gpu_task_clear_list_mutex_);
      gpu_task_clear_list_cv_.wait_for(
          lock,
          std::chrono::milliseconds(loop_thread_sleep_millis),
          [&]() -> bool { return terminated_.load(); });

      for (auto iter = gpu_task_clear_list_.begin();
           iter != gpu_task_clear_list_.end();) {
        auto task = *iter;
        if (!task->HasPrinted()) {
          // task->GetTraceMsg(start_event_);
          LOG(INFO) << task->GetTraceMsg(start_event_);
          task->SetPrint();
        }
        future = std::async(std::launch::async, [&]() { task->ClearRecord(); });
        if (future.wait_for(std::chrono::milliseconds(10)) ==
            std::future_status::timeout) {
          break;
        }
        iter = gpu_task_clear_list_.erase(iter);
      }
    }
  }
}
}  // namespace phi::distributed
