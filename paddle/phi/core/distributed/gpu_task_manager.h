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

#pragma once

#include <atomic>
#include <condition_variable>
#include <list>
#include <mutex>
#include <thread>

#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/distributed/gpu_task.h"

namespace phi {
namespace distributed {

class GPUTaskManager {
 public:
  GPUTaskManager();
  ~GPUTaskManager();

 public:
  static GPUTaskManager& GetInstance() {
    static GPUTaskManager instance;
    return instance;
  }
  void SetStartTime();
  void GPUTaskEnqueue(std::shared_ptr<GPUTask> gpu_task);
  void GPUTaskClearEnqueue(std::shared_ptr<GPUTask> gpu_task);
  void Stop();

 private:
  void GPUTaskLoop();
  void GPUTaskClearLoop();

  static std::thread gpu_task_loop_thread_;
  static std::thread gpu_task_clear_loop_thread_;
  static const int64_t loop_thread_sleep_millis;

  static std::atomic<bool> terminated_;

  static std::mutex gpu_task_list_mutex_;
  static std::condition_variable gpu_task_list_cv_;
  static std::list<std::shared_ptr<GPUTask>> gpu_task_list_;

  static std::mutex gpu_task_clear_list_mutex_;
  static std::condition_variable gpu_task_clear_list_cv_;
  static std::list<std::shared_ptr<GPUTask>> gpu_task_clear_list_;

  static gpuEvent_t start_event_;
};

}  // namespace distributed
}  // namespace phi
