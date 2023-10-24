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

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "paddle/phi/core/distributed/comm_context.h"
#include "paddle/phi/core/distributed/comm_task.h"
#include "paddle/phi/core/macros.h"

namespace phi {
namespace distributed {

enum ErrorHandlingMode { NoHandling = 0, TearDown = 1 };

class Store;

class CommTaskManager {
 public:
  CommTaskManager();
  ~CommTaskManager();

 public:
  static CommTaskManager& GetInstance() {
    static CommTaskManager instance;
    return instance;
  }

  void CommTaskEnqueue(std::shared_ptr<CommTask> comm_task);
  void Stop();

 private:
  void CommTaskLoop();

  static std::thread comm_task_loop_thread_;
  static const int64_t loop_thread_sleep_millis;

  static std::atomic<bool> terminated_;

  static std::mutex comm_task_list_mutex_;
  static std::condition_variable comm_task_list_cv_;
  static std::list<std::shared_ptr<CommTask>> comm_task_list_;
  // not start task
  static std::unordered_map<std::string, std::shared_ptr<CommTask>>
      init_comm_task_map_;
  // start but not finish task
  static std::unordered_map<std::string, std::shared_ptr<CommTask>>
      start_comm_task_map_;
  std::shared_ptr<Store> store_;
  bool store_error_{false};
};

}  // namespace distributed
}  // namespace phi
