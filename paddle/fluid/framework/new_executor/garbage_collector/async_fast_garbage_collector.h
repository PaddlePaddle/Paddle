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

#include "paddle/fluid/framework/new_executor/garbage_collector/garbage_collector.h"

namespace paddle {
namespace framework {

class SingleThreadLockFreeWorker {
 public:
  using Task = std::function<void()>;

  explicit SingleThreadLockFreeWorker(int capacity);

  ~SingleThreadLockFreeWorker() { Wait(); }

  void AddTask(Task task);

  void Wait();

 private:
  void WorkerLoop();

  const int capacity_;
  std::thread worker_;
  std::vector<Task> tasks_queue_;
  std::atomic<int> head_;
  std::atomic<int> tail_;
  std::atomic<bool> running_;
};

class InterpreterCoreAsyncFastGarbageCollector {
 public:
  explicit InterpreterCoreAsyncFastGarbageCollector(int num_instructions);

  void Add(const std::vector<Variable*>& vars);

 private:
  std::unique_ptr<SingleThreadLockFreeWorker> async_worker_;
};
}  // namespace framework
}  // namespace paddle
