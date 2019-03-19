// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstddef>
#include <cstdint>

#include <mutex>  // NOLINT
#include <queue>

namespace paddle {
namespace imperative {

struct Runnable {
  PreparedOp op_;
  std::function<void()> callback_;
};

class Engine {
 public:
  virtual ~Engine() {}

  virtual void Run(Runnable* runnable) = 0;
};

struct ReadyQueue {
  // TODO(minqiyang): change to priority queue with work-stealing algo
  std::queue<Runnable*> queue_;
  std::condition_variable not_empty_;
  std::mutex mutex_;

  void push(Runnable* runnable);
  Runnable* pop();
};

class AsyncEngine : public Engine {
 public:
  void Run(Runnable* runnable) override;

 private:
  void Enqueue(Runnable* runnable);

  void thread_start();
  void execute();

 private:
  ReadyQueue ready_queue_;
};

Engine* GetEngine();

}  // namespace imperative
}  // namespace paddle
