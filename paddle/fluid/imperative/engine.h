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

#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <queue>
#include <thread>  // NOLINT
#include <vector>

namespace paddle {
namespace imperative {

struct Runnable {
  virtual ~Runnable() {}

  virtual void operator()() = 0;

  std::vector<std::function<void()>> callbacks_;
};

class Engine {
 public:
  virtual ~Engine() {}

  virtual void Run(Runnable* runnable) = 0;

  virtual void Sync() = 0;
};

struct ReadyQueue {
  // TODO(minqiyang): change to priority queue with work-stealing algo
  std::queue<Runnable*> queue_;
  std::condition_variable not_empty_;
  std::mutex not_empty_mutex_;

  void Push(Runnable* runnable);
  Runnable* Pop();
  bool Empty() const;
};

class ImperativeEngine : public Engine {
 public:
  ImperativeEngine() : ready_queue_(new ReadyQueue()), async_(false) {}

  void Run(Runnable* runnable) override;

  void Sync() override;

 private:
  void Enqueue(Runnable* runnable);

  void ThreadStart();
  void Execute();

  void ExecuteInternal(Runnable* runnable);

 private:
  std::unique_ptr<ReadyQueue> ready_queue_;
  bool async_;
  std::condition_variable empty_;
  std::mutex mutex_;
};

Engine* GetEngine();

}  // namespace imperative
}  // namespace paddle
