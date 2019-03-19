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

#include "paddle/fluid/imperative/engine.h"

#include <mutex>  // NOLINT
#include <queue>

#include "glog/logging.h"

namespace paddle {
namespace imperative {

static std::once_flag init_engine;
static Engine* engine;

struct ReadyQueue {
  // TODO(minqiyang): change to priority queue with work-stealing algo
  std::queue<Runnable*> queue_;
  std::condition_variable not_empty_;
  std::mutex mutex_;

  void push(Runnable* runnable);
  Runnable* pop();
};

void ReadyQueue::push(Runnable* runnable) {
  {
    std::lock_guard<std::mutex> lock(mutex);
    queue_.push(runnable);
  }
  not_empty_.notify_one();
}

Runnable* ReadyQueue::pop() {
  std::unique_lock<std::mutex> lock(mutex_);
  not_empty_.wait(lock, [this] { return !queue_.empty(); });
  auto runnable = queue_.top();
  queue_.pop();
  return runnable;
}

class AsyncEngine : public Engine {
 public:
  void Enqueue(Runnable* runnable) override {
    std::call_once(init_engine, thread_start());

    queued_runnables_.push_back(runnable);
  }

  size_t Size() const override { return queued_runnables_.size(); }

  void thread_start() {
    // TODO(minqiyang): Only one thread now, change to multi-thread
    std::thread t(this, execute);
    t.detach();
  }

  void execute() {
    while (true) {
      Runnable* r = queued_runnables_.top();

      PreparedOp* op = r->op;

      framework::Scope scope;
      op->func(framework::ExecutionContext(op->op, scope, *op->dev_ctx,
                                           prepared_op->ctx,
                                           prepared_op->kernel_configs))

          queued_runnables_.pop();
    }
  }

 private:
  std::queue<Runnable*> queued_runnables_;
};

Engine* GetEngine() {
  std::call_once(init_engine, []() { engine = new AsyncEngine(); });
  return engine;
}

}  // namespace imperative
}  // namespace paddle
