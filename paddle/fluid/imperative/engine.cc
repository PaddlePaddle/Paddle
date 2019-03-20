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

#include "paddle/fluid/framework/python_headers.h"

namespace paddle {
namespace imperative {

static std::once_flag init_engine;
static Engine* engine;

static std::once_flag start_engine;

void ReadyQueue::Push(Runnable* runnable) {
  {
    std::lock_guard<std::mutex> lock(not_empty_mutex_);
    queue_.push(runnable);
  }
  not_empty_.notify_one();
}

bool ReadyQueue::Empty() const { return queue_.empty(); }

Runnable* ReadyQueue::Pop() {
  std::unique_lock<std::mutex> lock(not_empty_mutex_);
  not_empty_.wait(lock, [this] { return !queue_.empty(); });
  auto runnable = queue_.front();
  queue_.pop();
  return runnable;
}

void AsyncEngine::Run(Runnable* runnable) {
  std::call_once(start_engine, &AsyncEngine::ThreadStart, this);

  Enqueue(runnable);
}

void AsyncEngine::Sync() {
  VLOG(5) << "Sync Engine";

  if (!ready_queue_->Empty()) {
    // NOTE(minqiyang): backward refs invokers should acquire GIL lock
    // so we should release the lock in main thread to avoid dead lock
    pybind11::gil_scoped_release release;
    std::unique_lock<std::mutex> lock(mutex_);
    empty_.wait(lock, [this] { return ready_queue_->Empty(); });
  }
}

void AsyncEngine::Enqueue(Runnable* runnable) { ready_queue_->Push(runnable); }

void AsyncEngine::ThreadStart() {
  // TODO(minqiyang): Only one thread one queue now, should change to
  // multi-thread and multi-queue.
  std::thread t(&AsyncEngine::Execute, this);
  t.detach();
}

void AsyncEngine::Execute() {
  while (true) {
    Runnable* r = ready_queue_->Pop();

    r->operator()();

    for (auto callback : r->callbacks_) {
      callback();
    }

    delete r;

    if (ready_queue_->Empty()) {
      empty_.notify_one();
    }

    VLOG(10) << "Run op end";
  }
}

Engine* GetEngine() {
  std::call_once(init_engine, []() { engine = new AsyncEngine(); });
  return engine;
}

}  // namespace imperative
}  // namespace paddle
