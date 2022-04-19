// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <future>
#include <map>
#include <thread>
#include <vector>

#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace distributed {

class TaskLoop {
 public:
  static TaskLoop* GetTaskLoopOfCurrentThread();

  using Functor = std::function<void()>;

  TaskLoop();
  ~TaskLoop();

  void Loop();
  void Quit();

  void RunInLoop(Functor cb);
  void QueueInLoop(Functor cb);

  template <class F, class... Args>
  auto Enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<return_type> task_future = task->get_future();

    tasks_.Push([task]() { (*task)(); });
    return task_future;
  }

  void WakeUp();

  bool IsInLoopThread() const {
    return thread_id_ == std::this_thread::get_id();
  }

  void AssertInLoopThread() {
    if (!IsInLoopThread()) {
      AbortNotInLoopThread();
    }
  }

 private:
  DISABLE_COPY_AND_ASSIGN(TaskLoop);

  void AbortNotInLoopThread();

  static thread_local TaskLoop* thread_local_loop_;

  bool looping_;
  std::atomic<bool> quit_;
  std::thread::id thread_id_;

  framework::BlockingQueue<Functor> tasks_;
};

}  // namespace distributed
}  // namespace paddle
