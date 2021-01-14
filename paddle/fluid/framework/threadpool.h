/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <condition_variable>  // NOLINT
#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <queue>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace framework {

struct ExceptionHandler {
  mutable std::future<std::unique_ptr<platform::EnforceNotMet>> future_;
  explicit ExceptionHandler(
      std::future<std::unique_ptr<platform::EnforceNotMet>>&& f)
      : future_(std::move(f)) {}
  void operator()() const {
    auto ex = this->future_.get();
    if (ex != nullptr) {
      PADDLE_THROW(platform::errors::Fatal(
          "The exception is thrown inside the thread pool. You "
          "should use RunAndGetException to handle the exception."
          "The exception is:\n %s.",
          ex->what()));
    }
  }
};

// ThreadPool maintains a queue of tasks, and runs them using a fixed
// number of threads.
class ThreadPool {
 public:
  explicit ThreadPool(int num_threads);

  using Task = std::packaged_task<std::unique_ptr<platform::EnforceNotMet>()>;

  // Returns the singleton of ThreadPool.
  static ThreadPool* GetInstance();

  ~ThreadPool();

  // Run pushes a function to the task queue and returns a std::future
  // object. To wait for the completion of the task, call
  // std::future::wait().
  template <typename Callback>
  std::future<void> Run(Callback fn) {
    auto f = this->RunAndGetException(fn);
    return std::async(std::launch::deferred, ExceptionHandler(std::move(f)));
  }

  template <typename Callback>
  std::future<std::unique_ptr<platform::EnforceNotMet>> RunAndGetException(
      Callback fn) {
    Task task([fn]() -> std::unique_ptr<platform::EnforceNotMet> {
      try {
        fn();
      } catch (platform::EnforceNotMet& ex) {
        return std::unique_ptr<platform::EnforceNotMet>(
            new platform::EnforceNotMet(ex));
      } catch (const std::exception& e) {
        PADDLE_THROW(platform::errors::Fatal(
            "Unexpected exception is catched in thread pool. All "
            "throwable exception in Paddle should be an EnforceNotMet."
            "The exception is:\n %s.",
            e.what()));
      }
      return nullptr;
    });
    std::future<std::unique_ptr<platform::EnforceNotMet>> f = task.get_future();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!running_) {
        PADDLE_THROW(platform::errors::Unavailable(
            "Task is enqueued into stopped ThreadPool."));
      }
      tasks_.push(std::move(task));
    }
    scheduled_.notify_one();
    return f;
  }

 private:
  DISABLE_COPY_AND_ASSIGN(ThreadPool);

  // The constructor starts threads to run TaskLoop, which retrieves
  // and runs tasks from the queue.
  void TaskLoop();

  // Init is called by GetInstance.
  static void Init();

 private:
  static std::unique_ptr<ThreadPool> threadpool_;
  static std::once_flag init_flag_;

  std::vector<std::unique_ptr<std::thread>> threads_;

  std::queue<Task> tasks_;
  std::mutex mutex_;
  bool running_;
  std::condition_variable scheduled_;
};

class ThreadPoolIO : ThreadPool {
 public:
  static ThreadPool* GetInstanceIO();
  static void InitIO();

 private:
  // NOTE: threadpool in base will be inhereted here.
  static std::unique_ptr<ThreadPool> io_threadpool_;
  static std::once_flag io_init_flag_;
};

// Run a function asynchronously.
// NOTE: The function must return void. If the function need to return a value,
// you can use lambda to capture a value pointer.
template <typename Callback>
std::future<void> Async(Callback callback) {
  return ThreadPool::GetInstance()->Run(callback);
}

template <typename Callback>
std::future<void> AsyncIO(Callback callback) {
  return ThreadPoolIO::GetInstanceIO()->Run(callback);
}

}  // namespace framework
}  // namespace paddle
