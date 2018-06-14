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
#include <mutex>   // NOLINT
#include <queue>
#include <thread>  // NOLINT
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
      LOG(FATAL) << "The exception is thrown inside the thread pool. You "
                    "should use RunAndGetException to handle the exception.\n"
                    "The default exception handler is LOG(FATAL)."
                 << ex->what();
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

  // Returns the number of threads created by the constructor.
  size_t Threads() const { return total_threads_; }

  // Returns the number of currently idle threads.
  size_t IdleThreads() {
    std::unique_lock<std::mutex> lock(mutex_);
    return idle_threads_;
  }

  // Run pushes a function to the task queue and returns a std::future
  // object.  To wait for the completion of the task, call
  // std::future::wait().
  template <typename Callback>
  std::future<void> Run(Callback fn) {
    auto f = this->RunAndGetException(fn);
    return std::async(std::launch::deferred, ExceptionHandler(std::move(f)));
  }

  template <typename Callback>
  std::future<std::unique_ptr<platform::EnforceNotMet>> RunAndGetException(
      Callback fn) {
    std::unique_lock<std::mutex> lock(mutex_);
    Task task([fn]() -> std::unique_ptr<platform::EnforceNotMet> {
      try {
        fn();
      } catch (platform::EnforceNotMet ex) {
        return std::unique_ptr<platform::EnforceNotMet>(
            new platform::EnforceNotMet(ex));
      } catch (const std::exception& e) {
        LOG(FATAL) << "Unexpected exception is catched in thread pool. All "
                      "throwable exception in Fluid should be an EnforceNotMet."
                   << e.what();
      }
      return nullptr;
    });
    std::future<std::unique_ptr<platform::EnforceNotMet>> f = task.get_future();
    tasks_.push(std::move(task));
    lock.unlock();
    scheduled_.notify_one();
    return f;
  }

  // Wait until all the tasks are completed.
  void Wait();

 private:
  DISABLE_COPY_AND_ASSIGN(ThreadPool);

  // If the task queue is empty and avaialbe is equal to the number of
  // threads, means that all tasks are completed.  Note: this function
  // is not thread-safe.  Returns true if all tasks are completed.
  // Note: don't delete the data member total_threads_ and use
  // threads_.size() instead; because you'd need to lock the mutex
  // before accessing threads_.
  bool Done() { return tasks_.empty() && idle_threads_ == total_threads_; }

  // The constructor starts threads to run TaskLoop, which retrieves
  // and runs tasks from the queue.
  void TaskLoop();

  // Init is called by GetInstance.
  static void Init();

 private:
  static std::unique_ptr<ThreadPool> threadpool_;
  static std::once_flag init_flag_;

  std::vector<std::unique_ptr<std::thread>> threads_;
  const size_t total_threads_;
  size_t idle_threads_;

  std::queue<Task> tasks_;
  std::mutex mutex_;
  bool running_;
  std::condition_variable scheduled_;
  std::condition_variable completed_;
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
