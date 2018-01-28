/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "paddle/platform/enforce.h"

namespace paddle {
namespace framework {

// ThreadPool maintains a queue of tasks, and runs them using a fixed
// number of threads.
class ThreadPool {
 public:
  typedef std::packaged_task<void()> Task;

  // Returns the singleton of ThreadPool.
  static ThreadPool* GetInstance();

  ~ThreadPool();

  int GetNumThreads() const { return num_threads_; }

  int GetAvailable() {
    std::unique_lock<std::mutex> lock(mutex_);
    return available_;
  }

  // Push a function to the queue, and will be scheduled and executed
  // if a thread is available.Returns std::future<void>, we could wait
  // for the task finished by f.wait().
  template <typename Callback>
  std::future<void> Run(Callback fn) {
    std::unique_lock<std::mutex> lock(mutex_);
    Task task(std::bind(fn));
    std::future<void> f = task.get_future();
    tasks_.push(std::move(task));
    lock.unlock();
    scheduled_.notify_one();
    return f;
  }

  // Wait until all the tasks are completed.
  void Wait();

 private:
  DISABLE_COPY_AND_ASSIGN(ThreadPool);

  explicit ThreadPool(int num_threads);

  // If the task queue is empty and avaialbe is equal to the number of
  // threads, means that all tasks are completed.  Note: this function
  // is not thread-safe.  Returns true if all tasks are completed.
  bool Done() { return tasks_.empty() && available_ == num_threads_; }

  // The constructor starts threads to run TaskLoop, which retrieves
  // and runs tasks from the queue.
  void TaskLoop();

  // Init is called by GetInstance.
  static void Init();

 private:
  static std::unique_ptr<ThreadPool> threadpool_;
  static std::once_flag init_flag_;

  int num_threads_;
  int available_;
  bool running_;
  std::queue<Task> tasks_;
  std::vector<std::unique_ptr<std::thread>> threads_;
  std::mutex mutex_;
  std::condition_variable scheduled_;
  std::condition_variable completed_;
};

// Run a function asynchronously.
// NOTE: The function must return void. If the function need to return a value,
// you can use lambda to capture a value pointer.
template <typename Callback>
std::future<void> Async(Callback callback) {
  return ThreadPool::GetInstance()->Run(callback);
}

}  // namespace framework
}  // namespace paddle
