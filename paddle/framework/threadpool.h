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

#include "paddle/platform/enforce.h"

namespace paddle {
namespace framework {

class ThreadPool {
 public:
  typedef std::packaged_task<void()> Task;

  /**
   * @brief   Get a instance of threadpool, the thread number will
   *          be specified as the number of hardware thread contexts
   */
  static ThreadPool* GetInstance() {
    std::call_once(init_flag, &ThreadPool::Init);
    return threadpool.get();
  }

  ~ThreadPool() {
    {
      // notify all threads to stop running
      running_ = false;
      scheduled_.notify_all();
    }

    for (auto& t : threads_) {
      t->join();
      t.reset(nullptr);
    }
  }

  int GetNumThreads() const { return num_threads_; }

  int GetAvailable() {
    std::unique_lock<std::mutex> lock(mutex_);
    return available_;
  }

  /**
   * @brief   Push a function to the queue, and will be scheduled and
   *          executed if a thread is available.
   * @param[in] Task, will be pushed to the task queue.
   * @return    std::future<void>, we could wait for the task finished by
   *            f.wait().
   */
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

  /**
   * @brief   Wait until all the tasks are completed.
   */
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    completed_.wait(lock, [=] { return Done() == true; });
  }

 private:
  DISABLE_COPY_AND_ASSIGN(ThreadPool);

  explicit ThreadPool(int num_threads)
      : num_threads_(num_threads), available_(num_threads), running_(true) {
    threads_.resize(num_threads);
    for (auto& thread : threads_) {
      // TODO(Yancey1989): binding the thread on the specify CPU number
      thread.reset(new std::thread(std::bind(&ThreadPool::TaskLoop, this)));
    }
  }

  /**
   * @brief   If the task queue is empty and avaialbe
   *          is equal to the number of threads, means that
   *          all tasks are completed.
   *
   *          Note: this function is not thread-safe.
   *
   * @return true if all tasks are completed.
   */
  bool Done() { return tasks_.empty() && available_ == num_threads_; }

  void TaskLoop() {
    while (running_) {
      std::unique_lock<std::mutex> lock(mutex_);
      scheduled_.wait(lock, [=] { return !tasks_.empty() || !running_; });

      if (!running_) {
        break;
      }
      // pop a task from the task queue
      auto task = std::move(tasks_.front());
      tasks_.pop();

      --available_;
      lock.unlock();

      // run the task
      task();

      {
        std::unique_lock<std::mutex> lock(mutex_);
        ++available_;
        if (Done()) {
          completed_.notify_all();
        }
      }
    }
  }

  static void Init() {
    if (threadpool.get() == nullptr) {
      // TODO(Yancey1989): specify the max threads number
      int num_threads = std::thread::hardware_concurrency();
      PADDLE_ENFORCE_GT(num_threads, 0);
      threadpool.reset(new ThreadPool(num_threads));
    }
  }

 private:
  static std::unique_ptr<ThreadPool> threadpool;
  static std::once_flag init_flag;

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
