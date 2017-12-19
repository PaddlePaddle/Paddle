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
#include <cstdio>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

namespace paddle {
namespace framework {

class ThreadPool {
 private:
  typedef std::function<void()> Task;
  int num_threads_;
  int available_;
  bool running_;
  bool done_;
  std::queue<Task> tasks_;
  std::vector<std::unique_ptr<std::thread>> threads_;
  std::mutex mutex_;
  std::condition_variable scheduled_;
  std::condition_variable completed_;

 public:
  /**
   * @brief   get a instance of threadpool, the thread number will
   *          be specified by the first time.
   * @param[in] num_threads   The available thread number.
   *            If num_threads <= 0, the thread pool wil be initilized
   *            with the number of concurrent threads supported.
   */
  static ThreadPool* Instance(size_t num_threads) {
    static std::unique_ptr<ThreadPool> threadpool;
    if (threadpool.get() == nullptr) {
      if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
      }
      threadpool.reset(new ThreadPool(num_threads));
    }
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

  // get the total thread number
  int GetNumThreads() const { return num_threads_; }

  // get the available thread number
  int GetAvailable() const { return available_; }

  // push a function to the queue, and will be scheduled and
  // executed if a thread is available
  void Run(const Task& fn) {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.push(fn);
    done_ = false;
    lock.unlock();
    scheduled_.notify_one();
  }

  // wait util all the tasks are completed
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    completed_.wait(lock, [=] { return done_ == true; });
  }

 private:
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(const ThreadPool&) = delete;

  ThreadPool(int num_threads)
      : num_threads_(num_threads),
        available_(num_threads),
        running_(true),
        done_(false) {
    threads_.resize(num_threads);
    for (auto& thread : threads_) {
      thread.reset(new std::thread(std::bind(&ThreadPool::TaskLoop, this)));
    }
  }

  void TaskLoop() {
    while (running_) {
      std::unique_lock<std::mutex> lock(mutex_);
      scheduled_.wait(lock, [=] { return !tasks_.empty() || !running_; });

      if (!running_) {
        break;
      }
      // pop a task from the task queue
      auto task = tasks_.front();
      tasks_.pop();

      --available_;
      lock.unlock();

      // run the task
      task();
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ++available_;
        if (tasks_.empty() && available_ == num_threads_) {
          done_ = true;
          lock.unlock();
          completed_.notify_all();
        }
      }
    }
  }
};
}  // namespace framework
}  // namespace paddle
