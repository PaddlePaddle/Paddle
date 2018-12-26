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

#include <glog/logging.h>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <queue>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

//#include "common.h"

namespace swiftcpp {
namespace thread {

// Concurrent blocking queue.
template <typename T>
class TaskQueue {
 public:
  TaskQueue() { killed_ = false; }
  ~TaskQueue() = default;

  void Push(T &&v);

  void Push(const T &v);

  bool Pop(T *rv);

  // Tell all blocking pop will return false.
  void SignalForKill();

  size_t Size() const;

 private:
  std::mutex mut_;
  std::condition_variable cv_;
  std::atomic<bool> killed_;
  // std::atomic<int> num_pending_tasks_;
  std::queue<T> queue_;

  // DISALLOW_COPY_AND_ASSIGN(TaskQueue);
};

class ThreadPool {
 public:
  typedef std::function<void()> task_t;
  ThreadPool(int n, const task_t &task) : task_(task) {
    for (int i = 0; i < n; i++) {
      workers_.emplace_back(task);
      LOG(WARNING) << "new thread " << workers_.back().get_id();
    }
  }

  size_t Size() const { return workers_.size(); }

  ~ThreadPool() {
    for (auto &t : workers_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

 private:
  std::vector<std::thread> workers_;
  task_t task_;

  ThreadPool() = delete;
  // DISALLOW_COPY_AND_ASSIGN(ThreadPool);
};

template <typename T>
void TaskQueue<T>::Push(T &&v) {
  std::lock_guard<std::mutex> l(mut_);
  queue_.emplace(std::forward<T>(v));
  cv_.notify_all();
  LOG(INFO) << "push one to " << this;
}

template <typename T>
void TaskQueue<T>::Push(const T &v) {
  std::lock_guard<std::mutex> l(mut_);
  queue_.emplace(v);
  cv_.notify_all();
}

template <typename T>
bool TaskQueue<T>::Pop(T *cv) {
  // LOG(INFO) << "wait to pop one from " << this;
  std::unique_lock<std::mutex> l(mut_);
  cv_.wait(l, [this] { return !queue_.empty() || killed_.load(); });
  if (killed_.load()) {
    LOG(WARNING) << "kill signal get, all threads will exit";
    return false;
  }
  *cv = std::move(queue_.front());
  queue_.pop();
  // LOG(INFO) << "pop one";
  return true;
}

template <typename T>
size_t TaskQueue<T>::Size() const {
  return queue_.size();
}

template <typename T>
void TaskQueue<T>::SignalForKill() {
  killed_ = true;
  LOG(INFO) << "queue singal for kill...";
  cv_.notify_all();
}

}  // namespace thread
}  // namespace swiftcpp
