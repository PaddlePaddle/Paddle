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
#include <condition_variable>
#include <forward_list>
#include <future>
#include <list>
#include <mutex>
#include <queue>
#include "paddle/fluid/distributed/common/thread_queue.h"

namespace paddle {
namespace distributed {

template <class T>
class ThreadPool {
 public:
  ThreadPool<T>(uint32_t thread_num) { Init(thread_num); }
  ~ThreadPool<T>() {
    _tasks.disable();

    for (std::thread &t : _threads) {
      t.join();
    }
  }

  void Init(size_t thread_num) {
    for (; thread_num; --thread_num) {
      _threads.emplace_front([this]() {
        for (;;) {
          std::packaged_task<T()> task = _tasks.pop();
          if (task.valid()) {
            task();
          } else {
            break;
          }
        }
      });
    }
  }

  void destroy() { delete this; }
  template <class Callable, class... Args>
  std::future<T> AddTask(Callable &&func, Args &&... args) {
    std::packaged_task<T()> task(
        std::bind(std::forward<Callable>(func), std::forward<Args>(args)...));
    std::future<T> result = task.get_future();
    _tasks.push(std::move(task));
    return result;
  }

 private:
  typedef thread_queue<std::packaged_task<T()>, store_value> queue_t;
  queue_t _tasks;

  std::forward_list<std::thread> _threads;
};
typedef ThreadPool<void> WorkerPool;

}  // namespace distributed
}  // namespace paddle
