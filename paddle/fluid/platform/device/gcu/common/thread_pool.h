/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <atomic>
#include <condition_variable>  // NOLINT [build/c++11]
#include <functional>
#include <future>  // NOLINT [build/c++11]
#include <memory>
#include <mutex>  // NOLINT [build/c++11]
#include <queue>
#include <stdexcept>
#include <thread>  // NOLINT [build/c++11]
#include <utility>
#include <vector>

#include "paddle/fluid/platform/device/gcu/utils/types.h"

namespace paddle {
namespace platform {
namespace gcu {
using ThreadTask = std::function<void()>;

class GcuThreadPool {
 public:
  explicit GcuThreadPool(uint32_t thread_num = 128);
  ~GcuThreadPool();

  template <class Func, class... Args>
  auto commit(Func &&func, Args &&...args)
      -> std::future<decltype(func(args...))> {
    VLOG(10) << "commit run task enter.";
    using retType = decltype(func(args...));
    std::future<retType> fail_future;
    if (is_stoped_.load()) {
      PADDLE_THROW(platform::errors::Fatal("thread pool has been stopped."));
      return fail_future;
    }

    auto bindFunc =
        std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
    auto task = std::make_shared<std::packaged_task<retType()>>(bindFunc);
    if (task == nullptr) {
      PADDLE_THROW(platform::errors::Fatal("Make shared failed."));
      return fail_future;
    }
    std::future<retType> future = task->get_future();
    {
      std::lock_guard<std::mutex> lock{m_lock_};
      tasks_.emplace([task]() { (*task)(); });
    }
    cond_var_.notify_one();
    VLOG(10) << "commit run task end";
    return future;
  }

  static void ThreadFunc(GcuThreadPool *thread_pool);

 private:
  std::vector<std::thread> pool_;
  std::queue<ThreadTask> tasks_;
  std::mutex m_lock_;
  std::condition_variable cond_var_;
  std::atomic<bool> is_stoped_;
  std::atomic<uint32_t> idle_thrd_num_;
};
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
