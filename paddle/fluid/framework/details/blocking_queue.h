//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <mutex>
#include <thread>

namespace paddle {
namespace framework {
namespace details {

template <typename T>
class FixedBlockingQueue {
 public:
  FixedBlockingQueue(size_t size) : q_(new std::deque<T>()), size_(size) {}

  void Push(T&& item, bool* ok = nullptr) {
    {
      std::unique_lock<std::mutex> lock(mtx_);

      while (q_ && q_->size() >= size_) {
        full_cv_.wait(lock);
      }

      if (q_) {
        q_->emplace_back(std::move(item));
      }
      if (ok) {
        *ok = (bool)(q_);
      }
    }
    empty_cv_.notify_one();
  }

  T Pop(bool* ok = nullptr) {
    T v;
    {
      std::unique_lock<std::mutex> lock(mtx_);

      while (q_ && q_->empty()) {
        empty_cv_.wait(lock);
      }

      if (ok) {
        *ok = (bool)(q_);
      }

      if (q_) {
        v = std::move(q_->front());
        q_->pop_front();
      }
    }

    full_cv_.notify_one();
    return v;
  }

  void Close() {
    {
      std::lock_guard<std::mutex> guard(mtx_);
      q_.reset();
    }
    empty_cv_.notify_all();
    full_cv_.notify_all();
  }

  void Reset() {
    Close();
    {
      std::lock_guard<std::mutex> guard(mtx_);
      q_.reset(new std::deque<T>());
    }
  }

 private:
  std::unique_ptr<std::deque<T>> q_;
  std::mutex mtx_;
  std::condition_variable empty_cv_;
  std::condition_variable full_cv_;
  size_t size_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
