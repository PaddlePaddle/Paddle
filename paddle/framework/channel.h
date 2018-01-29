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
#include <mutex>
#include <queue>

namespace paddle {
namespace framework {

template <typename T>
class Channel {
 public:
  explicit Channel(std::size_t capacity) : capacity_(capacity) {}

  void Send(T* channel_element) {
    std::unique_lock<std::mutex> lock(mu_);

    if (IsBounded()) {
      full_cond_var_.wait(lock, [this]() {
        bool capacity_valid = capacity_ > 0 ? !IsCapacityFull() : true;
        return capacity_valid;
      });
    }
    channel_.push_back(std::move(*channel_element));

    lock.unlock();
    empty_cond_var_.notify_one();
  }

  T* Receive() {
    std::unique_lock<std::mutex> lock(mu_);
    empty_cond_var_.wait(lock, [this]() { return !channel_.empty(); });

    T* channel_element = std::move(channel_.front());
    channel_.pop_front();

    NotifyAllSenders(&lock);
    return channel_element;
  }

  size_t Size() {
    std::unique_lock<std::mutex> lock(mu_);
    return channel_.size();
  }

  void Clear() {
    std::unique_lock<std::mutex> lock(mu_);
    channel_.clear();

    NotifyAllSenders(&lock);
  }

 private:
  std::size_t capacity_;
  std::mutex mu_;
  std::condition_variable empty_cond_var_;
  std::condition_variable full_cond_var_;
  std::deque<T> channel_;

 private:
  void NotifyAllSenders(std::unique_lock<std::mutex>* lock) {
    if (IsBounded()) {
      lock->unlock();
      full_cond_var_.notify_one();
    }
  }

  bool IsBounded() const { return capacity_ > 0; }

  bool IsCapacityFull() const { return channel_.size() >= capacity_; }
};

}  // namespace operator
}  // namespace paddle
