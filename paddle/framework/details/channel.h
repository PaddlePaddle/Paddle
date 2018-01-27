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
#include <deque>

namespace paddle {
namespace framework {

template <typename T>
Channel<T>* Channel<T>::Make(size_t buffer_size) {
  if (buffer_size > 0) {
    return new Buffered<T>(buffer_size);
  }
  return new UnBuffered<T>();
}

template <typename T>
void Channel<T>::Close(Channel<T>* ch) {
  if (ch->Cap() > 0) {
    delete dynamic_cast<Buffered<T>*>(ch);
  } else {
    delete dynamic_cast<UnBuffered<T>*>(ch);
  }
}

namespace details {

template <typename T>
class Buffered : public Channel<T> {
 public:
  explicit Buffered(std::size_t capacity) : capacity_(capacity) {}

  void Send(T* channel_element) {
    std::unique_lock<std::mutex> lock(mu_);
    full_cond_var_.wait(lock, [this]() { channel_.size() < capacity_; });
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
  void NotifyAllSenders(std::unique_lock<std::mutex>* lock) {
    lock->unlock();
    full_cond_var_.notify_one();
  }

  std::size_t capacity_;
  std::mutex mu_;
  std::condition_variable empty_cond_var_;
  std::condition_variable full_cond_var_;
  std::deque<T> channel_;
};


}  // namespace details
}  // namespace framework
}  // namespace paddle
