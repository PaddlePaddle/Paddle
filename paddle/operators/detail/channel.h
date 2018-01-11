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
#include <vector>

#include "paddle/framework/lod_tensor.h"
#include "paddle/platform/place.h"

namespace paddle {
namespace operators {
namespace detail {

template <typename T>
class Channel {
 public:
  explicit Channel(std::size_t capacity, std::size_t bytes_limit)
      : capacity_(capacity), bytes_limit_(bytes_limit), current_bytes_(0) {}

  void Put(ChannelElement* buffer_element) {
    std::unique_lock<std::mutex> lock(mu_);

    std::size_t element_bytes = GetElementBytes(*buffer_element);

    PADDLE_ENFORCE(bytes_limit_ > 0 && element_bytes > bytes_limit_,
                   "*Attempted to insert T with combined size of %d "
                   "bytes into Chnnel with a memory limit of %d.",
                   element_bytes, bytes_limit_);

    if (IsBounded()) {
      full_cond_var_.wait(lock, [element_bytes, this]() {
        bool bytes_limit_valid =
            bytes_limit_ > 0 ? !IsExceedMemoryLimit(element_bytes) : true;
        bool capacity_valid = capacity_ > 0 ? !IsCapacityFull() : true;

        return capacity_valid && bytes_limit_valid;
      });
    }

    current_bytes_ += element_bytes;
    channel_.push_back(std::move(*buffer_element));

    lock.unlock();
    empty_cond_var_.notify_all();
  }

  void Get(ChannelElement* buffer_element) {
    std::unique_lock<std::mutex> lock(mu_);

    empty_cond_var_.wait(lock, [this]() { return !channel_.empty(); });

    *buffer_element = std::move(channel_.front());
    channel_.pop_front();

    current_bytes_ -= GetElementBytes(*buffer_element);

    NotifyInserters(&lock);
  }

  size_t Size() {
    std::unique_lock<std::mutex> lock(mu_);
    return channel_.size();
  }

  void Clear() {
    std::unique_lock<std::mutex> lock(mu_);
    channel_.clear();
    current_bytes_ = 0;

    NotifyInserters(&lock);
  }

 private:
  using ChannelElement = std::vector<T>;

  std::size_t capacity_;
  std::size_t bytes_limit_;
  std::size_t current_bytes_;
  std::mutex mu_;
  std::condition_variable empty_cond_var_;
  std::condition_variable full_cond_var_;
  std::deque<ChannelElement> channel_;

 private:
  void NotifyInserters(std::unique_lock<std::mutex>* lock) {
    if (IsBounded()) {
      lock->unlock();
      full_cond_var_.notify_all();
    }
  }

  bool IsBounded() const { return capacity_ > 0 || bytes_limit_ > 0; }

  bool IsCapacityFull() const { return channel_.size() >= capacity_; }

  bool IsExceedMemoryLimit(std::size_t bytes) const {
    return bytes + current_bytes_ > bytes_limit_;
  }

  std::size_t GetElementBytes(const ChannelElement& tuple) {
    return std::accumulate(tuple.begin(), tuple.end(), 0,
                           [](const std::size_t& cal, const T& ele) {
                             return cal + ele.memory_size();
                           });
  }
};

}  // namespace detail
}  // namespace operator
}  // namespace paddle
