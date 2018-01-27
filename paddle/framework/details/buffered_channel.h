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

#include "paddle/framework/channel.h"

namespace paddle {
namespace framework {
namespace details {

template <typename T>
class Buffered : public paddle::framework::Channel<T> {
  friend Channel<T>* paddle::framework::MakeChannel<T>(size_t);
  friend void paddle::framework::CloseChannel<T>(Channel<T>*);
  
 public:
  virtual void Send(T*);
  virtual T* Receive();
  virtual size_t Cap() { return cap_; }
  
 private:
  size_t cap_;
  std::mutex mu_;
  std::condition_variable empty_cond_var_;
  std::condition_variable full_cond_var_;
  std::deque<T> channel_;

  Buffered(size_t cap) : cap_(cap) {}
  virtual ~Buffered();

  void NotifyAllSenders(std::unique_lock<std::mutex>*);
};

template <typename T>
void Buffered<T>::Send(T* channel_element) {
  std::unique_lock<std::mutex> lock(mu_);
  full_cond_var_.wait(lock, [this]() { return channel_.size() < cap_; });
  channel_.push_back(std::move(*channel_element));
  lock.unlock();
  empty_cond_var_.notify_one();
}

template <typename T>
T* Buffered<T>::Receive() {
  std::unique_lock<std::mutex> lock(mu_);
  empty_cond_var_.wait(lock, [this]() { return !channel_.empty(); });
  
  T* channel_element = std::move(channel_.front());
  channel_.pop_front();
  
  NotifyAllSenders(&lock);
  return channel_element;
}

template <typename T>
Buffered<T>::~Buffered() {
  std::unique_lock<std::mutex> lock(mu_);
  channel_.clear();
  NotifyAllSenders(&lock);
}

template <typename T>
void Buffered<T>::NotifyAllSenders(std::unique_lock<std::mutex>* lock) {
  lock->unlock();
  full_cond_var_.notify_one();
}


}  // namespace details
}  // namespace framework
}  // namespace paddle
