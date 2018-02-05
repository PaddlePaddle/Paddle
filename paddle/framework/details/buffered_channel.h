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
#include <deque>
#include <mutex>

#include "paddle/framework/channel.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace framework {
namespace details {

template <typename T>
class Buffered : public paddle::framework::Channel<T> {
  friend Channel<T>* paddle::framework::MakeChannel<T>(size_t);
  friend void paddle::framework::CloseChannel<T>(Channel<T>*);

 public:
  virtual bool Send(T*);
  virtual bool Receive(T*);
  virtual size_t Cap() { return cap_; }
  virtual void Close();
  virtual ~Buffered();

 private:
  size_t cap_;
  std::mutex mu_;
  std::condition_variable empty_cond_var_;
  std::condition_variable full_cond_var_;
  std::deque<T> channel_;
  bool closed_;

  Buffered(size_t cap) : cap_(cap), closed_(false) {
    PADDLE_ENFORCE_GT(cap, 0);
  }

  void NotifyAllParticipants(std::unique_lock<std::mutex>*);
};

template <typename T>
bool Buffered<T>::Send(T* item) {
  std::unique_lock<std::mutex> lock(mu_);
  full_cond_var_.wait(lock,
                      [this]() { return channel_.size() < cap_ || closed_; });
  bool ret = false;
  if (!closed_) {
    channel_.push_back(std::move(*item));
    lock.unlock();
    empty_cond_var_.notify_one();
    ret = true;
  }
  return ret;
}

template <typename T>
bool Buffered<T>::Receive(T* item) {
  std::unique_lock<std::mutex> lock(mu_);
  empty_cond_var_.wait(lock, [this]() { return !channel_.empty() || closed_; });
  bool ret = false;
  if (!closed_) {
    *item = std::move(channel_.front());
    channel_.pop_front();
    full_cond_var_.notify_one();
    ret = true;
  }
  return ret;
}

template <typename T>
void Buffered<T>::Close() {
  std::unique_lock<std::mutex> lock(mu_);
  closed_ = true;
  NotifyAllParticipants(&lock);
}

template <typename T>
Buffered<T>::~Buffered() {
  std::unique_lock<std::mutex> lock(mu_);
  closed_ = true;
  channel_.clear();
  NotifyAllParticipants(&lock);
}

template <typename T>
void Buffered<T>::NotifyAllParticipants(std::unique_lock<std::mutex>* lock) {
  lock->unlock();
  full_cond_var_.notify_all();
  empty_cond_var_.notify_all();
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
