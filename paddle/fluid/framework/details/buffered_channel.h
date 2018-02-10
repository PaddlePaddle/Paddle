/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <condition_variable>
#include <deque>
#include <mutex>

#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace details {

// Four of the properties of Buffered Channel:
// - A send to a full channel blocks temporarily until a receive from the
// channel or the channel is closed.
// - A receive from an empty channel blocks temporarily until a send to the
// channel or the channel is closed.
// - A send to a closed channel returns false immediately.
// - A receive from a closed channel returns false immediately.

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
  std::condition_variable destructor_cond_var_;
  std::deque<T> channel_;
  std::atomic<bool> closed_{false};
  std::atomic<unsigned> send_ctr{0};
  std::atomic<unsigned> recv_ctr{0};

  Buffered(size_t cap) : cap_(cap), closed_(false) {
    PADDLE_ENFORCE_GT(cap, 0);
  }

  void NotifyAllParticipants(std::unique_lock<std::mutex>*);
};

template <typename T>
bool Buffered<T>::Send(T* item) {
  bool ret = false;
  if (closed_) {
    return ret;
  }
  send_ctr++;
  std::unique_lock<std::mutex> lock(mu_);
  full_cond_var_.wait(lock,
                      [this]() { return channel_.size() < cap_ || closed_; });
  if (!closed_) {
    channel_.push_back(std::move(*item));
    lock.unlock();
    empty_cond_var_.notify_one();
    ret = true;
  }
  send_ctr--;
  destructor_cond_var_.notify_one();
  return ret;
}

template <typename T>
bool Buffered<T>::Receive(T* item) {
  bool ret = false;
  // Once the channel has been closed and all data has been consumed,
  // just return false. Don't even try acquiring the mutex.
  if (closed_ && channel_.empty()) {
    return false;
  }
  recv_ctr++;
  std::unique_lock<std::mutex> lock(mu_);
  empty_cond_var_.wait(lock, [this]() { return !channel_.empty() || closed_; });
  if (!channel_.empty()) {
    *item = std::move(channel_.front());
    channel_.pop_front();
    full_cond_var_.notify_one();
    ret = true;
  }
  recv_ctr--;
  destructor_cond_var_.notify_one();
  return ret;
}

template <typename T>
void Buffered<T>::Close() {
  if (closed_) {
    return;
  }
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

  // The destructor must wait for all readers and writers to complete their task
  // The channel has been closed, so we will not accept new readers and writers
  lock.lock();
  destructor_cond_var_.wait(
      lock, [this]() { return send_ctr == 0 && recv_ctr == 0; });
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
