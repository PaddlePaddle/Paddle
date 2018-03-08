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
#include <stddef.h>  // for size_t
#include <atomic>
#include <condition_variable>
#include <deque>
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

template <typename T>
class ChannelImpl : public paddle::framework::Channel<T> {
  friend Channel<T> *paddle::framework::MakeChannel<T>(size_t);
  friend void paddle::framework::CloseChannel<T>(Channel<T> *);

 public:
  virtual bool Send(T *);
  virtual bool Receive(T *);
  virtual size_t Cap() { return cap_; }
  virtual void Lock();
  virtual void Unlock();
  virtual void Close();

  ChannelImpl(size_t);
  virtual ~ChannelImpl();

 private:
  struct QueueMessage {
    T *data;
    std::condition_variable_any cond;
    bool chan_closed = false;
    bool completed = false;

    QueueMessage(T *item) : data(item) {}

    void Wait(std::unique_lock<std::recursive_mutex> &lock) {
      cond.wait(lock, [this]() { return completed; });
    }

    void Notify() {
      completed = true;
      cond.notify_all();
    }
  };

  bool send_return(bool value) {
    send_ctr--;
    destructor_cond_.notify_all();
    return value;
  }

  bool recv_return(bool value) {
    recv_ctr--;
    destructor_cond_.notify_all();
    return value;
  }

  size_t cap_;
  std::recursive_mutex mu_;
  bool closed_;
  std::deque<T> buf_;
  std::deque<std::shared_ptr<QueueMessage>> recvq;
  std::deque<std::shared_ptr<QueueMessage>> sendq;
  std::atomic<unsigned> send_ctr{0};
  std::atomic<unsigned> recv_ctr{0};
  std::condition_variable_any destructor_cond_;
};

template <typename T>
ChannelImpl<T>::ChannelImpl(size_t capacity)
    : cap_(capacity), closed_(false), send_ctr(0), recv_ctr(0) {
  PADDLE_ENFORCE_GE(capacity, 0);
}

template <typename T>
bool ChannelImpl<T>::Send(T *item) {
  send_ctr++;
  std::unique_lock<std::recursive_mutex> lock{mu_};

  // If channel is closed, do nothing
  if (closed_) {
    lock.unlock();
    // TODO(abhinavarora) Should panic on closed channel
    return send_return(false);
  }

  // If there is a receiver, directly pass the value we want
  // to send to the receiver, bypassing the channel buffer if any
  if (!recvq.empty()) {
    std::shared_ptr<QueueMessage> m = recvq.front();
    recvq.pop_front();
    // Do the data transfer
    *(m->data) = std::move(*item);
    // Wake up the blocked process and unlock
    m->Notify();
    lock.unlock();
    return send_return(true);
  }

  // Unbuffered channel will always bypass this
  // If buffered channel has space in buffer,
  // write the element to the buffer.
  if (buf_.size() < cap_) {
    // Copy to buffer
    buf_.push_back(std::move(*item));
    // Release lock and return true
    lock.unlock();
    return send_return(true);
  }

  // Block on channel, because some receiver will complete
  // the operation for us
  auto m = std::make_shared<QueueMessage>(item);
  sendq.push_back(m);
  m->Wait(lock);
  // TODO(abhinavarora) Should panic on closed channel
  return send_return(!m->chan_closed);
}

template <typename T>
bool ChannelImpl<T>::Receive(T *item) {
  recv_ctr++;
  std::unique_lock<std::recursive_mutex> lock{mu_};

  // If channel is closed and buffer is empty or
  // channel is unbuffered
  if (closed_ && buf_.empty()) {
    lock.unlock();
    return recv_return(false);
  }

  // If there is a sender, directly receive the value we want
  // from the sender, bypassing the channel buffer if any
  if (!sendq.empty()) {
    std::shared_ptr<QueueMessage> m = sendq.front();
    sendq.pop_front();
    // Do the data transfer
    *item = std::move(*(m->data));
    // Wake up the blocked process and unlock
    m->Notify();
    lock.unlock();
    return recv_return(true);
  }

  // If this is a buffered channel and there are items in buffer
  if (buf_.size() > 0) {
    // Directly read from buffer
    *item = std::move(buf_.front());
    buf_.pop_front();
    // Release lock and return true
    lock.unlock();
    return recv_return(true);
  }

  // No sender available, block on this channel
  // Some receiver will complete the option for us
  auto m = std::make_shared<QueueMessage>(item);
  recvq.push_back(m);
  m->Wait(lock);

  return recv_return(!m->chan_closed);
}

template <typename T>
void ChannelImpl<T>::Lock() {
  mu_.lock();
}

template <typename T>
void ChannelImpl<T>::Unlock() {
  mu_.unlock();
}

template <typename T>
void ChannelImpl<T>::Close() {
  std::unique_lock<std::recursive_mutex> lock{mu_};

  if (closed_) {
    // TODO(abhinavarora): closing an already closed channel should panic
    lock.unlock();
    return;
  }

  closed_ = true;

  // Empty the readers
  while (!recvq.empty()) {
    std::shared_ptr<QueueMessage> m = recvq.front();
    recvq.pop_front();
    m->chan_closed = true;
    m->Notify();
  }

  // Empty the senders
  while (!sendq.empty()) {
    std::shared_ptr<QueueMessage> m = sendq.front();
    sendq.pop_front();
    m->chan_closed = true;
    m->Notify();
  }
}

template <typename T>
ChannelImpl<T>::~ChannelImpl() {
  Close();
  // The destructor must wait for all readers and writers to complete their task
  // The channel has been closed, so we will not accept new readers and writers
  std::unique_lock<std::recursive_mutex> lock{mu_};
  destructor_cond_.wait(lock,
                        [this]() { return send_ctr == 0 && recv_ctr == 0; });
}

}  // namespace framework
}  // namespace paddle
