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
#include <condition_variable>  // NOLINT
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
  virtual bool CanSend();
  virtual bool CanReceive();
  virtual void Send(T *);
  virtual bool Receive(T *);
  virtual size_t Cap() { return cap_; }
  virtual void Lock();
  virtual void Unlock();
  virtual bool IsClosed();
  virtual void Close();
  explicit ChannelImpl(size_t);
  virtual ~ChannelImpl();

  virtual void AddToSendQ(const void *referrer, T *data,
                          std::shared_ptr<std::condition_variable_any> cond,
                          std::function<bool(ChannelAction)> cb);
  virtual void AddToReceiveQ(const void *referrer, T *data,
                             std::shared_ptr<std::condition_variable_any> cond,
                             std::function<bool(ChannelAction)> cb);

  virtual void RemoveFromSendQ(const void *referrer);
  virtual void RemoveFromReceiveQ(const void *referrer);

 private:
  struct QueueMessage {
    T *data;
    std::shared_ptr<std::condition_variable_any> cond;
    bool chan_closed = false;
    bool completed = false;
    const void *referrer;  // TODO(thuan): figure out better way to do this
    std::function<bool(ChannelAction)> callback;

    explicit QueueMessage(T *item)
        : data(item), cond(std::make_shared<std::condition_variable_any>()) {}

    QueueMessage(T *item, std::shared_ptr<std::condition_variable_any> cond)
        : data(item), cond(cond) {}

    void Wait(std::unique_lock<std::recursive_mutex> &lock) {
      cond->wait(lock, [this]() { return completed; });
    }

    void Notify() {
      completed = true;
      cond->notify_all();
    }
  };

  void send_return() {
    send_ctr--;
    destructor_cond_.notify_all();
  }

  bool recv_return(bool value) {
    recv_ctr--;
    destructor_cond_.notify_all();
    return value;
  }

  std::shared_ptr<QueueMessage> get_first_message(
      std::deque<std::shared_ptr<QueueMessage>> *queue, ChannelAction action) {
    while (!queue->empty()) {
      // Check whether this message was added by Select
      // If this was added by Select then execute the callback
      // to check if you can execute this message. The callback
      // can return false if some other case was executed in Select.
      // In that case just discard this QueueMessage and process next.
      std::shared_ptr<QueueMessage> m = queue->front();
      queue->pop_front();
      if (m->callback == nullptr || m->callback(action)) return m;
    }
    return nullptr;
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
bool ChannelImpl<T>::CanSend() {
  std::lock_guard<std::recursive_mutex> lock{mu_};
  return !closed_ && (!recvq.empty() || buf_.size() < cap_);
}

template <typename T>
bool ChannelImpl<T>::CanReceive() {
  std::lock_guard<std::recursive_mutex> lock{mu_};
  return !(closed_ && buf_.empty()) && (!sendq.empty() || buf_.size() > 0);
}

template <typename T>
void ChannelImpl<T>::Send(T *item) {
  send_ctr++;
  std::unique_lock<std::recursive_mutex> lock{mu_};

  // If channel is closed, throw exception
  if (closed_) {
    send_return();
    lock.unlock();
    PADDLE_THROW("Cannot send on closed channel");
  }

  // If there is a receiver, directly pass the value we want
  // to send to the receiver, bypassing the channel buffer if any
  if (!recvq.empty()) {
    std::shared_ptr<QueueMessage> m =
        get_first_message(&recvq, ChannelAction::SEND);

    if (m != nullptr) {
      *(m->data) = std::move(*item);
      m->Notify();
      send_return();
      return;
    } else {
      Send(item);
      send_return();
      return;
    }
  }

  // Unbuffered channel will always bypass this
  // If buffered channel has space in buffer,
  // write the element to the buffer.
  if (buf_.size() < cap_) {
    // Copy to buffer
    buf_.push_back(std::move(*item));
    send_return();
    return;
  }

  // Block on channel, because some receiver will complete
  // the operation for us
  auto m = std::make_shared<QueueMessage>(item);
  sendq.push_back(m);
  m->Wait(lock);
  if (m->chan_closed) {
    send_return();
    lock.unlock();
    PADDLE_THROW("Cannot send on closed channel");
  }
  send_return();
}

template <typename T>
bool ChannelImpl<T>::Receive(T *item) {
  recv_ctr++;
  std::unique_lock<std::recursive_mutex> lock{mu_};

  // If channel is closed and buffer is empty or
  // channel is unbuffered
  if (closed_ && buf_.empty()) return recv_return(false);

  // If there is a sender, directly receive the value we want
  // from the sender. In case of a buffered channel, read from
  // buffer and move front of send queue to the buffer
  if (!sendq.empty()) {
    std::shared_ptr<QueueMessage> m =
        get_first_message(&sendq, ChannelAction::RECEIVE);
    if (buf_.size() > 0) {
      // Case 1 : Channel is Buffered
      // Do Data transfer from front of buffer
      // and add a QueueMessage to the buffer
      *item = std::move(buf_.front());
      buf_.pop_front();
      // If first message from sendq is not null
      // add it to the buffer and notify it
      if (m != nullptr) {
        // Copy to buffer
        buf_.push_back(std::move(*(m->data)));
        m->Notify();
      }  // Ignore if there is no first message
    } else {
      // Case 2: Channel is Unbuffered
      // Do data transfer from front of SendQ
      // If front is nullptr, then recursively call itself
      if (m != nullptr) {
        *item = std::move(*(m->data));
        m->Notify();
      } else {
        return recv_return(Receive(item));
      }
    }
    return recv_return(true);
  }

  // If this is a buffered channel and there are items in buffer
  if (buf_.size() > 0) {
    // Directly read from buffer
    *item = std::move(buf_.front());
    buf_.pop_front();
    // return true
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
bool ChannelImpl<T>::IsClosed() {
  std::lock_guard<std::recursive_mutex> lock{mu_};
  return closed_;
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

    // Execute callback function (if any)
    if (m->callback != nullptr) {
      m->callback(ChannelAction::CLOSE);
    }

    m->Notify();
  }

  // Empty the senders
  while (!sendq.empty()) {
    std::shared_ptr<QueueMessage> m = sendq.front();
    sendq.pop_front();
    m->chan_closed = true;

    // Execute callback function (if any)
    if (m->callback != nullptr) {
      m->callback(ChannelAction::CLOSE);
    }

    m->Notify();
  }
}

template <typename T>
void ChannelImpl<T>::AddToSendQ(
    const void *referrer, T *data,
    std::shared_ptr<std::condition_variable_any> cond,
    std::function<bool(ChannelAction)> cb) {
  std::lock_guard<std::recursive_mutex> lock{mu_};
  auto m = std::make_shared<QueueMessage>(data, cond);
  m->referrer = referrer;
  m->callback = cb;
  sendq.push_back(m);
}

template <typename T>
void ChannelImpl<T>::AddToReceiveQ(
    const void *referrer, T *data,
    std::shared_ptr<std::condition_variable_any> cond,
    std::function<bool(ChannelAction)> cb) {
  std::lock_guard<std::recursive_mutex> lock{mu_};
  auto m = std::make_shared<QueueMessage>(data, cond);
  m->referrer = referrer;
  m->callback = cb;
  recvq.push_back(m);
}

template <typename T>
void ChannelImpl<T>::RemoveFromSendQ(const void *referrer) {
  std::lock_guard<std::recursive_mutex> lock{mu_};

  for (auto it = sendq.begin(); it != sendq.end();) {
    std::shared_ptr<QueueMessage> sendMsg = (std::shared_ptr<QueueMessage>)*it;

    if (sendMsg->referrer == referrer) {
      it = sendq.erase(it);
    } else {
      ++it;
    }
  }
}

template <typename T>
void ChannelImpl<T>::RemoveFromReceiveQ(const void *referrer) {
  std::lock_guard<std::recursive_mutex> lock{mu_};

  for (auto it = recvq.begin(); it != recvq.end();) {
    std::shared_ptr<QueueMessage> recvMsg = (std::shared_ptr<QueueMessage>)*it;

    if (recvMsg->referrer == referrer) {
      it = recvq.erase(it);
    } else {
      ++it;
    }
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
