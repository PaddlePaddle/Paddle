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
#include <mutex>

#include "paddle/fluid/framework/channel.h"

namespace paddle {
namespace framework {
namespace details {

// Four of the properties of UnBuffered Channel:
// - A send to a channel blocks temporarily until a receive from the
// channel or the channel is closed.
// - A receive from a channel blocks temporarily until a send to the
// channel or the channel is closed.
// - A send to a closed channel returns false immediately.
// - A receive from a closed channel returns false immediately.
template <typename T>
class UnBuffered : public paddle::framework::Channel<T> {
  friend Channel<T>* paddle::framework::MakeChannel<T>(size_t);
  friend void paddle::framework::CloseChannel<T>(Channel<T>*);

 public:
  virtual bool Send(T*);
  virtual bool Receive(T*);
  virtual size_t Cap() { return 0; }
  virtual void Close();
  virtual ~UnBuffered();

 private:
  std::mutex mu_ch_;
  // Mutex for readers and writers who are waiting for other reader
  // and writer to complete execution
  std::recursive_mutex mu_read_, mu_write_;
  // reader_found_ is set true when a reader is ready to accept data
  // writer_found_ is set true when a writer is ready to send data
  // A transaction occurs only when both are true
  std::atomic<bool> reader_found_{false}, writer_found_{false};
  std::condition_variable cv_channel_;
  std::condition_variable_any cv_reader_, cv_writer_, cv_destructor_;
  T* item{nullptr};
  std::atomic<bool> closed_{false};
  std::atomic<unsigned> send_ctr{0};
  std::atomic<unsigned> recv_ctr{0};

  UnBuffered() : closed_(false) {}

  void NotifyAllParticipants(std::unique_lock<std::mutex>*);
};

// This function implements the concept of how data should
// be sent from a writer to a reader.
template <typename T>
bool UnBuffered<T>::Send(T* data) {
  bool ret = false;
  if (closed_) {
    return ret;
  }
  send_ctr++;
  // Prevent other writers from entering
  std::unique_lock<std::recursive_mutex> writer_lock(mu_write_);
  writer_found_ = true;
  std::unique_lock<std::recursive_mutex> cv_lock(mu_write_);
  // If writer comes first, it should wait till a reader arrives
  cv_writer_.wait(cv_lock,
                  [this]() { return reader_found_ == true || closed_; });
  cv_reader_.notify_one();
  if (!closed_) {
    std::unique_lock<std::mutex> channel_lock(mu_ch_);
    item = data;
    channel_lock.unlock();
    cv_channel_.notify_one();
    channel_lock.lock();
    cv_channel_.wait(channel_lock,
                     [this]() { return item == nullptr || closed_; });
    ret = true;
  }
  writer_found_ = false;
  send_ctr--;
  cv_destructor_.notify_one();
  return ret;
}

// This function implements the concept of how
// data that was sent by a writer is read from a reader.
template <typename T>
bool UnBuffered<T>::Receive(T* data) {
  bool ret = false;
  // If channel is closed, we don't even want any reader to enter.
  // Unlike a buffered channel, an unbuffered channel does not allow
  // readers to read after closing because there is no buffer to be consumed.
  if (closed_) return ret;
  recv_ctr++;
  // Prevent other readers from entering
  std::unique_lock<std::recursive_mutex> read_lock{mu_read_};
  reader_found_ = true;
  std::unique_lock<std::recursive_mutex> cv_lock{mu_read_};
  // If reader comes first, it should wait till a writer arrives
  cv_reader_.wait(cv_lock,
                  [this]() { return writer_found_ == true || closed_; });
  cv_writer_.notify_one();
  if (!closed_) {
    std::unique_lock<std::mutex> lock_ch{mu_ch_};
    // Reader should wait for the writer to first write its data
    cv_channel_.wait(lock_ch, [this]() { return item != nullptr || closed_; });
    if (!closed_) {
      *data = std::move(*item);
      item = nullptr;
      lock_ch.unlock();
      ret = true;
    }
    cv_channel_.notify_one();
  }
  reader_found_ = false;
  recv_ctr--;
  cv_destructor_.notify_one();
  return ret;
}

// This function implements the sequence of events
// that take place once the channel is closed.
template <typename T>
void UnBuffered<T>::Close() {
  if (closed_) {
    return;
  }
  std::unique_lock<std::mutex> lock(mu_ch_);
  item = nullptr;
  closed_ = true;
  NotifyAllParticipants(&lock);
}

// This function implements the sequence of events
// that are executed once the object of an UnBuffered
// channel is destroyed.
template <typename T>
UnBuffered<T>::~UnBuffered() {
  std::unique_lock<std::mutex> lock(mu_ch_);
  item = nullptr;
  closed_ = true;
  NotifyAllParticipants(&lock);
  lock.lock();
  cv_destructor_.wait(lock,
                      [this]() { return send_ctr == 0 && recv_ctr == 0; });
}

// This function notifies all the readers, writers and
// the channel condition variables.
template <typename T>
void UnBuffered<T>::NotifyAllParticipants(std::unique_lock<std::mutex>* lock) {
  lock->unlock();
  cv_writer_.notify_all();
  cv_channel_.notify_all();
  cv_reader_.notify_all();
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
