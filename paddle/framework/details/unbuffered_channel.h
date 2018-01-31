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

#include "paddle/framework/channel.h"

namespace paddle {
namespace framework {
namespace details {

template <typename T>
class UnBuffered : public paddle::framework::Channel<T> {
  friend Channel<T>* paddle::framework::MakeChannel<T>(size_t);
  friend void paddle::framework::CloseChannel<T>(Channel<T>*);

 public:
   virtual void Send(T*);
   virtual void Receive(T*);
   virtual size_t Cap() { return 0; }
   virtual void Close();
   virtual ~UnBuffered();

 private:
   std::mutex mu_ch_;
   std::recursive_mutex mu_read_, mu_write_;
   std::atomic<bool> reader_found_{false}, writer_found_{false};
   std::condition_variable cv_channel_;
   std::condition_variable_any cv_reader_, cv_writer_;
   T* item{nullptr};
   std::atomic<bool> closed_{false};

   UnBuffered(): closed_(false) {}

   void NotifyAllSenders(std::unique_lock<std::mutex>*);
};

template <typename T>
void UnBuffered<T>::Send(T* data) {
  std::unique_lock<std::recursive_mutex> writer_lock(mu_write_);
  writer_found_ = true;
  std::unique_lock<std::recursive_mutex> cv_lock(mu_write_);
  cv_writer_.wait(cv_lock,
                  [this]() { return reader_found_ == true || closed_; });
  if (!closed_) {
    {
      std::unique_lock<std::mutex> channel_lock(mu_ch_);
      item = data;
    }
    cv_channel_.notify_one();
  }
  writer_found_ = false;
}

template <typename T>
void UnBuffered<T>::Receive(T* data) {
  std::unique_lock<std::recursive_mutex> read_lock{mu_read_};
  reader_found_ = true;
  std::unique_lock<std::recursive_mutex> cv_lock{mu_read_};
  cv_reader_.wait(cv_lock,
                  [this]() { return writer_found_ == true || closed_; });
  if (!closed_){
    std::unique_lock<std::mutex> lock_ch{mu_ch_};
    cv_channel_.wait(lock_ch,
                     [this]() { return item != nullptr || closed_; });
    if (!closed_){
      *data = std::move(*item);
      item = nullptr;
      lock_ch.unlock();
      cv_writer_.notify_one();
    }
  }
  reader_found_ = false;
}

template <typename T>
void UnBuffered<T>::Close() {
  std::unique_lock<std::mutex> lock(mu_ch_);
  item = nullptr;
  closed_ = true;
  NotifyAllSenders(&lock);
}

template <typename T>
UnBuffered<T>::~UnBuffered() {
  std::unique_lock<std::mutex> lock(mu_ch_);
  closed_ = true;
  NotifyAllSenders(&lock);
}

template <typename T>
void UnBuffered<T>::NotifyAllSenders(std::unique_lock<std::mutex>* lock) {
  lock->unlock();
  cv_writer_.notify_all();
  cv_channel_.notify_all();
  cv_reader_.notify_all();
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
