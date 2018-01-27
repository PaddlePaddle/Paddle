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

// Channel is the abstract class of buffered and un-buffered channels.
template <typename T>
class Channel {
 public:
  // Instantiate and delete channels.
  static Channel* Make(size_t buffer_size);
  static void Close(Channel* ch);
  
  virtual void Send(T*) = 0;
  virtual T* Receive() = 0;
  virtual size_t Cap() = 0;

  // Don't delete channels; instead, call Channel::Close.
 protected:
  virtual ~Channel() {}
};


// details::Buffered and details::UnBuffered are derived from Channel.
namespace details {

template <typename T>
class Buffered : public Channel<T> {
  friend Channel<T>* Channel<T>::Make(size_t);
  friend void Channel<T>::Close(Channel<T>*);
  
 public:
  virtual void Send(T*);
  virtual T* Receive();
  virtual size_t Cap() { return cap_; }
  
 private:
  size_t cap_;

  Buffered(size_t cap) : cap_(cap) {}
  virtual ~Buffered() {}
};

template <typename T>
class UnBuffered : public Channel<T> {
  friend Channel<T>* Channel<T>::Make(size_t);
  friend void Channel<T>::Close(Channel<T>*);
  
 public:
  virtual void Send(T*);
  virtual T* Receive();
  virtual size_t Cap() { return 0; }
    
 private:
  UnBuffered() {}
  virtual ~UnBuffered() {}
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

#include "paddle/framework/details/channel.h"
