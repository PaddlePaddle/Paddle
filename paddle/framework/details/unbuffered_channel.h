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
  UnBuffered() {}
};

template <typename T>
void UnBuffered<T>::Send(T* channel_element) {}

template <typename T>
void UnBuffered<T>::Receive(T*) {}

template <typename T>
void UnBuffered<T>::Close() {}

template <typename T>
UnBuffered<T>::~UnBuffered() {}

}  // namespace details
}  // namespace framework
}  // namespace paddle
