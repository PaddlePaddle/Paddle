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

#include <stddef.h>  // for size_t

namespace paddle {
namespace framework {

// Channel is the abstract class of buffered and un-buffered channels.
template <typename T>
class Channel {
 public:
  virtual void Send(T*) = 0;
  virtual void Receive(T*) = 0;
  virtual size_t Cap() = 0;
  virtual void Close() = 0;

  // Don't delete channels; instead, call Channel::Close.
 protected:
  virtual ~Channel() {}
};

// Forward declaration of channel implementations.
namespace details {
template <typename T>
class Buffered;
template <typename T>
class UnBuffered;
}  // namespace details

template <typename T>
Channel<T>* MakeChannel(size_t buffer_size) {
  if (buffer_size > 0) {
    return new details::Buffered<T>(buffer_size);
  }
  return new details::UnBuffered<T>();
}

template <typename T>
void CloseChannel(Channel<T>* ch) {
  ch->Close();
}

template <typename T>
void DeleteChannel(Channel<T>* ch) {
  if (ch->Cap() > 0) {
    delete dynamic_cast<details::Buffered<T>*>(ch);
  } else {
    delete dynamic_cast<details::UnBuffered<T>*>(ch);
  }
}

}  // namespace framework
}  // namespace paddle

#include "paddle/framework/details/buffered_channel.h"
#include "paddle/framework/details/unbuffered_channel.h"
