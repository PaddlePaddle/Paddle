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
  virtual bool Send(T*) = 0;
  virtual bool Receive(T*) = 0;
  virtual size_t Cap() = 0;
  virtual void Close() = 0;
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

}  // namespace framework
}  // namespace paddle

#include "paddle/framework/details/buffered_channel.h"
#include "paddle/framework/details/unbuffered_channel.h"
