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
#include <typeindex>
#include "paddle/fluid/platform/enforce.h"

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

/*
 * The ChannelHolder class serves two main purposes:
 * 1. It acts as a unified wrapper for the different kinds of
 *    channels, i.e. Buffered and Unbuffered channels. This is
 *    similar to the ReaderHolder class.
 * 2. It also helps us in TypeHiding. This is similar to the
 *    PlaceHolder implementations in variable.h and tensor.h.
 */
class ChannelHolder {
 public:
  template <typename T>
  void Reset(size_t buffer_size) {
    holder_.reset(new PlaceholderImpl<T>(buffer_size));
  }

  template <typename T>
  bool Send(T* data) {
    if (!IsInitialized()) return false;
    PADDLE_ENFORCE_EQ(holder_->Type(), std::type_index(typeid(T)));
    // Static cast should be safe because we have ensured that types are same
    Channel<T>* channel = static_cast<Channel<T>*>(holder_->Ptr());
    return channel != nullptr ? channel->Send(data) : false;
  }

  template <typename T>
  bool Receive(T* data) {
    if (!IsInitialized()) return false;
    PADDLE_ENFORCE_EQ(holder_->Type(), std::type_index(typeid(T)));
    Channel<T>* channel = static_cast<Channel<T>*>(holder_->Ptr());
    return channel != nullptr ? channel->Receive(data) : false;
  }

  void close() {
    if (IsInitialized()) holder_->Close();
  }

  inline bool IsInitialized() const { return holder_ != nullptr; }

 private:
  /**
   * @note    Placeholder hides type T, so it doesn't appear as a template
   *          parameter of ChannelHolder.
   */
  struct Placeholder {
    virtual ~Placeholder() {}
    virtual const std::type_index Type() const = 0;
    virtual void* Ptr() const = 0;
    virtual void Close() const = 0;
    std::type_info type_;
  };

  template <typename T>
  struct PlaceholderImpl : public Placeholder {
    PlaceholderImpl(size_t buffer_size) : type_(std::type_index(typeid(T))) {
      channel_.reset(MakeChannel<T>(buffer_size));
    }

    virtual const std::type_index Type() const { return type_; }
    virtual void* Ptr() const { return static_cast<void*>(channel_.get()); }
    virtual void Close() {
      if (channel_) channel_->Close();
    }

    std::unique_ptr<Channel<T>*> channel_;
    const std::type_index type_;
  };

  // Pointer to a PlaceholderImpl object
  std::unique_ptr<Placeholder> holder_;
};

}  // namespace framework
}  // namespace paddle

#include "paddle/fluid/framework/details/buffered_channel.h"
#include "paddle/fluid/framework/details/unbuffered_channel.h"
