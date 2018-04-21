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

#include <stddef.h>            // for size_t
#include <condition_variable>  // NOLINT
#include <typeindex>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

enum class ChannelAction {
  SEND = 0,
  RECEIVE = 1,
  CLOSE = 2,
};

// Channel is the abstract class of buffered and un-buffered channels.
template <typename T>
class Channel {
 public:
  virtual bool CanSend() = 0;
  virtual bool CanReceive() = 0;
  virtual void Send(T*) = 0;
  virtual bool Receive(T*) = 0;
  virtual size_t Cap() = 0;
  virtual void Lock() = 0;

  virtual void Unlock() = 0;
  virtual bool IsClosed() = 0;
  virtual void Close() = 0;
  virtual ~Channel() {}

  virtual void AddToSendQ(const void* referrer, T* data,
                          std::shared_ptr<std::condition_variable_any> cond,
                          std::function<bool(ChannelAction)> cb) = 0;
  virtual void AddToReceiveQ(const void* referrer, T* data,
                             std::shared_ptr<std::condition_variable_any> cond,
                             std::function<bool(ChannelAction)> cb) = 0;
  virtual void RemoveFromSendQ(const void* referrer) = 0;
  virtual void RemoveFromReceiveQ(const void* referrer) = 0;
};

// Forward declaration of channel implementations.
template <typename T>
class ChannelImpl;

template <typename T>
Channel<T>* MakeChannel(size_t buffer_size) {
  return new ChannelImpl<T>(buffer_size);
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
  void Send(T* data) {
    PADDLE_ENFORCE_EQ(IsInitialized(), true,
                      "The Channel hasn't been initialized");
    PADDLE_ENFORCE_EQ(
        holder_->Type(), std::type_index(typeid(T)),
        "Channel type is not same as the type of the data being sent");
    // Static cast should be safe because we have ensured that types are same
    Channel<T>* channel = static_cast<Channel<T>*>(holder_->Ptr());
    PADDLE_ENFORCE_EQ(channel != nullptr, true, "Channel should not be null.");
    channel->Send(data);
  }

  template <typename T>
  bool Receive(T* data) {
    PADDLE_ENFORCE_EQ(IsInitialized(), true,
                      "The Channel hasn't been initialized");
    PADDLE_ENFORCE_EQ(
        holder_->Type(), std::type_index(typeid(T)),
        "Channel type is not same as the type of the data being sent");
    Channel<T>* channel = static_cast<Channel<T>*>(holder_->Ptr());
    PADDLE_ENFORCE_EQ(channel != nullptr, true, "Channel should not be null.");
    return channel->Receive(data);
  }

  bool IsClosed() {
    PADDLE_ENFORCE_EQ(IsInitialized(), true,
                      "The Channel hasn't been initialized");
    return holder_->IsClosed();
  }

  bool CanSend() {
    PADDLE_ENFORCE_EQ(IsInitialized(), true,
                      "The Channel hasn't been initialized");
    return holder_->CanSend();
  }

  bool CanReceive() {
    PADDLE_ENFORCE_EQ(IsInitialized(), true,
                      "The Channel hasn't been initialized");
    return holder_->CanReceive();
  }

  void close() {
    PADDLE_ENFORCE_EQ(IsInitialized(), true,
                      "The Channel hasn't been initialized");
    holder_->Close();
  }

  size_t Cap() {
    PADDLE_ENFORCE_EQ(IsInitialized(), true,
                      "The Channel hasn't been initialized");
    return holder_->Cap();
  }

  void Lock() {
    PADDLE_ENFORCE_EQ(IsInitialized(), true,
                      "The Channel hasn't been initialized");
    holder_->Lock();
  }

  void Unlock() {
    PADDLE_ENFORCE_EQ(IsInitialized(), true,
                      "The Channel hasn't been initialized");
    holder_->Unlock();
  }

  template <typename T>
  void AddToSendQ(const void* referrer, T* data,
                  std::shared_ptr<std::condition_variable_any> cond,
                  std::function<bool(ChannelAction)> cb) {
    PADDLE_ENFORCE_EQ(IsInitialized(), true,
                      "The Channel hasn't been initialized");
    Channel<T>* channel = static_cast<Channel<T>*>(holder_->Ptr());
    if (channel != nullptr) {
      channel->AddToSendQ(referrer, data, cond, cb);
    }
  }

  template <typename T>
  void AddToReceiveQ(const void* referrer, T* data,
                     std::shared_ptr<std::condition_variable_any> cond,
                     std::function<bool(ChannelAction)> cb) {
    PADDLE_ENFORCE_EQ(IsInitialized(), true,
                      "The Channel hasn't been initialized");
    Channel<T>* channel = static_cast<Channel<T>*>(holder_->Ptr());
    if (channel != nullptr) {
      channel->AddToReceiveQ(referrer, data, cond, cb);
    }
  }

  void RemoveFromSendQ(const void* referrer) {
    PADDLE_ENFORCE_EQ(IsInitialized(), true,
                      "The Channel hasn't been initialized");
    holder_->RemoveFromSendQ(referrer);
  }

  void RemoveFromReceiveQ(const void* referrer) {
    PADDLE_ENFORCE_EQ(IsInitialized(), true,
                      "The Channel hasn't been initialized");
    holder_->RemoveFromReceiveQ(referrer);
  }

  inline bool IsInitialized() const { return holder_ != nullptr; }

  inline const std::type_index Type() {
    PADDLE_ENFORCE_EQ(IsInitialized(), true,
                      "The Channel hasn't been initialized");
    return holder_->Type();
  }

 private:
  /**
   * @note    Placeholder hides type T, so it doesn't appear as a template
   *          parameter of ChannelHolder.
   */
  struct Placeholder {
    virtual ~Placeholder() {}
    virtual const std::type_index Type() const = 0;
    virtual void* Ptr() const = 0;
    virtual bool IsClosed() = 0;
    virtual bool CanSend() = 0;
    virtual bool CanReceive() = 0;
    virtual void RemoveFromSendQ(const void* referrer) = 0;
    virtual void RemoveFromReceiveQ(const void* referrer) = 0;
    virtual void Close() = 0;
    virtual void Lock() = 0;
    virtual void Unlock() = 0;
    virtual size_t Cap() = 0;
  };

  template <typename T>
  struct PlaceholderImpl : public Placeholder {
    explicit PlaceholderImpl(size_t buffer_size)
        : type_(std::type_index(typeid(T))) {
      channel_.reset(MakeChannel<T>(buffer_size));
    }

    virtual const std::type_index Type() const { return type_; }

    virtual void* Ptr() const { return static_cast<void*>(channel_.get()); }

    virtual bool IsClosed() {
      if (channel_) {
        return channel_->IsClosed();
      }
      return false;
    }

    virtual bool CanSend() {
      if (channel_) {
        return channel_->CanSend();
      }
      return false;
    }

    virtual bool CanReceive() {
      if (channel_) {
        return channel_->CanReceive();
      }
      return false;
    }

    virtual void RemoveFromSendQ(const void* referrer) {
      if (channel_) {
        channel_->RemoveFromSendQ(referrer);
      }
    }

    virtual void RemoveFromReceiveQ(const void* referrer) {
      if (channel_) {
        channel_->RemoveFromReceiveQ(referrer);
      }
    }

    virtual void Close() {
      if (channel_) channel_->Close();
    }

    virtual size_t Cap() {
      if (channel_)
        return channel_->Cap();
      else
        return -1;
    }

    virtual void Lock() {
      if (channel_) channel_->Lock();
    }

    virtual void Unlock() {
      if (channel_) channel_->Unlock();
    }

    std::unique_ptr<Channel<T>> channel_;
    const std::type_index type_;
  };

  // Pointer to a PlaceholderImpl object
  std::unique_ptr<Placeholder> holder_;
};

}  // namespace framework
}  // namespace paddle

#include "paddle/fluid/framework/channel_impl.h"
