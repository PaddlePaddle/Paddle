//  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT
#include <utility>

#include "paddle/fluid/platform/enforce.h"

#pragma once

namespace paddle {
namespace framework {

class AsyncVariable {
 public:
  AsyncVariable();
  AsyncVariable(const AsyncVariable&) = delete;
  AsyncVariable& operator=(const AsyncVariable&) = delete;

  bool IsAvailable() const { return state_ == EnumState::kAvailable; }

  template <typename T>
  const T& Get() const {
    if (state_ != EnumState::kAvailable) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return state_ == EnumState::kAvailable; });
    }
    return *static_cast<const T*>(holder_);
  }

  template <typename T>
  T* GetMutable() {
    if (state_ != EnumState::kAvailable) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return state_ == EnumState::kAvailable; });
    }
    return static_cast<T*>(holder_);
  }

  template <typename T, typename... Args>
  void Emplace(Args&&... args) {
    PADDLE_ENFORCE_NE(
        state_, EnumState::kAvailable,
        platform::errors::Unimplemented("Emplace AsyncVariable for multiple "
                                        "times is not implemented now."));
    if (state_ == EnumState::kNotAvailable) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = EnumState::kAvailable;
        holder_ = new T(std::forward<Args>(args)...);
      }
      cv_.notify_all();
      return;
    }
    holder_ = new T(std::forward<Args>(args)...);
  }

  template <typename WaiterT>
  void AndThen(WaiterT&& waiter) {
    // TODO(zhhsplendid): Implement it.
  }

  ~AsyncVariable();

  // Enum representing the states for AsyncVaraible
  enum EnumState {
    // Reserve
    kUknown = 0,
    // Data is not available
    kNotAvailable = 1,
    // Data is available
    kAvailable = 2,
  };

 private:
  // Holder for data member
  void* holder_;
  // Enum state of the current AsyncVariable
  EnumState state_;
  // Condition variable used for wait-notify
  mutable std::condition_variable cv_;
  // Mutex used for wait-notify
  mutable std::mutex mutex_;
};

}  // namespace framework
}  // namespace paddle
