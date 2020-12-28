// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  //NOLINT
#include <utility>

#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {

/*
 * AsyncVariable is a future type containding a Variable value.
 *
 * Note: Should we design a Base class, and use derive mechanism?
 *
 */
class AsyncVariable {
 public:
  enum EnumState {
    kUnknown = 0,
    kNotAvailable = 1,
    kAvailable = 2,
  };

  AsyncVariable();

  // TODO(Aurelius84): use virtual if use derive machanism?
  ~AsyncVariable();

  bool isAvailable() const { return state_ == EnumState::kAvailable; }

  template <typename T>
  const T& Get() const {
    if (state_ != EnumState::kAvailable) {
      std::unique_lock<std::mutex> lock(mutex_);
      // TODO(Aurelius84): it will block until data is ready. Shall we consider
      // a lock-free
      // solution?
      cv_.wait(lock, [this] { return state_ == EnumState::kAvailable; });
    }
    PADDLE_ENFORCE_NOT_NULL(
        holder_, platform::errors::Unavailable(
                     "holder_ should not be null while calling Get<T>()."));
    return *static_cast<const T*>(holder_);
  }

  template <typename T>
  T* GetMutable() {
    if (state_ != EnumState::kAvailable) {
      std::unique_lock<std::mutex> lock(mutex_);
      // TODO(Aurelius84): it will block until data is ready. Shall we consider
      // a lock-free
      // solution?
      cv_.wait(lock, [this] { return state_ == EnumState::kAvailable; });
    }
    PADDLE_ENFORCE_NOT_NULL(
        holder_,
        platform::errors::Unavailable(
            "holder_ should not be null while calling GetMutable<T>()."));
    return static_cast<T*>(holder_);
  }

  template <typename T, typename... Args>
  void Emplace(Args&&... args) {
    PADDLE_ENFORCE_NE(
        state_, EnumState::kAvailable,
        platform::errors::Unimplemented("Emplace AsyncVariable for multiple "
                                        "times is not implemented now."));

    if (state_ == EnumState::kNotAvailable) {
      std::lock_guard<std::mutex> lock(mutex_);
      holder_ = new T(std::forward<Args>(args)...);
      state_ = EnumState::kAvailable;
    }
    cv_.notify_all();
  }

  // TODO(Aurelius84): use template ?
  template <typename WaiterT>
  void AndThen(WaiterT&& waiter);

 private:
  DISABLE_COPY_AND_ASSIGN(AsyncVariable);

  // void Destroy() {
  //     if(state_ == EnumState::kAvailable){
  //         delete holder_;
  //     }
  // }
  // TODO(Aurelius84): composited with pointer ?
  // paddle::framework::Variable var_;
  void* holder_;

  EnumState state_;

  mutable std::condition_variable cv_;
  mutable std::mutex mutex_;
};
}  // namespace framework
}  // namespace paddle
