// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <stdint.h>
#include <memory>
#include <mutex>  // NOLINT
namespace paddle {
namespace platform {

/**
 * LockGuard for std::unique_ptr<LockType>. It will do nothing when guarded ptr
 * is nullptr.
 *
 * The advantage of using `LockGuardPtr` instead of
 * std::unique<std::lock_guard<lock_type>> is this type is totally a stack
 * variable. There is no heap allocation at all.
 */
template <typename LockType>
class LockGuardPtr {
  using LockGuardType = std::lock_guard<LockType>;

 public:
  class LockGuardDeleter {
   public:
    void operator()(LockGuardType* guard) { guard->~LockGuardType(); }
  };

  explicit LockGuardPtr(std::unique_ptr<LockType>& lock_ptr)  // NOLINT
      : guard_ptr_(lock_ptr ? new (guard_buffer_) LockGuardType(*lock_ptr)
                            : nullptr) {}

  LockGuardPtr(const LockGuardPtr&) = delete;
  LockGuardPtr& operator=(const LockGuardPtr&) = delete;
  LockGuardPtr(LockGuardPtr&&) = delete;
  LockGuardPtr& operator=(LockGuardPtr&&) = delete;

 private:
  uint8_t guard_buffer_[sizeof(LockGuardType)];
  std::unique_ptr<LockGuardType, LockGuardDeleter> guard_ptr_;
};

}  // namespace platform
}  // namespace paddle
