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

#include <unordered_map>

namespace paddle {
namespace platform {

template <typename Lockable>
class RepeatedLockMutex {
 public:
  using native_handle_type = typename Lockable::native_handle_type;

  ~RepeatedLockMutex() { is_locked_map()->erase(&lock_); }

  void lock() {
    auto &locked = is_locked_ref();
    if (!locked) {
      lock_.lock();
      locked = true;
    }
  }

  bool try_lock() {
    auto &locked = is_locked_ref();
    if (!locked) {
      locked = lock_.try_lock();
    }
    return locked;
  }

  void unlock() {
    auto &locked = is_locked_ref();
    if (locked) {
      lock_.unlock();
      locked = false;
    }
  }

  bool is_locked() { return is_locked_ref(); }

  native_handle_type native_handle() { return lock_.native_handle(); }

 private:
  Lockable lock_;

  struct BoolWrapper {
    bool val_{false};
  };

  static std::unordered_map<Lockable *, BoolWrapper> *is_locked_map() {
    thread_local std::unordered_map<Lockable *, BoolWrapper> is_locked;
    return &is_locked;
  }

  bool &is_locked_ref() { return (*is_locked_map())[&lock_].val_; }
};

}  // namespace platform
}  // namespace paddle
