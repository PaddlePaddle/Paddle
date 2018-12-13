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

#if !defined(_WIN32)
#include <pthread.h>
#else
#include <mutex>  // NOLINT
#endif            // !_WIN32

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

#if !defined(_WIN32)
struct SpinLock {
  SpinLock() { pthread_spin_init(&lock_, PTHREAD_PROCESS_PRIVATE); }

  ~SpinLock() { pthread_spin_destroy(&lock_); }

  void Lock() {
    PADDLE_ENFORCE_EQ(pthread_spin_lock(&lock_), 0, "acquire spin lock failed");
  }

  void Unlock() {
    PADDLE_ENFORCE_EQ(pthread_spin_unlock(&lock_), 0,
                      "release spin lock failed");
  }

 private:
  pthread_spinlock_t lock_;
};
#else
// FIXME(minqiyang): use mutex here to do fake spin lock
struct SpinLock {
  void Lock() { mutex_.lock(); }

  void Unlock() { mutex_.lock(); }

 private:
  std::mutex mutex_;
};
#endif

class AutoSpinLock {
 public:
  explicit SpinLockGuard(SpinLock* spin_lock) : lock_(spin_lock) {
    lock_->Lock();
  }

  ~SpinLockGuard() { lock_->Unlock(); }

 private:
  SpinLock* lock_;
};

}  // namespace framework
}  // namespace paddle
