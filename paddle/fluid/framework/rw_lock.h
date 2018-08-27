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
#endif  // !_WIN32

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

#if !defined(_WIN32)
struct RWLock {
  RWLock() { pthread_rwlock_init(&lock_, nullptr); }

  ~RWLock() { pthread_rwlock_destroy(&lock_); }

  void RDLock() {
    PADDLE_ENFORCE_EQ(pthread_rwlock_rdlock(&lock_), 0,
                      "acquire read lock failed");
  }

  void WRLock() {
    PADDLE_ENFORCE_EQ(pthread_rwlock_wrlock(&lock_), 0,
                      "acquire write lock failed");
  }

  void UNLock() {
    PADDLE_ENFORCE_EQ(pthread_rwlock_unlock(&lock_), 0, "unlock failed");
  }

 private:
  pthread_rwlock_t lock_;
};
#else
// https://stackoverflow.com/questions/7125250/making-pthread-rwlock-wrlock-recursive
// In windows, rw_lock seems like a hack. Use empty object and do nothing.
struct RWLock {
  void RDLock() {}
  void WRLock() {}
  void UNLock() {}
};
#endif

}  // namespace framework
}  // namespace paddle
