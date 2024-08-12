/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/core/enforce.h"

namespace phi {

#if !defined(_WIN32)
struct RWLock {
  RWLock() { pthread_rwlock_init(&lock_, nullptr); }

  ~RWLock() { pthread_rwlock_destroy(&lock_); }

  inline void RDLock() {
    PADDLE_ENFORCE_EQ(
        pthread_rwlock_rdlock(&lock_),
        0,
        common::errors::External("The pthread failed to acquire read lock."));
  }

  inline void WRLock() {
    PADDLE_ENFORCE_EQ(
        pthread_rwlock_wrlock(&lock_),
        0,
        common::errors::External("The pthread failed to acquire write lock."));
  }

  inline void UNLock() {
    PADDLE_ENFORCE_EQ(
        pthread_rwlock_unlock(&lock_),
        0,
        common::errors::External("The pthread failed to unlock."));
  }

 private:
  pthread_rwlock_t lock_;
};
// TODO(paddle-dev): Support RWLock for WIN32 for correctness.
#else
// https://stackoverflow.com/questions/7125250/making-pthread-rwlock-wrlock-recursive
// In windows, rw_lock seems like a hack. Use empty object and do nothing.
struct RWLock {
  // FIXME(minqiyang): use mutex here to do fake lock
  inline void RDLock() { mutex_.lock(); }

  inline void WRLock() { mutex_.lock(); }

  inline void UNLock() { mutex_.unlock(); }

 private:
  std::mutex mutex_;
};
#endif

class AutoWRLock {
 public:
  explicit AutoWRLock(RWLock* rw_lock) : lock_(rw_lock) { Lock(); }

  ~AutoWRLock() { UnLock(); }

 private:
  inline void Lock() { lock_->WRLock(); }

  inline void UnLock() { lock_->UNLock(); }

 private:
  RWLock* lock_;
};

class AutoRDLock {
 public:
  explicit AutoRDLock(RWLock* rw_lock) : lock_(rw_lock) { Lock(); }

  ~AutoRDLock() { UnLock(); }

 private:
  inline void Lock() { lock_->RDLock(); }

  inline void UnLock() { lock_->UNLock(); }

 private:
  RWLock* lock_;
};

}  // namespace phi
