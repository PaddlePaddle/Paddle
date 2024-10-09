// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

#ifdef _LINUX
#include <pthread.h>
#include <semaphore.h>
#include <cstdlib>
#endif
#include <condition_variable>
#include <mutex>
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
class Barrier {
 public:
  explicit Barrier(int count = 1) {
#ifdef _LINUX
    PADDLE_ENFORCE_GE(
        count,
        1UL,
        common::errors::InvalidArgument("The count of barrier must not be less "
                                        "than 1. But received count = %d.",
                                        count));
    int ret = pthread_barrier_init(&_barrier, NULL, count);
    PADDLE_ENFORCE_EQ(
        0UL,
        ret,
        common::errors::InvalidArgument(
            "Fail to initialize the barrier with error code %d.", ret));
#endif
  }
  ~Barrier() noexcept(false) {
#ifdef _LINUX
    int ret = pthread_barrier_destroy(&_barrier);
    PADDLE_ENFORCE_EQ(
        0,
        ret,
        common::errors::PreconditionNotMet(
            "[error info] the result of "
            "pthread_barrier_destroy(&_barrier) should be zero.\n "
            "[result info] The value of current result is %d.",
            ret));
#endif
  }
  void reset(int count) {
#ifdef _LINUX
    PADDLE_ENFORCE_GE(count,
                      1UL,
                      common::errors::InvalidArgument(
                          "The count of reset must not be less than "
                          "1. But received count = %d.",
                          count));
    int ret = pthread_barrier_destroy(&_barrier);
    PADDLE_ENFORCE_EQ(
        0UL,
        ret,
        common::errors::InvalidArgument(
            "Fail to destroy the barrier with error code %d.", ret));
    ret = pthread_barrier_init(&_barrier, NULL, count);
    PADDLE_ENFORCE_EQ(
        0UL,
        ret,
        common::errors::InvalidArgument(
            "Fail to initialize the barrier with error code %d.", ret));
#endif
  }

  void wait() {
#ifdef _LINUX
    int err = pthread_barrier_wait(&_barrier);
    err = pthread_barrier_wait(&_barrier);
    PADDLE_ENFORCE_EQ(
        true,
        (err == 0 || err == PTHREAD_BARRIER_SERIAL_THREAD),
        common::errors::InvalidArgument(
            "pthread_barrier_wait failed. Expected err to be 0 or "
            "PTHREAD_BARRIER_SERIAL_THREAD, but got %d. This indicates a "
            "problem with the pthread barrier synchronization.",
            err));
#endif
  }

 private:
#ifdef _LINUX
  pthread_barrier_t _barrier;
#endif
};
// Call func(args...). If interrupted by signal, recall the function.
template <class FUNC, class... ARGS>
auto ignore_signal_call(FUNC &&func, ARGS &&...args) ->
    typename std::result_of<FUNC(ARGS...)>::type {
  for (;;) {
    auto err = func(args...);

    if (err < 0 && errno == EINTR) {
      LOG(INFO) << "Signal is caught. Ignored.";
      continue;
    }
    return err;
  }
}
class Semaphore {
 public:
  Semaphore() {
#ifdef _LINUX
    int ret = sem_init(&_sem, 0, 0);
    PADDLE_ENFORCE_EQ(
        0UL,
        ret,
        common::errors::InvalidArgument(
            "Fail to initialize the semaphore with error code %d.", ret));
#endif
  }
  ~Semaphore() noexcept(false) {
#ifdef _LINUX
    int ret = sem_destroy(&_sem);
    PADDLE_ENFORCE_EQ(
        0,
        ret,
        common::errors::PreconditionNotMet(
            "[error info] the result of sem_destroy(&_sem) should be zero.\n"
            "[result info] The value of current result is %d.",
            ret));
#endif
  }
  void post() {
#ifdef _LINUX
    int ret = sem_post(&_sem);
    PADDLE_ENFORCE_EQ(
        0UL,
        ret,
        common::errors::InvalidArgument(
            "Fail to post the semaphore with error code %d.", ret));
#endif
  }
  void wait() {
#ifdef _LINUX
    int ret = ignore_signal_call(sem_wait, &_sem);
    PADDLE_ENFORCE_EQ(0UL,
                      ret,
                      common::errors::InvalidArgument(
                          "Fail to ignore signal call to wait for "
                          "semaphore with error code %d.",
                          ret));
#endif
  }
  bool try_wait() {
    int err = 0;
#ifdef _LINUX
    err = ignore_signal_call(sem_trywait, &_sem);
    PADDLE_ENFORCE_EQ(true,
                      (err == 0 || errno == EAGAIN),
                      common::errors::InvalidArgument(
                          "Call to sem_trywait failed. Expected err to be 0 or "
                          "errno to be EAGAIN, but got err=%d and errno=%d. "
                          "This indicates a problem with semaphore operation.",
                          err,
                          errno));
#endif
    return err == 0;
  }

 private:
#ifdef _LINUX
  sem_t _sem;
#endif
};
class WaitGroup {
 public:
  WaitGroup() {}
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    counter_ = 0;
    cond_.notify_all();
  }
  void add(int delta) {
    if (delta == 0) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    counter_ += delta;
    if (counter_ == 0) {
      cond_.notify_all();
    }
  }
  void done() { add(-1); }
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);

    while (counter_ != 0) {
      cond_.wait(lock);
    }
  }
  int count(void) {
    std::unique_lock<std::mutex> lock(mutex_);
    return counter_;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cond_;
  int counter_ = 0;
};
}  // namespace framework
}  // namespace paddle
