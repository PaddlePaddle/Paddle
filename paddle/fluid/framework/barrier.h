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

#ifdef __LINUX__
#include <pthread.h>
#include <semaphore.h>
#endif
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
class Barrier {
 public:
  explicit Barrier(int count = 1) {
#ifdef __LINUX__
    CHECK_GE(count, 1);
    CHECK_EQ(pthread_barrier_init(&_barrier, NULL, count), 0);
#endif
  }

  ~Barrier() {
#ifdef __LINUX__
    CHECK_EQ(pthread_barrier_destroy(&_barrier), 0);
#endif
  }

  void reset(int count) {
#ifdef __LINUX__
    CHECK_GE(count, 1);
    CHECK_EQ(pthread_barrier_destroy(&_barrier), 0);
    CHECK_EQ(pthread_barrier_init(&_barrier, NULL, count), 0);
#endif
  }

  void wait() {
#ifdef __LINUX__
    int err = pthread_barrier_wait(&_barrier);
    if (err != 0 && err != PTHREAD_BARRIER_SERIAL_THREAD)) {
      CHECK_EQ(1, 0);
    }
#endif
  }

 private:
#ifdef __LINUX__
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
#ifdef __LINUX__
    CHECK_EQ(sem_init(&_sem, 0, 0), 0);
#endif
  }
  ~Semaphore() {
#ifdef __LINUX__
    CHECK_EQ(sem_destroy(&_sem), 0);
#endif
  }
  void post() {
#ifdef __LINUX__
    CHECK_EQ(sem_post(&_sem), 0);
#endif
  }
  void wait() {
#ifdef __LINUX__
    CHECK_EQ(ignore_signal_call(sem_wait, &_sem), 0);
#endif
  }
  bool try_wait() {
    int err = 0;
#ifdef __LINUX__
    CHECK((err = ignore_signal_call(sem_trywait, &_sem),
           err == 0 || errno == EAGAIN));
#endif
    return err == 0;
  }

 private:
#ifdef __LINUX__
  sem_t _sem;
#endif
};
}  // namespace framework
}  // namespace paddle
