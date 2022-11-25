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

#include <pthread.h>
#include <semaphore.h>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
class Barrier {
 public:
  explicit Barrier(int count = 1) {
    CHECK(count >= 1);
    CHECK(0 == pthread_barrier_init(&_barrier, NULL, count));
  }
  ~Barrier() { CHECK(0 == pthread_barrier_destroy(&_barrier)); }
  void reset(int count) {
    CHECK(count >= 1);
    CHECK(0 == pthread_barrier_destroy(&_barrier));
    CHECK(0 == pthread_barrier_init(&_barrier, NULL, count));
  }
  void wait() {
    int err = pthread_barrier_wait(&_barrier);
    CHECK((err = pthread_barrier_wait(&_barrier),
           err == 0 || err == PTHREAD_BARRIER_SERIAL_THREAD));
  }

 private:
  pthread_barrier_t _barrier;
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
  Semaphore() { CHECK(0 == sem_init(&_sem, 0, 0)); }
  ~Semaphore() { CHECK(0 == sem_destroy(&_sem)); }
  void post() { CHECK(0 == sem_post(&_sem)); }
  void wait() { CHECK(0 == ignore_signal_call(sem_wait, &_sem)); }
  bool try_wait() {
    int err = 0;
    CHECK((err = ignore_signal_call(sem_trywait, &_sem),
           err == 0 || errno == EAGAIN));
    return err == 0;
  }

 private:
  sem_t _sem;
};
}  // namespace framework
}  // namespace paddle
