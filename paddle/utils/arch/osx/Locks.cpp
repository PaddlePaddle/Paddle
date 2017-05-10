/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/utils/Locks.h"
#include <dispatch/dispatch.h>
#include <libkern/OSAtomic.h>
#include <atomic>
#include "paddle/utils/Logging.h"

namespace paddle {

class SemaphorePrivate {
public:
  ~SemaphorePrivate() { dispatch_release(sem); }

  dispatch_semaphore_t sem;
};

Semaphore::Semaphore(int initValue) : m(new SemaphorePrivate()) {
  m->sem = dispatch_semaphore_create(initValue);
}

Semaphore::~Semaphore() { delete m; }

bool Semaphore::timeWait(timespec *ts) {
  dispatch_time_t tm = dispatch_walltime(ts, 0);
  return (0 == dispatch_semaphore_wait(m->sem, tm));
}

void Semaphore::wait() {
  dispatch_semaphore_wait(m->sem, DISPATCH_TIME_FOREVER);
}

void Semaphore::post() { dispatch_semaphore_signal(m->sem); }

class SpinLockPrivate {
public:
  std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
  char padding_[64 - sizeof(lock_)];  // Padding to cache line size
};

SpinLock::SpinLock() : m(new SpinLockPrivate()) {}
SpinLock::~SpinLock() { delete m; }

void SpinLock::lock() {
  while (m->lock_.test_and_set(std::memory_order_acquire)) {
  }
}

void SpinLock::unlock() { m->lock_.clear(std::memory_order_release); }

class ThreadBarrierPrivate {
public:
  pthread_mutex_t mutex_;
  pthread_cond_t cond_;
  int count_;
  int tripCount_;

  inline explicit ThreadBarrierPrivate(int cnt) : count_(0), tripCount_(cnt) {
    CHECK_NE(cnt, 0);
    CHECK_GE(pthread_mutex_init(&mutex_, 0), 0);
    CHECK_GE(pthread_cond_init(&cond_, 0), 0);
  }

  inline ~ThreadBarrierPrivate() {
    pthread_cond_destroy(&cond_);
    pthread_mutex_destroy(&mutex_);
  }

  /**
   * @brief wait
   * @return true if the last wait
   */
  inline bool wait() {
    pthread_mutex_lock(&mutex_);
    ++count_;
    if (count_ >= tripCount_) {
      count_ = 0;
      pthread_cond_broadcast(&cond_);
      pthread_mutex_unlock(&mutex_);
      return true;
    } else {
      pthread_cond_wait(&cond_, &mutex_);
      pthread_mutex_unlock(&mutex_);
      return false;
    }
  }
};

ThreadBarrier::ThreadBarrier(int count) : m(new ThreadBarrierPrivate(count)) {}
ThreadBarrier::~ThreadBarrier() { delete m; }
void ThreadBarrier::wait() { m->wait(); }

}  // namespace paddle
