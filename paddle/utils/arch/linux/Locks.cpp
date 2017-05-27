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
#include <semaphore.h>
#include <unistd.h>
#include "paddle/utils/Logging.h"

namespace paddle {
class SemaphorePrivate {
public:
  sem_t sem;
};

Semaphore::Semaphore(int initValue) : m(new SemaphorePrivate()) {
  sem_init(&m->sem, 0, initValue);
}

Semaphore::~Semaphore() {
  sem_destroy(&m->sem);
  delete m;
}

bool Semaphore::timeWait(struct timespec* ts) {
  return (0 == sem_timedwait(&m->sem, ts));
}

void Semaphore::wait() { sem_wait(&m->sem); }

void Semaphore::post() { sem_post(&m->sem); }

#ifdef PADDLE_USE_PTHREAD_SPINLOCK

class SpinLockPrivate {
public:
  inline SpinLockPrivate() { pthread_spin_init(&lock_, 0); }
  inline ~SpinLockPrivate() { pthread_spin_destroy(&lock_); }

  inline void lock() { pthread_spin_lock(&lock_); }
  inline void unlock() { pthread_spin_unlock(&lock_); }

  pthread_spinlock_t lock_;
  char padding_[64 - sizeof(pthread_spinlock_t)];
};

#else
// clang-format off
#include <cstddef>
#include <atomic>
// clang-format on

class SpinLockPrivate {
public:
  inline void lock() {
    while (lock_.test_and_set(std::memory_order_acquire)) {
    }
  }
  inline void unlock() { lock_.clear(std::memory_order_release); }

  std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
  char padding_[64 - sizeof(lock_)];  // Padding to cache line size
};

#endif

SpinLock::SpinLock() : m(new SpinLockPrivate()) {}
SpinLock::~SpinLock() { delete m; }
void SpinLock::lock() { m->lock(); }
void SpinLock::unlock() { m->unlock(); }

#ifdef PADDLE_USE_PTHREAD_BARRIER

class ThreadBarrierPrivate {
public:
  pthread_barrier_t barrier_;

  inline explicit ThreadBarrierPrivate(int count) {
    pthread_barrier_init(&barrier_, nullptr, count);
  }

  inline ~ThreadBarrierPrivate() { pthread_barrier_destroy(&barrier_); }

  inline void wait() { pthread_barrier_wait(&barrier_); }
};

#else

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

#endif

ThreadBarrier::ThreadBarrier(int count) : m(new ThreadBarrierPrivate(count)) {}
ThreadBarrier::~ThreadBarrier() { delete m; }
void ThreadBarrier::wait() { m->wait(); }

}  // namespace paddle
