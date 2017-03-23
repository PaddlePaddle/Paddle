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

namespace paddle {
class SemaphorePrivate {
public:
  sem_t sem;
};

Semaphore::Semaphore(int initValue) : m(new SemaphorePrivate()) {
  sem_init(&m->sem, 0, initValue);
}

Semaphore::~Semaphore() { sem_destroy(&m->sem); }

bool Semaphore::timeWait(struct timespec* ts) {
  return (0 == sem_timedwait(&m->sem, ts));
}

void Semaphore::wait() { sem_wait(&m->sem); }

void Semaphore::post() { sem_post(&m->sem); }

class SpinLockPrivate {
public:
  inline SpinLockPrivate() {
#ifndef __ANDROID__
    pthread_spin_init(&lock_, 0);
#else
    lock_ = 0;
#endif
  }
  inline ~SpinLockPrivate() {
#ifndef __ANDROID__
    pthread_spin_destroy(&lock_);
#endif
  }
#ifndef __ANDROID__
  pthread_spinlock_t lock_;
#else
  unsigned long lock_;
#endif
  char padding_[64 - sizeof(lock_)];
};

SpinLock::SpinLock() : m(new SpinLockPrivate()) {}

SpinLock::~SpinLock() { delete m; }

void SpinLock::lock() {
#ifndef __ANDROID__
  pthread_spin_lock(&m->lock_);
#endif
}

void SpinLock::unlock() {
#ifndef __ANDROID__
  pthread_spin_unlock(&m->lock_);
#endif
}

class ThreadBarrierPrivate {
public:
#ifndef __ANDROID__
  pthread_barrier_t barrier_;
#else
  unsigned long barrier_;
#endif
};

ThreadBarrier::ThreadBarrier(int count) : m(new ThreadBarrierPrivate()) {
#ifndef __ANDROID__
  pthread_barrier_init(&m->barrier_, nullptr, count);
#endif
}

ThreadBarrier::~ThreadBarrier() {
#ifndef __ANDROID__
  pthread_barrier_destroy(&m->barrier_);
#endif
  delete m;
}

void ThreadBarrier::wait() {
#ifndef __ANDROID__
  pthread_barrier_wait(&m->barrier_);
#endif
}

}  // namespace paddle
