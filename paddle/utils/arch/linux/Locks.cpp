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
  inline SpinLockPrivate() { pthread_spin_init(&lock_, 0); }
  inline ~SpinLockPrivate() { pthread_spin_destroy(&lock_); }
  pthread_spinlock_t lock_;
  char padding_[64 - sizeof(pthread_spinlock_t)];
};

SpinLock::SpinLock() : m(new SpinLockPrivate()) {}

SpinLock::~SpinLock() { delete m; }

void SpinLock::lock() { pthread_spin_lock(&m->lock_); }

void SpinLock::unlock() { pthread_spin_unlock(&m->lock_); }

class ThreadBarrierPrivate {
public:
  pthread_barrier_t barrier_;
};

ThreadBarrier::ThreadBarrier(int count) : m(new ThreadBarrierPrivate()) {
  pthread_barrier_init(&m->barrier_, nullptr, count);
}

ThreadBarrier::~ThreadBarrier() {
  pthread_barrier_destroy(&m->barrier_);
  delete m;
}

void ThreadBarrier::wait() { pthread_barrier_wait(&m->barrier_); }

}  // namespace paddle
