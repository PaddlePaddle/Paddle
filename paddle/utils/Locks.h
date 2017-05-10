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

#pragma once

#include <pthread.h>
#include <sys/time.h>
#include <condition_variable>
#include <mutex>

#include "Common.h"

namespace paddle {

/**
 * A simple read-write lock.
 * The RWlock allows a number of readers or at most one writer
 * at any point in time.
 * The RWlock disable copy.
 *
 * Lock:
 *
 * Use lock() to lock on write mode, no other thread can get it
 * until unlock.
 *
 * Use lock_shared() to lock on read mode, other thread can get
 * it by using the same method lock_shared().
 *
 * Unlock:
 *
 * Use unlock() to unlock the lock.
 */
class RWLock {
public:
  RWLock() { pthread_rwlock_init(&rwlock_, NULL); }
  ~RWLock() { pthread_rwlock_destroy(&rwlock_); }
  RWLock(const RWLock&) = delete;
  RWLock& operator=(const RWLock&) = delete;

  /**
   * @brief lock on write mode.
   * @note the method will block the thread, if failed to get the lock.
   */
  // std::mutex interface
  void lock() { pthread_rwlock_wrlock(&rwlock_); }
  /**
   * @brief lock on read mode.
   * @note if another thread is writing, it can't get the lock,
   * and will block the thread.
   */
  void lock_shared() { pthread_rwlock_rdlock(&rwlock_); }
  void unlock() { pthread_rwlock_unlock(&rwlock_); }

protected:
  pthread_rwlock_t rwlock_;
};

/**
 * The ReadLockGuard is a read mode RWLock
 * using RAII management mechanism.
 */
class ReadLockGuard {
public:
  /**
   * @brief Construct Function. Lock on rwlock in read mode.
   */
  explicit ReadLockGuard(RWLock& rwlock) : rwlock_(&rwlock) {
    rwlock_->lock_shared();
  }

  /**
   * @brief Destruct Function.
   * @note This method just unlock the read mode rwlock,
   * won't destroy the lock.
   */
  ~ReadLockGuard() { rwlock_->unlock(); }

protected:
  RWLock* rwlock_;
};

/**
 * A simple wrapper for spin lock.
 * The lock() method of SpinLock is busy-waiting
 * which means it will keep trying to lock until lock on successfully.
 * The SpinLock disable copy.
 */
class SpinLockPrivate;
class SpinLock {
public:
  DISABLE_COPY(SpinLock);
  SpinLock();
  ~SpinLock();

  // std::mutext interface
  void lock();
  void unlock();

private:
  SpinLockPrivate* m;
};

/**
 * A simple wapper of semaphore which can only be shared in the same process.
 */
class SemaphorePrivate;
class Semaphore {
public:
  //! Disable copy & assign
  Semaphore(const Semaphore& other) = delete;
  Semaphore& operator=(const Semaphore&& other) = delete;

  //! Enable move.
  Semaphore(Semaphore&& other) : m(std::move(other.m)) {}

public:
  /**
   * @brief Construct Function.
   * @param[in] initValue the initial value of the
   * semaphore, default 0.
   */
  explicit Semaphore(int initValue = 0);

  ~Semaphore();

  /**
   * @brief The same as wait(), except if the decrement can not
   * be performed until ts return false install of blocking.
   * @param[in] ts an absolute timeout in seconds and nanoseconds
   * since the Epoch 1970-01-01 00:00:00 +0000(UTC).
   * @return ture if the decrement proceeds before ts,
   * else return false.
   */
  bool timeWait(struct timespec* ts);

  /**
   * @brief decrement the semaphore. If the semaphore's value is 0, then call
   * blocks.
   */
  void wait();

  /**
   * @brief increment the semaphore. If the semaphore's value
   * greater than 0, wake up a thread blocked in wait().
   */
  void post();

private:
  SemaphorePrivate* m;
};

/**
 * A simple wrapper of thread barrier.
 * The ThreadBarrier disable copy.
 */
class ThreadBarrierPrivate;
class ThreadBarrier {
public:
  DISABLE_COPY(ThreadBarrier);

  /**
   * @brief Construct Function. Initialize the barrier should
   * wait for count threads in wait().
   */
  explicit ThreadBarrier(int count);
  ~ThreadBarrier();

  /**
   * @brief .
   * If there were count - 1 threads waiting before,
   * then wake up all the count - 1 threads and continue run together.
   * Else block the thread until waked by other thread .
   */
  void wait();

private:
  ThreadBarrierPrivate* m;
};

/**
 * A wrapper for condition variable with mutex.
 */
class LockedCondition : public std::condition_variable {
public:
  /**
   * @brief execute op and notify one thread which was blocked.
   * @param[in] op a thread can do something in op before notify.
   */
  template <class Op>
  void notify_one(Op op) {
    std::lock_guard<std::mutex> guard(mutex_);
    op();
    std::condition_variable::notify_one();
  }

  /**
   * @brief execute op and notify all the threads which were blocked.
   * @param[in] op a thread can do something in op before notify.
   */
  template <class Op>
  void notify_all(Op op) {
    std::lock_guard<std::mutex> guard(mutex_);
    op();
    std::condition_variable::notify_all();
  }

  /**
   * @brief wait until pred return ture.
   * @tparam Predicate c++ concepts, describes a function object
   * that takes a single iterator argument
   * that is dereferenced and used to
   * return a value testable as a bool.
   * @note pred shall not apply any non-constant function
   * through the dereferenced iterator.
   */
  template <class Predicate>
  void wait(Predicate pred) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::condition_variable::wait(lock, pred);
  }

  /**
   * @brief get mutex.
   */
  std::mutex* mutex() { return &mutex_; }

protected:
  std::mutex mutex_;
};

}  // namespace paddle
