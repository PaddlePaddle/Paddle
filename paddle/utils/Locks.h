/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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
#include <semaphore.h>
#include <sys/time.h>
#include <unistd.h>

#include <condition_variable>
#include <mutex>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#ifdef __APPLE__
#ifndef PTHREAD_BARRIER_H_
#define PTHREAD_BARRIER_H_

#include <pthread.h>
#include <errno.h>

typedef int pthread_barrierattr_t;
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int tripCount;
} pthread_barrier_t;


int pthread_barrier_init(pthread_barrier_t *barrier,
   const pthread_barrierattr_t *attr, unsigned int count) {
    if (count == 0) {
        errno = EINVAL;
        return -1;
    }
    if (pthread_mutex_init(&barrier->mutex, 0) < 0) {
        return -1;
    }
    if (pthread_cond_init(&barrier->cond, 0) < 0) {
        pthread_mutex_destroy(&barrier->mutex);
        return -1;
    }
    barrier->tripCount = count;
    barrier->count = 0;

    return 0;
}

int pthread_barrier_destroy(pthread_barrier_t *barrier) {
    pthread_cond_destroy(&barrier->cond);
    pthread_mutex_destroy(&barrier->mutex);
    return 0;
}

int pthread_barrier_wait(pthread_barrier_t *barrier) {
    pthread_mutex_lock(&barrier->mutex);
    ++(barrier->count);
    if (barrier->count >= barrier->tripCount) {
        barrier->count = 0;
        pthread_cond_broadcast(&barrier->cond);
        pthread_mutex_unlock(&barrier->mutex);
        return 1;
    } else {
        pthread_cond_wait(&barrier->cond, &(barrier->mutex));
        pthread_mutex_unlock(&barrier->mutex);
        return 0;
    }
}

#endif  // PTHREAD_BARRIER_H_
typedef int pthread_spinlock_t;

int pthread_spin_init(pthread_spinlock_t *lock, int pshared) {
    __asm__ __volatile__("" ::: "memory");
    *lock = 0;
    return 0;
}

int pthread_spin_destroy(pthread_spinlock_t *lock) {
    return 0;
}

int pthread_spin_lock(pthread_spinlock_t *lock) {
    while (1) {
        int i;
        for (i=0; i < 10000; i++) {
            if (__sync_bool_compare_and_swap(lock, 0, 1)) {
                return 0;
            }
        }
        sched_yield();
    }
}

int pthread_spin_trylock(pthread_spinlock_t *lock) {
    if (__sync_bool_compare_and_swap(lock, 0, 1)) {
        return 0;
    }
    return EBUSY;
}

int pthread_spin_unlock(pthread_spinlock_t *lock) {
    __asm__ __volatile__("" ::: "memory");
    *lock = 0;
    return 0;
}
#endif



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
class SpinLock {
public:
  SpinLock() { pthread_spin_init(&lock_, 0); }
  ~SpinLock() { pthread_spin_destroy(&lock_); }
  SpinLock(const SpinLock&) = delete;
  SpinLock& operator=(const SpinLock&) = delete;

  // std::mutext interface
  void lock() { pthread_spin_lock(&lock_); }
  void unlock() { pthread_spin_unlock(&lock_); }

protected:
  pthread_spinlock_t lock_;
  char padding_[64 - sizeof(pthread_spinlock_t)];
};

/**
 * A simple wapper of semaphore which can only be shared in the same process.
 */

#ifdef __APPLE__
class Semaphore {
public:
    explicit Semaphore(int initValue = 0) {
        sem_ = dispatch_semaphore_create(initValue);
    }

    ~Semaphore() { dispatch_release(sem_); }
    bool timeWait(struct timespec* ts) {
        dispatch_time_t m = dispatch_walltime(ts, 0);
        return (0 == dispatch_semaphore_wait(sem_, m));
    }
    void wait() { dispatch_semaphore_wait(sem_, DISPATCH_TIME_FOREVER); }
    void post() { dispatch_semaphore_signal(sem_);}

protected:
 dispatch_semaphore_t sem_;
};
#else

class Semaphore {
public:
  /**
   * @brief Construct Function. 
   * @param[in] initValue the initial value of the 
   * semaphore, default 0.
   */
  explicit Semaphore(int initValue = 0) { sem_init(&sem_, 0, initValue); }

  ~Semaphore() { sem_destroy(&sem_); }

  /**
   * @brief The same as wait(), except if the decrement can not 
   * be performed until ts return false install of blocking.
   * @param[in] ts an absolute timeout in seconds and nanoseconds 
   * since the Epoch 1970-01-01 00:00:00 +0000(UTC).
   * @return ture if the decrement proceeds before ts, 
   * else return false.
   */
  bool timeWait(struct timespec* ts) { return (0 == sem_timedwait(&sem_, ts)); }

  /**
   * @brief decrement the semaphore. If the semaphore's value is 0, then call blocks.
   */
  void wait() { sem_wait(&sem_); }

  /**
   * @brief increment the semaphore. If the semaphore's value 
   * greater than 0, wake up a thread blocked in wait().
   */
  void post() { sem_post(&sem_); }

protected:
  sem_t sem_;
};

#endif

static_assert(sizeof(SpinLock) == 64, "Wrong padding");

/**
 * A simple wrapper of thread barrier.
 * The ThreadBarrier disable copy.
 */
class ThreadBarrier {
public:
  /**
   * @brief Construct Function. Initialize the barrier should
   * wait for count threads in wait().
   */
  explicit ThreadBarrier(int count) {
    pthread_barrier_init(&barrier_, NULL, count);
  }
  ~ThreadBarrier() { pthread_barrier_destroy(&barrier_); }
  ThreadBarrier(const ThreadBarrier&) = delete;
  ThreadBarrier& operator=(const ThreadBarrier&) = delete;

  /**
   * @brief . 
   * If there were count - 1 threads waiting before, 
   * then wake up all the count - 1 threads and continue run together. 
   * Else block the thread until waked by other thread .
   */
  void wait() { pthread_barrier_wait(&barrier_); }

protected:
  pthread_barrier_t barrier_;
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
