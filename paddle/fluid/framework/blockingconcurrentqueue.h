// Provides an efficient blocking version of moodycamel::ConcurrentQueue.
// ©2015-2016 Cameron Desrochers. Distributed under the terms of the simplified
// BSD license, available at the top of concurrentqueue.h.
// Uses Jeff Preshing's semaphore implementation (under the terms of its
// separate zlib license, embedded below).

#pragma once

#include <cerrno>
#include <chrono>
#include <ctime>
#include <memory>
#include <type_traits>
#include "concurrentqueue.h"

#if defined(_WIN32)
// Avoid including windows.h in a header; we only need a handful of
// items, so we'll redeclare them here (this is relatively safe since
// the API generally has to remain stable between Windows versions).
// I know this is an ugly hack but it still beats polluting the global
// namespace with thousands of generic names or adding a .cpp for nothing.
extern "C" {
struct _SECURITY_ATTRIBUTES;
__declspec(dllimport) void* __stdcall CreateSemaphoreW(
    _SECURITY_ATTRIBUTES* lpSemaphoreAttributes, long lInitialCount,
    long lMaximumCount, const wchar_t* lpName);
__declspec(dllimport) int __stdcall CloseHandle(void* hObject);
__declspec(dllimport) unsigned long __stdcall WaitForSingleObject(
    void* hHandle, unsigned long dwMilliseconds);
__declspec(dllimport) int __stdcall ReleaseSemaphore(void* hSemaphore,
                                                     long lReleaseCount,
                                                     long* lpPreviousCount);
}
#elif defined(__MACH__)
#include <mach/mach.h>
#elif defined(__unix__)
#include <semaphore.h>
#endif

namespace moodycamel {
namespace details {
// Code in the mpmc_sema namespace below is an adaptation of Jeff Preshing's
// portable + lightweight semaphore implementations, originally from
// https://github.com/preshing/cpp11-on-multicore/blob/master/common/sema.h
// LICENSE:
// Copyright (c) 2015 Jeff Preshing
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//	claim that you wrote the original software. If you use this software
//	in a product, an acknowledgement in the product documentation would be
//	appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//	misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
namespace mpmc_sema {
#if defined(_WIN32)
class Semaphore {
 private:
  void* m_hSema;

  Semaphore(const Semaphore& other) MOODYCAMEL_DELETE_FUNCTION;
  Semaphore& operator=(const Semaphore& other) MOODYCAMEL_DELETE_FUNCTION;

 public:
  Semaphore(int initialCount = 0) {
    assert(initialCount >= 0);
    const long maxLong = 0x7fffffff;
    m_hSema = CreateSemaphoreW(nullptr, initialCount, maxLong, nullptr);
  }

  ~Semaphore() { CloseHandle(m_hSema); }

  void wait() {
    const unsigned long infinite = 0xffffffff;
    WaitForSingleObject(m_hSema, infinite);
  }

  bool try_wait() {
    const unsigned long RC_WAIT_TIMEOUT = 0x00000102;
    return WaitForSingleObject(m_hSema, 0) != RC_WAIT_TIMEOUT;
  }

  bool timed_wait(std::uint64_t usecs) {
    const unsigned long RC_WAIT_TIMEOUT = 0x00000102;
    return WaitForSingleObject(m_hSema, (unsigned long)(usecs / 1000)) !=
           RC_WAIT_TIMEOUT;
  }

  void signal(int count = 1) { ReleaseSemaphore(m_hSema, count, nullptr); }
};
#elif defined(__MACH__)
//---------------------------------------------------------
// Semaphore (Apple iOS and OSX)
// Can't use POSIX semaphores due to
// http://lists.apple.com/archives/darwin-kernel/2009/Apr/msg00010.html
//---------------------------------------------------------
class Semaphore {
 private:
  semaphore_t m_sema;

  Semaphore(const Semaphore& other) MOODYCAMEL_DELETE_FUNCTION;
  Semaphore& operator=(const Semaphore& other) MOODYCAMEL_DELETE_FUNCTION;

 public:
  Semaphore(int initialCount = 0) {
    assert(initialCount >= 0);
    semaphore_create(mach_task_self(), &m_sema, SYNC_POLICY_FIFO, initialCount);
  }

  ~Semaphore() { semaphore_destroy(mach_task_self(), m_sema); }

  void wait() { semaphore_wait(m_sema); }

  bool try_wait() { return timed_wait(0); }

  bool timed_wait(std::uint64_t timeout_usecs) {
    mach_timespec_t ts;
    ts.tv_sec = static_cast<unsigned int>(timeout_usecs / 1000000);
    ts.tv_nsec = (timeout_usecs % 1000000) * 1000;

    // added in OSX 10.10:
    // https://developer.apple.com/library/prerelease/mac/documentation/General/Reference/APIDiffsMacOSX10_10SeedDiff/modules/Darwin.html
    kern_return_t rc = semaphore_timedwait(m_sema, ts);

    return rc != KERN_OPERATION_TIMED_OUT && rc != KERN_ABORTED;
  }

  void signal() { semaphore_signal(m_sema); }

  void signal(int count) {
    while (count-- > 0) {
      semaphore_signal(m_sema);
    }
  }
};
#elif defined(__unix__)
//---------------------------------------------------------
// Semaphore (POSIX, Linux)
//---------------------------------------------------------
class Semaphore {
 private:
  sem_t m_sema;

  Semaphore(const Semaphore& other) MOODYCAMEL_DELETE_FUNCTION;
  Semaphore& operator=(const Semaphore& other) MOODYCAMEL_DELETE_FUNCTION;

 public:
  Semaphore(int initialCount = 0) {
    assert(initialCount >= 0);
    sem_init(&m_sema, 0, initialCount);
  }

  ~Semaphore() { sem_destroy(&m_sema); }

  void wait() {
    // http://stackoverflow.com/questions/2013181/gdb-causes-sem-wait-to-fail-with-eintr-error
    int rc;
    do {
      rc = sem_wait(&m_sema);
    } while (rc == -1 && errno == EINTR);
  }

  bool try_wait() {
    int rc;
    do {
      rc = sem_trywait(&m_sema);
    } while (rc == -1 && errno == EINTR);
    return !(rc == -1 && errno == EAGAIN);
  }

  bool timed_wait(std::uint64_t usecs) {
    struct timespec ts;
    const int usecs_in_1_sec = 1000000;
    const int nsecs_in_1_sec = 1000000000;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += usecs / usecs_in_1_sec;
    ts.tv_nsec += (usecs % usecs_in_1_sec) * 1000;
    // sem_timedwait bombs if you have more than 1e9 in tv_nsec
    // so we have to clean things up before passing it in
    if (ts.tv_nsec >= nsecs_in_1_sec) {
      ts.tv_nsec -= nsecs_in_1_sec;
      ++ts.tv_sec;
    }

    int rc;
    do {
      rc = sem_timedwait(&m_sema, &ts);
    } while (rc == -1 && errno == EINTR);
    return !(rc == -1 && errno == ETIMEDOUT);
  }

  void signal() { sem_post(&m_sema); }

  void signal(int count) {
    while (count-- > 0) {
      sem_post(&m_sema);
    }
  }
};
#else
#error Unsupported platform! (No semaphore wrapper available)
#endif

//---------------------------------------------------------
// LightweightSemaphore
//---------------------------------------------------------
class LightweightSemaphore {
 public:
  typedef std::make_signed<std::size_t>::type ssize_t;

 private:
  std::atomic<ssize_t> m_count;
  Semaphore m_sema;

  bool waitWithPartialSpinning(std::int64_t timeout_usecs = -1) {
    ssize_t oldCount;
    // Is there a better way to set the initial spin count?
    // If we lower it to 1000, testBenaphore becomes 15x slower on my Core
    // i7-5930K Windows PC,
    // as threads start hitting the kernel semaphore.
    int spin = 10000;
    while (--spin >= 0) {
      oldCount = m_count.load(std::memory_order_relaxed);
      if ((oldCount > 0) &&
          m_count.compare_exchange_strong(oldCount, oldCount - 1,
                                          std::memory_order_acquire,
                                          std::memory_order_relaxed))
        return true;
      std::atomic_signal_fence(std::memory_order_acquire);  // Prevent the
                                                            // compiler from
                                                            // collapsing the
                                                            // loop.
    }
    oldCount = m_count.fetch_sub(1, std::memory_order_acquire);
    if (oldCount > 0) return true;
    if (timeout_usecs < 0) {
      m_sema.wait();
      return true;
    }
    if (m_sema.timed_wait((std::uint64_t)timeout_usecs)) return true;
    // At this point, we've timed out waiting for the semaphore, but the
    // count is still decremented indicating we may still be waiting on
    // it. So we have to re-adjust the count, but only if the semaphore
    // wasn't signaled enough times for us too since then. If it was, we
    // need to release the semaphore too.
    while (true) {
      oldCount = m_count.load(std::memory_order_acquire);
      if (oldCount >= 0 && m_sema.try_wait()) return true;
      if (oldCount < 0 &&
          m_count.compare_exchange_strong(oldCount, oldCount + 1,
                                          std::memory_order_relaxed,
                                          std::memory_order_relaxed))
        return false;
    }
  }

  ssize_t waitManyWithPartialSpinning(ssize_t max,
                                      std::int64_t timeout_usecs = -1) {
    assert(max > 0);
    ssize_t oldCount;
    int spin = 10000;
    while (--spin >= 0) {
      oldCount = m_count.load(std::memory_order_relaxed);
      if (oldCount > 0) {
        ssize_t newCount = oldCount > max ? oldCount - max : 0;
        if (m_count.compare_exchange_strong(oldCount, newCount,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed))
          return oldCount - newCount;
      }
      std::atomic_signal_fence(std::memory_order_acquire);
    }
    oldCount = m_count.fetch_sub(1, std::memory_order_acquire);
    if (oldCount <= 0) {
      if (timeout_usecs < 0)
        m_sema.wait();
      else if (!m_sema.timed_wait((std::uint64_t)timeout_usecs)) {
        while (true) {
          oldCount = m_count.load(std::memory_order_acquire);
          if (oldCount >= 0 && m_sema.try_wait()) break;
          if (oldCount < 0 &&
              m_count.compare_exchange_strong(oldCount, oldCount + 1,
                                              std::memory_order_relaxed,
                                              std::memory_order_relaxed))
            return 0;
        }
      }
    }
    if (max > 1) return 1 + tryWaitMany(max - 1);
    return 1;
  }

 public:
  LightweightSemaphore(ssize_t initialCount = 0) : m_count(initialCount) {
    assert(initialCount >= 0);
  }

  bool tryWait() {
    ssize_t oldCount = m_count.load(std::memory_order_relaxed);
    while (oldCount > 0) {
      if (m_count.compare_exchange_weak(oldCount, oldCount - 1,
                                        std::memory_order_acquire,
                                        std::memory_order_relaxed))
        return true;
    }
    return false;
  }

  void wait() {
    if (!tryWait()) waitWithPartialSpinning();
  }

  bool wait(std::int64_t timeout_usecs) {
    return tryWait() || waitWithPartialSpinning(timeout_usecs);
  }

  // Acquires between 0 and (greedily) max, inclusive
  ssize_t tryWaitMany(ssize_t max) {
    assert(max >= 0);
    ssize_t oldCount = m_count.load(std::memory_order_relaxed);
    while (oldCount > 0) {
      ssize_t newCount = oldCount > max ? oldCount - max : 0;
      if (m_count.compare_exchange_weak(oldCount, newCount,
                                        std::memory_order_acquire,
                                        std::memory_order_relaxed))
        return oldCount - newCount;
    }
    return 0;
  }

  // Acquires at least one, and (greedily) at most max
  ssize_t waitMany(ssize_t max, std::int64_t timeout_usecs) {
    assert(max >= 0);
    ssize_t result = tryWaitMany(max);
    if (result == 0 && max > 0)
      result = waitManyWithPartialSpinning(max, timeout_usecs);
    return result;
  }

  ssize_t waitMany(ssize_t max) {
    ssize_t result = waitMany(max, -1);
    assert(result > 0);
    return result;
  }

  void signal(ssize_t count = 1) {
    assert(count >= 0);
    ssize_t oldCount = m_count.fetch_add(count, std::memory_order_release);
    ssize_t toRelease = -oldCount < count ? -oldCount : count;
    if (toRelease > 0) {
      m_sema.signal((int)toRelease);
    }
  }

  ssize_t availableApprox() const {
    ssize_t count = m_count.load(std::memory_order_relaxed);
    return count > 0 ? count : 0;
  }
};
}  // end namespace mpmc_sema
}  // end namespace details

// This is a blocking version of the queue. It has an almost identical interface
// to
// the normal non-blocking version, with the addition of various wait_dequeue()
// methods
// and the removal of producer-specific dequeue methods.
template <typename T, typename Traits = ConcurrentQueueDefaultTraits>
class BlockingConcurrentQueue {
 private:
  typedef ::moodycamel::ConcurrentQueue<T, Traits> ConcurrentQueue;
  typedef details::mpmc_sema::LightweightSemaphore LightweightSemaphore;

 public:
  typedef typename ConcurrentQueue::producer_token_t producer_token_t;
  typedef typename ConcurrentQueue::consumer_token_t consumer_token_t;

  typedef typename ConcurrentQueue::index_t index_t;
  typedef typename ConcurrentQueue::size_t size_t;
  typedef typename std::make_signed<size_t>::type ssize_t;

  static const size_t BLOCK_SIZE = ConcurrentQueue::BLOCK_SIZE;
  static const size_t EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD =
      ConcurrentQueue::EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD;
  static const size_t EXPLICIT_INITIAL_INDEX_SIZE =
      ConcurrentQueue::EXPLICIT_INITIAL_INDEX_SIZE;
  static const size_t IMPLICIT_INITIAL_INDEX_SIZE =
      ConcurrentQueue::IMPLICIT_INITIAL_INDEX_SIZE;
  static const size_t INITIAL_IMPLICIT_PRODUCER_HASH_SIZE =
      ConcurrentQueue::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE;
  static const std::uint32_t EXPLICIT_CONSUMER_CONSUMPTION_QUOTA_BEFORE_ROTATE =
      ConcurrentQueue::EXPLICIT_CONSUMER_CONSUMPTION_QUOTA_BEFORE_ROTATE;
  static const size_t MAX_SUBQUEUE_SIZE = ConcurrentQueue::MAX_SUBQUEUE_SIZE;

 public:
  // Creates a queue with at least `capacity` element slots; note that the
  // actual number of elements that can be inserted without additional memory
  // allocation depends on the number of producers and the block size (e.g. if
  // the block size is equal to `capacity`, only a single block will be
  // allocated
  // up-front, which means only a single producer will be able to enqueue
  // elements
  // without an extra allocation -- blocks aren't shared between producers).
  // This method is not thread safe -- it is up to the user to ensure that the
  // queue is fully constructed before it starts being used by other threads
  // (this
  // includes making the memory effects of construction visible, possibly with a
  // memory barrier).
  explicit BlockingConcurrentQueue(size_t capacity = 6 * BLOCK_SIZE)
      : inner(capacity),
        sema(create<LightweightSemaphore>(),
             &BlockingConcurrentQueue::template destroy<LightweightSemaphore>) {
    assert(reinterpret_cast<ConcurrentQueue*>((BlockingConcurrentQueue*)1) ==
               &((BlockingConcurrentQueue*)1)->inner &&
           "BlockingConcurrentQueue must have ConcurrentQueue as its first "
           "member");
    if (!sema) {
      MOODYCAMEL_THROW(std::bad_alloc());
    }
  }

  BlockingConcurrentQueue(size_t minCapacity, size_t maxExplicitProducers,
                          size_t maxImplicitProducers)
      : inner(minCapacity, maxExplicitProducers, maxImplicitProducers),
        sema(create<LightweightSemaphore>(),
             &BlockingConcurrentQueue::template destroy<LightweightSemaphore>) {
    assert(reinterpret_cast<ConcurrentQueue*>((BlockingConcurrentQueue*)1) ==
               &((BlockingConcurrentQueue*)1)->inner &&
           "BlockingConcurrentQueue must have ConcurrentQueue as its first "
           "member");
    if (!sema) {
      MOODYCAMEL_THROW(std::bad_alloc());
    }
  }

  // Disable copying and copy assignment
  BlockingConcurrentQueue(BlockingConcurrentQueue const&)
      MOODYCAMEL_DELETE_FUNCTION;
  BlockingConcurrentQueue& operator=(BlockingConcurrentQueue const&)
      MOODYCAMEL_DELETE_FUNCTION;

  // Moving is supported, but note that it is *not* a thread-safe operation.
  // Nobody can use the queue while it's being moved, and the memory effects
  // of that move must be propagated to other threads before they can use it.
  // Note: When a queue is moved, its tokens are still valid but can only be
  // used with the destination queue (i.e. semantically they are moved along
  // with the queue itself).
  BlockingConcurrentQueue(BlockingConcurrentQueue&& other) MOODYCAMEL_NOEXCEPT
      : inner(std::move(other.inner)),
        sema(std::move(other.sema)) {}

  inline BlockingConcurrentQueue& operator=(BlockingConcurrentQueue&& other)
      MOODYCAMEL_NOEXCEPT {
    return swap_internal(other);
  }

  // Swaps this queue's state with the other's. Not thread-safe.
  // Swapping two queues does not invalidate their tokens, however
  // the tokens that were created for one queue must be used with
  // only the swapped queue (i.e. the tokens are tied to the
  // queue's movable state, not the object itself).
  inline void swap(BlockingConcurrentQueue& other) MOODYCAMEL_NOEXCEPT {
    swap_internal(other);
  }

 private:
  BlockingConcurrentQueue& swap_internal(BlockingConcurrentQueue& other) {
    if (this == &other) {
      return *this;
    }

    inner.swap(other.inner);
    sema.swap(other.sema);
    return *this;
  }

 public:
  // Enqueues a single item (by copying it).
  // Allocates memory if required. Only fails if memory allocation fails (or
  // implicit
  // production is disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE
  // is 0,
  // or Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
  // Thread-safe.
  inline bool enqueue(T const& item) {
    if ((details::likely)(inner.enqueue(item))) {
      sema->signal();
      return true;
    }
    return false;
  }

  // Enqueues a single item (by moving it, if possible).
  // Allocates memory if required. Only fails if memory allocation fails (or
  // implicit
  // production is disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE
  // is 0,
  // or Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
  // Thread-safe.
  inline bool enqueue(T&& item) {
    if ((details::likely)(inner.enqueue(std::move(item)))) {
      sema->signal();
      return true;
    }
    return false;
  }

  // Enqueues a single item (by copying it) using an explicit producer token.
  // Allocates memory if required. Only fails if memory allocation fails (or
  // Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
  // Thread-safe.
  inline bool enqueue(producer_token_t const& token, T const& item) {
    if ((details::likely)(inner.enqueue(token, item))) {
      sema->signal();
      return true;
    }
    return false;
  }

  // Enqueues a single item (by moving it, if possible) using an explicit
  // producer token.
  // Allocates memory if required. Only fails if memory allocation fails (or
  // Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
  // Thread-safe.
  inline bool enqueue(producer_token_t const& token, T&& item) {
    if ((details::likely)(inner.enqueue(token, std::move(item)))) {
      sema->signal();
      return true;
    }
    return false;
  }

  // Enqueues several items.
  // Allocates memory if required. Only fails if memory allocation fails (or
  // implicit production is disabled because
  // Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE
  // is 0, or Traits::MAX_SUBQUEUE_SIZE has been defined and would be
  // surpassed).
  // Note: Use std::make_move_iterator if the elements should be moved instead
  // of copied.
  // Thread-safe.
  template <typename It>
  inline bool enqueue_bulk(It itemFirst, size_t count) {
    if ((details::likely)(
            inner.enqueue_bulk(std::forward<It>(itemFirst), count))) {
      sema->signal((LightweightSemaphore::ssize_t)(ssize_t)count);
      return true;
    }
    return false;
  }

  // Enqueues several items using an explicit producer token.
  // Allocates memory if required. Only fails if memory allocation fails
  // (or Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
  // Note: Use std::make_move_iterator if the elements should be moved
  // instead of copied.
  // Thread-safe.
  template <typename It>
  inline bool enqueue_bulk(producer_token_t const& token, It itemFirst,
                           size_t count) {
    if ((details::likely)(
            inner.enqueue_bulk(token, std::forward<It>(itemFirst), count))) {
      sema->signal((LightweightSemaphore::ssize_t)(ssize_t)count);
      return true;
    }
    return false;
  }

  // Enqueues a single item (by copying it).
  // Does not allocate memory. Fails if not enough room to enqueue (or implicit
  // production is disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE
  // is 0).
  // Thread-safe.
  inline bool try_enqueue(T const& item) {
    if (inner.try_enqueue(item)) {
      sema->signal();
      return true;
    }
    return false;
  }

  // Enqueues a single item (by moving it, if possible).
  // Does not allocate memory (except for one-time implicit producer).
  // Fails if not enough room to enqueue (or implicit production is
  // disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE is 0).
  // Thread-safe.
  inline bool try_enqueue(T&& item) {
    if (inner.try_enqueue(std::move(item))) {
      sema->signal();
      return true;
    }
    return false;
  }

  // Enqueues a single item (by copying it) using an explicit producer token.
  // Does not allocate memory. Fails if not enough room to enqueue.
  // Thread-safe.
  inline bool try_enqueue(producer_token_t const& token, T const& item) {
    if (inner.try_enqueue(token, item)) {
      sema->signal();
      return true;
    }
    return false;
  }

  // Enqueues a single item (by moving it, if possible) using an explicit
  // producer token.
  // Does not allocate memory. Fails if not enough room to enqueue.
  // Thread-safe.
  inline bool try_enqueue(producer_token_t const& token, T&& item) {
    if (inner.try_enqueue(token, std::move(item))) {
      sema->signal();
      return true;
    }
    return false;
  }

  // Enqueues several items.
  // Does not allocate memory (except for one-time implicit producer).
  // Fails if not enough room to enqueue (or implicit production is
  // disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE is 0).
  // Note: Use std::make_move_iterator if the elements should be moved
  // instead of copied.
  // Thread-safe.
  template <typename It>
  inline bool try_enqueue_bulk(It itemFirst, size_t count) {
    if (inner.try_enqueue_bulk(std::forward<It>(itemFirst), count)) {
      sema->signal((LightweightSemaphore::ssize_t)(ssize_t)count);
      return true;
    }
    return false;
  }

  // Enqueues several items using an explicit producer token.
  // Does not allocate memory. Fails if not enough room to enqueue.
  // Note: Use std::make_move_iterator if the elements should be moved
  // instead of copied.
  // Thread-safe.
  template <typename It>
  inline bool try_enqueue_bulk(producer_token_t const& token, It itemFirst,
                               size_t count) {
    if (inner.try_enqueue_bulk(token, std::forward<It>(itemFirst), count)) {
      sema->signal((LightweightSemaphore::ssize_t)(ssize_t)count);
      return true;
    }
    return false;
  }

  // Attempts to dequeue from the queue.
  // Returns false if all producer streams appeared empty at the time they
  // were checked (so, the queue is likely but not guaranteed to be empty).
  // Never allocates. Thread-safe.
  template <typename U>
  inline bool try_dequeue(U& item) {
    if (sema->tryWait()) {
      while (!inner.try_dequeue(item)) {
        continue;
      }
      return true;
    }
    return false;
  }

  // Attempts to dequeue from the queue using an explicit consumer token.
  // Returns false if all producer streams appeared empty at the time they
  // were checked (so, the queue is likely but not guaranteed to be empty).
  // Never allocates. Thread-safe.
  template <typename U>
  inline bool try_dequeue(consumer_token_t& token, U& item) {
    if (sema->tryWait()) {
      while (!inner.try_dequeue(token, item)) {
        continue;
      }
      return true;
    }
    return false;
  }

  // Attempts to dequeue several elements from the queue.
  // Returns the number of items actually dequeued.
  // Returns 0 if all producer streams appeared empty at the time they
  // were checked (so, the queue is likely but not guaranteed to be empty).
  // Never allocates. Thread-safe.
  template <typename It>
  inline size_t try_dequeue_bulk(It itemFirst, size_t max) {
    size_t count = 0;
    max =
        (size_t)sema->tryWaitMany((LightweightSemaphore::ssize_t)(ssize_t)max);
    while (count != max) {
      count += inner.template try_dequeue_bulk<It&>(itemFirst, max - count);
    }
    return count;
  }

  // Attempts to dequeue several elements from the queue using an explicit
  // consumer token.
  // Returns the number of items actually dequeued.
  // Returns 0 if all producer streams appeared empty at the time they
  // were checked (so, the queue is likely but not guaranteed to be empty).
  // Never allocates. Thread-safe.
  template <typename It>
  inline size_t try_dequeue_bulk(consumer_token_t& token, It itemFirst,
                                 size_t max) {
    size_t count = 0;
    max =
        (size_t)sema->tryWaitMany((LightweightSemaphore::ssize_t)(ssize_t)max);
    while (count != max) {
      count +=
          inner.template try_dequeue_bulk<It&>(token, itemFirst, max - count);
    }
    return count;
  }

  // Blocks the current thread until there's something to dequeue, then
  // dequeues it.
  // Never allocates. Thread-safe.
  template <typename U>
  inline void wait_dequeue(U& item) {
    sema->wait();
    while (!inner.try_dequeue(item)) {
      continue;
    }
  }

  // Blocks the current thread until either there's something to dequeue
  // or the timeout (specified in microseconds) expires. Returns false
  // without setting `item` if the timeout expires, otherwise assigns
  // to `item` and returns true.
  // Using a negative timeout indicates an indefinite timeout,
  // and is thus functionally equivalent to calling wait_dequeue.
  // Never allocates. Thread-safe.
  template <typename U>
  inline bool wait_dequeue_timed(U& item, std::int64_t timeout_usecs) {
    if (!sema->wait(timeout_usecs)) {
      return false;
    }
    while (!inner.try_dequeue(item)) {
      continue;
    }
    return true;
  }

  // Blocks the current thread until either there's something to dequeue
  // or the timeout expires. Returns false without setting `item` if the
  // timeout expires, otherwise assigns to `item` and returns true.
  // Never allocates. Thread-safe.
  template <typename U, typename Rep, typename Period>
  inline bool wait_dequeue_timed(
      U& item, std::chrono::duration<Rep, Period> const& timeout) {
    return wait_dequeue_timed(
        item,
        std::chrono::duration_cast<std::chrono::microseconds>(timeout).count());
  }

  // Blocks the current thread until there's something to dequeue, then
  // dequeues it using an explicit consumer token.
  // Never allocates. Thread-safe.
  template <typename U>
  inline void wait_dequeue(consumer_token_t& token, U& item) {
    sema->wait();
    while (!inner.try_dequeue(token, item)) {
      continue;
    }
  }

  // Blocks the current thread until either there's something to dequeue
  // or the timeout (specified in microseconds) expires. Returns false
  // without setting `item` if the timeout expires, otherwise assigns
  // to `item` and returns true.
  // Using a negative timeout indicates an indefinite timeout,
  // and is thus functionally equivalent to calling wait_dequeue.
  // Never allocates. Thread-safe.
  template <typename U>
  inline bool wait_dequeue_timed(consumer_token_t& token, U& item,
                                 std::int64_t timeout_usecs) {
    if (!sema->wait(timeout_usecs)) {
      return false;
    }
    while (!inner.try_dequeue(token, item)) {
      continue;
    }
    return true;
  }

  // Blocks the current thread until either there's something to dequeue
  // or the timeout expires. Returns false without setting `item` if the
  // timeout expires, otherwise assigns to `item` and returns true.
  // Never allocates. Thread-safe.
  template <typename U, typename Rep, typename Period>
  inline bool wait_dequeue_timed(
      consumer_token_t& token, U& item,
      std::chrono::duration<Rep, Period> const& timeout) {
    return wait_dequeue_timed(
        token, item,
        std::chrono::duration_cast<std::chrono::microseconds>(timeout).count());
  }

  // Attempts to dequeue several elements from the queue.
  // Returns the number of items actually dequeued, which will
  // always be at least one (this method blocks until the queue
  // is non-empty) and at most max.
  // Never allocates. Thread-safe.
  template <typename It>
  inline size_t wait_dequeue_bulk(It itemFirst, size_t max) {
    size_t count = 0;
    max = (size_t)sema->waitMany((LightweightSemaphore::ssize_t)(ssize_t)max);
    while (count != max) {
      count += inner.template try_dequeue_bulk<It&>(itemFirst, max - count);
    }
    return count;
  }

  // Attempts to dequeue several elements from the queue.
  // Returns the number of items actually dequeued, which can
  // be 0 if the timeout expires while waiting for elements,
  // and at most max.
  // Using a negative timeout indicates an indefinite timeout,
  // and is thus functionally equivalent to calling wait_dequeue_bulk.
  // Never allocates. Thread-safe.
  template <typename It>
  inline size_t wait_dequeue_bulk_timed(It itemFirst, size_t max,
                                        std::int64_t timeout_usecs) {
    size_t count = 0;
    max = (size_t)sema->waitMany((LightweightSemaphore::ssize_t)(ssize_t)max,
                                 timeout_usecs);
    while (count != max) {
      count += inner.template try_dequeue_bulk<It&>(itemFirst, max - count);
    }
    return count;
  }

  // Attempts to dequeue several elements from the queue.
  // Returns the number of items actually dequeued, which can
  // be 0 if the timeout expires while waiting for elements,
  // and at most max.
  // Never allocates. Thread-safe.
  template <typename It, typename Rep, typename Period>
  inline size_t wait_dequeue_bulk_timed(
      It itemFirst, size_t max,
      std::chrono::duration<Rep, Period> const& timeout) {
    return wait_dequeue_bulk_timed<It&>(
        itemFirst, max,
        std::chrono::duration_cast<std::chrono::microseconds>(timeout).count());
  }

  // Attempts to dequeue several elements from the queue using an explicit
  // consumer token.
  // Returns the number of items actually dequeued, which will
  // always be at least one (this method blocks until the queue
  // is non-empty) and at most max.
  // Never allocates. Thread-safe.
  template <typename It>
  inline size_t wait_dequeue_bulk(consumer_token_t& token, It itemFirst,
                                  size_t max) {
    size_t count = 0;
    max = (size_t)sema->waitMany((LightweightSemaphore::ssize_t)(ssize_t)max);
    while (count != max) {
      count +=
          inner.template try_dequeue_bulk<It&>(token, itemFirst, max - count);
    }
    return count;
  }

  // Attempts to dequeue several elements from the queue using an explicit
  // consumer token.
  // Returns the number of items actually dequeued, which can
  // be 0 if the timeout expires while waiting for elements,
  // and at most max.
  // Using a negative timeout indicates an indefinite timeout,
  // and is thus functionally equivalent to calling wait_dequeue_bulk.
  // Never allocates. Thread-safe.
  template <typename It>
  inline size_t wait_dequeue_bulk_timed(consumer_token_t& token, It itemFirst,
                                        size_t max,
                                        std::int64_t timeout_usecs) {
    size_t count = 0;
    max = (size_t)sema->waitMany((LightweightSemaphore::ssize_t)(ssize_t)max,
                                 timeout_usecs);
    while (count != max) {
      count +=
          inner.template try_dequeue_bulk<It&>(token, itemFirst, max - count);
    }
    return count;
  }

  // Attempts to dequeue several elements from the queue using an explicit
  // consumer token.
  // Returns the number of items actually dequeued, which can
  // be 0 if the timeout expires while waiting for elements,
  // and at most max.
  // Never allocates. Thread-safe.
  template <typename It, typename Rep, typename Period>
  inline size_t wait_dequeue_bulk_timed(
      consumer_token_t& token, It itemFirst, size_t max,
      std::chrono::duration<Rep, Period> const& timeout) {
    return wait_dequeue_bulk_timed<It&>(
        token, itemFirst, max,
        std::chrono::duration_cast<std::chrono::microseconds>(timeout).count());
  }

  // Returns an estimate of the total number of elements currently in the queue.
  // This
  // estimate is only accurate if the queue has completely stabilized before it
  // is called
  // (i.e. all enqueue and dequeue operations have completed and their memory
  // effects are
  // visible on the calling thread, and no further operations start while this
  // method is
  // being called).
  // Thread-safe.
  inline size_t size_approx() const { return (size_t)sema->availableApprox(); }

  // Returns true if the underlying atomic variables used by
  // the queue are lock-free (they should be on most platforms).
  // Thread-safe.
  static bool is_lock_free() { return ConcurrentQueue::is_lock_free(); }

 private:
  template <typename U>
  static inline U* create() {
    auto p = (Traits::malloc)(sizeof(U));
    return p != nullptr ? new (p) U : nullptr;
  }

  template <typename U, typename A1>
  static inline U* create(A1&& a1) {
    auto p = (Traits::malloc)(sizeof(U));
    return p != nullptr ? new (p) U(std::forward<A1>(a1)) : nullptr;
  }

  template <typename U>
  static inline void destroy(U* p) {
    if (p != nullptr) {
      p->~U();
    }
    (Traits::free)(p);
  }

 private:
  ConcurrentQueue inner;
  std::unique_ptr<LightweightSemaphore, void (*)(LightweightSemaphore*)> sema;
};

template <typename T, typename Traits>
inline void swap(BlockingConcurrentQueue<T, Traits>& a,
                 BlockingConcurrentQueue<T, Traits>& b) MOODYCAMEL_NOEXCEPT {
  a.swap(b);
}

}  // end namespace moodycamel
