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

#include <condition_variable>
#include <deque>
#include <mutex>

#include "Locks.h"

namespace paddle {

/**
 * A thread-safe queue that automatically grows but never shrinks.
 * Dequeue a empty queue will block current thread. Enqueue an element
 * will wake up another thread that blocked by dequeue method.
 *
 * For example.
 * @code{.cpp}
 *
 * paddle::Queue<int> q;
 * END_OF_JOB=-1
 * void thread1() {
 *   while (true) {
 *     auto job = q.dequeue();
 *     if (job == END_OF_JOB) {
 *       break;
 *     }
 *     processJob(job);
 *   }
 * }
 *
 * void thread2() {
 *   while (true) {
 *      auto job = getJob();
 *      q.enqueue(job);
 *      if (job == END_OF_JOB) {
 *        break;
 *      }
 *   }
 * }
 *
 * @endcode
 */
template <class T>
class Queue {
public:
  /**
   * @brief Construct Function. Default capacity of Queue is zero.
   */
  Queue() : numElements_(0) {}

  ~Queue() {}

  /**
   * @brief enqueue an element into Queue.
   * @param[in] el The enqueue element.
   * @note This method is thread-safe, and will wake up another blocked thread.
   */
  void enqueue(const T& el) {
    std::unique_lock<std::mutex> lock(queueLock_);
    elements_.emplace_back(el);
    numElements_++;

    queueCV_.notify_all();
  }

  /**
   * @brief enqueue an element into Queue.
   * @param[in] el The enqueue element. rvalue reference .
   * @note This method is thread-safe, and will wake up another blocked thread.
   */
  void enqueue(T&& el) {
    std::unique_lock<std::mutex> lock(queueLock_);
    elements_.emplace_back(std::move(el));
    numElements_++;

    queueCV_.notify_all();
  }

  /**
   * Dequeue from a queue and return a element.
   * @note this method will be blocked until not empty.
   */
  T dequeue() {
    std::unique_lock<std::mutex> lock(queueLock_);
    queueCV_.wait(lock, [this]() { return numElements_ != 0; });
    T el;

    using std::swap;
    // Becuase of the previous statement, the right swap() can be found
    // via argument-dependent lookup (ADL).
    swap(elements_.front(), el);

    elements_.pop_front();
    numElements_--;
    if (numElements_ == 0) {
      queueCV_.notify_all();
    }
    return el;
  }

  /**
   * Return size of queue.
   *
   * @note This method is not thread safe. Obviously this number
   * can change by the time you actually look at it.
   */
  inline int size() const { return numElements_; }

  /**
   * @brief is empty or not.
   * @return true if empty.
   * @note This method is not thread safe.
   */
  inline bool empty() const { return numElements_ == 0; }

  /**
   * @brief wait util queue is empty
   */
  void waitEmpty() {
    std::unique_lock<std::mutex> lock(queueLock_);
    queueCV_.wait(lock, [this]() { return numElements_ == 0; });
  }

  /**
   * @brief wait queue is not empty at most for some seconds.
   * @param seconds wait time limit.
   * @return true if queue is not empty. false if timeout.
   */
  bool waitNotEmptyFor(int seconds) {
    std::unique_lock<std::mutex> lock(queueLock_);
    return queueCV_.wait_for(lock, std::chrono::seconds(seconds), [this] {
      return numElements_ != 0;
    });
  }

private:
  std::deque<T> elements_;
  int numElements_;
  std::mutex queueLock_;
  std::condition_variable queueCV_;
};

/*
 * A thread-safe circular queue that
 * automatically blocking calling thread if capacity reached.
 *
 * For example.
 * @code{.cpp}
 *
 * paddle::BlockingQueue<int> q(capacity);
 * END_OF_JOB=-1
 * void thread1() {
 *   while (true) {
 *     auto job = q.dequeue();
 *     if (job == END_OF_JOB) {
 *       break;
 *     }
 *     processJob(job);
 *   }
 * }
 *
 * void thread2() {
 *   while (true) {
 *      auto job = getJob();
 *      q.enqueue(job); //Block until q.size() < capacity .
 *      if (job == END_OF_JOB) {
 *        break;
 *      }
 *   }
 * }
 */
template <typename T>
class BlockingQueue {
public:
  /**
   * @brief Construct Function.
   * @param[in] capacity the max numer of elements the queue can have.
   */
  explicit BlockingQueue(size_t capacity) : capacity_(capacity) {}

  /**
   * @brief enqueue an element into Queue.
   * @param[in] x The enqueue element, pass by reference .
   * @note This method is thread-safe, and will wake up another thread
   * who was blocked because of the queue is empty.
   * @note If it's size() >= capacity before enqueue,
   * this method will block and wait until size() < capacity.
   */
  void enqueue(const T& x) {
    std::unique_lock<std::mutex> lock(mutex_);
    notFull_.wait(lock, [&] { return queue_.size() < capacity_; });
    queue_.push_back(x);
    notEmpty_.notify_one();
  }

  /**
   * Dequeue from a queue and return a element.
   * @note this method will be blocked until not empty.
   * @note this method will wake up another thread who was blocked because
   * of the queue is full.
   */
  T dequeue() {
    std::unique_lock<std::mutex> lock(mutex_);
    notEmpty_.wait(lock, [&] { return !queue_.empty(); });

    T front(queue_.front());
    queue_.pop_front();
    notFull_.notify_one();
    return front;
  }

  /**
   * Return size of queue.
   *
   * @note This method is thread safe.
   * The size of the queue won't change until the method return.
   */
  size_t size() {
    std::lock_guard<std::mutex> guard(mutex_);
    return queue_.size();
  }

  /**
   * @brief is empty or not.
   * @return true if empty.
   * @note This method is thread safe.
   */
  size_t empty() {
    std::lock_guard<std::mutex> guard(mutex_);
    return queue_.empty();
  }

private:
  std::mutex mutex_;
  std::condition_variable notEmpty_;
  std::condition_variable notFull_;
  std::deque<T> queue_;
  size_t capacity_;
};

}  // namespace paddle
