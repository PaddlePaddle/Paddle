/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

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
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <map>
#include <mutex>  // NOLINT
#include <random>

namespace paddle {
namespace platform {

static pid_t GetTID() {
#if defined(__APPLE__) || defined(__OSX__)
  // syscall is deprecated: first deprecated in macOS 10.12.
  // syscall is unsupported;
  // syscall pid_t tid = syscall(SYS_thread_selfid);
  uint64_t tid;
  pthread_threadid_np(NULL, &tid);
#else
#ifndef __NR_gettid
#define __NR_gettid 224
#endif
  pid_t tid = syscall(__NR_gettid);
#endif
  CHECK_NE((int)tid, -1);
  return tid;
}

/**
 * Thread local storage for object.
 * Example:
 *
 * Declarartion:
 * ThreadLocal<vector<int>> vec_;
 *
 * Use in thread:
 * vector<int>& vec = *vec; // obtain the thread specific object
 * vec.resize(100);
 *
 * Note that this ThreadLocal will desconstruct all internal data when thread
 * exits
 * This class is suitable for cases when frequently creating and deleting
 * threads.
 *
 * Consider implementing a new ThreadLocal if one needs to frequently create
 * both instances and threads.
 *
 * see also ThreadLocalD
 */
template <class T>
class ThreadLocal {
 public:
  ThreadLocal() {
    CHECK_EQ(pthread_key_create(&thread_specific_key_, DataDestructor), 0);
  }
  ~ThreadLocal() { pthread_key_delete(thread_specific_key_); }

  /**
   * @brief get thread local object.
   * @param if createLocal is true and thread local object is never created,
   * return a new object. Otherwise, return nullptr.
   */
  T* Get(bool createLocal = true) {
    T* p = reinterpret_cast<T*>(pthread_getspecific(thread_specific_key_));
    if (!p && createLocal) {
      p = new T();
      int ret = pthread_setspecific(thread_specific_key_, p);
      CHECK_EQ(ret, 0);
    }
    return p;
  }

  /**
   * @brief set (overwrite) thread local object. If there is a thread local
   * object before, the previous object will be destructed before.
   *
   */
  void Set(T* p) {
    if (T* q = Get(false)) {
      DataDestructor(q);
    }
    CHECK_EQ(pthread_setspecific(thread_specific_key_, p), 0);
  }

  /**
   * return reference.
   */
  T& operator*() { return *Get(); }

  /**
   * Implicit conversion to T*
   */
  operator T*() { return Get(); }

 private:
  static void DataDestructor(void* p) { delete reinterpret_cast<T*>(p); }

  pthread_key_t thread_specific_key_;
};

/**
 * Almost the same as ThreadLocal, but note that this ThreadLocalD will
 * destruct all internal data when ThreadLocalD instance destructs.
 *
 * This class is suitable for cases when frequently creating and deleting
 * objects.
 *
 * see also ThreadLocal
 *
 * @note The type T must implemented default constructor.
 */
template <class T>
class ThreadLocalD {
 public:
  ThreadLocalD() {
    CHECK_EQ(pthread_key_create(&thread_specific_key_, NULL), 0);
  }
  ~ThreadLocalD() {
    pthread_key_delete(thread_specific_key_);
    for (auto t : thread_map_) {
      DataDestructor(t.second);
    }
  }

  /**
   * @brief Get thread local object. If not exists, create new one.
   */
  T* Get() {
    T* p = reinterpret_cast<T*>(pthread_getspecific(thread_specific_key_));
    if (!p) {
      p = new T();
      CHECK_EQ(pthread_setspecific(thread_specific_key_, p), 0);
      UpdateMap(p);
    }
    return p;
  }

  /**
   * @brief Set thread local object. If there is an object create before, the
   * old object will be destructed.
   */
  void Set(T* p) {
    if (T* q =
            reinterpret_cast<T*>(pthread_getspecific(thread_specific_key_))) {
      DataDestructor(q);
    }
    CHECK_EQ(pthread_setspecific(thread_specific_key_, p), 0);
    UpdateMap(p);
  }

  /**
   * @brief Get reference of the thread local object.
   */
  T& operator*() { return *Get(); }

 private:
  static void DataDestructor(void* p) { delete reinterpret_cast<T*>(p); }

  void UpdateMap(T* p) {
    pid_t tid = GetTID();
    CHECK_NE(tid, -1);
    std::lock_guard<std::mutex> guard(mutex_);
    auto ret = thread_map_.insert(std::make_pair(tid, p));
    if (!ret.second) {
      ret.first->second = p;
    }
  }

  pthread_key_t thread_specific_key_;
  std::mutex mutex_;
  std::map<pid_t, T*> thread_map_;
};

}  // namespace platform
}  // namespace paddle
