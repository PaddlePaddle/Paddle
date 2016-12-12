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
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <map>
#include <mutex>
#include <random>
#include "Logging.h"
#include "Util.h"

namespace paddle {

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
    PCHECK(pthread_key_create(&threadSpecificKey_, dataDestructor) == 0);
  }
  ~ThreadLocal() { pthread_key_delete(threadSpecificKey_); }

  /**
   * @brief get thread local object.
   * @param if createLocal is true and thread local object is never created,
   * return a new object. Otherwise, return nullptr.
   */
  T* get(bool createLocal = true) {
    T* p = (T*)pthread_getspecific(threadSpecificKey_);
    if (!p && createLocal) {
      p = new T();
      int ret = pthread_setspecific(threadSpecificKey_, p);
      PCHECK(ret == 0);
    }
    return p;
  }

  /**
   * @brief set (overwrite) thread local object. If there is a thread local
   * object before, the previous object will be destructed before.
   *
   */
  void set(T* p) {
    if (T* q = get(false)) {
      dataDestructor(q);
    }
    PCHECK(pthread_setspecific(threadSpecificKey_, p) == 0);
  }

  /**
   * return reference.
   */
  T& operator*() { return *get(); }

  /**
   * Implicit conversion to T*
   */
  operator T*() { return get(); }

private:
  static void dataDestructor(void* p) { delete (T*)p; }

  pthread_key_t threadSpecificKey_;
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
  ThreadLocalD() { PCHECK(pthread_key_create(&threadSpecificKey_, NULL) == 0); }
  ~ThreadLocalD() {
    pthread_key_delete(threadSpecificKey_);
    for (auto t : threadMap_) {
      dataDestructor(t.second);
    }
  }

  /**
   * @brief Get thread local object. If not exists, create new one.
   */
  T* get() {
    T* p = (T*)pthread_getspecific(threadSpecificKey_);
    if (!p) {
      p = new T();
      PCHECK(pthread_setspecific(threadSpecificKey_, p) == 0);
      updateMap(p);
    }
    return p;
  }

  /**
   * @brief Set thread local object. If there is an object create before, the
   * old object will be destructed.
   */
  void set(T* p) {
    if (T* q = (T*)pthread_getspecific(threadSpecificKey_)) {
      dataDestructor(q);
    }
    PCHECK(pthread_setspecific(threadSpecificKey_, p) == 0);
    updateMap(p);
  }

  /**
   * @brief Get reference of the thread local object.
   */
  T& operator*() { return *get(); }

private:
  static void dataDestructor(void* p) { delete (T*)p; }

  void updateMap(T* p) {
    pid_t tid = getTID();
    CHECK_NE(tid, -1);
    std::lock_guard<std::mutex> guard(mutex_);
    auto ret = threadMap_.insert(std::make_pair(tid, p));
    if (!ret.second) {
      ret.first->second = p;
    }
  }

  pthread_key_t threadSpecificKey_;
  std::mutex mutex_;
  std::map<pid_t, T*> threadMap_;
};

/**
 * @brief Thread-safe C-style random API.
 */
class ThreadLocalRand {
public:
  /**
   * initSeed just like srand,
   * called by main thread,
   * init defaultSeed for all thread
   */
  static void initSeed(unsigned int seed) { defaultSeed_ = seed; }

  /**
   * initThreadSeed called by each thread,
   * init seed to defaultSeed + *tid*
   * It should be called after main initSeed and before using rand()
   * It's optional, getSeed will init seed if it's not initialized.
   */
  static void initThreadSeed(int tid) {
    seed_.set(new unsigned int(defaultSeed_ + tid));
  }

  /// thread get seed, then can call rand_r many times.
  /// Caller thread can modify the seed value if it's necessary.
  ///
  /// if flag thread_local_rand_use_global_seed set,
  /// the seed will be set to defaultSeed in thread's first call.
  static unsigned int* getSeed();

  /// like ::rand
  static int rand() { return rand_r(getSeed()); }

  /**
   * Get defaultSeed for all thread.
   */
  static int getDefaultSeed() { return defaultSeed_; }

protected:
  static unsigned int defaultSeed_;
  static ThreadLocal<unsigned int> seed_;
};

/**
 * @brief Thread-safe C++ style random engine.
 */
class ThreadLocalRandomEngine {
public:
  /**
   * get random_engine for each thread.
   *
   * Engine's seed will be initialized by ThreadLocalRand.
   */
  static std::default_random_engine& get();

protected:
  static ThreadLocal<std::default_random_engine> engine_;
};

}  // namespace paddle
