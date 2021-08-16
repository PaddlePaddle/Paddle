// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 the spinlock implementation is borrowed from Doug Lea's malloc, released to the
 public domain, as explained at
  http://creativecommons.org/licenses/publicdomain.  Send questions,
  comments, complaints, performance data, etc to dl@cs.oswego.edu
*/

#pragma once

#ifndef WIN32
#include <pthread.h>
#if defined(__SVR4) && defined(__sun) /* solaris */
#include <thread.h>
#endif /* solaris */
#else
#ifndef _M_AMD64
/* These are already defined on AMD64 builds */
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
LONG __cdecl _InterlockedCompareExchange(LONG volatile* Dest, LONG Exchange,
                                         LONG Comp);
LONG __cdecl _InterlockedExchange(LONG volatile* Target, LONG Value);
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* _M_AMD64 */
#pragma intrinsic(_InterlockedCompareExchange)
#pragma intrinsic(_InterlockedExchange)
#define interlockedcompareexchange _InterlockedCompareExchange
#define interlockedexchange _InterlockedExchange
#endif /* Win32 */

#ifndef FORCEINLINE
#if defined(__GNUC__)
#define FORCEINLINE __inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define FORCEINLINE __forceinline
#endif
#endif
#ifndef NOINLINE
#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE
#endif
#endif

#ifdef __cplusplus
extern "C" {
#ifndef FORCEINLINE
#define FORCEINLINE inline
#endif
#endif /* __cplusplus */
#ifndef FORCEINLINE
#define FORCEINLINE
#endif
#ifdef __cplusplus
};     /* end of extern "C" */
#endif /* __cplusplus */

#ifndef WIN32

/* Custom pthread-style spin locks on x86 and x64 for gcc */
struct pthread_mlock_t {
  volatile unsigned int l;
  unsigned int c;
  pthread_t threadid;
};
#define MLOCK_T struct pthread_mlock_t
#define CURRENT_THREAD pthread_self()
#define INITIAL_LOCK(sl) ((sl)->threadid = 0, (sl)->l = (sl)->c = 0, 0)
#define ACQUIRE_LOCK(sl) pthread_acquire_lock(sl)
#define RELEASE_LOCK(sl) pthread_release_lock(sl)
#define TRY_LOCK(sl) pthread_try_lock(sl)
#define SPINS_PER_YIELD 63

// static MLOCK_T malloc_global_mutex = { 0, 0, 0};

static FORCEINLINE int pthread_acquire_lock(MLOCK_T* sl) {
  int spins = 0;
  volatile unsigned int* lp = &sl->l;
  for (;;) {
    if (*lp != 0) {
      if (sl->threadid == CURRENT_THREAD) {
        ++sl->c;
        return 0;
      }
    } else {
      /* place args to cmpxchgl in locals to evade oddities in some gccs */
      int cmp = 0;
      int val = 1;
      int ret;
      __asm__ __volatile__("lock; cmpxchgl %1, %2"
                           : "=a"(ret)
                           : "r"(val), "m"(*(lp)), "0"(cmp)
                           : "memory", "cc");
      if (!ret) {
        assert(!sl->threadid);
        sl->threadid = CURRENT_THREAD;
        sl->c = 1;
        return 0;
      }
    }
    if ((++spins & SPINS_PER_YIELD) == 0) {
#if defined(__SVR4) && defined(__sun) /* solaris */
      thr_yield();
#else
#if defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
      sched_yield();
#else  /* no-op yield on unknown systems */
      ;  // NOLINT
#endif /* __linux__ || __FreeBSD__ || __APPLE__ */
#endif /* solaris */
    }
  }
}

static FORCEINLINE void pthread_release_lock(MLOCK_T* sl) {
  volatile unsigned int* lp = &sl->l;
  assert(*lp != 0);
  assert(sl->threadid == CURRENT_THREAD);
  if (--sl->c == 0) {
    sl->threadid = 0;
    int prev = 0;
    int ret;
    __asm__ __volatile__("lock; xchgl %0, %1"
                         : "=r"(ret)
                         : "m"(*(lp)), "0"(prev)
                         : "memory");
  }
}

static FORCEINLINE int pthread_try_lock(MLOCK_T* sl) {
  volatile unsigned int* lp = &sl->l;
  if (*lp != 0) {
    if (sl->threadid == CURRENT_THREAD) {
      ++sl->c;
      return 1;
    }
  } else {
    int cmp = 0;
    int val = 1;
    int ret;
    __asm__ __volatile__("lock; cmpxchgl %1, %2"
                         : "=a"(ret)
                         : "r"(val), "m"(*(lp)), "0"(cmp)
                         : "memory", "cc");
    if (!ret) {
      assert(!sl->threadid);
      sl->threadid = CURRENT_THREAD;
      sl->c = 1;
      return 1;
    }
  }
  return 0;
}

#else /* WIN32 */
/* Custom win32-style spin locks on x86 and x64 for MSC */
struct win32_mlock_t {
  volatile long l;  // NOLINT
  unsigned int c;
  long threadid;  // NOLINT
};

#define MLOCK_T struct win32_mlock_t
#define CURRENT_THREAD GetCurrentThreadId()
#define INITIAL_LOCK(sl) ((sl)->threadid = 0, (sl)->l = (sl)->c = 0, 0)
#define ACQUIRE_LOCK(sl) win32_acquire_lock(sl)
#define RELEASE_LOCK(sl) win32_release_lock(sl)
#define TRY_LOCK(sl) win32_try_lock(sl)
#define SPINS_PER_YIELD 63

// static MLOCK_T malloc_global_mutex = { 0, 0, 0};

static FORCEINLINE int win32_acquire_lock(MLOCK_T *sl) {
  int spins = 0;
  for (;;) {
    if (sl->l != 0) {
      if (sl->threadid == CURRENT_THREAD) {
        ++sl->c;
        return 0;
      }
    } else {
      if (!interlockedexchange(&sl->l, 1)) {
        assert(!sl->threadid);
        sl->threadid = CURRENT_THREAD;
        sl->c = 1;
        return 0;
      }
    }
    if ((++spins & SPINS_PER_YIELD) == 0) SleepEx(0, FALSE);
  }
}

static FORCEINLINE void win32_release_lock(MLOCK_T *sl) {
  assert(sl->threadid == CURRENT_THREAD);
  assert(sl->l != 0);
  if (--sl->c == 0) {
    sl->threadid = 0;
    interlockedexchange(&sl->l, 0);
  }
}

static FORCEINLINE int win32_try_lock(MLOCK_T *sl) {
  if (sl->l != 0) {
    if (sl->threadid == CURRENT_THREAD) {
      ++sl->c;
      return 1;
    }
  } else {
    if (!interlockedexchange(&sl->l, 1)) {
      assert(!sl->threadid);
      sl->threadid = CURRENT_THREAD;
      sl->c = 1;
      return 1;
    }
  }
  return 0;
}

#endif /* WIN32 */
