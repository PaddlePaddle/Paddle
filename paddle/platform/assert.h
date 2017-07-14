#pragma once

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#if defined(__APPLE__) && defined(__CUDA_ARCH__) && !defined(NDEBUG)
#include <stdio.h>
#define PADDLE_ASSERT(e)                                           \
  do {                                                             \
    if (!(e)) {                                                    \
      printf("%s:%d Assertion `%s` failed.\n", __FILE__, __LINE__, \
             TOSTRING(e));                                         \
      asm("trap;");                                                \
    }                                                              \
  } while (0)

#define PADDLE_ASSERT_MSG(e, m)                                         \
  do {                                                                  \
    if (!(e)) {                                                         \
      printf("%s:%d Assertion `%s` failed (%s).\n", __FILE__, __LINE__, \
             TOSTRING(e), m);                                           \
      asm("trap;");                                                     \
    }                                                                   \
  } while (0)
#else
#include <assert.h>
#define PADDLE_ASSERT(e) assert(e)
#define PADDLE_ASSERT_MSG(e, m) assert((e) && (m))
#endif
