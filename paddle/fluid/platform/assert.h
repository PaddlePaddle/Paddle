/*   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#if defined(__CUDA_ARCH__)
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
// For cuda, the assertions can affect performance and it is therefore
// recommended to disable them in production code
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#assertion
#define PADDLE_ASSERT(e) assert((e))
#define PADDLE_ASSERT_MSG(e, m) assert((e) && (m))
#endif
