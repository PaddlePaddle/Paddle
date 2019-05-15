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

// For cuda, the assertions can affect performance and it is therefore
// recommended to disable them in production code
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#assertion
#if defined(__CUDA_ARCH__)
#include <stdio.h>
#define EXIT() asm("trap;")
#else
#include <assert.h>
#define EXIT() throw std::runtime_error("Exception encounter.")
#endif

#define PADDLE_ASSERT(_IS_NOT_ERROR)                                          \
  do {                                                                        \
    if (!(_IS_NOT_ERROR)) {                                                   \
      printf("Exception: %s:%d Assertion `%s` failed.\n", __FILE__, __LINE__, \
             TOSTRING(_IS_NOT_ERROR));                                        \
      EXIT();                                                                 \
    }                                                                         \
  } while (0)

// NOTE: PADDLE_ASSERT is mainly used in CUDA Kernel or HOSTDEVICE function.
#define PADDLE_ASSERT_MSG(_IS_NOT_ERROR, __MSG, __VAL)                       \
  do {                                                                       \
    if (!(_IS_NOT_ERROR)) {                                                  \
      printf("Exception: %s:%d Assertion `%s` failed (%s %ld).\n", __FILE__, \
             __LINE__, TOSTRING(_IS_NOT_ERROR), __MSG, __VAL);               \
      EXIT();                                                                \
    }                                                                        \
  } while (0)
