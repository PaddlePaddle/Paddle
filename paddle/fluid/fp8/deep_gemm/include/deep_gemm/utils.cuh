// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// The file has been adapted from DeepSeek DeepGEMM project
// Copyright (c) 2025 DeepSeek
// Licensed under the MIT License -
// https://github.com/deepseek-ai/DeepGEMM/blob/main/LICENSE

#pragma once

#include <exception>

#ifdef __CLION_IDE__
__host__ __device__ __forceinline__ void host_device_printf(const char* format,
                                                            ...) {
  asm volatile("trap;");
}
#define printf host_device_printf
#endif

class AssertionException : public std::exception {
 private:
  std::string message{};

 public:
  explicit AssertionException(const std::string& message) : message(message) {}

  const char* what() const noexcept override { return message.c_str(); }
};

#ifndef DG_HOST_ASSERT
#define DG_HOST_ASSERT(cond)                                \
  do {                                                      \
    if (not(cond)) {                                        \
      printf("Assertion failed: %s:%d, condition: %s\n",    \
             __FILE__,                                      \
             __LINE__,                                      \
             #cond);                                        \
      throw AssertionException("Assertion failed: " #cond); \
    }                                                       \
  } while (0)
#endif

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond)                           \
  do {                                                   \
    if (not(cond)) {                                     \
      printf("Assertion failed: %s:%d, condition: %s\n", \
             __FILE__,                                   \
             __LINE__,                                   \
             #cond);                                     \
      asm("trap;");                                      \
    }                                                    \
  } while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

template <typename T>
__device__ __host__ constexpr T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}
