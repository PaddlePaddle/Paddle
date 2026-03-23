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

// The file has been adapted from DeepSeek DeepEP project
// Copyright (c) 2025 DeepSeek
// Licensed under the MIT License -
// https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

#pragma once

#include <exception>
#include <string>

#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

class EPException : public std::exception {
 private:
  std::string message = {};

 public:
  explicit EPException(const char* name,
                       const char* file,
                       const int line,
                       const std::string& error) {
    message = std::string("Failed: ") + name + " error " + file + ":" +
              std::to_string(line) + " '" + error + "'";
  }

  const char* what() const noexcept override { return message.c_str(); }
};

#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd)                                                     \
  do {                                                                      \
    cudaError_t e = (cmd);                                                  \
    if (e != cudaSuccess) {                                                 \
      throw EPException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }                                                                       \
  } while (0)
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                     \
  do {                                                           \
    if (not(cond)) {                                             \
      throw EPException("Assertion", __FILE__, __LINE__, #cond); \
    }                                                            \
  } while (0)
#endif

#ifndef EP_DEVICE_ASSERT
#define EP_DEVICE_ASSERT(cond)                           \
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
