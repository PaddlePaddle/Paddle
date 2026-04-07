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

#pragma once

#include <exception>
#include <string>

#if defined(PADDLE_WITH_HIP)
#include <hip/hip_runtime.h>
#elif defined(PADDLE_WITH_CUDA)
#include <cuda_runtime.h>
#endif

class CompatException : public std::exception {
 private:
  std::string message = {};

 public:
  explicit CompatException(const char* name,
                           const char* file,
                           const int line,
                           const std::string& error) {
    message = std::string("Failed: ") + name + " error " + file + ":" +
              std::to_string(line) + " '" + error + "'";
  }

  const char* what() const noexcept override { return message.c_str(); }
};

#ifndef C10_CUDA_CHECK
#if defined(PADDLE_WITH_HIP)
#define C10_CUDA_CHECK(cmd)                                                   \
  do {                                                                        \
    hipError_t e = (cmd);                                                     \
    if (e != hipSuccess) {                                                    \
      throw CompatException("HIP", __FILE__, __LINE__, hipGetErrorString(e)); \
    }                                                                         \
  } while (0)
#elif defined(PADDLE_WITH_CUDA)
#define C10_CUDA_CHECK(cmd)                                   \
  do {                                                        \
    cudaError_t e = (cmd);                                    \
    if (e != cudaSuccess) {                                   \
      throw CompatException(                                  \
          "CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }                                                         \
  } while (0)
#endif
#endif
