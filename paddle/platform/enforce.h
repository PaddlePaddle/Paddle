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

#include <paddle/string/printf.h>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef PADDLE_ONLY_CPU

#include "paddle/platform/dynload/cublas.h"
#include "paddle/platform/dynload/cudnn.h"
#include "paddle/platform/dynload/curand.h"

#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#endif  // PADDLE_ONLY_CPU

namespace paddle {
namespace platform {

// Because most enforce conditions would evaluate to true, we can use
// __builtin_expect to instruct the C++ compiler to generate code that
// always forces branch prediction of true.
// This generates faster binary code. __builtin_expect is since C++11.
// For more details, please check https://stackoverflow.com/a/43870188/724872.
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)

#ifndef PADDLE_ONLY_CPU

template <typename... Args>
inline void throw_on_error(cudaError_t e, const Args&... args) {
  if (UNLIKELY(e)) {
    // clang-format off
    throw thrust::system_error(
        e, thrust::cuda_category(),
        string::Sprintf(args...) +
        string::Sprintf(" at [%s:%s];", __FILE__, __LINE__));
    // clang-format on
  }
}

template <typename... Args>
inline void throw_on_error(curandStatus_t stat, const Args&... args) {
  if (stat != CURAND_STATUS_SUCCESS) {
    // clang-format off
    throw thrust::system_error(
        cudaErrorLaunchFailure, thrust::cuda_category(),
        string::Sprintf(args...) +
        string::Sprintf(" at [%s:%s];", __FILE__, __LINE__));
    // clang-format on
  }
}

template <typename... Args>
inline void throw_on_error(cudnnStatus_t stat, const Args&... args) {
  if (stat == CUDNN_STATUS_SUCCESS) {
    return;
  } else {
    // clang-format off
    throw std::runtime_error(
        platform::dynload::cudnnGetErrorString(stat) +
        string::Sprintf(args...) +
        string::Sprintf(" at [%s:%s];", __FILE__, __LINE__));
    // clang-format on
  }
}

template <typename... Args>
inline void throw_on_error(cublasStatus_t stat, const Args&... args) {
  std::string err;
  if (stat == CUBLAS_STATUS_SUCCESS) {
    return;
  } else if (stat == CUBLAS_STATUS_NOT_INITIALIZED) {
    err = "CUBLAS: not initialized, ";
  } else if (stat == CUBLAS_STATUS_ALLOC_FAILED) {
    err = "CUBLAS: alloc failed, ";
  } else if (stat == CUBLAS_STATUS_INVALID_VALUE) {
    err = "CUBLAS: invalid value, ";
  } else if (stat == CUBLAS_STATUS_ARCH_MISMATCH) {
    err = "CUBLAS: arch mismatch, ";
  } else if (stat == CUBLAS_STATUS_MAPPING_ERROR) {
    err = "CUBLAS: mapping error, ";
  } else if (stat == CUBLAS_STATUS_EXECUTION_FAILED) {
    err = "CUBLAS: execution failed, ";
  } else if (stat == CUBLAS_STATUS_INTERNAL_ERROR) {
    err = "CUBLAS: internal error, ";
  } else if (stat == CUBLAS_STATUS_NOT_SUPPORTED) {
    err = "CUBLAS: not supported, ";
  } else if (stat == CUBLAS_STATUS_LICENSE_ERROR) {
    err = "CUBLAS: license error, ";
  }
  throw std::runtime_error(err + string::Sprintf(args...) +
                           string::Sprintf(" at [%s:%s];", __FILE__, __LINE__));
}

#endif  // PADDLE_ONLY_CPU

template <typename... Args>
inline void throw_on_error(int stat, const Args&... args) {
  if (UNLIKELY(!(stat))) {
    throw std::runtime_error(
        string::Sprintf(args...) +
        string::Sprintf(" at [%s:%s];", __FILE__, __LINE__));
  }
}

#define PADDLE_THROW(...)                                     \
  do {                                                        \
    throw std::runtime_error(                                 \
        string::Sprintf(__VA_ARGS__) +                        \
        string::Sprintf(" at [%s:%s];", __FILE__, __LINE__)); \
  } while (0)

/**
 * @brief Enforce a condition, otherwise throw an EnforceNotMet
 */
#define PADDLE_ENFORCE(condition, ...)                          \
  do {                                                          \
    ::paddle::platform::throw_on_error(condition, __VA_ARGS__); \
  } while (0)

}  // namespace platform
}  // namespace paddle
