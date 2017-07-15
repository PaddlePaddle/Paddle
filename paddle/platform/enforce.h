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

/**
 * @brief Enforce exception. Inherits std::exception
 *
 * All enforce condition not met, will throw an EnforceNotMet exception.
 */
class EnforceNotMet : public std::exception {
 public:
  EnforceNotMet(const std::string& msg, const char* file, int fileline) {
    std::ostringstream sout;
    sout << msg << " at [" << file << ":" << fileline << "];";
    all_msg_ = sout.str();
  }

  const char* what() const noexcept override { return all_msg_.c_str(); }

 private:
  std::string all_msg_;
};

// From https://stackoverflow.com/questions/30130930/
// __buildin_expect is in C++ 11 standard. Since the condition which enforced
// should be true in most situation, it will make the compiler generate faster
// code by adding `UNLIKELY` macro.
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)

/**
 * @brief Throw a EnforceNotMet exception, automatically filled __FILE__ &
 * __LINE__
 *
 * This macro take __VA_ARGS__, user can pass any type if that type can
 * serialize to std::ostream
 */
#define PADDLE_THROW(...)                                            \
  do {                                                               \
    throw ::paddle::platform::EnforceNotMet(                         \
        ::paddle::string::Sprintf(__VA_ARGS__), __FILE__, __LINE__); \
  } while (0)

#ifndef PADDLE_ONLY_CPU

template <typename... Args>
inline void throw_on_error(cudaError_t e, const Args&... args) {
  if (e) {
    std::stringstream ss;
    ss << ::paddle::string::Sprintf(args...);
    ss << ::paddle::string::Sprintf(" at [%s:%s];", __FILE__, __LINE__);
    throw thrust::system_error(e, thrust::cuda_category(), ss.str());
  }
}

template <typename... Args>
inline void throw_on_error(curandStatus_t stat, const Args&... args) {
  if (stat != CURAND_STATUS_SUCCESS) {
    std::stringstream ss;
    ss << ::paddle::string::Sprintf(args...);
    ss << ::paddle::string::Sprintf(" at [%s:%s];", __FILE__, __LINE__);
    throw thrust::system_error(cudaErrorLaunchFailure, thrust::cuda_category(),
                               ss.str());
  }
}

template <typename... Args>
inline void throw_on_error(cudnnStatus_t stat, const Args&... args) {
  if (stat == CUDNN_STATUS_SUCCESS) {
    return;
  } else {
    std::stringstream ss;
    ss << ::paddle::platform::dynload::cudnnGetErrorString(stat);
    ss << ", " << ::paddle::string::Sprintf(args...);
    ss << ::paddle::string::Sprintf(" at [%s:%s];", __FILE__, __LINE__);
    throw std::runtime_error(ss.str());
  }
}

template <typename... Args>
inline void throw_on_error(cublasStatus_t stat, const Args&... args) {
  std::stringstream ss;
  if (stat == CUBLAS_STATUS_SUCCESS) {
    return;
  } else if (stat == CUBLAS_STATUS_NOT_INITIALIZED) {
    ss << "CUBLAS: not initialized";
  } else if (stat == CUBLAS_STATUS_ALLOC_FAILED) {
    ss << "CUBLAS: alloc failed";
  } else if (stat == CUBLAS_STATUS_INVALID_VALUE) {
    ss << "CUBLAS: invalid value";
  } else if (stat == CUBLAS_STATUS_ARCH_MISMATCH) {
    ss << "CUBLAS: arch mismatch";
  } else if (stat == CUBLAS_STATUS_MAPPING_ERROR) {
    ss << "CUBLAS: mapping error";
  } else if (stat == CUBLAS_STATUS_EXECUTION_FAILED) {
    ss << "CUBLAS: execution failed";
  } else if (stat == CUBLAS_STATUS_INTERNAL_ERROR) {
    ss << "CUBLAS: internal error";
  } else if (stat == CUBLAS_STATUS_NOT_SUPPORTED) {
    ss << "CUBLAS: not supported";
  } else if (stat == CUBLAS_STATUS_LICENSE_ERROR) {
    ss << "CUBLAS: license error";
  }
  ss << ", " << ::paddle::string::Sprintf(args...);
  ss << ::paddle::string::Sprintf(" at [%s:%s];", __FILE__, __LINE__);
  throw std::runtime_error(ss.str());
}

#endif  // PADDLE_ONLY_CPU

template <typename... Args>
inline void throw_on_error(int stat, const Args&... args) {
  if (UNLIKELY(!(stat))) {
    PADDLE_THROW(args...);
  }
}

/**
 * @brief Enforce a condition, otherwise throw an EnforceNotMet
 */
#define PADDLE_ENFORCE(condition, ...)                          \
  do {                                                          \
    ::paddle::platform::throw_on_error(condition, __VA_ARGS__); \
  } while (0)

}  // namespace platform
}  // namespace paddle
