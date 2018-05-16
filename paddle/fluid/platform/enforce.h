/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <dlfcn.h>     // for dladdr
#include <execinfo.h>  // for backtrace

#ifdef __GNUC__
#include <cxxabi.h>  // for __cxa_demangle
#endif               // __GNUC__

#ifdef PADDLE_WITH_CUDA
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#endif  // PADDLE_WITH_CUDA

#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "glog/logging.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/to_string.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cublas.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/fluid/platform/dynload/curand.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#endif

namespace paddle {
namespace platform {

#ifdef __GNUC__
inline std::string demangle(std::string name) {
  int status = -4;  // some arbitrary value to eliminate the compiler warning
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(name.c_str(), NULL, NULL, &status), std::free};
  return (status == 0) ? res.get() : name;
}
#else
inline std::string demangle(std::string name) { return name; }
#endif

struct EnforceNotMet : public std::exception {
  std::exception_ptr exp_;
  std::string err_str_;
  EnforceNotMet(std::exception_ptr e, const char* f, int l) : exp_(e) {
    static constexpr int TRACE_STACK_LIMIT = 100;
    try {
      std::rethrow_exception(exp_);
    } catch (const std::exception& exp) {
      std::ostringstream sout;

      sout << string::Sprintf("%s at [%s:%d]", exp.what(), f, l) << std::endl;
      sout << "PaddlePaddle Call Stacks: " << std::endl;

      void* call_stack[TRACE_STACK_LIMIT];
      auto size = backtrace(call_stack, TRACE_STACK_LIMIT);
      auto symbols = backtrace_symbols(call_stack, size);

      Dl_info info;
      for (int i = 0; i < size; ++i) {
        if (dladdr(call_stack[i], &info) && info.dli_sname) {
          auto demangled = demangle(info.dli_sname);
          auto addr_offset = static_cast<char*>(call_stack[i]) -
                             static_cast<char*>(info.dli_saddr);
          sout << string::Sprintf("%-3d %*0p %s + %zd\n", i,
                                  2 + sizeof(void*) * 2, call_stack[i],
                                  demangled, addr_offset);
        } else {
          sout << string::Sprintf("%-3d %*0p\n", i, 2 + sizeof(void*) * 2,
                                  call_stack[i]);
        }
      }
      free(symbols);
      err_str_ = sout.str();
    }
  }

  const char* what() const noexcept { return err_str_.c_str(); }
};

// Because most enforce conditions would evaluate to true, we can use
// __builtin_expect to instruct the C++ compiler to generate code that
// always forces branch prediction of true.
// This generates faster binary code. __builtin_expect is since C++11.
// For more details, please check https://stackoverflow.com/a/43870188/724872.
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    bool stat, const Args&... args) {
  if (UNLIKELY(!(stat))) {
    throw std::runtime_error(string::Sprintf(args...));
  }
}

#ifdef PADDLE_WITH_CUDA

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    cudaError_t e, const Args&... args) {
  if (UNLIKELY(e)) {
    throw thrust::system_error(e, thrust::cuda_category(),
                               string::Sprintf(args...));
  }
}

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    curandStatus_t stat, const Args&... args) {
  if (stat != CURAND_STATUS_SUCCESS) {
    throw thrust::system_error(cudaErrorLaunchFailure, thrust::cuda_category(),
                               string::Sprintf(args...));
  }
}

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    cudnnStatus_t stat, const Args&... args) {
  if (stat == CUDNN_STATUS_SUCCESS) {
    return;
  } else {
    throw std::runtime_error(platform::dynload::cudnnGetErrorString(stat) +
                             string::Sprintf(args...));
  }
}

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    cublasStatus_t stat, const Args&... args) {
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
  throw std::runtime_error(err + string::Sprintf(args...));
}

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    ncclResult_t stat, const Args&... args) {
  if (stat == ncclSuccess) {
    return;
  } else {
    throw std::runtime_error(platform::dynload::ncclGetErrorString(stat) +
                             string::Sprintf(args...));
  }
}

#endif  // PADDLE_WITH_CUDA

template <typename T>
inline void throw_on_error(T e) {
  throw_on_error(e, "");
}

#define PADDLE_THROW(...)                                              \
  do {                                                                 \
    throw ::paddle::platform::EnforceNotMet(                           \
        std::make_exception_ptr(                                       \
            std::runtime_error(paddle::string::Sprintf(__VA_ARGS__))), \
        __FILE__, __LINE__);                                           \
  } while (false)

#define PADDLE_ENFORCE(...)                                             \
  do {                                                                  \
    try {                                                               \
      ::paddle::platform::throw_on_error(__VA_ARGS__);                  \
    } catch (...) {                                                     \
      throw ::paddle::platform::EnforceNotMet(std::current_exception(), \
                                              __FILE__, __LINE__);      \
    }                                                                   \
  } while (false)

/*
 * Some enforce helpers here, usage:
 *    int a = 1;
 *    int b = 2;
 *    PADDLE_ENFORCE_EQ(a, b);
 *
 *    will raise an expression described as follows:
 *    "enforce a == b failed, 1 != 2" with detailed stack information.
 *
 *    extra messages is also supported, for example:
 *    PADDLE_ENFORCE(a, b, "some simple enforce failed between %d numbers", 2)
 */

#define PADDLE_ENFORCE_EQ(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, ==, !=, __VA_ARGS__)
#define PADDLE_ENFORCE_NE(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, !=, ==, __VA_ARGS__)
#define PADDLE_ENFORCE_GT(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, >, <=, __VA_ARGS__)
#define PADDLE_ENFORCE_GE(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, >=, <, __VA_ARGS__)
#define PADDLE_ENFORCE_LT(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, <, >=, __VA_ARGS__)
#define PADDLE_ENFORCE_LE(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, <=, >, __VA_ARGS__)
#define PADDLE_ENFORCE_NOT_NULL(__VAL, ...)                  \
  do {                                                       \
    if (UNLIKELY(nullptr == (__VAL))) {                      \
      PADDLE_THROW(#__VAL " should not be null\n%s",         \
                   paddle::string::Sprintf("" __VA_ARGS__)); \
    }                                                        \
  } while (0)

#define __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, __CMP, __INV_CMP, ...)  \
  do {                                                                  \
    if (UNLIKELY(!((__VAL0)__CMP(__VAL1)))) {                           \
      PADDLE_THROW("enforce %s " #__CMP " %s failed, %s " #__INV_CMP    \
                   " %s\n%s",                                           \
                   #__VAL0, #__VAL1, paddle::string::to_string(__VAL0), \
                   paddle::string::to_string(__VAL1),                   \
                   paddle::string::Sprintf("" __VA_ARGS__));            \
    }                                                                   \
  } while (0)

}  // namespace platform
}  // namespace paddle
