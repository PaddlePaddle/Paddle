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
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/to_string.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cublas.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/fluid/platform/dynload/curand.h"
#if !defined(__APPLE__) && !defined(_WIN32)
#include "paddle/fluid/platform/dynload/nccl.h"
#endif  // __APPLE__
#endif  // PADDLE_WITH_CUDA

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
  std::string err_str_;
  EnforceNotMet(std::exception_ptr e, const char* f, int l) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception& e) {
      Init(e.what(), f, l);
    }
  }

  template <typename... ARGS>
  EnforceNotMet(const char* f, int l, ARGS... args) {
    Init(string::Sprintf(args...), f, l);
  }

  const char* what() const noexcept override { return err_str_.c_str(); }

 private:
  template <typename StrType>
  inline void Init(StrType what, const char* f, int l) {
    static constexpr int TRACE_STACK_LIMIT = 100;
    std::ostringstream sout;

    sout << string::Sprintf("%s at [%s:%d]", what, f, l) << std::endl;
    sout << "PaddlePaddle Call Stacks: " << std::endl;
#if !defined(_WIN32)
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
                                2 + sizeof(void*) * 2, call_stack[i], demangled,
                                addr_offset);
      } else {
        sout << string::Sprintf("%-3d %*0p\n", i, 2 + sizeof(void*) * 2,
                                call_stack[i]);
      }
    }
    free(symbols);
#else
    sout << "Windows not support stack backtrace yet.";
#endif
    err_str_ = sout.str();
  }
};

struct EOFException : public std::exception {
  std::string err_str_;
  EOFException(const char* err_msg, const char* f, int l) {
    err_str_ = string::Sprintf("%s at [%s:%d]", err_msg, f, l);
  }

  const char* what() const noexcept { return err_str_.c_str(); }
};

// Because most enforce conditions would evaluate to true, we can use
// __builtin_expect to instruct the C++ compiler to generate code that
// always forces branch prediction of true.
// This generates faster binary code. __builtin_expect is since C++11.
// For more details, please check https://stackoverflow.com/a/43870188/724872.
#if !defined(_WIN32)
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
#else
// there is no equivalent intrinsics in msvc.
#define UNLIKELY(condition) (condition)
#endif

#if !defined(_WIN32)
#define LIKELY(condition) __builtin_expect(static_cast<bool>(condition), 1)
#else
// there is no equivalent intrinsics in msvc.
#define LIKELY(condition) (condition)
#endif

inline bool is_error(bool stat) { return !stat; }

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    bool stat, const Args&... args) {
#ifndef REPLACE_ENFORCE_GLOG
  throw std::runtime_error(string::Sprintf(args...));
#else
  LOG(FATAL) << string::Sprintf(args...);
#endif
}

#ifdef PADDLE_WITH_CUDA

inline bool is_error(cudaError_t e) { return UNLIKELY(e); }

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    cudaError_t e, const Args&... args) {
#ifndef REPLACE_ENFORCE_GLOG
  throw thrust::system_error(e, thrust::cuda_category(),
                             string::Sprintf(args...));
#else
  LOG(FATAL) << string::Sprintf(args...);
#endif
}

inline bool is_error(curandStatus_t stat) {
  return stat != CURAND_STATUS_SUCCESS;
}

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    curandStatus_t stat, const Args&... args) {
#ifndef REPLACE_ENFORCE_GLOG
  throw thrust::system_error(cudaErrorLaunchFailure, thrust::cuda_category(),
                             string::Sprintf(args...));
#else
  LOG(FATAL) << string::Sprintf(args...);
#endif
}

inline bool is_error(cudnnStatus_t stat) {
  return stat != CUDNN_STATUS_SUCCESS;
}

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    cudnnStatus_t stat, const Args&... args) {
#ifndef REPLACE_ENFORCE_GLOG
  throw std::runtime_error(platform::dynload::cudnnGetErrorString(stat) +
                           string::Sprintf(args...));
#else
  LOG(FATAL) << string::Sprintf(args...);
#endif
}

inline bool is_error(cublasStatus_t stat) {
  return stat != CUBLAS_STATUS_SUCCESS;
}

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    cublasStatus_t stat, const Args&... args) {
  std::string err;
  if (stat == CUBLAS_STATUS_NOT_INITIALIZED) {
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
#ifndef REPLACE_ENFORCE_GLOG
  throw std::runtime_error(err + string::Sprintf(args...));
#else
  LOG(FATAL) << err << string::Sprintf(args...);
#endif
}

#if !defined(__APPLE__) && !defined(_WIN32)
template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    ncclResult_t stat, const Args&... args) {
  if (stat == ncclSuccess) {
    return;
  } else {
#ifndef REPLACE_ENFORCE_GLOG
    throw std::runtime_error(platform::dynload::ncclGetErrorString(stat) +
                             string::Sprintf(args...));
#else
    LOG(FATAL) << platform::dynload::ncclGetErrorString(stat)
               << string::Sprintf(args...);
#endif
  }
}
#endif  // __APPLE__ and windows
#endif  // PADDLE_WITH_CUDA

template <typename T>
inline void throw_on_error(T e) {
  throw_on_error(e, "");
}

#define PADDLE_THROW(...) \
  throw ::paddle::platform::EnforceNotMet(__FILE__, __LINE__, __VA_ARGS__)

#define __PADDLE_THROW_ERROR_I(_, _9, _8, _7, _6, _5, _4, _3, _2, X_, ...) X_;

#define __THROW_ON_ERROR_ONE_ARG(COND, ARG) \
  ::paddle::platform::throw_on_error(COND, ::paddle::string::Sprintf(ARG));

#ifdef _WIN32
#define __PADDLE_THROW_ON_ERROR(COND, ...) \
  __THROW_ON_ERROR_ONE_ARG(COND, __VA_ARGS__)
#else  // _WIN32
#define __PADDLE_THROW_ON_ERROR(COND, ...)                                \
  __PADDLE_THROW_ERROR_I(                                                 \
      __VA_ARGS__, ::paddle::platform::throw_on_error(COND, __VA_ARGS__), \
      ::paddle::platform::throw_on_error(COND, __VA_ARGS__),              \
      ::paddle::platform::throw_on_error(COND, __VA_ARGS__),              \
      ::paddle::platform::throw_on_error(COND, __VA_ARGS__),              \
      ::paddle::platform::throw_on_error(COND, __VA_ARGS__),              \
      ::paddle::platform::throw_on_error(COND, __VA_ARGS__),              \
      ::paddle::platform::throw_on_error(COND, __VA_ARGS__),              \
      ::paddle::platform::throw_on_error(COND, __VA_ARGS__),              \
      __THROW_ON_ERROR_ONE_ARG(COND, __VA_ARGS__))
#endif  // _WIN32

#define __PADDLE_UNARY_COMPARE(COND, ...)                 \
  do {                                                    \
    auto __cond = COND;                                   \
    if (UNLIKELY(::paddle::platform::is_error(__cond))) { \
      __PADDLE_THROW_ON_ERROR(__cond, __VA_ARGS__);       \
    }                                                     \
  } while (0)

#ifndef REPLACE_ENFORCE_GLOG
#define __PADDLE_ENFORCE_I(COND, ...)                                   \
  do {                                                                  \
    try {                                                               \
      __PADDLE_UNARY_COMPARE(COND, __VA_ARGS__);                        \
    } catch (...) {                                                     \
      throw ::paddle::platform::EnforceNotMet(std::current_exception(), \
                                              __FILE__, __LINE__);      \
    }                                                                   \
  } while (0)

#else
#define __PADDLE_ENFORCE_I(COND, ...) __PADDLE_UNARY_COMPARE(COND, __VA_ARGS__);
#endif  // REPLACE_ENFORCE_GLOG

#define __PADDLE_ENFORCE(__args) __PADDLE_ENFORCE_I __args
#define PADDLE_ENFORCE(...) __PADDLE_ENFORCE((__VA_ARGS__))

#define PADDLE_THROW_EOF()                                                     \
  do {                                                                         \
    throw ::paddle::platform::EOFException("There is no next data.", __FILE__, \
                                           __LINE__);                          \
  } while (false)

/*
 * Some enforce helpers here, usage:
 *    int a = 1;
 *    int b = 2;
 *    PADDLE_ENFORCE_EQ(a, b);
 *
 *    will raise an expression described as follows:
 *    "Enforce failed. Expected input a == b, but received a(1) != b(2)."
 *      with detailed stack information.
 *
 *    extra messages is also supported, for example:
 *    PADDLE_ENFORCE(a, b, "some simple enforce failed between %d numbers", 2)
 */
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
      PADDLE_THROW("Enforce failed. Expected %s " #__CMP                \
                   " %s, but received %s:%s " #__INV_CMP " %s:%s.\n%s", \
                   #__VAL0, #__VAL1, #__VAL0,                           \
                   paddle::string::to_string(__VAL0), #__VAL1,          \
                   paddle::string::to_string(__VAL1),                   \
                   paddle::string::Sprintf("" __VA_ARGS__));            \
    }                                                                   \
  } while (0)

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

}  // namespace platform
}  // namespace paddle
