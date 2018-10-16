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

#if defined(_WIN32)
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#define GOOGLE_GLOG_DLL_DECL
#endif

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

#ifdef PADDLE_WITH_HIP

#include "paddle/fluid/platform/dynload/hipblas.h"
#include "paddle/fluid/platform/dynload/miopen.h"
#include "paddle/fluid/platform/dynload/hiprand.h"
#include "paddle/fluid/platform/dynload/rccl.h"

#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include <miopen/miopen.h>
#include <hiprand.h>
#include <rccl.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

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
                                  2 + sizeof(void*) * 2, call_stack[i],
                                  demangled, addr_offset);
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
  }

  const char* what() const noexcept { return err_str_.c_str(); }
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
#define UNLIKELY(condition) (condition == 0)
#endif

#if !defined(_WIN32)
#define LIKELY(condition) __builtin_expect(static_cast<bool>(condition), 1)
#else
// there is no equivalent intrinsics in msvc.
#define LIKELY(condition) (condition != 0)
#endif

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    bool stat, const Args&... args) {
  if (UNLIKELY(!(stat))) {
#ifndef REPLACE_ENFORCE_GLOG
    throw std::runtime_error(string::Sprintf(args...));
#else
    LOG(FATAL) << string::Sprintf(args...);
#endif
  }
}

#ifdef PADDLE_WITH_CUDA

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    cudaError_t e, const Args&... args) {
  if (UNLIKELY(e)) {
#ifndef REPLACE_ENFORCE_GLOG
    throw thrust::system_error(e, thrust::cuda_category(),
                               string::Sprintf(args...));
#else
    LOG(FATAL) << string::Sprintf(args...);
#endif
  }
}

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    curandStatus_t stat, const Args&... args) {
  if (stat != CURAND_STATUS_SUCCESS) {
#ifndef REPLACE_ENFORCE_GLOG
    throw thrust::system_error(cudaErrorLaunchFailure, thrust::cuda_category(),
                               string::Sprintf(args...));
#else
    LOG(FATAL) << string::Sprintf(args...);
#endif
  }
}

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    cudnnStatus_t stat, const Args&... args) {
  if (stat == CUDNN_STATUS_SUCCESS) {
    return;
  } else {
#ifndef REPLACE_ENFORCE_GLOG
    throw std::runtime_error(platform::dynload::cudnnGetErrorString(stat) +
                             string::Sprintf(args...));
#else
    LOG(FATAL) << string::Sprintf(args...);
#endif
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


#ifdef PADDLE_WITH_HIP

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    hipError_t e, const Args&... args) {
  if (UNLIKELY(e)) {
    throw thrust::system_error(e, thrust::cuda_category(),
                               string::Sprintf(args...));
  }
}

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    hiprandStatus_t stat, const Args&... args) {
  if (stat != HIPRAND_STATUS_SUCCESS) {
    throw thrust::system_error(hipErrorLaunchFailure, thrust::cuda_category(),
                               string::Sprintf(args...));
  }
}

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    miopenStatus_t stat, const Args&... args) {
  if (stat == miopenStatusSuccess) {
    return;
  } else {
    throw std::runtime_error(platform::dynload::miopenGetErrorString(stat) +
                             string::Sprintf(args...));
  }
}

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    hipblasStatus_t stat, const Args&... args) {
  std::string err;
  if (stat == HIPBLAS_STATUS_SUCCESS) {
    return;
  } else if (stat == HIPBLAS_STATUS_NOT_INITIALIZED) {
    err = "CUBLAS: not initialized, ";
  } else if (stat == HIPBLAS_STATUS_ALLOC_FAILED) {
    err = "CUBLAS: alloc failed, ";
  } else if (stat == HIPBLAS_STATUS_INVALID_VALUE) {
    err = "CUBLAS: invalid value, ";
  } else if (stat == HIPBLAS_STATUS_MAPPING_ERROR) {
    err = "CUBLAS: mapping error, ";
  } else if (stat == HIPBLAS_STATUS_EXECUTION_FAILED) {
    err = "CUBLAS: execution failed, ";
  } else if (stat == HIPBLAS_STATUS_INTERNAL_ERROR) {
    err = "CUBLAS: internal error, ";
  } else if (stat == HIPBLAS_STATUS_NOT_SUPPORTED) {
    err = "CUBLAS: not supported, ";
  }
  throw std::runtime_error(err + string::Sprintf(args...));
}

template <typename... Args>
inline typename std::enable_if<sizeof...(Args) != 0, void>::type throw_on_error(
    rcclResult_t stat, const Args&... args) {
  if (stat == rcclSuccess) {
    return;
  } else {
    throw std::runtime_error(string::Sprintf(args...));
  }
}

#endif  // PADDLE_WITH_HIP

template <typename T>
inline void throw_on_error(T e) {
  throw_on_error(e, "");
}

#if !defined(_WIN32)
#define PADDLE_THROW(...)                                              \
  do {                                                                 \
    throw ::paddle::platform::EnforceNotMet(                           \
        std::make_exception_ptr(                                       \
            std::runtime_error(paddle::string::Sprintf(__VA_ARGS__))), \
        __FILE__, __LINE__);                                           \
  } while (false)

#ifndef REPLACE_ENFORCE_GLOG
#define PADDLE_ENFORCE(...)                                             \
  do {                                                                  \
    try {                                                               \
      ::paddle::platform::throw_on_error(__VA_ARGS__);                  \
    } catch (...) {                                                     \
      throw ::paddle::platform::EnforceNotMet(std::current_exception(), \
                                              __FILE__, __LINE__);      \
    }                                                                   \
  } while (false)

#else
#define PADDLE_ENFORCE(...) ::paddle::platform::throw_on_error(__VA_ARGS__);
#endif  // REPLACE_ENFORCE_GLOG

#else  // !_WIN32
// disable enforce, caused by the varardic macro exception error
#define PADDLE_THROW(x)                                      \
  do {                                                       \
    throw std::make_exception_ptr(                           \
        std::runtime_error("Windows disable the enforce.")); \
  } while (false)

#define PADDLE_ENFORCE(x, ...) x
#endif  // !_WIN32

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
#if !defined(_WIN32)
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
      PADDLE_THROW("Enforce failed. Expected %s " #__CMP                \
                   " %s, but received %s:%s " #__INV_CMP " %s:%s.\n%s", \
                   #__VAL0, #__VAL1, #__VAL0,                           \
                   paddle::string::to_string(__VAL0), #__VAL1,          \
                   paddle::string::to_string(__VAL1),                   \
                   paddle::string::Sprintf("" __VA_ARGS__));            \
    }                                                                   \
  } while (0)
#else
#define PADDLE_ENFORCE_EQ(__VAL0, __VAL1, ...) ((__VAL0) == (__VAL1))
#define PADDLE_ENFORCE_NE(__VAL0, __VAL1, ...) ((__VAL0) != (__VAL1))
#define PADDLE_ENFORCE_GT(__VAL0, __VAL1, ...) ((__VAL0) > (__VAL1))
#define PADDLE_ENFORCE_GE(__VAL0, __VAL1, ...) ((__VAL0) >= (__VAL1))
#define PADDLE_ENFORCE_LT(__VAL0, __VAL1, ...) ((__VAL0) < (__VAL1))
#define PADDLE_ENFORCE_LE(__VAL0, __VAL1, ...) ((__VAL0) <= (__VAL1))

#define __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, __CMP, __INV_CMP, ...) \
  do {                                                                 \
    if (!((__VAL0)__CMP(__VAL1))) {                                    \
      PADDLE_THROW("Windows disable the enforce. Enforce failed.");    \
    }                                                                  \
  } while (0)
#define PADDLE_ENFORCE_NOT_NULL(__VAL1, ...)                       \
  do {                                                             \
    if (nullptr == (__VAL1)) {                                     \
      PADDLE_THROW("Windows disable the enforce. Enforce failed"); \
    }                                                              \
  } while (0)
#endif  // !_WIN32

}  // namespace platform
}  // namespace paddle
