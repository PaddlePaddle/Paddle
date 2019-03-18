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
#include <type_traits>
#include <utility>

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
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

  EnforceNotMet(const std::string& str, const char* f, int l) {
    Init(str, f, l);
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

inline void throw_on_error(bool stat, const std::string& msg) {
#ifndef REPLACE_ENFORCE_GLOG
  throw std::runtime_error(msg);
#else
  LOG(FATAL) << msg;
#endif
}

#ifdef PADDLE_WITH_CUDA

inline bool is_error(cudaError_t e) { return e != cudaSuccess; }

inline void throw_on_error(cudaError_t e, const std::string& msg) {
#ifndef REPLACE_ENFORCE_GLOG
  throw thrust::system_error(e, thrust::cuda_category(), msg);
#else
  LOG(FATAL) << msg;
#endif
}

inline bool is_error(curandStatus_t stat) {
  return stat != CURAND_STATUS_SUCCESS;
}

inline void throw_on_error(curandStatus_t stat, const std::string& msg) {
#ifndef REPLACE_ENFORCE_GLOG
  throw thrust::system_error(cudaErrorLaunchFailure, thrust::cuda_category(),
                             msg);
#else
  LOG(FATAL) << msg;
#endif
}

inline bool is_error(cudnnStatus_t stat) {
  return stat != CUDNN_STATUS_SUCCESS;
}

inline void throw_on_error(cudnnStatus_t stat, const std::string& msg) {
#ifndef REPLACE_ENFORCE_GLOG
  throw std::runtime_error(platform::dynload::cudnnGetErrorString(stat) + msg);
#else
  LOG(FATAL) << platform::dynload::cudnnGetErrorString(stat) << msg;
#endif
}

inline bool is_error(cublasStatus_t stat) {
  return stat != CUBLAS_STATUS_SUCCESS;
}

inline void throw_on_error(cublasStatus_t stat, const std::string& msg) {
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
  throw std::runtime_error(err + msg);
#else
  LOG(FATAL) << err << msg;
#endif
}

#if !defined(__APPLE__) && !defined(_WIN32)
inline bool is_error(ncclResult_t nccl_result) {
  return nccl_result != ncclSuccess;
}

inline void throw_on_error(ncclResult_t stat, const std::string& msg) {
#ifndef REPLACE_ENFORCE_GLOG
  throw std::runtime_error(platform::dynload::ncclGetErrorString(stat) + msg);
#else
  LOG(FATAL) << platform::dynload::ncclGetErrorString(stat) << msg;
#endif
}
#endif  // __APPLE__ and windows
#endif  // PADDLE_WITH_CUDA

#define PADDLE_THROW(...)                                            \
  do {                                                               \
    throw ::paddle::platform::EnforceNotMet(                         \
        ::paddle::string::Sprintf(__VA_ARGS__), __FILE__, __LINE__); \
  } while (0)

#define PADDLE_ENFORCE(COND, ...)                                         \
  do {                                                                    \
    auto __cond__ = (COND);                                               \
    if (UNLIKELY(::paddle::platform::is_error(__cond__))) {               \
      try {                                                               \
        ::paddle::platform::throw_on_error(                               \
            __cond__, ::paddle::string::Sprintf(__VA_ARGS__));            \
      } catch (...) {                                                     \
        throw ::paddle::platform::EnforceNotMet(std::current_exception(), \
                                                __FILE__, __LINE__);      \
      }                                                                   \
    }                                                                     \
  } while (0)

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
#define PADDLE_ENFORCE_NOT_NULL(__VAL, ...)                 \
  do {                                                      \
    if (UNLIKELY(nullptr == (__VAL))) {                     \
      PADDLE_THROW(#__VAL " should not be null\n%s",        \
                   ::paddle::string::Sprintf(__VA_ARGS__)); \
    }                                                       \
  } while (0)

namespace details {
template <typename T>
inline constexpr bool IsArithmetic() {
  return std::is_arithmetic<T>::value;
}

template <typename T1, typename T2, bool kIsArithmetic /* = true */>
struct TypeConverterImpl {
  using Type1 = typename std::common_type<T1, T2>::type;
  using Type2 = Type1;
};

template <typename T1, typename T2>
struct TypeConverterImpl<T1, T2, false> {
  using Type1 = T1;
  using Type2 = T2;
};

template <typename T1, typename T2>
struct TypeConverter {
 private:
  static constexpr bool kIsArithmetic =
      IsArithmetic<T1>() && IsArithmetic<T2>();

 public:
  using Type1 = typename TypeConverterImpl<T1, T2, kIsArithmetic>::Type1;
  using Type2 = typename TypeConverterImpl<T1, T2, kIsArithmetic>::Type2;
};

template <typename T1, typename T2>
using CommonType1 = typename std::add_lvalue_reference<
    typename std::add_const<typename TypeConverter<T1, T2>::Type1>::type>::type;

template <typename T1, typename T2>
using CommonType2 = typename std::add_lvalue_reference<
    typename std::add_const<typename TypeConverter<T1, T2>::Type2>::type>::type;
}  // namespace details

#define __PADDLE_BINARY_COMPARE(__VAL1, __VAL2, __CMP, __INV_CMP, ...)  \
  do {                                                                  \
    auto __val1 = (__VAL1);                                             \
    auto __val2 = (__VAL2);                                             \
    using __TYPE1__ = decltype(__val1);                                 \
    using __TYPE2__ = decltype(__val2);                                 \
    using __COMMON_TYPE1__ =                                            \
        ::paddle::platform::details::CommonType1<__TYPE1__, __TYPE2__>; \
    using __COMMON_TYPE2__ =                                            \
        ::paddle::platform::details::CommonType2<__TYPE1__, __TYPE2__>; \
    bool __is_not_error = (static_cast<__COMMON_TYPE1__>(__val1))__CMP( \
        static_cast<__COMMON_TYPE2__>(__val2));                         \
    if (UNLIKELY(!__is_not_error)) {                                    \
      PADDLE_THROW("Enforce failed. Expected %s " #__CMP                \
                   " %s, but received %s:%s " #__INV_CMP " %s:%s.\n%s", \
                   #__VAL1, #__VAL2, #__VAL1,                           \
                   ::paddle::string::to_string(__val1), #__VAL2,        \
                   ::paddle::string::to_string(__val2),                 \
                   ::paddle::string::Sprintf(__VA_ARGS__));             \
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
