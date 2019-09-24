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

#include <fstream>
#include <iomanip>
#include <iostream>
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

template <typename StrType>
inline std::string GetTraceBackString(StrType&& what, const char* file,
                                      int line) {
  static constexpr int TRACE_STACK_LIMIT = 100;
  std::ostringstream sout;

  sout << string::Sprintf("%s at [%s:%d]", std::forward<StrType>(what), file,
                          line)
       << std::endl;
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
      sout << string::Sprintf("%-3d %*0p %s + %zd\n", i, 2 + sizeof(void*) * 2,
                              call_stack[i], demangled, addr_offset);
    } else {
      sout << string::Sprintf("%-3d %*0p\n", i, 2 + sizeof(void*) * 2,
                              call_stack[i]);
    }
  }
  free(symbols);
#else
  sout << "Windows not support stack backtrace yet.";
#endif
  return sout.str();
}

struct EnforceNotMet : public std::exception {
  std::string err_str_;
  EnforceNotMet(std::exception_ptr e, const char* file, int line) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception& e) {
      err_str_ = GetTraceBackString(e.what(), file, line);
    }
  }

  EnforceNotMet(const std::string& str, const char* file, int line)
      : err_str_(GetTraceBackString(str, file, line)) {}

  const char* what() const noexcept override { return err_str_.c_str(); }
};

struct EOFException : public std::exception {
  std::string err_str_;
  EOFException(const char* err_msg, const char* file, int line) {
    err_str_ = string::Sprintf("%s at [%s:%d]", err_msg, file, line);
  }

  const char* what() const noexcept override { return err_str_.c_str(); }
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

#ifdef PADDLE_WITH_CUDA
namespace details {

template <typename T>
struct CudaStatusType {};

#define DEFINE_CUDA_STATUS_TYPE(type, success_value) \
  template <>                                        \
  struct CudaStatusType<type> {                      \
    using Type = type;                               \
    static constexpr Type kSuccess = success_value;  \
  }

DEFINE_CUDA_STATUS_TYPE(cudaError_t, cudaSuccess);
DEFINE_CUDA_STATUS_TYPE(curandStatus_t, CURAND_STATUS_SUCCESS);
DEFINE_CUDA_STATUS_TYPE(cudnnStatus_t, CUDNN_STATUS_SUCCESS);
DEFINE_CUDA_STATUS_TYPE(cublasStatus_t, CUBLAS_STATUS_SUCCESS);

#if !defined(__APPLE__) && !defined(_WIN32)
DEFINE_CUDA_STATUS_TYPE(ncclResult_t, ncclSuccess);
#endif

}  // namespace details
#endif

#define PADDLE_THROW(...)                                            \
  do {                                                               \
    throw ::paddle::platform::EnforceNotMet(                         \
        ::paddle::string::Sprintf(__VA_ARGS__), __FILE__, __LINE__); \
  } while (0)

#if defined(__CUDA_ARCH__)
// For cuda, the assertions can affect performance and it is therefore
// recommended to disable them in production code
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#assertion
#define PADDLE_ENFORCE(_IS_NOT_ERROR, __FORMAT, ...)                   \
  do {                                                                 \
    if (!(_IS_NOT_ERROR)) {                                            \
      printf("Exception: %s:%d Assertion `%s` failed. " __FORMAT "\n", \
             __FILE__, __LINE__, #_IS_NOT_ERROR, ##__VA_ARGS__);       \
      asm("trap;");                                                    \
    }                                                                  \
  } while (0)
#else
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
#endif

#ifdef PADDLE_WITH_CUDA
#define PADDLE_ENFORCE_CUDA_SUCCESS(COND, ...)                            \
  do {                                                                    \
    auto __cond__ = (COND);                                               \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);                      \
    constexpr auto __success_type__ =                                     \
        ::paddle::platform::details::CudaStatusType<                      \
            __CUDA_STATUS_TYPE__>::kSuccess;                              \
    if (UNLIKELY(__cond__ != __success_type__)) {                         \
      try {                                                               \
        ::paddle::platform::throw_on_error(                               \
            __cond__, ::paddle::string::Sprintf(__VA_ARGS__));            \
      } catch (...) {                                                     \
        throw ::paddle::platform::EnforceNotMet(std::current_exception(), \
                                                __FILE__, __LINE__);      \
      }                                                                   \
    }                                                                     \
  } while (0)

#undef DEFINE_CUDA_STATUS_TYPE
#endif

#define PADDLE_THROW_EOF()                                                     \
  do {                                                                         \
    throw ::paddle::platform::EOFException("There is no next data.", __FILE__, \
                                           __LINE__);                          \
  } while (0)

#define PADDLE_THROW_BAD_ALLOC(...)                                  \
  do {                                                               \
    throw ::paddle::memory::allocation::BadAlloc(                    \
        ::paddle::string::Sprintf(__VA_ARGS__), __FILE__, __LINE__); \
  } while (0)

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

// Here, we use SFINAE to check whether T can be converted to std::string
template <typename T>
struct CanToString {
 private:
  using YesType = uint8_t;
  using NoType = uint16_t;

  template <typename U>
  static YesType Check(decltype(std::cout << std::declval<U>())) {
    return 0;
  }

  template <typename U>
  static NoType Check(...) {
    return 0;
  }

 public:
  static constexpr bool kValue =
      std::is_same<YesType, decltype(Check<T>(std::cout))>::value;
};

template <bool kCanToString /* = true */>
struct BinaryCompareMessageConverter {
  template <typename T>
  static std::string Convert(const char* expression, const T& value) {
    return expression + std::string(":") + string::to_string(value);
  }
};

template <>
struct BinaryCompareMessageConverter<false> {
  template <typename T>
  static const char* Convert(const char* expression, const T& value) {
    return expression;
  }
};

}  // namespace details

#define __PADDLE_BINARY_COMPARE(__VAL1, __VAL2, __CMP, __INV_CMP, ...)         \
  do {                                                                         \
    auto __val1 = (__VAL1);                                                    \
    auto __val2 = (__VAL2);                                                    \
    using __TYPE1__ = decltype(__val1);                                        \
    using __TYPE2__ = decltype(__val2);                                        \
    using __COMMON_TYPE1__ =                                                   \
        ::paddle::platform::details::CommonType1<__TYPE1__, __TYPE2__>;        \
    using __COMMON_TYPE2__ =                                                   \
        ::paddle::platform::details::CommonType2<__TYPE1__, __TYPE2__>;        \
    bool __is_not_error = (static_cast<__COMMON_TYPE1__>(__val1))__CMP(        \
        static_cast<__COMMON_TYPE2__>(__val2));                                \
    if (UNLIKELY(!__is_not_error)) {                                           \
      constexpr bool __kCanToString__ =                                        \
          ::paddle::platform::details::CanToString<__TYPE1__>::kValue &&       \
          ::paddle::platform::details::CanToString<__TYPE2__>::kValue;         \
      PADDLE_THROW("Enforce failed. Expected %s " #__CMP                       \
                   " %s, but received %s " #__INV_CMP " %s.\n%s",              \
                   #__VAL1, #__VAL2,                                           \
                   ::paddle::platform::details::BinaryCompareMessageConverter< \
                       __kCanToString__>::Convert(#__VAL1, __val1),            \
                   ::paddle::platform::details::BinaryCompareMessageConverter< \
                       __kCanToString__>::Convert(#__VAL2, __val2),            \
                   ::paddle::string::Sprintf(__VA_ARGS__));                    \
    }                                                                          \
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

#define __PADDLE_INFERSHAPE_BINARY_COMPARE(__CTX, __VAL1, __VAL2, __CMP, \
                                           __INV_CMP, ...)               \
  do {                                                                   \
    auto __val1 = (__VAL1);                                              \
    auto __val2 = (__VAL2);                                              \
    if (!__CTX->IsRuntime()) {                                           \
      if (__val1 == -1 || __val2 == -1) {                                \
        break;                                                           \
      }                                                                  \
    }                                                                    \
    using __TYPE1__ = decltype(__val1);                                  \
    using __TYPE2__ = decltype(__val2);                                  \
    using __COMMON_TYPE1__ =                                             \
        ::paddle::platform::details::CommonType1<__TYPE1__, __TYPE2__>;  \
    using __COMMON_TYPE2__ =                                             \
        ::paddle::platform::details::CommonType2<__TYPE1__, __TYPE2__>;  \
    bool __is_not_error = (static_cast<__COMMON_TYPE1__>(__val1))__CMP(  \
        static_cast<__COMMON_TYPE2__>(__val2));                          \
    if (UNLIKELY(!__is_not_error)) {                                     \
      PADDLE_THROW("Enforce failed. Expected %s " #__CMP                 \
                   " %s, but received %s:%s " #__INV_CMP " %s:%s.\n%s",  \
                   #__VAL1, #__VAL2, #__VAL1,                            \
                   ::paddle::string::to_string(__val1), #__VAL2,         \
                   ::paddle::string::to_string(__val2),                  \
                   ::paddle::string::Sprintf(__VA_ARGS__));              \
    }                                                                    \
  } while (0)

#define PADDLE_INFERSHAPE_ENFORCE_EQ(__CTX, __VAL0, __VAL1, ...) \
  __PADDLE_INFERSHAPE_BINARY_COMPARE(__CTX, __VAL0, __VAL1, ==, !=, __VA_ARGS__)
#define PADDLE_INFERSHAPE_ENFORCE_NE(__CTX, __VAL0, __VAL1, ...) \
  __PADDLE_INFERSHAPE_BINARY_COMPARE(__CTX, __VAL0, __VAL1, !=, ==, __VA_ARGS__)
#define PADDLE_INFERSHAPE_ENFORCE_GT(__CTX, __VAL0, __VAL1, ...) \
  __PADDLE_INFERSHAPE_BINARY_COMPARE(__CTX, __VAL0, __VAL1, >, <=, __VA_ARGS__)
#define PADDLE_INFERSHAPE_ENFORCE_GE(__CTX, __VAL0, __VAL1, ...) \
  __PADDLE_INFERSHAPE_BINARY_COMPARE(__CTX, __VAL0, __VAL1, >=, <, __VA_ARGS__)
#define PADDLE_INFERSHAPE_ENFORCE_LT(__CTX, __VAL0, __VAL1, ...) \
  __PADDLE_INFERSHAPE_BINARY_COMPARE(__CTX, __VAL0, __VAL1, <, >=, __VA_ARGS__)
#define PADDLE_INFERSHAPE_ENFORCE_LE(__CTX, __VAL0, __VAL1, ...) \
  __PADDLE_INFERSHAPE_BINARY_COMPARE(__CTX, __VAL0, __VAL1, <=, >, __VA_ARGS__)

}  // namespace platform
}  // namespace paddle
