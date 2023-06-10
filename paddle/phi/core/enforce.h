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

#if !defined(_WIN32)
#include <dlfcn.h>   // dladdr
#include <unistd.h>  // sleep, usleep
#else                // _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#include <windows.h>  // GetModuleFileName, Sleep
#endif

#ifdef PADDLE_WITH_CUDA
#include <cublas_v2.h>
#include <cudnn.h>
#include <cufft.h>
#include <curand.h>
#include <cusparse.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include <hiprand.h>
#include <miopen/miopen.h>
#include <rocblas.h>
#include <thrust/system/hip/error.h>
#include <thrust/system_error.h>  // NOLINT
#endif

#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include "paddle/phi/core/macros.h"
#if !defined(_WIN32) && !defined(PADDLE_WITH_MUSL)
#include <execinfo.h>
#endif

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include "paddle/phi/core/errors.h"

#include "paddle/phi/backends/dynload/port.h"
#include "paddle/utils/string/printf.h"
#include "paddle/utils/string/to_string.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/backends/dynload/curand.h"
#include "paddle/phi/backends/dynload/cusolver.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
#include <error.h>

#include "paddle/phi/backends/dynload/nccl.h"
#endif  // __APPLE__
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/hipfft.h"
#include "paddle/phi/backends/dynload/hiprand.h"
#include "paddle/phi/backends/dynload/miopen.h"
#include "paddle/phi/backends/dynload/rocblas.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
#include <error.h>  // NOLINT

#include "paddle/phi/backends/dynload/rccl.h"
#endif  // __APPLE__
#endif  // PADDLE_WITH_HIP

// Note: these headers for simplify demangle type string
#include "paddle/phi/core/type_defs.h"
// Note: this header for simplify HIP and CUDA type string
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/gpu_types.h"
#endif

#include "paddle/utils/variant.h"

namespace phi {
class ErrorSummary;
}  // namespace phi

namespace phi {
namespace proto {}  // namespace proto
}  // namespace phi

namespace phi {
namespace enforce {

/** HELPER MACROS AND FUNCTIONS **/
#ifndef PADDLE_MAY_THROW
#define PADDLE_MAY_THROW noexcept(false)
#endif

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

#if defined _WIN32 && defined PADDLE_ON_INFERENCE && defined PADDLE_NO_PYTHON
#define HANDLE_THE_ERROR try {
#define END_HANDLE_THE_ERROR            \
  }                                     \
  catch (const std::exception& e) {     \
    std::cout << e.what() << std::endl; \
    throw;                              \
  }
#else
#define HANDLE_THE_ERROR
#define END_HANDLE_THE_ERROR
#endif

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
  static constexpr bool kIsArithmetic =
      IsArithmetic<T1>() && IsArithmetic<T2>();
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
    return expression + std::string(":") + paddle::string::to_string(value);
  }
};

template <>
struct BinaryCompareMessageConverter<false> {
  template <typename T>
  static const char* Convert(const char* expression, const T& value UNUSED) {
    return expression;
  }
};
}  // namespace details

int GetCallStackLevel();
std::string GetCurrentTraceBackString(bool for_signal = false);
std::string SimplifyErrorTypeFormat(const std::string& str);

template <typename StrType>
static std::string GetErrorSumaryString(StrType&& what,
                                        const char* file,
                                        int line) {
  std::ostringstream sout;
  if (GetCallStackLevel() > 1) {
    sout << "\n----------------------\nError Message "
            "Summary:\n----------------------\n";
  }
  sout << paddle::string::Sprintf(
              "%s (at %s:%d)", std::forward<StrType>(what), file, line)
       << std::endl;
  return sout.str();
}

template <typename StrType>
std::string GetCompleteTraceBackString(StrType&& what,
                                       const char* file,
                                       int line) {
  std::ostringstream sout;
  sout << "\n----------------------\nError Message "
          "Summary:\n----------------------\n";
  sout << paddle::string::Sprintf(
              "%s (at %s:%d)", std::forward<StrType>(what), file, line)
       << std::endl;
  return GetCurrentTraceBackString() + sout.str();
}

template <typename StrType>
static std::string GetTraceBackString(StrType&& what,
                                      const char* file,
                                      int line) {
  if (GetCallStackLevel() > 1) {
    // FLAGS_call_stack_level>1 means showing c++ call stack
    return GetCurrentTraceBackString() + GetErrorSumaryString(what, file, line);
  } else {
    return GetErrorSumaryString(what, file, line);
  }
}

inline bool is_error(bool stat) { return !stat; }

// Note: This Macro can only be used within enforce.h
#define __THROW_ERROR_INTERNAL__(__ERROR_SUMMARY)                             \
  do {                                                                        \
    HANDLE_THE_ERROR                                                          \
    throw ::phi::enforce::EnforceNotMet(__ERROR_SUMMARY, __FILE__, __LINE__); \
    END_HANDLE_THE_ERROR                                                      \
  } while (0)

/**
 * [Why declare function ThrowWarnInternal instead of defining macro
 * __THROW_WARN_INTERNAL__?]
 * ThrowWarnInternal uses `LOG` macro to display warning message, which depends
 * on third-party header file "logging.h". However, "logging.h" has not been
 * exposed to site-package yet, so that error will occur when we include
 * "enforce.h" header file. Hence, we declare function in enforce.h and define
 * it in enforce.cc file.
 */
void ThrowWarnInternal(const std::string& message);

/** ENFORCE EXCEPTION AND MACROS **/

struct EnforceNotMet : public std::exception {
 public:
  EnforceNotMet(std::exception_ptr e, const char* file, int line) {
    try {
      std::rethrow_exception(e);
    } catch (EnforceNotMet& e) {
      code_ = e.code();
      err_str_ = GetTraceBackString(e.what(), file, line);
      simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
    } catch (std::exception& e) {
      err_str_ = GetTraceBackString(e.what(), file, line);
      simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
    }
  }

  EnforceNotMet(const std::string& str, const char* file, int line)
      : err_str_(GetTraceBackString(str, file, line)) {
    simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
  }

  EnforceNotMet(const phi::ErrorSummary& error, const char* file, int line)
      : code_(error.code()),
        err_str_(GetTraceBackString(error.to_string(), file, line)) {
    simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
  }

  const char* what() const noexcept override {
    if (GetCallStackLevel() > 1) {
      return err_str_.c_str();
    } else {
      return simple_err_str_.c_str();
    }
  }

  phi::ErrorCode code() const { return code_; }

  const std::string& error_str() const { return err_str_; }

  const std::string& simple_error_str() const { return simple_err_str_; }

  void set_error_str(std::string str) {
    if (GetCallStackLevel() > 1) {
      err_str_ = str;
    } else {
      simple_err_str_ = str;
    }
  }

  ~EnforceNotMet() override = default;

 private:
  // Used to determine the final type of exception thrown
  phi::ErrorCode code_ = phi::ErrorCode::LEGACY;
  // Complete error message
  // e.g. InvalidArgumentError: ***
  std::string err_str_;
  // Simple error message used when no C++ stack and python compile stack
  // e.g. (InvalidArgument) ***
  std::string simple_err_str_;
};

#define PADDLE_THROW(...)                                      \
  do {                                                         \
    HANDLE_THE_ERROR                                           \
    throw ::phi::enforce::EnforceNotMet(                       \
        ::phi::ErrorSummary(__VA_ARGS__), __FILE__, __LINE__); \
    END_HANDLE_THE_ERROR                                       \
  } while (0)

#if defined(__CUDA_ARCH__)
// For cuda, the assertions can affect performance and it is therefore
// recommended to disable them in production code
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#assertion
#define PADDLE_ENFORCE(_IS_NOT_ERROR, __FORMAT, ...)               \
  do {                                                             \
    if (!(_IS_NOT_ERROR)) {                                        \
      printf("Error: %s:%d Assertion `%s` failed. " __FORMAT "\n", \
             __FILE__,                                             \
             __LINE__,                                             \
             #_IS_NOT_ERROR,                                       \
             ##__VA_ARGS__);                                       \
      asm("trap;");                                                \
    }                                                              \
  } while (0)
#elif defined(__HIPCC__)
#define PADDLE_ENFORCE(_IS_NOT_ERROR, __FORMAT, ...)               \
  do {                                                             \
    if (!(_IS_NOT_ERROR)) {                                        \
      printf("Error: %s:%d Assertion `%s` failed. " __FORMAT "\n", \
             __FILE__,                                             \
             __LINE__,                                             \
             #_IS_NOT_ERROR,                                       \
             ##__VA_ARGS__);                                       \
      abort();                                                     \
    }                                                              \
  } while (0)
#else
#define PADDLE_ENFORCE(COND, ...)                               \
  do {                                                          \
    auto __cond__ = (COND);                                     \
    if (UNLIKELY(::phi::is_error(__cond__))) {                  \
      __THROW_ERROR_INTERNAL__(phi::ErrorSummary(__VA_ARGS__)); \
    }                                                           \
  } while (0)
#endif

/*
 * Some enforce helpers here, usage:
 *    int a = 1;
 *    int b = 2;
 *    PADDLE_ENFORCE_EQ(a, b);
 *
 *    will raise an expression described as follows:
 *    "Expected input a == b, but received a(1) != b(2)."
 *      with detailed stack information.
 *
 *    extra messages is also supported, for example:
 *    PADDLE_ENFORCE(a, b, "some simple enforce failed between %d numbers", 2)
 */

#define PADDLE_ENFORCE_NOT_NULL(__VAL, ...)                    \
  do {                                                         \
    if (UNLIKELY(nullptr == (__VAL))) {                        \
      auto __summary__ = phi::ErrorSummary(__VA_ARGS__);       \
      auto __message__ = ::paddle::string::Sprintf(            \
          "%s\n  [Hint: " #__VAL " should not be null.]",      \
          __summary__.error_message());                        \
      __THROW_ERROR_INTERNAL__(                                \
          phi::ErrorSummary(__summary__.code(), __message__)); \
    }                                                          \
  } while (0)

#define PADDLE_WARN_NOT_NULL(__VAL, ...)                  \
  do {                                                    \
    if (UNLIKELY(nullptr == (__VAL))) {                   \
      auto __summary__ = phi::ErrorSummary(__VA_ARGS__);  \
      auto __message__ = ::paddle::string::Sprintf(       \
          "%s\n  [Hint: " #__VAL " should not be null.]", \
          __summary__.error_message());                   \
      ::phi::enforce::ThrowWarnInternal(__message__);     \
    }                                                     \
  } while (0)

#define __PADDLE_BINARY_COMPARE(__VAL1, __VAL2, __CMP, __INV_CMP, ...)  \
  do {                                                                  \
    auto __val1 = (__VAL1);                                             \
    auto __val2 = (__VAL2);                                             \
    using __TYPE1__ = decltype(__val1);                                 \
    using __TYPE2__ = decltype(__val2);                                 \
    using __COMMON_TYPE1__ =                                            \
        ::phi::details::CommonType1<__TYPE1__, __TYPE2__>;              \
    using __COMMON_TYPE2__ =                                            \
        ::phi::details::CommonType2<__TYPE1__, __TYPE2__>;              \
    bool __is_not_error = (static_cast<__COMMON_TYPE1__>(__val1))__CMP( \
        static_cast<__COMMON_TYPE2__>(__val2));                         \
    if (UNLIKELY(!__is_not_error)) {                                    \
      auto __summary__ = phi::ErrorSummary(__VA_ARGS__);                \
      constexpr bool __kCanToString__ =                                 \
          ::phi::details::CanToString<__TYPE1__>::kValue &&             \
          ::phi::details::CanToString<__TYPE2__>::kValue;               \
      auto __message__ = ::paddle::string::Sprintf(                     \
          "%s\n  [Hint: Expected %s " #__CMP                            \
          " %s, but received %s " #__INV_CMP " %s.]",                   \
          __summary__.error_message(),                                  \
          #__VAL1,                                                      \
          #__VAL2,                                                      \
          ::phi::details::BinaryCompareMessageConverter<                \
              __kCanToString__>::Convert(#__VAL1, __val1),              \
          ::phi::details::BinaryCompareMessageConverter<                \
              __kCanToString__>::Convert(#__VAL2, __val2));             \
      __THROW_ERROR_INTERNAL__(                                         \
          phi::ErrorSummary(__summary__.code(), __message__));          \
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

/** EXTENDED TOOL FUNCTIONS WITH CHECKING **/

/*
 * Summary: This macro is used to get Variable or internal type
 *   data (such as LoDTensor or SelectedRows) of the Input and
 *   Output in op, generally used when call scope.FindVar(Input/
 *   Output("Name")) or ctx.Input<LoDTensor>().
 *   Firstly this macro check whether the obtained pointer is null,
 *   and then return data if it is not null.
 *
 * Note: This macro is only suitable for specific scenarios and
 *   does not intended to be widely used. If it cannot meet the
 *   requirements, please use other PADDLE_ENFORCE** check macro.
 *
 * Parameters:
 *     __PTR: pointer
 *     __ROLE: (string), Input or Output
 *     __NAME: (string), Input or Output name
 *     __OP_TYPE: (string), the op type
 *
 * Return: The data pointed to by the pointer.
 *
 * Examples:
 *    GET_DATA_SAFELY(ctx.Input<LoDTensor>("X"), "Input", "X", "Mul");
 */
#define GET_DATA_SAFELY(__PTR, __ROLE, __NAME, __OP_TYPE)               \
  (([&]() -> std::add_lvalue_reference<decltype(*(__PTR))>::type {      \
    auto* __ptr = (__PTR);                                              \
    if (UNLIKELY(nullptr == __ptr)) {                                   \
      auto __summary__ = phi::errors::NotFound(                         \
          "Unable to get %s data of %s %s in operator %s. "             \
          "Possible reasons are:\n"                                     \
          "  1. The %s is not the %s of operator %s;\n"                 \
          "  2. The %s has no corresponding variable passed in;\n"      \
          "  3. The %s corresponding variable is not initialized.",     \
          phi::demangle(                                                \
              typeid(std::add_lvalue_reference<decltype(*__ptr)>::type) \
                  .name()),                                             \
          __ROLE,                                                       \
          __NAME,                                                       \
          __OP_TYPE,                                                    \
          __NAME,                                                       \
          __ROLE,                                                       \
          __OP_TYPE,                                                    \
          __NAME,                                                       \
          __NAME);                                                      \
      auto __message__ = ::paddle::string::Sprintf(                     \
          "%s\n  [Hint: pointer " #__PTR " should not be null.]",       \
          __summary__.error_message());                                 \
      __THROW_ERROR_INTERNAL__(                                         \
          phi::ErrorSummary(__summary__.code(), __message__));          \
    }                                                                   \
    return *__ptr;                                                      \
  })())

/*
 * Summary: This PADDLE_GET(_**) series macros are used to call paddle::get
 *   safely. paddle::get is not a completely safe api, although it will not
 *   go wrong in most cases, but in extreme cases, it may fail and directly
 *   throw a paddle::bad_variant_access const exception, without any stack
 *information.
 *   This kind of problems is difficult to debug, so add these macros to
 *   enrich paddle::get error information. At the same time, we restrict
 *   the direct use of paddle::get by CI rule.
 *
 * Parameters:
 *     __TYPE: the target variable type
 *     __VALUE: the target variable to get
 *
 * Examples:
 *     - unsafe writing: int x = paddle::get<int>(y);
 *     - safe writing: int x = PADDLE_GET(int, y);
 *
 * Note: GCC 4.8 cannot select right overloaded function here, so need
 *    to define different functions and macros here, after we upgrade
 *    CI gcc version, we can only define one PADDLE_GET macro.
 */
namespace details {

#define DEFINE_SAFE_PADDLE_GET(                                              \
    __InputType, __OutputType, __OutputTypePtr, __FuncName)                  \
  template <typename OutputType, typename InputType>                         \
  auto __FuncName(                                                           \
      __InputType input, const char* expression, const char* file, int line) \
      ->typename std::conditional<std::is_pointer<InputType>::value,         \
                                  __OutputTypePtr,                           \
                                  __OutputType>::type {                      \
    try {                                                                    \
      return paddle::get<OutputType>(input);                                 \
    } catch (paddle::bad_variant_access const&) {                            \
      HANDLE_THE_ERROR                                                       \
      throw ::phi::enforce::EnforceNotMet(                                   \
          phi::errors::InvalidArgument(                                      \
              "paddle::get failed, cannot get value "                        \
              "(%s) by type %s, its type is %s.",                            \
              expression,                                                    \
              phi::enforce::demangle(typeid(OutputType).name()),             \
              phi::enforce::demangle(input.type().name())),                  \
          file,                                                              \
          line);                                                             \
      END_HANDLE_THE_ERROR                                                   \
    }                                                                        \
  }

DEFINE_SAFE_PADDLE_GET(InputType&, OutputType&, OutputType*, SafeBoostGet);
DEFINE_SAFE_PADDLE_GET(const InputType&,
                       const OutputType&,
                       const OutputType*,
                       SafeBoostGetConst);
DEFINE_SAFE_PADDLE_GET(InputType&&,
                       OutputType,
                       OutputType*,
                       SafeBoostGetMutable);

}  // namespace details

#define PADDLE_GET(__TYPE, __VALUE)            \
  phi::enforce::details::SafeBoostGet<__TYPE>( \
      __VALUE, #__VALUE, __FILE__, __LINE__)
#define PADDLE_GET_CONST(__TYPE, __VALUE)           \
  phi::enforce::details::SafeBoostGetConst<__TYPE>( \
      __VALUE, #__VALUE, __FILE__, __LINE__)
#define PADDLE_GET_MUTABLE(__TYPE, __VALUE)           \
  phi::enforce::details::SafeBoostGetMutable<__TYPE>( \
      __VALUE, #__VALUE, __FILE__, __LINE__)

/**************************************************************************/
/**************************** NVIDIA ERROR ********************************/
#ifdef PADDLE_WITH_CUDA

namespace details {

template <typename T>
struct ExternalApiType {};

#define DEFINE_EXTERNAL_API_TYPE(type, success_value) \
  template <>                                         \
  struct ExternalApiType<type> {                      \
    using Type = type;                                \
    static constexpr Type kSuccess = success_value;   \
  }

DEFINE_EXTERNAL_API_TYPE(cudaError_t, cudaSuccess);
DEFINE_EXTERNAL_API_TYPE(curandStatus_t, CURAND_STATUS_SUCCESS);
DEFINE_EXTERNAL_API_TYPE(cudnnStatus_t, CUDNN_STATUS_SUCCESS);
DEFINE_EXTERNAL_API_TYPE(cublasStatus_t, CUBLAS_STATUS_SUCCESS);
DEFINE_EXTERNAL_API_TYPE(cusparseStatus_t, CUSPARSE_STATUS_SUCCESS);
DEFINE_EXTERNAL_API_TYPE(cusolverStatus_t, CUSOLVER_STATUS_SUCCESS);
DEFINE_EXTERNAL_API_TYPE(cufftResult_t, CUFFT_SUCCESS);
DEFINE_EXTERNAL_API_TYPE(CUresult, CUDA_SUCCESS);

#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
DEFINE_EXTERNAL_API_TYPE(ncclResult_t, ncclSuccess);
#endif

}  // namespace details

template <typename T>
std::string GetExternalErrorMsg(T status);

/*************** CUDA ERROR ***************/
inline bool is_error(cudaError_t e) { return e != cudaSuccess; }

inline std::string build_nvidia_error_msg(cudaError_t e) {
  std::ostringstream sout;
  sout << "CUDA error(" << e << "), " << cudaGetErrorString(e) << ". "
       << GetExternalErrorMsg(e);
  return sout.str();
}

/*************** CURAND ERROR ***************/
inline bool is_error(curandStatus_t stat) {
  return stat != CURAND_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(curandStatus_t stat) {
  std::ostringstream sout;
  sout << "CURAND error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUDNN ERROR ***************/
inline bool is_error(cudnnStatus_t stat) {
  return stat != CUDNN_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(cudnnStatus_t stat) {
  std::ostringstream sout;
  sout << "CUDNN error(" << stat << "), "
       << phi::dynload::cudnnGetErrorString(stat) << ". "
       << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUBLAS ERROR ***************/
inline bool is_error(cublasStatus_t stat) {
  return stat != CUBLAS_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(cublasStatus_t stat) {
  std::ostringstream sout;
  sout << "CUBLAS error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUSPARSE ERROR ***************/
inline bool is_error(cusparseStatus_t stat) {
  return stat != CUSPARSE_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(cusparseStatus_t stat) {
  std::ostringstream sout;
  sout << "CUSparse error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUSOLVER ERROR ***************/
inline bool is_error(cusolverStatus_t stat) {
  return stat != CUSOLVER_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(cusolverStatus_t stat) {
  std::ostringstream sout;
  sout << "CUSOLVER error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUFFT ERROR ***************/
inline bool is_error(cufftResult_t stat) { return stat != CUFFT_SUCCESS; }

inline std::string build_nvidia_error_msg(cufftResult_t stat) {
  std::ostringstream sout;
  sout << "CUFFT error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUresult ERROR ***************/
inline bool is_error(CUresult stat) { return stat != CUDA_SUCCESS; }

inline std::string build_nvidia_error_msg(CUresult stat) {
  std::ostringstream sout;
  sout << "CU error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/**************** NCCL ERROR ****************/
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
inline bool is_error(ncclResult_t nccl_result) {
  return nccl_result != ncclSuccess;
}

inline std::string build_nvidia_error_msg(ncclResult_t nccl_result) {
  std::ostringstream sout;
  sout << "NCCL error(" << nccl_result << "), "
       << phi::dynload::ncclGetErrorString(nccl_result) << ". ";
  if (errno == ENOSPC || errno == EAGAIN) {
    std::string detail(strerror(errno));
    detail += "\nPlease try one of the following solutions:";
    detail += "\n1. export NCCL_SHM_DISABLE=1;";
    detail += "\n2. export NCCL_P2P_LEVEL=SYS;";
    detail +=
        "\n3. Increase shared memory by setting the -shm-size "
        "option when starting docker container, e.g., setting "
        " -shm-size=2g.\n";
    sout << " Detail: " + detail;
  }
  sout << GetExternalErrorMsg(nccl_result);
  return sout.str();
}
#endif  // not(__APPLE__) and PADDLE_WITH_NCCL

#define PADDLE_ENFORCE_GPU_SUCCESS(COND)                     \
  do {                                                       \
    auto __cond__ = (COND);                                  \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);         \
    constexpr auto __success_type__ =                        \
        ::phi::enforce::details::ExternalApiType<            \
            __CUDA_STATUS_TYPE__>::kSuccess;                 \
    if (UNLIKELY(__cond__ != __success_type__)) {            \
      auto __summary__ = phi::errors::External(              \
          ::phi::enforce::build_nvidia_error_msg(__cond__)); \
      __THROW_ERROR_INTERNAL__(__summary__);                 \
    }                                                        \
  } while (0)

#define PADDLE_WARN_GPU_SUCCESS(COND)                        \
  do {                                                       \
    auto __cond__ = (COND);                                  \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);         \
    constexpr auto __success_type__ =                        \
        ::phi::enforce::details::ExternalApiType<            \
            __CUDA_STATUS_TYPE__>::kSuccess;                 \
    if (UNLIKELY(__cond__ != __success_type__)) {            \
      ::phi::enforce::ThrowWarnInternal(                     \
          ::phi::enforce::build_nvidia_error_msg(__cond__)); \
    }                                                        \
  } while (0)

#define PADDLE_ENFORCE_CUDA_LAUNCH_SUCCESS(OP)                              \
  do {                                                                      \
    auto res = cudaGetLastError();                                          \
    if (UNLIKELY(res != cudaSuccess)) {                                     \
      auto msg = ::phi::enforce::build_nvidia_error_msg(res);               \
      PADDLE_THROW(                                                         \
          phi::errors::Fatal("CUDA error after kernel (%s): %s", OP, msg)); \
    }                                                                       \
  } while (0)

inline void retry_sleep(unsigned milliseconds) {
#ifdef _WIN32
  Sleep(milliseconds);
#else
  if (milliseconds < 1000) {
    // usleep argument must be less than 1,000,000. Reference:
    // https://pubs.opengroup.org/onlinepubs/7908799/xsh/usleep.html
    usleep(milliseconds * 1000);
  } else {
    // clip to sleep in seconds because we can not and don't have to
    // sleep for exact milliseconds
    sleep(milliseconds / 1000);
  }
#endif
}

#define PADDLE_RETRY_CUDA_SUCCESS(COND)                                 \
  do {                                                                  \
    auto __cond__ = (COND);                                             \
    int retry_count = 1;                                                \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);                    \
    constexpr auto __success_type__ =                                   \
        ::phi::enforce::details::ExternalApiType<                       \
            __CUDA_STATUS_TYPE__>::kSuccess;                            \
    while (UNLIKELY(__cond__ != __success_type__) && retry_count < 5) { \
      phi::enforce::retry_sleep(10000);                                 \
      __cond__ = (COND);                                                \
      ++retry_count;                                                    \
    }                                                                   \
    if (UNLIKELY(__cond__ != __success_type__)) {                       \
      auto __summary__ = phi::errors::External(                         \
          ::phi::enforce::build_nvidia_error_msg(__cond__));            \
      __THROW_ERROR_INTERNAL__(__summary__);                            \
    }                                                                   \
  } while (0)

#undef DEFINE_EXTERNAL_API_TYPE
#endif  // PADDLE_WITH_CUDA

/**************************************************************************/
/***************************** HIP ERROR **********************************/
#ifdef PADDLE_WITH_HIP

/***** HIP ERROR *****/
inline bool is_error(hipError_t e) { return e != hipSuccess; }

inline std::string build_rocm_error_msg(hipError_t e) {
  std::ostringstream sout;
  sout << " Hip error(" << e << "), " << hipGetErrorString(e) << ".";
  return sout.str();
}

/***** HIPRAND ERROR *****/
inline bool is_error(hiprandStatus_t stat) {
  return stat != HIPRAND_STATUS_SUCCESS;
}

inline const char* hiprandGetErrorString(hiprandStatus_t stat) {
  switch (stat) {
    case HIPRAND_STATUS_SUCCESS:
      return "HIPRAND_STATUS_SUCCESS";
    case HIPRAND_STATUS_VERSION_MISMATCH:
      return "HIPRAND_STATUS_VERSION_MISMATCH";
    case HIPRAND_STATUS_NOT_INITIALIZED:
      return "HIPRAND_STATUS_NOT_INITIALIZED";
    case HIPRAND_STATUS_ALLOCATION_FAILED:
      return "HIPRAND_STATUS_ALLOCATION_FAILED";
    case HIPRAND_STATUS_TYPE_ERROR:
      return "HIPRAND_STATUS_TYPE_ERROR";
    case HIPRAND_STATUS_OUT_OF_RANGE:
      return "HIPRAND_STATUS_OUT_OF_RANGE";
    case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE";
    case HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case HIPRAND_STATUS_LAUNCH_FAILURE:
      return "HIPRAND_STATUS_LAUNCH_FAILURE";
    case HIPRAND_STATUS_PREEXISTING_FAILURE:
      return "HIPRAND_STATUS_PREEXISTING_FAILURE";
    case HIPRAND_STATUS_INITIALIZATION_FAILED:
      return "HIPRAND_STATUS_INITIALIZATION_FAILED";
    case HIPRAND_STATUS_ARCH_MISMATCH:
      return "HIPRAND_STATUS_ARCH_MISMATCH";
    case HIPRAND_STATUS_INTERNAL_ERROR:
      return "HIPRAND_STATUS_INTERNAL_ERROR";
    case HIPRAND_STATUS_NOT_IMPLEMENTED:
      return "HIPRAND_STATUS_NOT_IMPLEMENTED";
    default:
      return "Unknown hiprand status";
  }
}

inline std::string build_rocm_error_msg(hiprandStatus_t stat) {
  std::string msg(" Hiprand error, ");
  return msg + hiprandGetErrorString(stat) + " ";
}

/***** MIOPEN ERROR *****/
inline bool is_error(miopenStatus_t stat) {
  return stat != miopenStatusSuccess;
}

inline std::string build_rocm_error_msg(miopenStatus_t stat) {
  std::string msg(" Miopen error, ");
  return msg + phi::dynload::miopenGetErrorString(stat) + " ";
}

/***** ROCBLAS ERROR *****/
inline bool is_error(rocblas_status stat) {
  return stat != rocblas_status_success;
}

inline const char* rocblasGetErrorString(rocblas_status stat) {
  switch (stat) {
    case rocblas_status_invalid_handle:
      return "rocblas_status_invalid_handle";
    case rocblas_status_memory_error:
      return "rocblas_status_memory_error";
    case rocblas_status_invalid_value:
      return "rocblas_status_invalid_value";
    case rocblas_status_not_implemented:
      return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
      return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
      return "rocblas_status_invalid_size";
    case rocblas_status_internal_error:
      return "rocblas_status_internal_error";
    default:
      return "Unknown cublas status";
  }
}

inline std::string build_rocm_error_msg(rocblas_status stat) {
  std::string msg(" Rocblas error, ");
  return msg + rocblasGetErrorString(stat) + " ";
}

/****** RCCL ERROR ******/
#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
inline bool is_error(ncclResult_t nccl_result) {
  return nccl_result != ncclSuccess;
}

inline std::string build_rocm_error_msg(ncclResult_t nccl_result) {
  std::string msg(" Rccl error, ");
  return msg + phi::dynload::ncclGetErrorString(nccl_result) + " ";
}
#endif  // not(__APPLE__) and PADDLE_WITH_NCCL

/***** HIPFFT ERROR *****/
inline bool is_error(hipfftResult_t stat) { return stat != HIPFFT_SUCCESS; }

inline std::string build_rocm_error_msg(hipfftResult_t stat) {
  std::string msg(" HIPFFT error, ");
  return msg + phi::dynload::hipfftGetErrorString(stat) + " ";
}

namespace details {

template <typename T>
struct ExternalApiType {};

#define DEFINE_EXTERNAL_API_TYPE(type, success_value) \
  template <>                                         \
  struct ExternalApiType<type> {                      \
    using Type = type;                                \
    static constexpr Type kSuccess = success_value;   \
  }

DEFINE_EXTERNAL_API_TYPE(hipError_t, hipSuccess);
DEFINE_EXTERNAL_API_TYPE(hiprandStatus_t, HIPRAND_STATUS_SUCCESS);
DEFINE_EXTERNAL_API_TYPE(miopenStatus_t, miopenStatusSuccess);
DEFINE_EXTERNAL_API_TYPE(rocblas_status, rocblas_status_success);
DEFINE_EXTERNAL_API_TYPE(hipfftResult_t, HIPFFT_SUCCESS);

#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
DEFINE_EXTERNAL_API_TYPE(ncclResult_t, ncclSuccess);
#endif

}  // namespace details

#define PADDLE_ENFORCE_GPU_SUCCESS(COND)                   \
  do {                                                     \
    auto __cond__ = (COND);                                \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);       \
    constexpr auto __success_type__ =                      \
        ::phi::enforce::details::ExternalApiType<          \
            __CUDA_STATUS_TYPE__>::kSuccess;               \
    if (UNLIKELY(__cond__ != __success_type__)) {          \
      auto __summary__ = phi::errors::External(            \
          ::phi::enforce::build_rocm_error_msg(__cond__)); \
      __THROW_ERROR_INTERNAL__(__summary__);               \
    }                                                      \
  } while (0)

#define PADDLE_WARN_GPU_SUCCESS(COND)                      \
  do {                                                     \
    auto __cond__ = (COND);                                \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);       \
    constexpr auto __success_type__ =                      \
        ::phi::enforce::details::ExternalApiType<          \
            __CUDA_STATUS_TYPE__>::kSuccess;               \
    if (UNLIKELY(__cond__ != __success_type__)) {          \
      ::phi::enforce::ThrowWarnInternal(                   \
          ::phi::enforce::build_rocm_error_msg(__cond__)); \
    }                                                      \
  } while (0)

inline void retry_sleep(unsigned millisecond) {
#ifdef _WIN32
  Sleep(millisecond);
#else
  sleep(millisecond);
#endif
}

#define PADDLE_RETRY_CUDA_SUCCESS(COND)                                 \
  do {                                                                  \
    auto __cond__ = (COND);                                             \
    int retry_count = 1;                                                \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);                    \
    constexpr auto __success_type__ =                                   \
        ::phi::enforce::details::ExternalApiType<                       \
            __CUDA_STATUS_TYPE__>::kSuccess;                            \
    while (UNLIKELY(__cond__ != __success_type__) && retry_count < 5) { \
      ::phi::enforce::retry_sleep(10000);                               \
      __cond__ = (COND);                                                \
      ++retry_count;                                                    \
    }                                                                   \
    if (UNLIKELY(__cond__ != __success_type__)) {                       \
      auto __summary__ = phi::errors::External(                         \
          ::phi::enforce::build_rocm_error_msg(__cond__));              \
      __THROW_ERROR_INTERNAL__(__summary__);                            \
    }                                                                   \
  } while (0)

#undef DEFINE_EXTERNAL_API_TYPE
#endif  // PADDLE_WITH_HIP

}  // namespace enforce
using namespace enforce;  // NOLINT
}  // namespace phi
