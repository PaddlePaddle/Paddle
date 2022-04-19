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

#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#if !defined(_WIN32) && !defined(PADDLE_WITH_MUSL)
#include <execinfo.h>
#endif

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/phi/core/errors.h"
#include "paddle/utils/string/printf.h"
#include "paddle/utils/string/to_string.h"

// Note: these headers for simplify demangle type string
#include "paddle/phi/core/compat/type_defs.h"

namespace phi {
class ErrorSummary;
}  // namespace phi

DECLARE_int32(call_stack_level);
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
  static const char* Convert(const char* expression, const T& value) {
    return expression;
  }
};
}  // namespace details

template <typename T>
inline std::string ReplaceComplexTypeStr(std::string str,
                                         const std::string& type_name) {
  auto demangle_type_str = demangle(typeid(T).name());
  size_t start_pos = 0;
  while ((start_pos = str.find(demangle_type_str, start_pos)) !=
         std::string::npos) {
    str.replace(start_pos, demangle_type_str.length(), type_name);
    start_pos += type_name.length();
  }
  return str;
}

#define __REPLACE_COMPLEX_TYPE_STR__(__TYPENAME, __STR)                      \
  do {                                                                       \
    __STR =                                                                  \
        phi::enforce::ReplaceComplexTypeStr<__TYPENAME>(__STR, #__TYPENAME); \
  } while (0)

inline std::string SimplifyDemangleStr(std::string str) {
  // the older is important, you have to put complex types in front
  __REPLACE_COMPLEX_TYPE_STR__(paddle::framework::AttributeMap, str);
  __REPLACE_COMPLEX_TYPE_STR__(paddle::framework::Attribute, str);
  __REPLACE_COMPLEX_TYPE_STR__(paddle::imperative::NameVariableWrapperMap, str);
  __REPLACE_COMPLEX_TYPE_STR__(paddle::imperative::NameVarBaseMap, str);
  __REPLACE_COMPLEX_TYPE_STR__(std::string, str);
  return str;
}

inline std::string GetCurrentTraceBackString(bool for_signal = false) {
  std::ostringstream sout;

  if (!for_signal) {
    sout << "\n\n--------------------------------------\n";
    sout << "C++ Traceback (most recent call last):";
    sout << "\n--------------------------------------\n";
  }
#if !defined(_WIN32) && !defined(PADDLE_WITH_MUSL)
  static constexpr int TRACE_STACK_LIMIT = 100;

  void* call_stack[TRACE_STACK_LIMIT];
  auto size = backtrace(call_stack, TRACE_STACK_LIMIT);
  auto symbols = backtrace_symbols(call_stack, size);
  Dl_info info;
  int idx = 0;
  // `for_signal` used to remove the stack trace introduced by
  // obtaining the error stack trace when the signal error occurred,
  // that is not related to the signal error self, remove it to
  // avoid misleading users and developers
  int end_idx = for_signal ? 2 : 0;
  for (int i = size - 1; i >= end_idx; --i) {
    if (dladdr(call_stack[i], &info) && info.dli_sname) {
      auto demangled = demangle(info.dli_sname);
      std::string path(info.dli_fname);
      // C++ traceback info are from core.so
      if (path.substr(path.length() - 3).compare(".so") == 0) {
        sout << paddle::string::Sprintf(
            "%-3d %s\n", idx++, SimplifyDemangleStr(demangled));
      }
    }
  }
  free(symbols);
#else
  sout << "Not support stack backtrace yet.\n";
#endif
  return sout.str();
}

template <typename StrType>
inline std::string GetErrorSumaryString(StrType&& what,
                                        const char* file,
                                        int line) {
  std::ostringstream sout;
  if (FLAGS_call_stack_level > 1) {
    sout << "\n----------------------\nError Message "
            "Summary:\n----------------------\n";
  }
  sout << paddle::string::Sprintf(
              "%s (at %s:%d)", std::forward<StrType>(what), file, line)
       << std::endl;
  return sout.str();
}

template <typename StrType>
inline std::string GetTraceBackString(StrType&& what,
                                      const char* file,
                                      int line) {
  if (FLAGS_call_stack_level > 1) {
    // FLAGS_call_stack_level>1 means showing c++ call stack
    return GetCurrentTraceBackString() + GetErrorSumaryString(what, file, line);
  } else {
    return GetErrorSumaryString(what, file, line);
  }
}

inline std::string SimplifyErrorTypeFormat(const std::string& str) {
  std::ostringstream sout;
  size_t type_end_pos = str.find(":", 0);
  if (type_end_pos == std::string::npos) {
    sout << str;
  } else {
    // Remove "Error:", add "()""
    sout << "(" << str.substr(0, type_end_pos - 5) << ")"
         << str.substr(type_end_pos + 1);
  }
  return sout.str();
}

inline bool is_error(bool stat) { return !stat; }

// Note: This Macro can only be used within enforce.h
#define __THROW_ERROR_INTERNAL__(__ERROR_SUMMARY)                             \
  do {                                                                        \
    HANDLE_THE_ERROR                                                          \
    throw ::phi::enforce::EnforceNotMet(__ERROR_SUMMARY, __FILE__, __LINE__); \
    END_HANDLE_THE_ERROR                                                      \
  } while (0)

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
    if (FLAGS_call_stack_level > 1) {
      return err_str_.c_str();
    } else {
      return simple_err_str_.c_str();
    }
  }

  phi::ErrorCode code() const { return code_; }

  const std::string& error_str() const { return err_str_; }

  const std::string& simple_error_str() const { return simple_err_str_; }

  void set_error_str(std::string str) {
    if (FLAGS_call_stack_level > 1) {
      err_str_ = str;
    } else {
      simple_err_str_ = str;
    }
  }

 private:
  // Used to determine the final type of exception thrown
  phi::ErrorCode code_ = phi::ErrorCode::LEGACY;
  // Complete error message
  // e.g. InvalidArgumentError: ***
  std::string err_str_;
  // Simple errror message used when no C++ stack and python compile stack
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
 * Summary: This macro is used to check whether op has specified
 * Input or Output Variables. Because op's Input and Output
 * checking are written similarly, so abstract this macro.
 *
 * Parameters:
 *     __EXPR: (bool), the bool expression
 *     __ROLE: (string), Input or Output
 *     __NAME: (string), Input or Output name
 *     __OP_TYPE: (string), the op type
 *
 * Examples:
 *    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Mul");
 */
#define OP_INOUT_CHECK(__EXPR, __ROLE, __NAME, __OP_TYPE)                    \
  do {                                                                       \
    PADDLE_ENFORCE_EQ(                                                       \
        __EXPR,                                                              \
        true,                                                                \
        phi::errors::NotFound(                                               \
            "No %s(%s) found for %s operator.", __ROLE, __NAME, __OP_TYPE)); \
  } while (0)

}  // namespace enforce
using namespace enforce;  // NOLINT
}  // namespace phi
