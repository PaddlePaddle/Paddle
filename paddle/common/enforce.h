// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef __GNUC__
#include <cxxabi.h>  // for __cxa_demangle
#endif               // __GNUC__
#include <exception>
#include <iostream>
#if !defined(_WIN32)
#include <dlfcn.h>   // dladdr
#include <unistd.h>  // sleep, usleep
#else                // _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#include <windows.h>  // GetModuleFileName, Sleep
#endif

#include "paddle/common/errors.h"
#include "paddle/common/macros.h"

#if !defined(_WIN32) && !defined(PADDLE_WITH_MUSL)
#include <execinfo.h>
#endif

// msvc conflict logging with windows.h
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "paddle/utils/string/printf.h"
#include "paddle/utils/string/to_string.h"
#include "paddle/utils/test_macros.h"
#include "paddle/utils/variant.h"

namespace common {
class CommonNotMetException : public std::exception {
 public:
  explicit CommonNotMetException(const std::string& str) : err_str_(str) {}

  const char* what() const noexcept override { return err_str_.c_str(); }

 private:
  std::string err_str_;
};
}  // namespace common

namespace common {
namespace enforce {

/** HELPER MACROS AND FUNCTIONS **/
#ifndef PADDLE_MAY_THROW
#define PADDLE_MAY_THROW noexcept(false)
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

#define COMMON_THROW(...)                                               \
  do {                                                                  \
    HANDLE_THE_ERROR                                                    \
    throw common::CommonNotMetException(                                \
        paddle::string::Sprintf("Error occured at: %s:%d :\n%s",        \
                                __FILE__,                               \
                                __LINE__,                               \
                                paddle::string::Sprintf(__VA_ARGS__))); \
    END_HANDLE_THE_ERROR                                                \
  } while (0)

#define __COMMON_BINARY_COMPARE(__VAL1, __VAL2, __CMP, __INV_CMP, ...)         \
  do {                                                                         \
    auto __val1 = (__VAL1);                                                    \
    auto __val2 = (__VAL2);                                                    \
    using __TYPE1__ = decltype(__val1);                                        \
    using __TYPE2__ = decltype(__val2);                                        \
    using __COMMON_TYPE1__ =                                                   \
        common::enforce::CommonType1<__TYPE1__, __TYPE2__>;                    \
    using __COMMON_TYPE2__ =                                                   \
        ::common::enforce::CommonType2<__TYPE1__, __TYPE2__>;                  \
    bool __is_not_error = (static_cast<__COMMON_TYPE1__>(__val1))__CMP(        \
        static_cast<__COMMON_TYPE2__>(__val2));                                \
    if (UNLIKELY(!__is_not_error)) {                                           \
      auto __message__ =                                                       \
          ::paddle::string::Sprintf("%s\n  [Hint: Expected %s " #__CMP         \
                                    " %s,but received %s " #__INV_CMP " %s.]", \
                                    #__VAL1,                                   \
                                    #__VAL2,                                   \
                                    #__VAL1,                                   \
                                    #__VAL2);                                  \
      try {                                                                    \
        throw common::CommonNotMetException(__message__);                      \
      } catch (const std::exception& e) {                                      \
        std::cout << e.what() << std::endl;                                    \
        throw;                                                                 \
      }                                                                        \
    }                                                                          \
  } while (0)

#define COMMON_ENFORCE_EQ(__VAL0, __VAL1, ...) \
  __COMMON_BINARY_COMPARE(__VAL0, __VAL1, ==, !=, __VA_ARGS__)
#define COMMON_ENFORCE_NE(__VAL0, __VAL1, ...) \
  __COMMON_BINARY_COMPARE(__VAL0, __VAL1, !=, ==, __VA_ARGS__)
#define COMMON_ENFORCE_GT(__VAL0, __VAL1, ...) \
  __COMMON_BINARY_COMPARE(__VAL0, __VAL1, >, <=, __VA_ARGS__)
#define COMMON_ENFORCE_GE(__VAL0, __VAL1, ...) \
  __COMMON_BINARY_COMPARE(__VAL0, __VAL1, >=, <, __VA_ARGS__)
#define COMMON_ENFORCE_LT(__VAL0, __VAL1, ...) \
  __COMMON_BINARY_COMPARE(__VAL0, __VAL1, <, >=, __VA_ARGS__)
#define COMMON_ENFORCE_LE(__VAL0, __VAL1, ...) \
  __COMMON_BINARY_COMPARE(__VAL0, __VAL1, <=, >, __VA_ARGS__)

}  // namespace enforce
}  // namespace common

// TODO(zhangbopd): This is a copy from pir, and shoud be removed after merge
// this into common enfoce namespace above.
template <typename T>
inline bool is_error(const T& stat) {
  return !stat;
}

namespace pir {
class IrNotMetException : public std::exception {
 public:
  explicit IrNotMetException(const std::string& str) : err_str_(str) {}

  const char* what() const noexcept override { return err_str_.c_str(); }

 private:
  std::string err_str_;
};

#define IR_THROW(...)                                                     \
  do {                                                                    \
    try {                                                                 \
      throw pir::IrNotMetException(                                       \
          paddle::string::Sprintf("Error occured at: %s:%d :\n%s",        \
                                  __FILE__,                               \
                                  __LINE__,                               \
                                  paddle::string::Sprintf(__VA_ARGS__))); \
    } catch (const std::exception& e) {                                   \
      std::cout << e.what() << std::endl;                                 \
      throw;                                                              \
    }                                                                     \
  } while (0)

#define IR_ENFORCE(COND, ...)                                               \
  do {                                                                      \
    bool __cond__(COND);                                                    \
    if (UNLIKELY(is_error(__cond__))) {                                     \
      try {                                                                 \
        throw pir::IrNotMetException(                                       \
            paddle::string::Sprintf("Error occured at: %s:%d :\n%s",        \
                                    __FILE__,                               \
                                    __LINE__,                               \
                                    paddle::string::Sprintf(__VA_ARGS__))); \
      } catch (const std::exception& e) {                                   \
        std::cout << e.what() << std::endl;                                 \
        throw;                                                              \
      }                                                                     \
    }                                                                       \
  } while (0)

}  // namespace pir
