// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#if !defined(_WIN32)
#include <dlfcn.h>   // dladdr
#include <unistd.h>  // sleep, usleep
#else                // _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#include <windows.h>  // GetModuleFileName, Sleep
#endif

#if !defined(_WIN32) && !defined(PADDLE_WITH_MUSL)
#include <execinfo.h>
#endif

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cinn {
namespace utils {

namespace enforce {

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

static std::string GetErrorSumaryString(const std::string& what,
                                        const char* file,
                                        int line) {
  std::ostringstream sout;
  sout << "\n----------------------\nError Message "
          "Summary:\n----------------------\n";
  sout << what << "(at " << file << " : " << line << ")" << std::endl;
  return sout.str();
}

static std::string GetCurrentTraceBackString() {
  std::ostringstream sout;
  sout << "\n\n--------------------------------------\n";
  sout << "C++ Traceback (most recent call last):";
  sout << "\n--------------------------------------\n";
#if !defined(_WIN32) && !defined(PADDLE_WITH_MUSL)
  static constexpr int TRACE_STACK_LIMIT = 100;

  void* call_stack[TRACE_STACK_LIMIT];
  auto size = backtrace(call_stack, TRACE_STACK_LIMIT);
  auto symbols = backtrace_symbols(call_stack, size);
  Dl_info info;
  int idx = 0;
  int end_idx = 0;
  for (int i = size - 1; i >= end_idx; --i) {
    if (dladdr(call_stack[i], &info) && info.dli_sname) {
      auto demangled = demangle(info.dli_sname);
      std::string path(info.dli_fname);
      // C++ traceback info are from core.so
      if (path.substr(path.length() - 3).compare(".so") == 0) {
        sout << idx++ << " " << demangled << "\n";
      }
    }
  }
  free(symbols);
#else
  sout << "Not support stack backtrace yet.\n";
#endif
  return sout.str();
}

static std::string GetTraceBackString(const std::string& what,
                                      const char* file,
                                      int line) {
  return GetCurrentTraceBackString() + GetErrorSumaryString(what, file, line);
}

struct EnforceNotMet : public std::exception {
 public:
  EnforceNotMet(const std::string& str, const char* file, int line)
      : err_str_(GetTraceBackString(str, file, line)) {}

  const char* what() const noexcept override { return err_str_.c_str(); }

 private:
  std::string err_str_;
};

#ifdef PADDLE_THROW
#define CINN_THROW PADDLE_THROW
#else
#define CINN_THROW(...)                                                     \
  do {                                                                      \
    try {                                                                   \
      throw utils::enforce::EnforceNotMet(__VA_ARGS__, __FILE__, __LINE__); \
    } catch (const std::exception& e) {                                     \
      std::cout << e.what() << std::endl;                                   \
      throw;                                                                \
    }                                                                       \
  } while (0)
#endif
}  // namespace enforce

/**
 *  \brief Indicates the level of printing error message in the current
 * operation
 */
enum class ErrorMessageLevel : int32_t {
  /** \brief  Print an error message in short mode.
   * Short mode shows which and where the error happens*/
  kGeneral = 0,
  /** \brief Print an error message in detailed mode.
   * Detailed mode shows which and where the error happens, and the
   * detailed input parameters.
   */
  kDetailed = 1,
};

/**
 * This handler is a base class dealing with the errors happen in in the current
 * operation.
 */
class ErrorHandler {
 public:
  /**
   * \brief Returns a short error message corresponding to the kGeneral error
   * level.
   */
  virtual std::string GeneralErrorMessage() const = 0;

  /**
   * \brief Returns a detailed error message corresponding to the kDetailed
   * error level.
   */
  virtual std::string DetailedErrorMessage() const = 0;

  /**
   * \brief Format the error message.
   */
  std::string FormatErrorMessage(const ErrorMessageLevel& err_msg_level) const;

 protected:
  const std::string indent_str_{"  "};
};

}  // namespace utils
}  // namespace cinn
