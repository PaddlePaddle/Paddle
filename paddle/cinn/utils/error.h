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
