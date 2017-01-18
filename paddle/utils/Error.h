/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <glog/logging.h>
#include <stdio.h>
#include <memory>
#include <string>
#include "Compiler.h"

namespace paddle {

/**
 * Status is Paddle error code. It only contain a std::string as error message.
 * Although Status inherits the std::exception, but do not throw it except you
 * know what you are doing.
 *
 *
 * There are two styles to return status in Paddle.
 *
 * 1. Return Status
 *    When method return a status, the return must use `__must_check` attribute.
 *    Example as below.
 * @code{cpp}
 * Error __must_check foo();
 *
 * Error __must_check bar() {
 *   // do something.
 *   Status s = foo();  // invoke other method return status.
 *   if (!s.isOK()) return s;
 *   // do something else.
 *   return Status();
 * }
 * @endcode{cpp}
 *
 * 2. Return by parameter.
 *    It is another way to return a status, by using a pointer parameter.
 *    Example as below.
 *
 * @code{cpp}
 * Error bar();
 *
 * int foo(Error* status) {
 *   // Do something.
 *   Status s = bar();
 *   if (!s.isOK()) {
 *     *status = s;
 *     return 0;
 *   }
 *   // Do something else.
 *   if (someInternalErrorHappend) {
 *     *status = ErrorF("Some dimension is too large, %d", dimension);
 *     return 0;
 *   }
 *   // End of method.
 *   return someValue;
 * }
 *
 * Error foobar() {
 *   Error s;
 *   // do something.
 *   foo(&s);
 *   if (!s.isOK()) return s;
 * }
 * @endcode{cpp}
 *
 *
 * Currently there is a helper method 'check' in status, because Paddle always
 * use log(FATAL) or CHECK to make program exit before. When we clean all
 * log(FATAL) and CHECK in Paddle, 'check' method will be removed.
 */
class Error final : public std::exception {
public:
  /**
   * Default Status. OK
   */
  Error() noexcept {}

  /**
   * @brief what will return the error message. If status is OK, return nullptr.
   */
  const char* what() const noexcept override {
    if (errMsg_) {
      return errMsg_->data();
    } else {
      return nullptr;
    }
  }

  /**
   * @brief isOK
   * @return true if OK.
   */
  inline bool isOK() const noexcept { return errMsg_ == nullptr; }

  /**
   * @brief check this status by glog.
   * @note It is a temp method used during cleaning Paddle code. It will be
   *       removed later.
   */
  inline void check() const { CHECK(isOK()) << what(); }

  /**
   * friend method to create Error.
   */
  template <typename... ARGS>
  friend Error __must_check ErrorF(const char* fmt, ARGS... args);

private:
  std::shared_ptr<std::string> errMsg_;
};

/**
 * ErrorF will create an Error by printf syntax.
 *
 * Specialize this method because clang will give a warning when use printf(fmt)
 * without arguments.
 */
template <>
inline Error __must_check ErrorF(const char* msg) {
  Error e;
  e.errMsg_.reset(new std::string(msg));
  return e;
}

/**
 * ErrorF will create an Error by printf syntax.
 *
 * Examples:
 * @code{cpp}
 * auto err = ErrorF("SomeError");
 * auto err2 = ErrorF("SomeErrorWithParameter %f %d", real_val, int_val);
 * @endcode{cpp}
 */
template <typename... ARGS>
inline Error __must_check ErrorF(const char* fmt, ARGS... args) {
  constexpr size_t kBufferSize = 1024;
  char buffer[kBufferSize];
  snprintf(buffer, kBufferSize, fmt, args...);
  Error e;
  e.errMsg_.reset(new std::string(buffer));
  return e;
}

}  // namespace paddle
