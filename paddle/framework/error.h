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

#include <stdarg.h>
#include <stdio.h>
#include <memory>
#include <string>
#include "paddle/platform/must_check.h"

namespace paddle {
namespace framework {

/**
 * Error is Paddle error code. It only contain a std::string as error message.
 *
 * There are two styles to return error in Paddle.
 *
 * 1. Return Error
 *    When method return a status, the return must use `__must_check` attribute.
 *    Example as below.
 * @code{cpp}
 * Error __must_check foo();
 *
 * Error __must_check bar() {
 *   // do something.
 *   Error err = foo();  // invoke other method return status.
 *   if (err) return err;
 *   // do something else.
 *   return Error();
 * }
 * @endcode{cpp}
 *
 * 2. Return by parameter.
 *    It is another way to return an error, by using a pointer parameter.
 *    Example as below.
 *
 * @code{cpp}
 * Error __must_check bar();
 *
 * int foo(Error* error) {
 *   // Do something.
 *   Error err = bar();
 *   if (err) {
 *     *error = s;
 *     return 0;
 *   }
 *   // Do something else.
 *   if (someInternalErrorHappend) {
 *     *error = Error("Some dimension is too large, %d", dimension);
 *     return 0;
 *   }
 *   // End of method.
 *   return someValue;
 * }
 *
 * Error __must_check foobar() {
 *   Error err;
 *   // do something.
 *   foo(&err);
 *   if (err) return err;
 * }
 * @endcode{cpp}
 *
 */
class Error {
 public:
  /**
   * Construct a no-error value.
   */
  Error() {}

  /**
   * @brief Create an Error use printf syntax.
   */
  explicit Error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    constexpr size_t kBufferSize = 1024;
    char buffer[kBufferSize];
    vsnprintf(buffer, kBufferSize, fmt, ap);
    this->msg_.reset(new std::string(buffer));
    va_end(ap);
  }

  /**
   * @brief msg will return the error message. If no error, return nullptr.
   */
  const char *msg() const {
    if (msg_) {
      return msg_->c_str();
    } else {
      return nullptr;
    }
  }

  /**
   * @brief operator bool, return True if there is something error.
   */
  operator bool() const { return !this->isOK(); }

  /**
   * @brief isOK return True if there is no error.
   * @return True if no error.
   */
  bool isOK() const { return msg_ == nullptr; }

  bool operator==(const Error &o) const {
    if (msg_ == o.msg_) {
      return true;
    } else if (msg_ != nullptr && o.msg_ != nullptr) {
      return *msg_ == *o.msg_;
    } else {  // one is nullptr, the other is not.
      return false;
    }
  }

 private:
  std::shared_ptr<std::string> msg_;
};

}  // namespace framework

using framework::Error;  // Error is a global name for paddle namespace, not
                         // only for framework package.

}  // namespace paddle
