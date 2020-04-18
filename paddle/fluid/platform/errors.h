/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

#include "paddle/fluid/platform/error_codes.pb.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace platform {

typedef ::paddle::platform::error::Code Code;

class ErrorSummary {
 public:
  // Note(chenweihang): Final deprecated constructor
  //   This constructor is only used to be compatible with
  //   current existing no error message PADDLE_ENFORCE_*
  ErrorSummary() {
    code_ = paddle::platform::error::LEGACY;
    msg_ =
        "An error occurred here. There is no accurate error hint for this "
        "error yet. We are continuously in the process of increasing hint for "
        "this kind of error check. It would be helpful if you could inform us "
        "of how this conversion went by opening a github issue. And we will "
        "resolve it with high priority.\n"
        "  - New issue link: "
        "https://github.com/PaddlePaddle/Paddle/issues/new\n"
        "  - Recommended issue content: all error stack information";
  }

  // Note(chenweihang): Final deprecated constructor
  //   This constructor is used to be compatible with
  //   current existing untyped PADDLE_ENFORCE_*
  //   PADDLE_ENFORCE
  template <typename... Args>
  explicit ErrorSummary(Args... args) {
    code_ = paddle::platform::error::LEGACY;
    msg_ = paddle::string::Sprintf(args...);
  }

  // Note(chenweihang): Recommended constructor
  explicit ErrorSummary(Code code, std::string msg) : code_(code), msg_(msg) {}

  Code code() const { return code_; }

  const std::string& error_message() const { return msg_; }

  std::string ToString() const;

 private:
  Code code_;
  std::string msg_;
};

namespace errors {

#define REGISTER_ERROR(FUNC, CONST, ...)                                       \
  template <typename... Args>                                                  \
  ::paddle::platform::ErrorSummary FUNC(Args... args) {                        \
    return ::paddle::platform::ErrorSummary(                                   \
        ::paddle::platform::error::CONST, ::paddle::string::Sprintf(args...)); \
  }

REGISTER_ERROR(InvalidArgument, INVALID_ARGUMENT)
REGISTER_ERROR(NotFound, NOT_FOUND)
REGISTER_ERROR(OutOfRange, OUT_OF_RANGE)
REGISTER_ERROR(AlreadyExists, ALREADY_EXISTS)
REGISTER_ERROR(ResourceExhausted, RESOURCE_EXHAUSTED)
REGISTER_ERROR(PreconditionNotMet, PRECONDITION_NOT_MET)
REGISTER_ERROR(PermissionDenied, PERMISSION_DENIED)
REGISTER_ERROR(ExecutionTimeout, EXECUTION_TIMEOUT)
REGISTER_ERROR(Unimplemented, UNIMPLEMENTED)
REGISTER_ERROR(Unavailable, UNAVAILABLE)
REGISTER_ERROR(Fatal, FATAL)
REGISTER_ERROR(External, EXTERNAL)

#undef REGISTER_ERROR

}  // namespace errors
}  // namespace platform
}  // namespace paddle
