/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/pten/core/error_codes.pb.h"
#include "paddle/pten/core/string/printf.h"

namespace pten {
typedef ::pten::proto::Code Code;

class ErrorSummary {
 public:
  // Note(chenweihang): Final deprecated constructor
  //   This constructor is used to be compatible with
  //   current existing untyped PADDLE_ENFORCE_*
  //   PADDLE_ENFORCE
  // Note(chenweihang): Windows openblas need this
  //   constructor for compiling PADDLE_ENFORCE in *.cu,
  //   this is a bug cause we can't remove this
  //   constructor now.
  template <typename... Args>
  explicit ErrorSummary(Args... args) {
    code_ = pten::proto::LEGACY;
    msg_ = pten::string::Sprintf(args...);
  }

  // Note(chenweihang): Only recommended constructor
  //   No longer supports PADDLE_ENFORCE without type or without error message
  explicit ErrorSummary(Code code, std::string msg) : code_(code), msg_(msg) {}

  Code code() const { return code_; }

  const std::string& error_message() const { return msg_; }

  std::string to_string() const;

 private:
  Code code_;
  std::string msg_;
};

namespace errors {

#define REGISTER_ERROR(FUNC, CONST, ...)                           \
  template <typename... Args>                                      \
  ::pten::ErrorSummary FUNC(Args... args) {                        \
    return ::pten::ErrorSummary(::pten::CONST,                     \
                                ::pten::string::Sprintf(args...)); \
  }

REGISTER_ERROR(InvalidArgument, proto::INVALID_ARGUMENT)
REGISTER_ERROR(NotFound, proto::NOT_FOUND)
REGISTER_ERROR(OutOfRange, proto::OUT_OF_RANGE)
REGISTER_ERROR(AlreadyExists, proto::ALREADY_EXISTS)
REGISTER_ERROR(ResourceExhausted, proto::RESOURCE_EXHAUSTED)
REGISTER_ERROR(PreconditionNotMet, proto::PRECONDITION_NOT_MET)
REGISTER_ERROR(PermissionDenied, proto::PERMISSION_DENIED)
REGISTER_ERROR(ExecutionTimeout, proto::EXECUTION_TIMEOUT)
REGISTER_ERROR(Unimplemented, proto::UNIMPLEMENTED)
REGISTER_ERROR(Unavailable, proto::UNAVAILABLE)
REGISTER_ERROR(Fatal, proto::FATAL)
REGISTER_ERROR(External, proto::EXTERNAL)

#undef REGISTER_ERROR

}  // namespace errors
}  // namespace pten
