/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/utils/string/printf.h"
#include "paddle/utils/test_macros.h"

namespace common {
enum ErrorCode {
  // Legacy error.
  // Error type string: "Error"
  LEGACY = 0,

  // Client specified an invalid argument.
  // Error type string: "InvalidArgumentError"
  INVALID_ARGUMENT = 1,

  // Some requested entity (e.g., file or directory) was not found.
  // Error type string: "NotFoundError"
  NOT_FOUND = 2,

  // Operation tried to iterate past the valid input range.  E.g., seeking or
  // reading past end of file.
  // Error type string: "OutOfRangeError"
  OUT_OF_RANGE = 3,

  // Some entity that we attempted to create (e.g., file or directory)
  // already exists.
  // Error type string: "AlreadyExistsError"
  ALREADY_EXISTS = 4,

  // Some resource has been exhausted, perhaps a per-user quota, or
  // perhaps the entire file system is out of space.
  // Error type string: "ResourceExhaustedError"
  RESOURCE_EXHAUSTED = 5,

  // Operation was rejected because the system is not in a state
  // required for the operation's execution.
  // Error type string: "PreconditionNotMetError"
  PRECONDITION_NOT_MET = 6,

  // The caller does not have permission to execute the specified
  // operation.
  // Error type string: "PermissionDeniedError"
  PERMISSION_DENIED = 7,

  // Deadline expired before operation could complete.
  // Error type string: "ExecutionTimeout"
  EXECUTION_TIMEOUT = 8,

  // Operation is not implemented or not supported/enabled in this service.
  // Error type string: "UnimplementedError"
  UNIMPLEMENTED = 9,

  // The service is currently unavailable.  This is a most likely a
  // transient condition and may be corrected by retrying with
  // a backoff.
  // Error type string: "UnavailableError"
  UNAVAILABLE = 10,

  // Fatal errors.  Means some invariant expected by the underlying
  // system has been broken.  If you see one of these errors,
  // something is very broken.
  // Error type string: "FatalError"
  FATAL = 11,

  // Third-party library error.
  // Error type string: "ExternalError"
  EXTERNAL = 12,

  // Client specified an unmatched type.
  // Error type string: "InvalidTypeError"
  INVALID_TYPE = 13,
};

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
    code_ = common::ErrorCode::LEGACY;
    msg_ = paddle::string::Sprintf(args...);
  }

  // Note(chenweihang): Only recommended constructor
  //   No longer supports PADDLE_ENFORCE without type or without error message
  explicit ErrorSummary(ErrorCode code, std::string msg)
      : code_(code), msg_(msg) {}

  ErrorCode code() const { return code_; }

  const std::string& error_message() const { return msg_; }

  TEST_API std::string to_string() const;

 private:
  ErrorCode code_;
  std::string msg_;
};

namespace errors {

#define REGISTER_ERROR(FUNC, CONST, ...)                               \
  template <typename... Args>                                          \
  ::common::ErrorSummary FUNC(Args... args) {                          \
    return ::common::ErrorSummary(::common::CONST,                     \
                                  ::paddle::string::Sprintf(args...)); \
  }

REGISTER_ERROR(InvalidArgument, ErrorCode::INVALID_ARGUMENT)
REGISTER_ERROR(NotFound, ErrorCode::NOT_FOUND)
REGISTER_ERROR(OutOfRange, ErrorCode::OUT_OF_RANGE)
REGISTER_ERROR(AlreadyExists, ErrorCode::ALREADY_EXISTS)
REGISTER_ERROR(ResourceExhausted, ErrorCode::RESOURCE_EXHAUSTED)
REGISTER_ERROR(PreconditionNotMet, ErrorCode::PRECONDITION_NOT_MET)
REGISTER_ERROR(PermissionDenied, ErrorCode::PERMISSION_DENIED)
REGISTER_ERROR(ExecutionTimeout, ErrorCode::EXECUTION_TIMEOUT)
REGISTER_ERROR(Unimplemented, ErrorCode::UNIMPLEMENTED)
REGISTER_ERROR(Unavailable, ErrorCode::UNAVAILABLE)
REGISTER_ERROR(Fatal, ErrorCode::FATAL)
REGISTER_ERROR(External, ErrorCode::EXTERNAL)
REGISTER_ERROR(InvalidType, ErrorCode::INVALID_TYPE)

#undef REGISTER_ERROR

}  // namespace errors
}  // namespace common

namespace phi {
namespace errors = ::common::errors;
using ErrorCode = ::common::ErrorCode;
using ErrorSummary = ::common::ErrorSummary;
}  // namespace phi
