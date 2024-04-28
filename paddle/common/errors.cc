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

#include "paddle/common/errors.h"

#include <stdexcept>

namespace common {
std::string error_name(ErrorCode code) {
  switch (code) {
    case ErrorCode::LEGACY:
      return "Error";
    case ErrorCode::INVALID_ARGUMENT:
      return "InvalidArgumentError";
    case ErrorCode::NOT_FOUND:
      return "NotFoundError";
    case ErrorCode::OUT_OF_RANGE:
      return "OutOfRangeError";
    case ErrorCode::ALREADY_EXISTS:
      return "AlreadyExistsError";
    case ErrorCode::RESOURCE_EXHAUSTED:
      return "ResourceExhaustedError";
    case ErrorCode::PRECONDITION_NOT_MET:
      return "PreconditionNotMetError";
    case ErrorCode::PERMISSION_DENIED:
      return "PermissionDeniedError";
    case ErrorCode::EXECUTION_TIMEOUT:
      return "ExecutionTimeoutError";
    case ErrorCode::UNIMPLEMENTED:
      return "UnimplementedError";
    case ErrorCode::UNAVAILABLE:
      return "UnavailableError";
    case ErrorCode::FATAL:
      return "FatalError";
    case ErrorCode::EXTERNAL:
      return "ExternalError";
    case ErrorCode::INVALID_TYPE:
      return "InvalidTypeError";
    default:
      throw std::invalid_argument("The error type is undefined.");
  }
}

std::string ErrorSummary::to_string() const {
  std::string result(error_name(code()));
  result += ": ";
  result += error_message();
  return result;
}
}  // namespace common
