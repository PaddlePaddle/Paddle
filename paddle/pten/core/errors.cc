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

#include "paddle/pten/core/errors.h"

#include <stdexcept>

namespace pten {
std::string error_name(ErrorCode code) {
  switch (code) {
    case ErrorCode::LEGACY:
      return "Error";
      break;
    case ErrorCode::INVALID_ARGUMENT:
      return "InvalidArgumentError";
      break;
    case ErrorCode::NOT_FOUND:
      return "NotFoundError";
      break;
    case ErrorCode::OUT_OF_RANGE:
      return "OutOfRangeError";
      break;
    case ErrorCode::ALREADY_EXISTS:
      return "AlreadyExistsError";
      break;
    case ErrorCode::RESOURCE_EXHAUSTED:
      return "ResourceExhaustedError";
      break;
    case ErrorCode::PRECONDITION_NOT_MET:
      return "PreconditionNotMetError";
      break;
    case ErrorCode::PERMISSION_DENIED:
      return "PermissionDeniedError";
      break;
    case ErrorCode::EXECUTION_TIMEOUT:
      return "ExecutionTimeoutError";
      break;
    case ErrorCode::UNIMPLEMENTED:
      return "UnimplementedError";
      break;
    case ErrorCode::UNAVAILABLE:
      return "UnavailableError";
      break;
    case ErrorCode::FATAL:
      return "FatalError";
      break;
    case ErrorCode::EXTERNAL:
      return "ExternalError";
      break;
    default:
      throw std::invalid_argument("The error type is undefined.");
      break;
  }
}

std::string ErrorSummary::to_string() const {
  std::string result(error_name(code()));
  result += ": ";
  result += error_message();
  return result;
}
}  // namespace pten
