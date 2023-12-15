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

#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/utils/error.h"

namespace cinn {
namespace hlir {
namespace framework {

/**
 * This handler is used to deal with the errors during the compilation process
 */
class CompileErrorHandler : public utils::ErrorHandler {
 public:
  /**
   * \brief constructor
   * \param err_msg the error message
   */
  explicit CompileErrorHandler(const CompilationStatus& status,
                               const std::string& err_msg,
                               const std::string& detail_info,
                               const char* file,
                               int line)
      : status_(status),
        err_msg_(err_msg),
        detail_info_(detail_info),
        file_(file),
        line_(line) {}

  /**
   * \brief Returns a short error message corresponding to the kGeneral error
   * level.
   */
  std::string GeneralErrorMessage() const;

  /**
   * \brief Returns a detailed error message corresponding to the kDetailed
   * error level.
   */
  std::string DetailedErrorMessage() const;

  CompilationStatus Status() const { return status_; }

 private:
  CompilationStatus status_;
  std::string err_msg_;
  std::string detail_info_;
  const char* file_;
  int line_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
