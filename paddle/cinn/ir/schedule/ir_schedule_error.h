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

#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace ir {

/**
 * This handler is dealing with the errors happen in in the current
 * Scheduling.
 */
class IRScheduleErrorHandler : public utils::ErrorHandler {
 public:
  /**
   * \brief constructor
   * \param err_msg the error message
   */
  explicit IRScheduleErrorHandler(const std::string& primitive,
                                  const std::string& err_msg,
                                  const ModuleExpr& module_expr)
      : primitive_(primitive), err_msg_(err_msg), module_expr_(module_expr) {}

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

 private:
  std::string primitive_;
  std::string err_msg_;
  ModuleExpr module_expr_;
};

}  // namespace ir
}  // namespace cinn
