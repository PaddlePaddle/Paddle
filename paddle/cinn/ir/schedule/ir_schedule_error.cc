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

#include "paddle/cinn/ir/schedule/ir_schedule_error.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/utils/ir_printer.h"

namespace cinn {
namespace ir {

std::string IRScheduleErrorHandler::GeneralErrorMessage() const {
  std::ostringstream os;
  os << "[IRScheduleError] An error occurred in the scheduel primitive < "
     << this->primitive_ << " >. " << std::endl;
  os << indent_str_ << "[Error info] " << this->err_msg_;
  return os.str();
}

std::string IRScheduleErrorHandler::DetailedErrorMessage() const {
  std::ostringstream os;
  os << GeneralErrorMessage();
  os << indent_str_ << "[Expr info] The Expr of current schedule is:\n"
     << this->module_expr_.GetExprs() << std::endl;
  return os.str();
}

}  // namespace ir
}  // namespace cinn
