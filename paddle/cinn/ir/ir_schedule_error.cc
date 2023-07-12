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

#include "paddle/cinn/ir/ir_schedule_error.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"

namespace cinn {
namespace ir {

std::string IRScheduleErrorHandler::GeneralErrorMessage() const {
  return this->err_msg_;
}

std::string IRScheduleErrorHandler::DetailedErrorMessage() const {
  std::ostringstream os;
  os << GeneralErrorMessage();
  os << "[Expr info] The Expr of current schedule is: "
     << this->module_expr_.GetExprs() << std::endl;
  return os.str();
}

std::string IRScheduleErrorHandler::FormatErrorMessage(
    const std::string& primitive,
    const ScheduleErrorMessageLevel& err_msg_level) const {
  std::ostringstream os;
  std::string err_msg = err_msg_level == ScheduleErrorMessageLevel::kDetailed
                            ? DetailedErrorMessage()
                            : GeneralErrorMessage();

  os << "[IRScheduleError] An error occurred in the scheduel primitive <"
     << primitive << ">. " << std::endl;
  os << "[Error info] " << err_msg;
  return os.str();
}

std::string NegativeFactorErrorMessage(const int64_t& factor,
                                       const size_t& idx) {
  std::ostringstream os;
  os << "The params in factors of Split should be positive. However, the "
        "factor at position "
     << idx << " is " << factor << std::endl;
  return os.str();
}

std::string InferFactorErrorMessage() {
  std::ostringstream os;
  os << "The params in factors of Split should not be less than -1 or have "
        "more than one -1!"
     << std::endl;
  return os.str();
}

std::string FactorProductErrorMessage() {
  std::ostringstream os;
  os << "In Split, the factors' product should be not larger than or equal "
        "to original loop's extent!"
     << std::endl;
  return os.str();
}

}  // namespace ir
}  // namespace cinn
