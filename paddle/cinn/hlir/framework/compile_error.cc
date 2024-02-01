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

#include "paddle/cinn/hlir/framework/compile_error.h"
#include "paddle/cinn/utils/enum_string.h"

namespace cinn {
namespace hlir {
namespace framework {

std::string CompileErrorHandler::GeneralErrorMessage() const {
  std::ostringstream os;
  os << "[CompileError] An error occurred during compilation with the error "
        "code: "
     << utils::Enum2String(status_) << std::endl;
  os << "(at " << file_ << " : " << line_ << ")" << std::endl;
  os << indent_str_ << "[Error info] " << this->err_msg_ << std::endl;
  return os.str();
}

std::string CompileErrorHandler::DetailedErrorMessage() const {
  std::ostringstream os;
  os << GeneralErrorMessage();
  os << indent_str_ << "[Detail info] " << detail_info_ << std::endl;
  return os.str();
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
