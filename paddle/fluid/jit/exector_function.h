// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"

#include "paddle/fluid/jit/base_function.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/layer_utils.h"

namespace paddle {
namespace jit {

class ExectorFunction : public BaseFunction {
 public:
  ExectorFunction(const std::shared_ptr<FunctionInfo> &info,
                  const Name2VariableMap &params_dict,
                  const phi::Place &place)
      : info_(info), place_(place), inner_exe_(place_) {
    ShareParamsIntoScope(info_->GetParamNames(), params_dict, &scope_);
    VLOG(6) << framework::GenScopeTreeDebugInfo(&scope_);
  }

  ~ExectorFunction() noexcept {}

  std::vector<Variable> operator()(const std::vector<Variable> &inputs) {
    ShareInputsIntoScope(info_->GetInputArgNames(), inputs, &scope_);
    inner_exe_.Run(info_->GetProgramDesc(),
                   &scope_,
                   /*blockID=*/0,
                   false,
                   true,
                   info_->GetOutputArgNames());
    VLOG(6) << framework::GenScopeTreeDebugInfo(&scope_);
    std::vector<Variable> res;
    FetchVarsByNames(info_->GetOutputArgNames(), scope_, &res);
    return res;
  }

 private:
  std::shared_ptr<FunctionInfo> info_;
  framework::Scope scope_;
  phi::Place place_;
  framework::Executor inner_exe_;
};

}  // namespace jit
}  // namespace paddle
