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

#include "paddle/fluid/jit/base_function.h"

namespace paddle {
namespace jit {

class ExectorFunction : public BaseFunction {
 public:
  ExectorFunction(const framework::ProgramDesc &program_desc,
                  const std::vector<std::string> param_names_for_program,
                  const VariableNameMap &params_dict)
      : BaseFunction(program_desc, param_names_for_program, params_dict),
        inner_exe_(phi::CPUPlace()) {}

  ~ExectorFunction() {}

  std::vector<Variable> operator()(const std::vector<Variable> &inputs) {
    // share input into scope
    ShareInputsIntoScope(inputs);
    // run program
    inner_exe_.Run(program_desc_, &scope_, /*blockID=*/0, false, true,
                   schema_.GetOutputArgNames());
    VLOG(6) << framework::GenScopeTreeDebugInfo(&scope_);
    // fetch outputs
    std::vector<Variable> res;
    FetchOutput(&res);
    return res;
  }

 private:
  // TODO(dev): support other devices exe
  framework::Executor inner_exe_;
};

}  // namespace jit
}  // namespace paddle
