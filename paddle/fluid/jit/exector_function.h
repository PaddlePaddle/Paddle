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
#include "paddle/fluid/jit/ivalue.h"

namespace paddle {
namespace jit {

class ExectorFunction : public BaseFunction {
 public:
  ExectorFunction(const framework::ProgramDesc &program_desc,
                  const std::vector<std::string> param_name_for_program,
                  const std::map<std::string, IValue> &all_param)
      : BaseFunction(program_desc, param_name_for_program, all_param) {}

  ~ExectorFunction() {}

  std::vector<IValue> operator()(const std::vector<IValue> &args) {
    framework::Executor inner_exe_{phi::CPUPlace()};
    // share input into scope
    ShareIntoScope(args);
    // run program
    inner_exe_.Run(program_desc_, &scope_, /*blockID=*/0, false, true,
                   skip_var_name_);
    VLOG(6) << framework::GenScopeTreeDebugInfo(&scope_);
    // fetch outputs
    std::vector<IValue> res;
    FetchOutput(&res);
    return res;
  }
};

}  // namespace jit
}  // namespace paddle
