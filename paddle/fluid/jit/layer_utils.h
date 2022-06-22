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

#include <string>
#include <vector>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace jit {

using Variable = paddle::framework::Variable;
using VariableNameMap = std::map<std::string, Variable>;
using DenseTensor = phi::DenseTensor;

void FetchVarsByNames(const std::vector<std::string> &names,
                      const framework::Scope &scope,
                      std::vector<Variable> *outs);

void ShareInputsIntoScope(const std::vector<std::string> &ordered_input_names,
                          const std::vector<Variable> &vars,
                          framework::Scope *scope);

void ShareParamsIntoScope(const std::vector<std::string> &param_names,
                          const VariableNameMap &params_dict,
                          framework::Scope *scope);

void RemoveFeedFetch(framework::ProgramDesc *program_desc);

}  // namespace jit
}  // namespace paddle
