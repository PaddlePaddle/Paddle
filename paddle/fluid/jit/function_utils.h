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
#include <unordered_map>
#include <vector>

#include "paddle/fluid/jit/function_schema.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/place.h"
#include "paddle/pir/include/core/program.h"

namespace paddle {

namespace framework {
class Variable;
class ProgramDesc;
class Scope;
}  // namespace framework

namespace jit {
using Variable = paddle::framework::Variable;
using VariableMap = std::unordered_map<std::string, std::shared_ptr<Variable>>;
using DenseTensor = phi::DenseTensor;
using Tensor = paddle::Tensor;

namespace utils {

std::vector<DenseTensor> ToDenseTensors(const std::vector<Tensor> &tensors);
std::vector<Tensor> ToTensors(const std::vector<DenseTensor> &tensors);

void FetchOuts(const std::vector<std::string> &names,
               const framework::Scope &scope,
               std::vector<DenseTensor> *outs);

void ShareIntoScope(const std::vector<std::string> &ordered_input_names,
                    const std::vector<DenseTensor> &vars,
                    framework::Scope *scope);

void ShareParamsIntoScope(const std::vector<std::string> &param_names,
                          const std::shared_ptr<VariableMap> &params_dict,
                          framework::Scope *scope);

void RemoveFeedFetch(framework::ProgramDesc *program_desc);

template <typename T>
std::shared_ptr<T> MakeEngine(const std::shared_ptr<FunctionInfo> &info,
                              const std::shared_ptr<VariableMap> &params_dict,
                              const phi::Place &place) {
  return std::make_shared<T>(info, params_dict, place);
}

template <typename T>
std::shared_ptr<T> MakePirEngine(
    const std::shared_ptr<PirFunctionInfo> &info,
    const std::shared_ptr<VariableMap> &params_dict,
    const phi::Place &place,
    const std::shared_ptr<pir::Program> &prog) {
  return std::make_shared<T>(info, params_dict, place, prog);
}

}  // namespace utils
}  // namespace jit
}  // namespace paddle
