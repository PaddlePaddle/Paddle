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

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/place.h"

#include "paddle/fluid/jit/function_schema.h"

namespace paddle {

namespace framework {
class Variable;
class ProgramDesc;
class Scope;
}  // namespace framework

namespace jit {
using Variable = paddle::framework::Variable;
<<<<<<< HEAD
using Name2VariableMap =
    std::unordered_map<std::string, std::shared_ptr<Variable>>;
=======
using VariableMap = std::unordered_map<std::string, std::shared_ptr<Variable>>;
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
using DenseTensor = phi::DenseTensor;
using Tensor = paddle::experimental::Tensor;

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
<<<<<<< HEAD
                          const Name2VariableMap &params_dict,
=======
                          const VariableMap &params_dict,
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                          framework::Scope *scope);

void RemoveFeedFetch(framework::ProgramDesc *program_desc);

template <typename T>
<<<<<<< HEAD
std::shared_ptr<T> MakeFunction(const std::shared_ptr<FunctionInfo> &info,
                                const Name2VariableMap &params_dict,
                                const phi::Place &place) {
=======
std::shared_ptr<T> MakeEngine(const std::shared_ptr<FunctionInfo> &info,
                              const VariableMap &params_dict,
                              const phi::Place &place) {
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  return std::make_shared<T>(info, params_dict, place);
}

}  // namespace utils
}  // namespace jit
}  // namespace paddle
