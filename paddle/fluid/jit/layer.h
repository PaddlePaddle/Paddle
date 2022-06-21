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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/jit/ast.h"
#include "paddle/fluid/jit/base_function.h"
#include "paddle/fluid/jit/compilation_unit.h"
#include "paddle/fluid/jit/exector_function.h"
#include "paddle/fluid/jit/object.h"
#include "paddle/fluid/jit/pe_function.h"

namespace paddle {
namespace jit {
using Variable = paddle::framework::Variable;
using VariableNameMap = std::map<std::string, Variable>;
using DenseTensor = phi::DenseTensor;

class Layer {
 public:
  // TODO(dev): Make vector<string>, num_slot as in argument
  // Layer(const std::shared_ptr<ClassType>& type) : obj_(type, /*num_slot*/ 0U)
  // {}
  // TODO(dev): consider make `func_name, program_desc, param_nams` as a class
  Layer(
      const std::vector<std::string>& func_names,
      const std::vector<framework::ProgramDesc>& program_descs,
      const std::vector<std::vector<std::string>>& param_names_for_each_program,
      const VariableNameMap& params_dict,
      const phi::Place& place);

  std::shared_ptr<BaseFunction> GetFunction(const std::string& name) const;

  std::vector<Variable> forward(const std::vector<Variable>& inputs);

 private:
  // internal::Object obj_;
  // std::vector<framework::ProgramDesc> all_program_desc_;
  // std::vector<std::vector<std::string>> param_name_for_each_program_;
  // std::vector<Variable> all_param_;
  std::map<std::string, std::shared_ptr<BaseFunction>> function_dict;
};

}  // namespace jit
}  // namespace paddle
