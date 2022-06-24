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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/common/place.h"

#include "paddle/fluid/jit/base_function.h"
#include "paddle/fluid/jit/compilation_unit.h"
#include "paddle/fluid/jit/function_schema.h"

namespace paddle {
namespace jit {
using Variable = paddle::framework::Variable;
using Name2VariableMap = std::unordered_map<std::string, Variable>;

class Layer {
 public:
  // TODO(dev): Make vector<string>, num_slot as in argument
  // Layer(const std::shared_ptr<ClassType>& type) : obj_(type, /*num_slot*/ 0U)
  // {}
  Layer(const std::vector<std::shared_ptr<FunctionInfo>>& infos,
        const Name2VariableMap& params_dict,
        const phi::Place& place);

  std::shared_ptr<BaseFunction> Function(const std::string& name) const;

  Variable Attribute(const std::string& name) const;

  std::vector<Variable> forward(const std::vector<Variable>& inputs);

  void to(const phi::Place& place);

  void SetFunction(const std::string& name,
                   const std::shared_ptr<BaseFunction>& function);

 private:
  // internal::Object obj_;
  Name2VariableMap params_dict_;
  Name2VariableMap attrs_dict_;
  CompilationUnit unit_;
};

}  // namespace jit
}  // namespace paddle
