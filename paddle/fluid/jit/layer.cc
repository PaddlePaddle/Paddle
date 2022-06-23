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

#include "paddle/fluid/jit/layer.h"

namespace paddle {
namespace jit {
// TODO(dev): Make vector<string>, num_slot as in argument
// Layer(const std::shared_ptr<ClassType>& type) : obj_(type, /*num_slot*/ 0U)
// {}
Layer::Layer(const std::vector<std::shared_ptr<FunctionInfo>>& infos,
             const Name2VariableMap& params_dict,
             const phi::Place& place)
    : params_dict_(params_dict) {
  VLOG(3) << "infos size: " << infos.size();
  // Layer manage the life time of all parameter.
  for (size_t i = 0; i < infos.size(); ++i) {
    // TODO(dev): choose exector or pe by flag
    unit_.AddExecutorFunction(
        infos[i]->GetFunctionName(), infos[i], params_dict_, place);
  }
}

std::shared_ptr<BaseFunction> Layer::GetFunction(
    const std::string& name) const {
  return unit_.GetFunction(name);
}

std::vector<Variable> Layer::forward(const std::vector<Variable>& inputs) {
  auto func = GetFunction("forward");
  return (*func)(inputs);
}

}  // namespace jit
}  // namespace paddle
