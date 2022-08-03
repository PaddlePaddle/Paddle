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

#include "paddle/fluid/framework/variable.h"

#include "paddle/fluid/jit/compilation_unit.h"
#include "paddle/fluid/jit/function/base_function.h"
#include "paddle/fluid/jit/function_schema.h"

namespace paddle {
namespace jit {
Layer::Layer(const Name2VariableMap& params_dict, const phi::Place& place)
    : params_dict_(params_dict) {
  unit_.reset(new CompilationUnit());
}

std::shared_ptr<BaseFunction> Layer::Function(const std::string& name) const {
  return unit_->Function(name);
}

std::vector<Tensor> Layer::forward(const std::vector<Tensor>& inputs) {
  auto func = Function("forward");
  return (*func)(inputs);
}

std::vector<DenseTensor> Layer::forward(
    const std::vector<DenseTensor>& inputs) {
  auto func = Function("forward");
  return (*func)(inputs);
}

void Layer::to(const phi::Place& place) {}

void Layer::SetFunction(const std::string& name,
                        const std::shared_ptr<BaseFunction>& function) {
  unit_->SetFunction(name, function);
}

std::vector<std::string> Layer::FunctionNames() const {
  return unit_->FunctionNames();
}

const Name2FunctionMap& Layer::FunctionMap() const {
  return unit_->FunctionMap();
}

}  // namespace jit
}  // namespace paddle
