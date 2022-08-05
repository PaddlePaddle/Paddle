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
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

#include "paddle/fluid/jit/compilation_unit.h"
#include "paddle/fluid/jit/engine/base_engine.h"
#include "paddle/fluid/jit/function.h"
#include "paddle/fluid/jit/function_schema.h"

namespace paddle {
namespace jit {

Layer::Layer(const Name2VariableMap& params_dict,
             const Name2VariableMap& attrs_dict,
             const phi::Place& place)
    : params_dict_(params_dict), attrs_dict_(attrs_dict) {
  unit_.reset(new CompilationUnit());
}

jit::Function Layer::Function(const std::string& name) const {
  return jit::Function(unit_->GetEngine(name).get());
}

std::vector<Tensor> Layer::forward(const std::vector<Tensor>& inputs) {
  auto func = this->Function("forward");
  return func(inputs);
}

std::vector<DenseTensor> Layer::forward(
    const std::vector<DenseTensor>& inputs) {
  auto func = this->Function("forward");
  return func(inputs);
}

void Layer::to(const phi::Place& place) {}

void Layer::SetEngine(const std::string& name,
                      const std::shared_ptr<BaseEngine>& engine) {
  unit_->SetEngine(name, engine);
}

const Name2EngineMap& Layer::EngineMap() const { return unit_->EngineMap(); }

#define PD_SPECIALZE_ATTRIBUTE_TYPE(T)                                \
  template <>                                                         \
  T Layer::Attribute<T>(const std::string& name) const {              \
    if (attrs_dict_.find(name) == attrs_dict_.end()) {                \
      PADDLE_THROW(phi::errors::NotFound(                             \
          "Attribute can not found %s, please check if it exists.")); \
      return T();                                                     \
    }                                                                 \
    auto var = attrs_dict_.at(name);                                  \
    T ret = var->Get<T>();                                            \
    return ret;                                                       \
  }

PD_SPECIALZE_ATTRIBUTE_TYPE(int)
PD_SPECIALZE_ATTRIBUTE_TYPE(float)
PD_SPECIALZE_ATTRIBUTE_TYPE(std::string)
PD_SPECIALZE_ATTRIBUTE_TYPE(std::vector<int>)
PD_SPECIALZE_ATTRIBUTE_TYPE(std::vector<float>)
PD_SPECIALZE_ATTRIBUTE_TYPE(std::vector<std::string>)

}  // namespace jit
}  // namespace paddle
