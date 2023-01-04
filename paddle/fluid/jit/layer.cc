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

Layer::Layer(const VariableMap& params_map,
             const VariableMap& attrs_map,
             const FunctionInfoMap& info_map,
             const phi::Place& place)
    : params_map_(params_map), attrs_map_(attrs_map), info_map_(info_map) {
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

const std::shared_ptr<jit::FunctionInfo>& Layer::FunctionInfo(
    const std::string& name) const {
  PADDLE_ENFORCE_EQ(
      info_map_.count(name),
      1,
      phi::errors::InvalidArgument(
          "FuncitonInfo named %s is not existed in info_map_.", name));
  return info_map_.at(name);
}

std::vector<std::string> Layer::FunctionNames() const {
  std::vector<std::string> names;
  for (auto it = info_map_.begin(); it != info_map_.end(); ++it) {
    names.emplace_back(it->first);
  }
  return names;
}

#define PD_SPECIALZE_ATTRIBUTE_TYPE(T)                                \
  template <>                                                         \
  T Layer::Attribute<T>(const std::string& name) const {              \
    if (attrs_map_.find(name) == attrs_map_.end()) {                  \
      PADDLE_THROW(phi::errors::NotFound(                             \
          "Attribute can not found %s, please check if it exists.")); \
      return T();                                                     \
    }                                                                 \
    auto var = attrs_map_.at(name);                                   \
    T ret = var->Get<T>();                                            \
    return ret;                                                       \
  }

PD_SPECIALZE_ATTRIBUTE_TYPE(int)
PD_SPECIALZE_ATTRIBUTE_TYPE(float)
PD_SPECIALZE_ATTRIBUTE_TYPE(framework::String)
PD_SPECIALZE_ATTRIBUTE_TYPE(std::vector<int>)
PD_SPECIALZE_ATTRIBUTE_TYPE(std::vector<float>)
PD_SPECIALZE_ATTRIBUTE_TYPE(std::vector<std::string>)

}  // namespace jit
}  // namespace paddle
