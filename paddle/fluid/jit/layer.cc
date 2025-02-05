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

#include "paddle/common/errors.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/core/enforce.h"

#include "paddle/fluid/jit/compilation_unit.h"
#include "paddle/fluid/jit/engine/base_engine.h"
#include "paddle/fluid/jit/function.h"
#include "paddle/fluid/jit/function_schema.h"

namespace paddle::jit {

Layer::Layer(const std::shared_ptr<VariableMap>& params_map,
             const std::shared_ptr<VariableMap>& attrs_map,
             const BaseFunctionInfoMap& info_map,
             const phi::Place& place)
    : params_map_(params_map),
      attrs_map_(attrs_map),
      info_map_(info_map),
      place_(place) {
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

const std::shared_ptr<jit::BaseFunctionInfo>& Layer::FunctionInfo(
    const std::string& name) const {
  PADDLE_ENFORCE_EQ(
      info_map_.count(name),
      1,
      common::errors::InvalidArgument(
          "FunctionInfo named %s is not existed in info_map_.", name));
  return info_map_.at(name);
}

std::vector<std::string> Layer::FunctionNames() const {
  std::vector<std::string> names;
  for (const auto& info : info_map_) {
    names.emplace_back(info.first);
  }
  return names;
}

#define PD_SPECIALIZE_ATTRIBUTE_TYPE(T)                               \
  template <>                                                         \
  T Layer::Attribute<T>(const std::string& name) const {              \
    if (attrs_map_->find(name) == attrs_map_->end()) {                \
      PADDLE_THROW(common::errors::NotFound(                          \
          "Attribute can not found %s, please check if it exists.")); \
      return T();                                                     \
    }                                                                 \
    auto var = attrs_map_->at(name);                                  \
    T ret = var->Get<T>();                                            \
    return ret;                                                       \
  }

PD_SPECIALIZE_ATTRIBUTE_TYPE(int)
PD_SPECIALIZE_ATTRIBUTE_TYPE(float)
PD_SPECIALIZE_ATTRIBUTE_TYPE(framework::String)
PD_SPECIALIZE_ATTRIBUTE_TYPE(std::vector<int>)
PD_SPECIALIZE_ATTRIBUTE_TYPE(std::vector<float>)
PD_SPECIALIZE_ATTRIBUTE_TYPE(std::vector<std::string>)

std::shared_ptr<Layer> Layer::Clone(void* stream) {
  std::shared_ptr<Layer> x =
      std::make_shared<Layer>(params_map_, attrs_map_, info_map_, place_);
  x->unit_ = unit_->Clone(stream);
  return x;
}

}  // namespace paddle::jit
