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

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/place.h"

#include "function.h"  //NOLINT

namespace paddle {

namespace framework {
class Variable;
}  // namespace framework

namespace jit {
class CompilationUnit;
class BaseFunctionInfo;

using DenseTensor = phi::DenseTensor;
using Tensor = paddle::Tensor;
using Variable = paddle::framework::Variable;
using VariableMap = std::unordered_map<std::string, std::shared_ptr<Variable>>;
using BaseFunctionInfoMap =
    std::unordered_map<std::string, std::shared_ptr<BaseFunctionInfo>>;

class Layer {
 public:
  Layer(const std::shared_ptr<VariableMap>& params_map,
        const std::shared_ptr<VariableMap>& attrs_map_,
        const BaseFunctionInfoMap& info_map,
        const phi::Place& place);

  jit::Function Function(const std::string& name) const;

  template <typename T>

  T Attribute(const std::string& name) const;

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs);

  std::vector<DenseTensor> forward(const std::vector<DenseTensor>& inputs);

  void to(const phi::Place& place);

  void SetEngine(const std::string& name,
                 const std::shared_ptr<BaseEngine>& engine);

  const std::shared_ptr<jit::BaseFunctionInfo>& FunctionInfo(
      const std::string& name) const;

  std::vector<std::string> FunctionNames() const;

  std::shared_ptr<Layer> Clone(void* stream = nullptr);

 private:
  std::shared_ptr<VariableMap> params_map_;
  std::shared_ptr<VariableMap> attrs_map_;
  BaseFunctionInfoMap info_map_;
  phi::Place place_;
  std::shared_ptr<CompilationUnit> unit_;
};

}  // namespace jit
}  // namespace paddle
