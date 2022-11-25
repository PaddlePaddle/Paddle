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

<<<<<<< HEAD
#include <memory>
=======
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/place.h"

<<<<<<< HEAD
#include "function.h"  //NOLINT
=======
#include "base_function.h"
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

namespace paddle {

namespace framework {
class Variable;
}  // namespace framework

namespace jit {
class CompilationUnit;
<<<<<<< HEAD
class FunctionInfo;
=======
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

using DenseTensor = phi::DenseTensor;
using Tensor = paddle::experimental::Tensor;
using Variable = paddle::framework::Variable;
<<<<<<< HEAD
using VariableMap = std::unordered_map<std::string, std::shared_ptr<Variable>>;
using FunctionInfoMap =
    std::unordered_map<std::string, std::shared_ptr<FunctionInfo>>;

class Layer {
 public:
  Layer(const VariableMap& params_map,
        const VariableMap& attrs_map_,
        const FunctionInfoMap& info_map,
        const phi::Place& place);

  jit::Function Function(const std::string& name) const;

  template <typename T>
  T Attribute(const std::string& name) const;
=======
using Name2VariableMap =
    std::unordered_map<std::string, std::shared_ptr<Variable>>;
using Name2FunctionMap =
    std::unordered_map<std::string, std::shared_ptr<BaseFunction>>;

class Layer {
 public:
  Layer(const Name2VariableMap& params_dict, const phi::Place& place);

  std::shared_ptr<BaseFunction> Function(const std::string& name) const;

  Variable Attribute(const std::string& name) const;
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs);

  std::vector<DenseTensor> forward(const std::vector<DenseTensor>& inputs);

  void to(const phi::Place& place);

<<<<<<< HEAD
  void SetEngine(const std::string& name,
                 const std::shared_ptr<BaseEngine>& engine);

  const std::shared_ptr<jit::FunctionInfo>& FunctionInfo(
      const std::string& name) const;

  std::vector<std::string> FunctionNames() const;

 private:
  VariableMap params_map_;
  VariableMap attrs_map_;
  FunctionInfoMap info_map_;
=======
  void SetFunction(const std::string& name,
                   const std::shared_ptr<BaseFunction>& function);

  std::vector<std::string> FunctionNames() const;

  const Name2FunctionMap& FunctionMap() const;

 private:
  Name2VariableMap params_dict_;
  Name2VariableMap attrs_dict_;
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
  std::shared_ptr<CompilationUnit> unit_;
};

}  // namespace jit
}  // namespace paddle
