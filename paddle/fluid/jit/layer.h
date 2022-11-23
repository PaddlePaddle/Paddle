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
=======
#include <memory>
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/place.h"

<<<<<<< HEAD
#include "base_function.h"
=======
#include "function.h"  //NOLINT
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

namespace paddle {

namespace framework {
class Variable;
}  // namespace framework

namespace jit {
class CompilationUnit;
<<<<<<< HEAD
=======
class FunctionInfo;
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

using DenseTensor = phi::DenseTensor;
using Tensor = paddle::experimental::Tensor;
using Variable = paddle::framework::Variable;
<<<<<<< HEAD
using Name2VariableMap =
    std::unordered_map<std::string, std::shared_ptr<Variable>>;
using Name2FunctionMap =
    std::unordered_map<std::string, std::shared_ptr<BaseFunction>>;

class Layer {
 public:
  Layer(const Name2VariableMap& params_dict, const phi::Place& place);

  std::shared_ptr<BaseFunction> Function(const std::string& name) const;

  Variable Attribute(const std::string& name) const;
=======
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
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs);

  std::vector<DenseTensor> forward(const std::vector<DenseTensor>& inputs);

  void to(const phi::Place& place);

<<<<<<< HEAD
  void SetFunction(const std::string& name,
                   const std::shared_ptr<BaseFunction>& function);

  std::vector<std::string> FunctionNames() const;

  const Name2FunctionMap& FunctionMap() const;

 private:
  Name2VariableMap params_dict_;
  Name2VariableMap attrs_dict_;
=======
  void SetEngine(const std::string& name,
                 const std::shared_ptr<BaseEngine>& engine);

  const std::shared_ptr<jit::FunctionInfo>& FunctionInfo(
      const std::string& name) const;

  std::vector<std::string> FunctionNames() const;

 private:
  VariableMap params_map_;
  VariableMap attrs_map_;
  FunctionInfoMap info_map_;
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  std::shared_ptr<CompilationUnit> unit_;
};

}  // namespace jit
}  // namespace paddle
