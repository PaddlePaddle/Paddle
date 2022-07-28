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

#include "paddle/fluid/jit/compilation_unit.h"

#include "paddle/phi/core/enforce.h"

#include "paddle/fluid/jit/base_function.h"

namespace paddle {
namespace jit {

std::shared_ptr<BaseFunction> CompilationUnit::Function(
    const std::string &name) const {
  PADDLE_ENFORCE_EQ(
      function_map_.count(name),
      1,
      phi::errors::InvalidArgument(
          "Funciton name %s is not exist in function_map_.", name));
  return function_map_.at(name);
}

void CompilationUnit::SetFunction(
    const std::string &name, const std::shared_ptr<BaseFunction> &function) {
  function_map_[name] = function;
}

std::vector<std::string> CompilationUnit::FunctionNames() const {
  std::vector<std::string> names;
  for (auto it = function_map_.begin(); it != function_map_.end(); it++) {
    names.emplace_back(it->first);
  }
  return names;
}

const Name2FunctionMap &CompilationUnit::FunctionMap() const {
  return function_map_;
}

}  // namespace jit
}  // namespace paddle
