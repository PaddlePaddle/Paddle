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

#include <string>
#include <unordered_map>

#include "paddle/fluid/jit/base_function.h"

namespace paddle {
namespace jit {
using FunctionMap =
    std::unordered_map<std::string, std::shared_ptr<BaseFunction>>;

class CompilationUnit {
 public:
  CompilationUnit() = default;
  ~CompilationUnit() {}

  std::shared_ptr<BaseFunction> Function(const std::string &name) const;

  void SetFunction(const std::string &name,
                   const std::shared_ptr<BaseFunction> &function);

  std::vector<std::string> FunctionNames() const;

  const FunctionMap &FunctionDict() const;

 private:
  FunctionMap function_dict_;
};

}  // namespace jit
}  // namespace paddle
