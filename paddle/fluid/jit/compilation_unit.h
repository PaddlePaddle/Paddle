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

namespace paddle {
namespace jit {
<<<<<<< HEAD
class BaseFunction;
using Name2FunctionMap =
    std::unordered_map<std::string, std::shared_ptr<BaseFunction>>;
=======
class BaseEngine;
using EngineMap = std::unordered_map<std::string, std::shared_ptr<BaseEngine>>;
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

class CompilationUnit {
 public:
  CompilationUnit() = default;
  ~CompilationUnit() {}

<<<<<<< HEAD
  std::shared_ptr<BaseFunction> Function(const std::string &name) const;

  void SetFunction(const std::string &name,
                   const std::shared_ptr<BaseFunction> &function);

  std::vector<std::string> FunctionNames() const;

  const Name2FunctionMap &FunctionMap() const;

 private:
  Name2FunctionMap function_map_;
=======
  std::shared_ptr<BaseEngine> GetEngine(const std::string &name) const;

  void SetEngine(const std::string &name,
                 const std::shared_ptr<BaseEngine> &engine);

  const jit::EngineMap &EngineMap() const;

 private:
  jit::EngineMap engine_map_;
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
};

}  // namespace jit
}  // namespace paddle
