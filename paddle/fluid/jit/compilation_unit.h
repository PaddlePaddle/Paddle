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
class BaseEngine;
using EngineMap = std::unordered_map<std::string, std::shared_ptr<BaseEngine>>;
=======
class BaseFunction;
using Name2FunctionMap =
    std::unordered_map<std::string, std::shared_ptr<BaseFunction>>;
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

class CompilationUnit {
 public:
  CompilationUnit() = default;
  ~CompilationUnit() {}

<<<<<<< HEAD
  std::shared_ptr<BaseEngine> GetEngine(const std::string &name) const;

  void SetEngine(const std::string &name,
                 const std::shared_ptr<BaseEngine> &engine);

  const jit::EngineMap &EngineMap() const;

 private:
  jit::EngineMap engine_map_;
=======
  std::shared_ptr<BaseFunction> Function(const std::string &name) const;

  void SetFunction(const std::string &name,
                   const std::shared_ptr<BaseFunction> &function);

  std::vector<std::string> FunctionNames() const;

  const Name2FunctionMap &FunctionMap() const;

 private:
  Name2FunctionMap function_map_;
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
};

}  // namespace jit
}  // namespace paddle
