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
class BaseEngine;
using EngineMap = std::unordered_map<std::string, std::shared_ptr<BaseEngine>>;

class CompilationUnit {
 public:
  CompilationUnit() = default;
  ~CompilationUnit() {}

  std::shared_ptr<BaseEngine> GetEngine(const std::string &name) const;

  void SetEngine(const std::string &name,
                 const std::shared_ptr<BaseEngine> &engine);

  const jit::EngineMap &EngineMap() const;

 private:
  jit::EngineMap engine_map_;
};

}  // namespace jit
}  // namespace paddle
