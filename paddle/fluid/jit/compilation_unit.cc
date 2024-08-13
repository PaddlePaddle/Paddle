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

#include "paddle/fluid/jit/engine/base_engine.h"

namespace paddle::jit {

std::shared_ptr<BaseEngine> CompilationUnit::GetEngine(
    const std::string &name) const {
  PADDLE_ENFORCE_EQ(
      engine_map_.count(name),
      1,
      common::errors::InvalidArgument(
          "Function named %s is not existed in engine_map_.", name));
  return engine_map_.at(name);
}

void CompilationUnit::SetEngine(const std::string &name,
                                const std::shared_ptr<BaseEngine> &engine) {
  engine_map_[name] = engine;
}

const jit::EngineMap &CompilationUnit::EngineMap() const { return engine_map_; }

std::shared_ptr<CompilationUnit> CompilationUnit::Clone(void *stream) {
  auto x = std::make_shared<CompilationUnit>();
  for (auto &it : engine_map_) {
    x->SetEngine(it.first, it.second->Clone(stream));
  }
  return x;
}

}  // namespace paddle::jit
