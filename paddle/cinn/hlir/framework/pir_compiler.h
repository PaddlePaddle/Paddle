// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/hlir/framework/pir/compilation_task.h"

namespace cinn {
namespace hlir {
namespace framework {

class PirCompiler final {
 public:
  using CompileResult = std::vector<pir::CINNKernelInfo>;
  PirCompiler(const Target& target) : target_(target) {}

  CompileResult Build(const std::vector<pir::OpLoweringGroupPtr>& groups);

 private:
  CINN_DISALLOW_COPY_AND_ASSIGN(PirCompiler);

  Target target_;
  std::vector<GroupCompilationContext> group_compilation_contexts_;
};

class PirCompilerManager {
 public:
  static PirCompilerManager& Instance() {
    static PirCompilerManager instance;
    return instance;
  }

  static std::shared_ptr<PirCompiler> Create(const Target& target) {
    std::shared_ptr<PirCompiler> compiler =
        std::make_shared<PirCompiler>(target);
    PirCompilerManager::Instance().insert(compiler);
    return compiler;
  }

  void insert(const std::shared_ptr<PirCompiler>& compiler) {
    compilers_.push_back(compiler);
  }

  void clear() { compilers_.clear(); }

 private:
  std::vector<std::shared_ptr<PirCompiler>> compilers_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
