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
#include <unordered_map>
#include "paddle/cinn/common/macros.h"
#include "paddle/pir/core/program.h"

#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/hlir/framework/pir/compilation_task.h"

namespace cinn {
namespace hlir {
namespace framework {

// TODO(Aurelius84): Need abstract this logic to implement Proxy for
// the co-existance with GraphCompiler.
class PirCompiler final {
 public:
  PirCompiler(const ::pir::Program& prog,
              const Target& target,
              const std::shared_ptr<Scope>& scope)
      : program_(prog),
        m_builder_("Pir", target),
        target_(target),
        scope_(scope) {}

  std::unique_ptr<Program> Build();

  std::vector<pir::CINNKernelInfo> BuildCUDAJITInfo(
      const std::vector<pir::GroupPtr>& groups);

  std::unique_ptr<Program> Build(const std::vector<pir::GroupPtr>& groups);

 private:
  CINN_DISALLOW_COPY_AND_ASSIGN(PirCompiler);

  std::vector<ir::LoweredFunc> GetOpFunc(const ::pir::Operation& op, int idx);

  void ProcessFunction(const std::vector<ir::LoweredFunc>& lowered_funcs);

  std::vector<std::unique_ptr<Instruction>> BuildInstructions(
      const std::vector<pir::GroupPtr>& groups);

  const ::pir::Program& program_;
  ir::Module::Builder m_builder_;
  std::unique_ptr<backends::Compiler> compiler_{nullptr};
  Target target_;
  std::shared_ptr<Scope> scope_;
  std::unordered_map<std::string, std::string> func_names_;
  std::vector<GroupCompilationContext> group_compilation_contexts_;
};

// TODO(phlrain): pir compiler don't need Scope, need to remove this
std::shared_ptr<Scope> BuildScope(const Target&, const ::pir::Program&);

class PirCompilerManager {
 public:
  static PirCompilerManager& Instance() {
    static PirCompilerManager instance;
    return instance;
  }

  static std::shared_ptr<PirCompiler> Create(
      const ::pir::Program& prog,
      const Target& target,
      const std::shared_ptr<Scope>& scope) {
    std::shared_ptr<PirCompiler> compiler =
        std::make_shared<PirCompiler>(prog, target, scope);
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
