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
#include "paddle/ir/core/program.h"

#include "paddle/cinn/hlir/framework/graph_compiler.h"

namespace cinn {
namespace hlir {
namespace framework {

struct CompatibleInfo {
  static constexpr char* kInputPrefix = "input_";
  static constexpr char* kOutputPrefix = "output_";
  // TODO(Aurelius): Need add name mapping logic in REGISTER_CINN_OP
  // macros or attempt to unify Op name with Paddle and CINN.
  static const std::unordered_map<std::string, std::string> OP_NAMES;
};

// TODO(Aurelius84): Need abstract this logic to implement Proxy for
// the co-existance with GraphCompiler.
class NewIRCompiler final {
 public:
  NewIRCompiler(const ::ir::Program& prog,
                const Target& target,
                const std::shared_ptr<Scope>& scope)
      : program_(prog),
        m_builder_("NewIR", target),
        target_(target),
        scope_(scope) {}

  std::unique_ptr<Program> Build();
  std::vector<ir::LoweredFunc> GetOpFunc(const ::ir::Operation& op, int idx);
  void ProcessFunction(const std::vector<ir::LoweredFunc>& lowered_funcs);

  std::vector<std::unique_ptr<Instruction>> BuildInstructions(
      const std::vector<std::vector<::ir::Operation*>>& groups);

 protected:
  const std::string& GenOpFuncName(const ::ir::Operation& op, int idx);

  std::vector<std::string> OpGetInputNames(const ::ir::Operation& op);

  std::vector<std::string> OpGetOutputNames(const ::ir::Operation& op);

 private:
  CINN_DISALLOW_COPY_AND_ASSIGN(NewIRCompiler);

  const ::ir::Program& program_;
  ir::Module::Builder m_builder_;
  std::unique_ptr<backends::Compiler> compiler_{nullptr};
  Target target_;
  std::shared_ptr<Scope> scope_;
  std::unordered_map<std::string, std::string> func_names_;
};

std::shared_ptr<Scope> BuildScope(const Target&, const ::ir::Program&);

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
