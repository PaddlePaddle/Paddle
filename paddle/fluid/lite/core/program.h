// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <list>
#include <string>
#include <vector>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {

// A program is used to represent a code program, in Paddle, a code program
// contains:
// - main block, which is a list of OpLite
// - scope: which contains all the weights
struct Program {
  std::list<std::string> tmp_vars;
  std::list<std::string> weights;
  std::list<std::shared_ptr<OpLite>> ops;
  // the scope to run the kernels, NOTE not the root scope.
  std::shared_ptr<lite::Scope> scope;
  // Runtime scope.
  lite::Scope* exec_scope{};

  explicit Program(const std::shared_ptr<Scope>& root) { scope = root; }
  Program(const framework::ProgramDesc& desc,
          const std::shared_ptr<Scope>& root,
          const std::vector<Place>& valid_places) {
    scope = root;
    PrepareWorkspace(desc);
    Build(desc, valid_places);
  }

  std::unique_ptr<Program> Clone() const {
    std::unique_ptr<Program> res(new Program(scope));
    res->tmp_vars = tmp_vars;
    res->weights = weights;
    res->ops = ops;
    return res;
  }

 private:
  // Build from a program and scope.
  void Build(const framework::ProgramDesc& program,
             const std::vector<Place>& valid_places) {
    CHECK(ops.empty()) << "Executor duplicate Build found";

    // Create operators.
    for (auto* op_desc : program.Block(0).AllOps()) {
      auto op_type = op_desc->Type();
      if (op_type == "feed" || op_type == "fetch") continue;
      LOG(INFO) << "create Op [" << op_type << "]";
      ops.emplace_back(LiteOpRegistry::Global().Create(op_type));
      // pick initial kernel
      ops.back()->PickKernel(valid_places);
      ops.back()->Attach(*op_desc, exec_scope);
    }
  }

  // Create temporary variables.
  void PrepareWorkspace(const framework::ProgramDesc& program) {
    CHECK(!exec_scope) << "Duplicate PrepareWorkspace found";
    exec_scope = &scope->NewScope();

    for (auto var_desc : program.Block(0).AllVars()) {
      if (!var_desc->Persistable()) {
        auto* var = exec_scope->Var(var_desc->Name());
        LOG(INFO) << "create tmp var " << var_desc->Name() << " " << var;
      }
    }
  }
};

}  // namespace lite
}  // namespace paddle
