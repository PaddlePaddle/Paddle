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
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/scope.h"

namespace paddle {
namespace lite {

// The Executor is used to run the operators.
class Executor {
 public:
  Executor(lite::Scope* scope, const std::vector<OpLite::Place>& valid_places)
      : scope_(scope), valid_places_(valid_places) {}

  // Create temporary variables.
  void PrepareWorkspace(framework::ProgramDesc& program, lite::Scope* scope) {
    CHECK(!exec_scope_) << "Duplicate PrepareWorkspace found";
    exec_scope_ = &scope_->NewScope();

    for (auto var_desc : program.Block(0).AllVars()) {
      if (!var_desc->Persistable()) {
        auto* var = exec_scope_->Var(var_desc->Name());
        LOG(INFO) << "create tmp var " << var_desc->Name() << " " << var;
      }
    }
  }

  // Build from a program and scope.
  void Build(framework::ProgramDesc& program) {
    CHECK(ops_.empty()) << "Executor duplicate Build found";

    // Create operators.
    for (auto* op_desc : program.Block(0).AllOps()) {
      auto op_type = op_desc->Type();
      if (op_type == "feed" || op_type == "fetch") continue;
      LOG(INFO) << "create Op [" << op_type << "]";
      ops_.emplace_back(LiteOpRegistry::Global().Create(op_type));
      // pick initial kernel
      ops_.back()->PickKernel(valid_places_);
      ops_.back()->Attach(*op_desc, exec_scope_);
    }
  }

  // Run the program.
  void Run() {
    for (auto& op : ops_) {
      LOG(INFO) << op->DebugString();
      // TODO(Superjomn) check only once
      op->CheckShape();
      op->InferShape();
      op->Run();
    }
  }

 private:
  std::vector<std::unique_ptr<OpLite>> ops_;
  lite::Scope* scope_{};
  std::vector<OpLite::Place> valid_places_;
  lite::Scope* exec_scope_{};
};

}  // namespace lite
}  // namespace paddle
