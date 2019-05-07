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
#include "paddle/fluid/lite/core/program.h"
#include "paddle/fluid/lite/core/program.h"
#include "paddle/fluid/lite/core/scope.h"

namespace paddle {
namespace lite {

/*
// The Executor is used to run the operators.
class Executor {
 public:
  Executor(const framework::ProgramDesc& desc,
           const std::shared_ptr<lite::Scope>& scope,
           const std::vector<Place>& valid_places)
      : valid_places_(valid_places) {
    program_.reset(new Program(desc, scope, valid_places));
  }

  // Run the program.
  void Run() {
    for (auto& op : program_->ops) {
      LOG(INFO) << op->DebugString();
      // TODO(Superjomn) check only once
      op->CheckShape();
      op->InferShape();
      op->Run();
    }
  }

  const Program& program() const { return *program_; }

 private:
  std::vector<Place> valid_places_;
  std::unique_ptr<Program> program_;
};

class RuntimeExecutor {
 public:
  RuntimeExecutor(RuntimeProgram* program) : program_(program) {}

  void Run() {
    CHECK(program_);
    program_->Run();
  }

 private:
  RuntimeProgram* program_{};
};
 */

}  // namespace lite
}  // namespace paddle
