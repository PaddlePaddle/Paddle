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
#include <memory>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * GenerateProgramPass will build the execution program for executor from a mir
 * graph.
 */
class GenerateProgramPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph> &graph) override;

  std::unique_ptr<RuntimeProgram> GenProgram() {
    LOG(INFO) << "insts.size " << insts_.size();
    std::unique_ptr<RuntimeProgram> program(
        new RuntimeProgram(std::move(insts_)));
    return program;
  }

 private:
  std::vector<Instruction> insts_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
