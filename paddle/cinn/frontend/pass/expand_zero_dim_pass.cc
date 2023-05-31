// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/program_pass.h"
#include "glog/logging.h"

namespace cinn {
namespace frontend {
namespace pass {

class ExpandZeroDimPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    NetBuilder builder("expand_zero_dim_builder");
    for (auto var : program->GetInputs()) {
      if (var->shape.empty()) {
        var->shape.push_back(1);
      }
      builder.CreateInput(var);
    }
    for (int i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];
      for (auto& input : instr->inputs) {
        if (input->shape.empty()) {
          VLOG(4) << "Change input 0D-Tensor " << input->id << " to 1D-Tensor";
          input->shape.push_back(1);
        }
      }
      for (auto& output : instr->outputs) {
        if (output->shape.empty()) {
          VLOG(4) << "Change output 0D-Tensor " << output->id
                  << " to 1D-Tensor";
          output->shape.push_back(1);
        }
      }
      builder.AppendInstruction(instr);
    }
    *program = builder.Build();
  }

  void Clear() override {}
};

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(ExpandZeroDim) {
  CINN_REGISTER_PROGRAM_PASS(ExpandZeroDim,
                             cinn::frontend::pass::ExpandZeroDimPass);

  return true;
}
