// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <string>
#include <unordered_set>

#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/program_pass.h"

namespace cinn {
namespace frontend {
namespace pass {

// Program maybe has some unused instructions. `DeadCodeEliminate` will remove
// these instructions. The way to find unused instructions is to traverse all
// instructions to determine whether its output is used by other instructions in
// the same subgraph or in the `fetch_ids`.
class DeadCodeEliminatePass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  void Clear() override {}

  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    if (!CheckFetchIds(*program, fetch_ids)) {
      return;
    }

    std::unordered_set<std::string> inputs;
    std::unordered_set<int> remove_idxs;
    for (int i = program->size() - 1; i >= 0; --i) {
      const auto& instr = (*program)[i];
      bool can_remove = true;
      for (const auto& out : instr->outputs) {
        if (inputs.count(out->id) || fetch_ids.count(out->id)) {
          can_remove = false;
          break;
        }
      }
      if (can_remove) {
        VLOG(3) << "Remove the " << i << "-th instruction: " << instr;
        remove_idxs.insert(i);
      } else {
        for (const auto& in : instr->inputs) {
          inputs.insert(in->id);
        }
      }
    }

    VLOG(3) << "Total remove " << remove_idxs.size() << " instructions.";
    if (remove_idxs.size() == 0) {
      return;
    }

    NetBuilder builder("dead_code_eliminate_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }
    for (int i = 0; i < program->size(); i++) {
      if (!remove_idxs.count(i)) {
        builder.AppendInstruction((*program)[i]);
      }
    }
    *program = builder.Build();
  }

 private:
  bool CheckFetchIds(const Program& program,
                     const std::unordered_set<std::string>& fetch_ids) {
    if (fetch_ids.empty()) {
      // If fetch_ids is not specified, all output vars are considered as fetch
      // vars.
      return false;
    }

    std::unordered_set<std::string> outputs;
    for (int i = 0; i < program.size(); i++) {
      const auto& instr = program[i];
      for (auto& var : instr->outputs) {
        outputs.insert(var->id);
      }
    }

    bool res = true;
    for (auto& id : fetch_ids) {
      if (!outputs.count(id)) {
        LOG(WARNING)
            << id
            << " in fetch_ids is not output of any instruction in program.";
        res = false;
      }
    }

    return res;
  }
};

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(DeadCodeEliminate) {
  CINN_REGISTER_PROGRAM_PASS(DeadCodeEliminate,
                             cinn::frontend::pass::DeadCodeEliminatePass);

  return true;
}
