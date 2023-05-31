// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <unordered_set>

#include "paddle/cinn/frontend/decomposer_registry.h"
#include "paddle/cinn/frontend/program_pass.h"

namespace cinn {
namespace frontend {
namespace pass {

class DecomposerPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  void Clear() override {}

  void ApplyImpl(Program* prog,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) const override {
    // step 1: set the inputs of the origin program to the new program
    NetBuilder builder("decomposer_builder");
    for (auto& var : prog->GetInputs()) {
      builder.CreateInput(var);
    }

    // step 2: use primitive instructions to build the new program
    absl::flat_hash_map<std::string, Variable> var_map;
    DecomposerContext context(&builder, &var_map);
    for (size_t i = 0; i < prog->size(); i++) {
      auto instr = (*prog)[i];
      auto decomposer =
          InstrDecomposerRegistry::Global()->Find(instr->op_type, target);
      if (decomposer) {
        VLOG(3) << "Run decomposer of op " << instr->op_type;
        decomposer->Run(instr, context);
      } else {
        VLOG(3) << "Don't run decomposer of op " << instr->op_type;
        builder.AppendInstruction(instr);
      }
    }
    VLOG(3) << "Before builder.Build()";
    *prog = builder.Build();
    VLOG(3) << "After builder.Build()";
    // step 3: set the origin output to the output of decomposed operator.
    for (size_t i = 0; i < prog->size(); i++) {
      auto& outputs = (*prog)[i]->outputs;
      for (size_t j = 0; j < outputs.size(); j++) {
        auto it = var_map.find(outputs[j]->id);
        if (it != var_map.end()) {
          outputs[j] = it->second;
        }
      }
      auto& inputs = (*prog)[i]->inputs;
      for (size_t j = 0; j < inputs.size(); j++) {
        auto it = var_map.find(inputs[j]->id);
        if (it != var_map.end()) {
          inputs[j] = it->second;
        }
      }
    }
  }
};

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(Decomposer) {
  CINN_REGISTER_PROGRAM_PASS(Decomposer,
                             ::cinn::frontend::pass::DecomposerPass);

  return true;
}
