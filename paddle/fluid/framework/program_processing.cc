/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/program_processing.h"
#include "paddle/fluid/framework/block_desc.h"

namespace paddle {
namespace framework {

bool ProgramProcessor::IsControlFlowBlock(ProgramDesc *program,
                                          const BlockDesc &current_block) {
  // Determing if the input varible is created in control flow block.
  std::vector<std::string> inner_inputs;
  std::vector<std::string> removed_inner_inputs;
  for (OpDesc *op : current_block.AllOps()) {
    for (auto iname : op->InputNames())
      if (std::find(inner_inputs.begin(), inner_inputs.end(), iname) !=
          inner_inputs.end())
        inner_inputs.push_back(iname);
  }
  for (auto in_var_name : inner_inputs) {
    VarDesc *parent_block_var =
        program->Block(current_block.Parent()).FindVarRecursive(in_var_name);
    VarDesc *current_block_var;
    if (current_block.HasVar(in_var_name)) {
      current_block_var = current_block.FindVar(in_var_name);
    }
    if (parent_block_var == nullptr && current_block_var)
      removed_inner_inputs.push_back(in_var_name);
  }

  return !removed_inner_inputs.empty();
}

void ProgramProcessor::SSAProgram(ProgramDesc *program) {
  for (size_t i = 0; i < program->Size(); i++) {
    VLOG(3) << ">>>>>>>>>>>>>>>>>";
    VLOG(3) << "Block ID :" << program->Block(i).ID();
    VLOG(3) << "Block parent's ID :" << program->Block(i).Parent();
    if (IsControlFlowBlock(program, program->Block(i))) {
      VLOG(3) << "Block ID with whlie op:" << program->Block(i).ID();
      // ssa_processing(program, cur_block);
    } else {
      VLOG(3) << "Not a ControlFlow Block";
    }
  }
}

ProgramProcessor::ProgramProcessor() {}

}  // namespace framework
}  // namespace paddle
