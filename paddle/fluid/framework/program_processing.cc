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
      for (auto in_var_name : op->Inputs().find(iname)->second) {
        if (std::find(inner_inputs.begin(), inner_inputs.end(), in_var_name) !=
            inner_inputs.end())
          inner_inputs.push_back(in_var_name);
      }
  }
  for (auto in_var_name : inner_inputs) {
    VarDesc *parent_block_var =
        program->Block(current_block.Parent()).FindVarRecursive(in_var_name);
    VarDesc *current_block_var;
    if (current_block.HasVar(in_var_name)) {
      current_block_var = current_block.FindVar(in_var_name);
    }
    if (parent_block_var == nullptr && current_block_var &&
        current_block_var->GetType() == proto::VarType::LOD_TENSOR)
      removed_inner_inputs.push_back(in_var_name);
  }

  return !removed_inner_inputs.empty();
}

void ProgramProcessor::GetInputsOutputsInBlock(
    ProgramDesc *program, const BlockDesc &current_block,
    std::set<std::string> *inner_inputs, std::set<std::string> *inner_outputs) {
  // Step1: update inner_inputs and inner_outputs
  // NOTE: Here assumes that all variables are input or output of Ops,
  // but some variables are created without appendding a real op.
  // For example, in `arr = create_array(dtype)`, `arr` is not a output of a op.
  VLOG(3) << "GetInputsOutputsInBlock <<<<<<<:";
  std::set<std::string> removed_inner_inputs;
  VLOG(3) << "current_block.AllOps length:" << current_block.AllOps().size();
  for (OpDesc *op : current_block.AllOps()) {
    for (auto iname : op->InputNames())
      for (auto in_var_name : op->Inputs().find(iname)->second) {
        if ((*inner_inputs).find(in_var_name) == (*inner_inputs).end())
          (*inner_inputs).insert(in_var_name);
        VLOG(3) << "insert iname:" << in_var_name;
      }

    for (auto oname : op->OutputNames())
      for (auto out_var_name : op->Outputs().find(oname)->second) {
        VLOG(3) << "insert oame:" << out_var_name;
        (*inner_outputs).insert(out_var_name);
      }
  }

  //  Step2: Remove LOD_TENSOR_ARRAY created in current control flow block.
  BlockDesc *parent_block = program->MutableBlock(current_block.Parent());
  if (parent_block) {
    for (auto in_var_name : *inner_inputs) {
      VLOG(3) << "Step2 iname:" << in_var_name;
      VarDesc *parent_block_var = parent_block->FindVarRecursive(in_var_name);
      VLOG(3) << "Step2 FindVarRecursive:" << in_var_name;
      VarDesc *current_block_var;
      if (current_block.HasVar(in_var_name)) {
        current_block_var = current_block.FindVar(in_var_name);
      }
      if (parent_block_var == nullptr && current_block_var &&
          current_block_var->GetType() == proto::VarType::LOD_TENSOR)
        removed_inner_inputs.insert(in_var_name);
      VLOG(3) << "removed_inner_inputs iname:" << in_var_name;
    }
  }

  std::set_difference((*inner_inputs).begin(), (*inner_inputs).end(),
                      removed_inner_inputs.begin(), removed_inner_inputs.end(),
                      inserter((*inner_inputs), (*inner_inputs).begin()));
  VLOG(3) << "inner_inputs length:" << (*inner_inputs).size();
  VLOG(3) << "inner_outputs length:" << (*inner_outputs).size();
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
