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

void ProgramProcessor::GetInputsOutputsInBlock(
    ProgramDesc *program, const BlockDesc &current_block,
    std::set<std::string> *inner_inputs, std::set<std::string> *inner_outputs) {
  /* Find inputs and outputs in current control flow block.
  :param program: Program of current control flow block.
  :param current_block: Current control flow block.
  :param inner_inputs: Input var name of ops in current block.
  :param inner_outputs: Output var name of ops in current block. */

  // Step1: update inner_inputs and inner_outputs
  // NOTE: Here assumes that all variables are input or output of Ops,

  std::set<std::string> removed_inner_inputs;

  for (OpDesc *op : current_block.AllOps()) {
    for (auto iname : op->InputNames())
      for (auto in_var_name : op->Inputs().find(iname)->second) {
        if ((*inner_outputs).find(in_var_name) == (*inner_outputs).end())
          (*inner_inputs).insert(in_var_name);
      }

    for (auto oname : op->OutputNames())
      for (auto out_var_name : op->Outputs().find(oname)->second) {
        (*inner_outputs).insert(out_var_name);
      }
  }

  // Step2: Remove LOD_TENSOR_ARRAY created in current control flow block.

  BlockDesc *parent_block = program->MutableBlock(current_block.Parent());
  VarDesc *current_block_var;

  if (parent_block) {
    for (auto in_var_name : *inner_inputs) {
      VarDesc *parent_block_var = parent_block->FindVarRecursive(in_var_name);
      if (current_block.HasVar(in_var_name)) {
        current_block_var = current_block.FindVar(in_var_name);
      }
      if (parent_block_var == nullptr && current_block_var &&
          current_block_var->GetType() == proto::VarType::LOD_TENSOR)
        removed_inner_inputs.insert(in_var_name);
    }
  }

  std::set<std::string> inner_inputs_;
  std::set_difference((*inner_inputs).begin(), (*inner_inputs).end(),
                      removed_inner_inputs.begin(), removed_inner_inputs.end(),
                      inserter(inner_inputs_, inner_inputs_.begin()));

  (*inner_inputs).swap(inner_inputs_);
}

void ProgramProcessor::SSAProgram(ProgramDesc *program) {
  for (size_t i = 0; i < program->Size(); i++) {
    VLOG(3) << ">>>>>>>>>>>>>>>>>";
    VLOG(3) << "Block ID :" << program->Block(i).ID();
    VLOG(3) << "Block parent's ID :" << program->Block(i).Parent();
  }
}

ProgramProcessor::ProgramProcessor() {}

}  // namespace framework
}  // namespace paddle
