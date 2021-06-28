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

void ProgramProcessor::GetInputsOutputsInBlock(const BlockDesc &current_block,
                                               VariableNameMap *inner_inputs,
                                               VariableNameMap *inner_outputs) {
  /* Find inputs and outputs in current control flow block.
  :param current_block: Current control flow block.
  :param inner_inputs: Input VariableNameMap of ops in current block.
  :param inner_outputs: Output VariableNameMap of ops in current block. */

  // Step1: update inner_inputs and inner_outputs
  // NOTE: Here assumes that all variables are input or output of Ops,

  std::set<std::string> inner_inputs_var;
  std::set<std::string> inner_outputs_var;

  for (OpDesc *op : current_block.AllOps()) {
    for (auto iname : op->InputNames()) {
      bool AddInput = false;
      for (auto in_var_name : op->Input(iname)) {
        VLOG(3) << "in_var_name:" << in_var_name;
        if (inner_outputs_var.find(in_var_name) == inner_outputs_var.end()) {
          inner_inputs_var.insert(in_var_name);
          AddInput = true;  // here we assume if one var is not in the
                            // outputs_var list, we add this input into the
                            // inputs list.
        }
      }
      if (AddInput) inner_inputs->emplace(iname, op->Input(iname));
    }

    for (auto oname : op->OutputNames()) {
      for (auto out_var_name : op->Output(oname)) {
        VLOG(3) << "out_var_name:" << out_var_name;
        inner_outputs_var.insert(out_var_name);
      }
      inner_outputs->emplace(oname, op->Output(oname));
    }
  }

  // Step2: Remove LOD_TENSOR_ARRAY created in current control flow block.
  BlockDesc *parent_block = current_block.ParentBlock();
  VarDesc *current_block_var;

  if (parent_block) {
    for (auto in_var_name : inner_inputs_var) {
      VLOG(3) << "recursively find var:" << in_var_name;
      VarDesc *parent_block_var = parent_block->FindVarRecursive(in_var_name);
      if (current_block.HasVar(in_var_name)) {
        current_block_var = current_block.FindVar(in_var_name);
      }
      if (parent_block_var == nullptr && current_block_var &&
          current_block_var->GetType() == proto::VarType::LOD_TENSOR) {
        // if we find var created in current block, remove it in block input.
        auto removed_var = in_var_name;
        VLOG(3) << "removed_var" << removed_var;
        for (auto it1 = inner_inputs->begin(); it1 != inner_inputs->end();) {
          auto *var_vector = &it1->second;
          for (auto it2 = var_vector->begin(); it2 != var_vector->end();) {
            if (it2->compare(removed_var) == 0)
              it2 = var_vector->erase(it2);
            else
              it2++;
          }
          if (var_vector->size() == 0)
            it1 = inner_inputs->erase(it1);
          else
            it1++;
        }
      }
    }
  }
}

void ProgramProcessor::AddDepToBlockOp(const BlockDesc &block) {
  VLOG(3) << "Op size:" << block.AllOps().size();
  for (OpDesc *op : block.AllOps()) {
    if (op->HasAttr("sub_block")) {
      auto sub_block = op->GetAttr("sub_block");
      // TODO(huangxu96): sub_block is an Attr, how to get a BlockDesc*?
      // recursively processing
      // VLOG(3)<<"sub_block" << sub_block;
      // AddDepToBlockOp(sub_block);

      VariableNameMap sub_inputs;
      VariableNameMap sub_outputs;
      ProgramProcessor::GetInputsOutputsInBlock(block, &sub_inputs,
                                                &sub_outputs);
      VLOG(3) << "sub_inputs.size:" << sub_inputs.size();
      VLOG(3) << "sub_outputs.size:" << sub_outputs.size();
      // TODO(huangxu96): check sub_inputs and sub_outputs are in parent block.

      const std::vector<std::string> &op_inputs_name = op->InputNames();
      auto *op_inputs = op->MutableInputs();
      for (auto sub_input : sub_inputs) {
        if (op_inputs->find(sub_input.first) == op_inputs->end())
          op_inputs->insert(sub_input);
        VLOG(3) << "modified private inputs, inputs.size():"
                << op_inputs->size();
      }
      const std::vector<std::string> &op_outputs_name = op->OutputNames();
      auto *op_outputs = op->MutableOutputs();
      for (auto sub_output : sub_outputs) {
        if (op_outputs->find(sub_output.first) == op_outputs->end()) {
          op_outputs->insert(sub_output);
          VLOG(3) << "modified private outputs, outputs.size():"
                  << op_outputs->size();
        }
      }
    }
  }
}

void ProgramProcessor::ToSSAProgram(ProgramDesc *program) {
  // TODO(huangxu96)
}

ProgramProcessor::ProgramProcessor() {}

}  // namespace framework
}  // namespace paddle
