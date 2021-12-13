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
    const BlockDesc &current_block, std::set<std::string> *inner_inputs,
    std::set<std::string> *inner_outputs) {
  /* Find inputs and outputs in current control flow block.
  :param current_block: Current control flow block.
  :param inner_inputs: Input var vector of ops in current block.
  :param inner_outputs: Output var vector of ops in current block. */

  // Step1: update inner_inputs and inner_outputs
  // NOTE: Here assumes that all variables are input or output of Ops,

  for (OpDesc *op : current_block.AllOps()) {
    for (auto iname : op->InputNames()) {
      for (auto in_var_name : op->Input(iname)) {
        VLOG(3) << "insert inner_inputs_name:" << in_var_name;
        inner_inputs->insert(in_var_name);
      }
    }

    for (auto oname : op->OutputNames()) {
      for (auto out_var_name : op->Output(oname)) {
        VLOG(3) << "insert out_var_name:" << out_var_name;
        inner_outputs->insert(out_var_name);
      }
    }
  }

  // Step2: Remove variable created in current control flow block.
  BlockDesc *parent_block = current_block.ParentBlock();

  if (parent_block) {
    for (auto iter = inner_inputs->begin(); iter != inner_inputs->end();) {
      const std::string &in_var_name = *iter;
      if (current_block.HasVar(in_var_name)) {
        VLOG(3) << "remove inner intput var:" << in_var_name;
        iter = inner_inputs->erase(iter);
      } else {
        ++iter;
      }
    }

    for (auto iter = inner_outputs->begin(); iter != inner_outputs->end();) {
      const std::string &out_var_name = *iter;
      if (current_block.HasVar(out_var_name)) {
        VLOG(3) << "remove inner output  var:" << out_var_name;
        iter = inner_outputs->erase(iter);
      } else {
        ++iter;
      }
    }
  }
}

void ProgramProcessor::AddDepToBlockOp(const BlockDesc &block) {
  VLOG(3) << "Op size:" << block.AllOps().size();
  for (OpDesc *op : block.AllOps()) {
    if (op->HasAttr("sub_block")) {
      auto op_type = op->Type();
      BlockDesc *sub_block =
          BOOST_GET_MUTABLE(BlockDesc *, op->GetAttr("sub_block"));

      // recursively processing
      AddDepToBlockOp(*sub_block);

      std::set<std::string> sub_inputs;
      std::set<std::string> sub_outputs;
      ProgramProcessor::GetInputsOutputsInBlock(*sub_block, &sub_inputs,
                                                &sub_outputs);
      VLOG(3) << "sub_inputs.size:" << sub_inputs.size();
      VLOG(3) << "sub_outputs.size:" << sub_outputs.size();

      auto *op_inputs = op->MutableInputs();
      std::vector<std::string> *op_input_var_vec;
      VLOG(3) << "op_type:>>>>>>" << op_type;
      if (op_type.compare("while") == 0) {
        op_input_var_vec = &((*op_inputs)["kX"]);
      } else if (op_type.compare("conditional_block") == 0) {
        op_input_var_vec = &((*op_inputs)["kInputs"]);
      } else {
        // Only support while_op and conditinal_block_op now
        LOG(WARNING)
            << "Currently, only support while_op and conditinal_block_op.\n";
        continue;
      }

      for (auto sub_input : sub_inputs) {
        if (std::find(op_input_var_vec->begin(), op_input_var_vec->end(),
                      sub_input) == op_input_var_vec->end())
          op_input_var_vec->push_back(sub_input);
        VLOG(3) << "modified private inputs, inputs.size():"
                << op_input_var_vec->size();
      }

      auto *op_outputs = op->MutableOutputs();
      auto *op_output_var_vec = &((*op_outputs)["kOutputs"]);

      for (auto sub_output : sub_outputs) {
        if (std::find(op_output_var_vec->begin(), op_output_var_vec->end(),
                      sub_output) == op_output_var_vec->end())
          op_output_var_vec->push_back(sub_output);
        VLOG(3) << "modified private outputs, outputs.size():"
                << op_output_var_vec->size();
      }
    }
  }
}

ProgramProcessor::ProgramProcessor() {}

}  // namespace framework
}  // namespace paddle
