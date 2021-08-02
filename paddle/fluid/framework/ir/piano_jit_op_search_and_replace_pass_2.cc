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

#include "paddle/fluid/framework/ir/piano_jit_op_search_and_replace_pass.h"

namespace paddle {
namespace framework {
namespace ir {

constexpr char kPianoJitOpName[] = "PianoJitOp";

void PianoJitOpSearchAndReplacePass::ApplyImpl(ir::Graph* graph) const {
  // Step1: Collect direct connection information between jit-supported ops
  // through a var.
  PianoJitOpSearchHelper search_helper;
        if (search_helper.IsJitSupported(next_op_node->name())) {

  // Step2: Find out clusters through dfs method.
  std::vector<std::vector<const Node*>> clusters;

  // Step3: Add a new op node for each cluster.
  for (const auto& cluster : clusters) {
    std::unordered_set<const Node*> inputs_recoder;
    std::unordered_set<const Node*> outputs_recoder;
    for (const Node* op_node : cluster) {
      for (const Node* input_var_node : op_node->inputs) {
        inputs_recorder.insert(input_var_node);
      }
      for (const Node* output_var_node : op_node->outputs) {
        outputs_recorder.insert(output_var_node);
      }
    }

    std::vector<Node*> piano_jit_op_inputs;
    std::vector<Node*> piano_jit_op_outputs;
    for (const Node* var_node : outputs_recoder) {
      if (inputs_recoder.find(var_node) != inputs_recoder.end()) {
        inputs_recoder.erase(var_node);
        bool is_only_used_internel = true;
        for (const Node* next_op_node : var_node->outputs) {
          is_only_used_internel &=
              (cluster.find(next_op_node) != cluster.end());
        }
        if (is_only_used_internel) {
          graph->RemoveNode(var_node);
        } else {
          piano_jit_op_outputs.push_back(var_node);
        }
      } else {
        piano_jit_op_outputs.push_back(var_node);
      }
    }
    for (const Node* var_node : inputs_recoder) {
      piano_jit_op_inputs.push_back(var_node);
    }
    // TODO(levi): need to add attr to piano_jit_op.
    Node* piano_jit_op =
        graph->CreateEmptyNode(PianoJitOpName, Node::Type::kOperation);
    piano_jit_op->inputs = piano_jit_op_inputs;
    piano_jit_op->outputs = piano_jit_op_outputs;
  }
  // Step4: Delete merged ops.
  for (const Node* target_op_node : target_op_nodes) {
    graph->RemoveNode(target_op_node);
  }
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(piano_jit_op_search_and_replace_pass,
              paddle::framework::ir::PianoJitOpSearchAndReplacePass);
