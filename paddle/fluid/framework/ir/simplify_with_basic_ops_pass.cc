/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/simplify_with_basic_ops_pass.h"

#include <unordered_set>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

void SimplifyWithBasicOpsPass::ApplyImpl(Graph* graph) const {
  VLOG(3) << "Simplify the Graph with basic ops.";
  std::unordered_set<const Node*> del_node_set;
  for (Node* n : graph->Nodes()) {
    bool can_remove = false;
    if (n->IsOp() && n->Op()) {
      if (n->Op()->Type() == "dropout") {
        can_remove = SimplifyDropout(graph, n);
      }
    }
    if (can_remove) {
      del_node_set.insert(n);
    }
  }

  GraphSafeRemoveNodes(graph, del_node_set);
}

bool SimplifyWithBasicOpsPass::SimplifyDropout(Graph* graph, Node* n) const {
  OpDesc* op = n->Op();
  bool is_test = boost::get<bool>(op->GetAttr("is_test"));
  if (!is_test) {
    return false;
  }

  Node* dropout_x = GetInputVar(n, op->Input("X")[0]);
  Node* dropout_out = GetOutputVar(n, op->Output("Out")[0]);

  bool upscale_in_train =
      boost::get<std::string>(op->GetAttr("dropout_implementation")) ==
      "upscale_in_train";
  if (upscale_in_train) {
    // dropout_op can be delete.
    for (auto* next_op : dropout_out->outputs) {
      dropout_x->outputs.push_back(next_op);
      for (size_t i = 0; i < next_op->inputs.size(); ++i) {
        if (next_op->inputs[i] == dropout_out) {
          next_op->inputs[i] = dropout_x;
        }
      }
    }
  } else {
    // use a scale_op replaces the dropout_op
    float scale = 1.0f - boost::get<float>(op->GetAttr("dropout_prob"));

    framework::OpDesc new_op_desc;
    new_op_desc.SetType("scale");
    new_op_desc.SetInput("X", {dropout_x->Name()});
    new_op_desc.SetOutput("Out", {dropout_out->Name()});
    new_op_desc.SetAttr("scale", scale);
    new_op_desc.SetAttr("bias", static_cast<float>(0));
    new_op_desc.SetAttr("bias_after_scale", true);

    auto scale_node = graph->CreateOpNode(&new_op_desc);
    IR_NODE_LINK_TO(dropout_x, scale_node);
    IR_NODE_LINK_TO(scale_node, dropout_out);
  }
  return true;
}

Node* SimplifyWithBasicOpsPass::GetInputVar(Node* n,
                                            const std::string& name) const {
  for (auto* in : n->inputs) {
    if (in->Name() == name) {
      return in;
    }
  }
  return nullptr;
}

Node* SimplifyWithBasicOpsPass::GetOutputVar(Node* n,
                                             const std::string& name) const {
  for (auto* out : n->outputs) {
    if (out->Name() == name) {
      return out;
    }
  }
  return nullptr;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(simplify_with_basic_ops_pass,
              paddle::framework::ir::SimplifyWithBasicOpsPass);
