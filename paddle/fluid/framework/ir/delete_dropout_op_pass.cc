// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/ir/delete_dropout_op_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node_) GET_IR_NODE_FROM_SUBGRAPH(node_, node_, pattern)

void DeleteDropoutOpPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "delete_dropout_op_pattern";
  FusePassBase::Init(pattern_name, graph);
  int found_subgraph_count = 0;

  for (auto with_mask : {true, false}) {
    GraphPatternDetector gpd;
    patterns::DeleteDropoutOpPattern pattern(gpd.mutable_pattern(),
                                             pattern_name);
    pattern(with_mask);

    auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                       Graph* g) {
      GET_IR_NODE(dropout_op_x);
      GET_IR_NODE(dropout_op);
      GET_IR_NODE(dropout_op_out);

      // link dropout_op_x to next_op
      auto dropout_op_x_name = dropout_op_x->Var()->Name();
      auto dropout_op_out_name = dropout_op_out->Var()->Name();
      auto next_op_nodes = dropout_op_out->outputs;
      for (auto next_op_node : next_op_nodes) {
        auto next_op_desc = next_op_node->Op();
        auto next_op_inputs = next_op_desc->Inputs();
        for (auto& input_var : next_op_inputs) {
          auto names = input_var.second;
          for (size_t i = 0; i < names.size(); i++) {
            if (names[i] == dropout_op_out_name) {
              names[i] = dropout_op_x_name;
              next_op_desc->SetInput(input_var.first, names);
              break;
            }
          }
        }
        IR_NODE_LINK_TO(dropout_op_x, next_op_node);
      }
      // delete useless node
      std::unordered_set<const Node*> delete_nodes{dropout_op, dropout_op_out};
      if (with_mask) {
        GET_IR_NODE(dropout_op_mask);
        delete_nodes.insert(dropout_op_mask);
      }
      GraphSafeRemoveNodes(graph, delete_nodes);
      found_subgraph_count++;
    };
    gpd(graph, handler);
  }
  AddStatis(found_subgraph_count);
}

DeleteDropoutOpXPass::DeleteDropoutOpXPass() {
  AddOpCompat(OpCompat("scale"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("scale")
      .IsNumGE(0.f)
      .IsNumLE(1.f)
      .End()
      .AddAttr("bias")
      .IsNumEQ(0.f)
      .End()
      .AddAttr("bias_after_scale")
      .IsNumEQ(true)
      .End();
}

void DeleteDropoutOpXPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "delte dropout op.";
  std::unordered_set<const Node*> del_node_set;
  for (Node* n : graph->Nodes()) {
    if (n->IsOp() && n->Op()) {
      if (n->Op()->Type() == "dropout") {
        DelDropout(graph, n, &del_node_set);
      }
    }
  }

  GraphSafeRemoveNodes(graph, del_node_set);
}

bool DeleteDropoutOpXPass::DelDropout(
    Graph* graph,
    Node* n,
    std::unordered_set<const Node*>* del_node_set) const {
  OpDesc* dropout_op_desc = n->Op();

  Node* dropout_x = GetInputVar(n, dropout_op_desc->Input("X")[0]);
  Node* dropout_out = GetOutputVar(n, dropout_op_desc->Output("Out")[0]);

  bool upscale_in_train = false;
  // Once the dropout_implementation's AttrType is BOOLEAN, but now is STRING.
  if (dropout_op_desc->HasAttr("dropout_implementation")) {
    if (dropout_op_desc->GetAttrType("dropout_implementation") ==
        proto::AttrType::BOOLEAN) {
      upscale_in_train = PADDLE_GET_CONST(
          bool, dropout_op_desc->GetAttr("dropout_implementation"));
    } else if (dropout_op_desc->GetAttrType("dropout_implementation") ==
               proto::AttrType::STRING) {
      upscale_in_train =
          PADDLE_GET_CONST(std::string,
                           dropout_op_desc->GetAttr(
                               "dropout_implementation")) == "upscale_in_train";
    }
  }

  VLOG(3) << "upscale_in_train: " << upscale_in_train;

  if (upscale_in_train) {
    // delete dropout
    // dropout_op can be deleted.
    // dropout_x -> dropout_op -> dropout_out -> next_op -> next_out
    //   |
    //  \|/
    // dropout_x -> next_op -> next_out
    // Check whether dropout_x is some next_op's output
    bool dropout_x_is_reused_as_output = false;
    for (auto* next_op : dropout_out->outputs) {
      for (auto* next_out : next_op->outputs) {
        if (next_out == dropout_x ||
            next_out->Var()->Name() == dropout_x->Var()->Name()) {
          dropout_x_is_reused_as_output = true;
          break;
        }
      }
      if (dropout_x_is_reused_as_output) {
        break;
      }
    }
    if (dropout_x_is_reused_as_output) {
      VarDesc new_var_desc(*dropout_x->Var());
      new_var_desc.SetName("delete_dropout_x_pass_" + dropout_x->Name());
      auto* new_var_node = graph->CreateVarNode(&new_var_desc);
      for (auto* out_op : dropout_x->outputs) {
        if (out_op != n) {
          ReplaceInputVar(out_op, dropout_x, new_var_node);
        }
      }
      for (auto* in_op : dropout_x->inputs) {
        ReplaceOutputVar(in_op, dropout_x, new_var_node);
      }
      dropout_x = new_var_node;
    }
    for (auto* next_op : dropout_out->outputs) {
      ReplaceInputVar(next_op, dropout_out, dropout_x);
    }

    del_node_set->insert(dropout_out);
  } else {
    // keep dropout
    // Use a scale_op replaces the dropout_op
    // dropout_x -> dropout_op -> dropout_out -> next_op -> next_out
    //   |
    //  \|/
    // dropout_x -> scale_op -> dropout_out -> next_op -> next_out
    float scale = 1.0f - PADDLE_GET_CONST(
                             float, dropout_op_desc->GetAttr("dropout_prob"));

    framework::OpDesc new_op_desc(dropout_op_desc->Block());
    new_op_desc.SetType("scale");
    new_op_desc.SetInput("X", {dropout_x->Name()});
    new_op_desc.SetOutput("Out", {dropout_out->Name()});
    new_op_desc.SetAttr("scale", scale);
    new_op_desc.SetAttr("bias", static_cast<float>(0));
    new_op_desc.SetAttr("bias_after_scale", true);

    if (!IsCompat(new_op_desc)) {
      LOG(WARNING) << "Basic ops pass in scale op compat failed.";
      return false;
    }

    auto* scale_op_node = graph->CreateOpNode(&new_op_desc);
    IR_NODE_LINK_TO(dropout_x, scale_op_node);
    IR_NODE_LINK_TO(scale_op_node, dropout_out);
  }

  del_node_set->insert(n);
  return true;
}

Node* DeleteDropoutOpXPass::GetInputVar(Node* n,
                                        const std::string& name) const {
  for (auto* in : n->inputs) {
    if (in->Name() == name) {
      return in;
    }
  }
  return nullptr;
}

Node* DeleteDropoutOpXPass::GetOutputVar(Node* n,
                                         const std::string& name) const {
  for (auto* out : n->outputs) {
    if (out->Name() == name) {
      return out;
    }
  }
  return nullptr;
}

void DeleteDropoutOpXPass::ReplaceInputVar(Node* op,
                                           Node* old_var,
                                           Node* new_var) const {
  if (op->IsOp() && op->Op()) {
    new_var->outputs.push_back(op);
    for (size_t i = 0; i < op->inputs.size(); ++i) {
      if (op->inputs[i] == old_var) {
        op->inputs[i] = new_var;
        op->Op()->RenameInput(old_var->Name(), new_var->Name());
      }
    }
  }
}

void DeleteDropoutOpXPass::ReplaceOutputVar(Node* op,
                                            Node* old_var,
                                            Node* new_var) const {
  if (op->IsOp() && op->Op()) {
    new_var->inputs.push_back(op);
    for (size_t i = 0; i < op->outputs.size(); ++i) {
      if (op->outputs[i] == old_var) {
        op->outputs[i] = new_var;
        op->Op()->RenameOutput(old_var->Name(), new_var->Name());
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_dropout_op_pass,
              paddle::framework::ir::DeleteDropoutOpPass);
REGISTER_PASS_CAPABILITY(delete_dropout_op_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "dropout", 0));

REGISTER_PASS(delete_dropout_op_x_pass,
              paddle::framework::ir::DeleteDropoutOpXPass);
REGISTER_PASS_CAPABILITY(delete_dropout_op_x_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "scale", 0));
