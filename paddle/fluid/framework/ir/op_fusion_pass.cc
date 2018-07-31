/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <unordered_set>

#include "paddle/fluid/framework/ir/op_fusion_pass.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
namespace ir {
const char kOpDescs[] = "op_descs";
using OpDescs = std::vector<std::unique_ptr<OpDesc>>;

std::unique_ptr<ir::Graph> OpFusionPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  // The created OpDesc will be stored in graph.
  graph->Set<OpDescs>(kOpDescs, new OpDescs());

  std::vector<NodePtr> operations = ir::GetOperations(*graph.get());

  // internal_nodes is used to record the origin node
  // and fused node which has this origin node.
  std::unordered_map<NodePtr, InternalNodePtr> internal_nodes;
  // need_removed_nodes is used to record the unnecessary nodes which should be
  // released.
  std::unordered_set<NodePtr> need_removed_nodes;

  for (auto iter_node = operations.begin(); iter_node != operations.end();
       ++iter_node) {
    auto cur_node = *iter_node;

    // tobe_fused is used to record the nodes which can be fused with cur_node,
    // and these nodes are the generation operator of cur_node's input.
    std::unordered_set<NodePtr> tobe_fused;

    // Whether cur_node can fuse with it adjacent nodes(in-degree), if so,
    // add the adjacent node to tobe_fused.
    // Node: cur_node maybe has been fused, if so, get the fused
    // node from internal_nodes, and check whether the fused node
    // can fuse with cur_node's adjacent nodes(in-degree).
    if (SetUpFusion(cur_node, internal_nodes, &tobe_fused)) {
      // fuse cur_node and tobe_fused.
      NodePtr fused_node =
          FuseOperators(cur_node, tobe_fused, &need_removed_nodes, graph.get());

      tobe_fused.emplace(cur_node);

      // Add the map between tobe_fused nodes and the fused node.
      for (auto &sub_node : tobe_fused) {
        PADDLE_ENFORCE_EQ(internal_nodes.count(sub_node), 0);
        internal_nodes.emplace(sub_node, fused_node);
      }

      // Record these nodes that are no longer useful.
      need_removed_nodes.insert(tobe_fused.begin(), tobe_fused.end());
    }
  }

  // Release unnecessary nodes
  for (auto &node : need_removed_nodes) {
    graph->ReleaseNode(node);
  }
  return graph;
}

// Whether node's input
bool OpFusionPass::SetUpFusion(
    const NodePtr node,
    const std::unordered_map<NodePtr, InternalNodePtr> &internal_nodes,
    std::unordered_set<NodePtr> *tobe_fused) const {
  PADDLE_ENFORCE(node->IsOperation(), "Node should be an operation.");
  NodePtr cur_node = node;

  // If the input node has be fused, get the fused_node from
  // internal_nodes, and the fused_node become the cur_node.
  if (internal_nodes.count(node)) {
    cur_node = internal_nodes.at(node);
  }

  auto no_control_vars = ir::NoControlDepVar(cur_node->inputs);
  if (no_control_vars.empty()) return false;

  VLOG(10) << "Current op:" << cur_node->Name();

  bool need_fusion = false;
  for (auto &in_var : no_control_vars) {
    if (in_var->inputs.empty()) continue;
    PADDLE_ENFORCE(in_var->IsVariable(), "in_var should be a variable.");
    PADDLE_ENFORCE_EQ(in_var->inputs.size(), 1,
                      "in_var's generation op should be only one.");

    auto in_var_gen_op = in_var->inputs[0];
    if (IsFusible(cur_node, in_var_gen_op)) {
      need_fusion = true;
      tobe_fused->insert(in_var_gen_op);
      VLOG(10) << "Fuse: " << cur_node->Name() << ", " << in_var_gen_op->Name();
    }
  }
  return need_fusion;
}

// Fuse cur_node and tobe_fused nodes
NodePtr OpFusionPass::FuseOperators(
    const NodePtr cur_node, const std::unordered_set<NodePtr> &tobe_fused,
    std::unordered_set<NodePtr> *need_removed_nodes, ir::Graph *graph) const {
  //  Create OpDesc,
  graph->Get<OpDescs>(kOpDescs).emplace_back(new framework::OpDesc());
  auto *fused_op_desc = graph->Get<OpDescs>(kOpDescs).back().get();

  if (tobe_fused.size() == 1) {
    // Init OpDesc
    if (IsElemwiseAndActivation(cur_node, tobe_fused)) {
      FuseElemwiseAndActivation(cur_node, tobe_fused, fused_op_desc);
    } else {
      PADDLE_THROW(
          "Currently, only support fusing elementwise and activation "
          "operator.");
    }
  } else {
    PADDLE_ENFORCE("Currently only support fusing two operators.");
  }

  // Create Node
  fused_op_desc->Flush();
  auto fused_node = graph->CreateOpNode(fused_op_desc);

  // Adjust the link relationship between nodes

  // Get the input arguments of the fused_node.
  // Node: the in_args may have duplicated name.
  auto in_args = fused_node->Op()->InputArgumentNames();
  std::unordered_set<std::string> in_args_set;
  for (auto &in : in_args) {
    in_args_set.emplace(in);
  }

  auto replace_node = [](NodePtr cur_node, NodePtr new_node,
                         std::vector<NodePtr> *vars) {
    bool flag = false;
    for (auto &o : *vars) {
      if (o == cur_node) {
        o = new_node;
        flag = true;
      }
    }
    PADDLE_ENFORCE(flag);
  };

  // link fused_node's input.
  for (auto &var : cur_node->inputs) {
    PADDLE_ENFORCE_LE(var->inputs.size(), 1);
    bool no_need_merge =
        var->inputs.empty() || !tobe_fused.count(var->inputs[0]);
    if (no_need_merge) {
      if (in_args_set.count(var->Name()) || ir::IsControlDepVar(*var)) {
        fused_node->inputs.emplace_back(var);
        replace_node(cur_node, fused_node, &(var->outputs));
      } else {
        if (var->outputs.size() == 1) {
          // TODO(zcd): how to deal with this situation
          PADDLE_THROW("how to deal with this situation(%s).",
                       var->outputs[0]->Name());
        } else {
          std::vector<NodePtr> new_out;
          for (auto &o : var->outputs) {
            if (o != cur_node) {
              new_out.emplace_back(o);
            }
          }
          var->outputs = new_out;
        }
      }
    } else {
      auto &in_var_gen_node = var->inputs[0];
      need_removed_nodes->emplace(var);
      for (auto &in_var : in_var_gen_node->inputs) {
        PADDLE_ENFORCE(in_var->IsVariable());
        fused_node->inputs.emplace_back(in_var);
        replace_node(in_var_gen_node, fused_node, &(in_var->outputs));
      }

      for (auto &out_var : in_var_gen_node->outputs) {
        PADDLE_ENFORCE(out_var->IsVariable());
        if (ir::IsControlDepVar(*out_var)) {
          fused_node->outputs.emplace_back(out_var);
          out_var->inputs.clear();
          out_var->inputs.emplace_back(fused_node);
        } else {
          need_removed_nodes->emplace(out_var);
        }
      }
    }
  }

  // link fused_node's output.
  for (auto &cur_output : cur_node->outputs) {
    PADDLE_ENFORCE(cur_output->IsVariable());
    fused_node->outputs.emplace_back(cur_output);
    cur_output->inputs.clear();
    cur_output->inputs.emplace_back(fused_node);
  }

  return fused_node;
}

bool OpFusionPass::IsBackward(
    const NodePtr node, const std::unordered_set<NodePtr> &tobe_fused) const {
  auto op_role = boost::get<int>(
      node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));

  for (auto &tebe_node : tobe_fused) {
    PADDLE_ENFORCE_EQ(op_role, boost::get<int>(tebe_node->Op()->GetAttr(
                                   OpProtoAndCheckerMaker::OpRoleAttrName())),
                      "Currently, only support fusing the same role operators");
  }
  return op_role == static_cast<int>(OpRole::kBackward);
}

bool OpFusionPass::IsFusible(const NodePtr n1, const NodePtr n2) const {
  PADDLE_ENFORCE(n1->IsOperation(), "n1 should be an operation.");
  PADDLE_ENFORCE(n2->IsOperation(), "n2 should be an operation.");

  // TODO(zcd): hard code
  bool case1 = (n1->Op()->Type() == "scale" || n1->Op()->Type() == "relu") &&
               (n2->Op()->Type() == "elementwise_add");
  bool case2 = (n1->Op()->Type() == "elementwise_add") &&
               (n2->Op()->Type() == "scale" || n2->Op()->Type() == "relu");
  bool case3 =
      (n1->Op()->Type() == "elementwise_add_grad") &&
      (n2->Op()->Type() == "scale_grad" || n2->Op()->Type() == "relu_grad");
  //  bool case4 =
  //    (n1->Op()->Type() == "scale_grad" || n1->Op()->Type() == "relu_grad") &&
  //    (n2->Op()->Type() == "elementwise_add_grad");

  return case1 || case2 || case3;
}

// temporally
static bool IsActivation(std::string op_type) {
  static std::unordered_set<std::string> activations = {
      "relu", "scale", "relu_grad", "scale_grad"};
  return activations.count(op_type) == 1;
}

// temporally
static bool IsElemwise(std::string op_type) {
  static std::unordered_set<std::string> elementwise = {"elementwise_add",
                                                        "elementwise_add_grad"};
  return elementwise.count(op_type) == 1;
}

bool OpFusionPass::IsElemwiseAndActivation(
    const NodePtr node, const std::unordered_set<NodePtr> &tobe_fused) const {
  PADDLE_ENFORCE_EQ(tobe_fused.size(), 1);
  auto outside_op_name = node->Op()->Type();
  auto inside_op_name = (*tobe_fused.begin())->Op()->Type();

  return (IsActivation(outside_op_name) && IsElemwise(inside_op_name)) ||
         (IsActivation(inside_op_name) && IsElemwise(outside_op_name));
}

void OpFusionPass::FuseElemwiseAndActivation(
    const NodePtr outside_node, const std::unordered_set<NodePtr> &tobe_fused,
    OpDesc *op_desc) const {
  auto intra_node = *tobe_fused.begin();
  auto intra_op_type = intra_node->Op()->Type();
  auto outside_op_type = outside_node->Op()->Type();

  auto intra_node_in_args = intra_node->Op()->InputArgumentNames();
  auto intra_node_out_args = intra_node->Op()->OutputArgumentNames();
  auto outside_node_in_args = outside_node->Op()->InputArgumentNames();

  // Set Input
  if (IsBackward(outside_node, tobe_fused)) {
    op_desc->SetType("fused_elemwise_activation_grad");
    op_desc->SetAttr("functor_list", intra_op_type + "," + outside_op_type);

    if (IsElemwise(outside_op_type)) {
      PADDLE_ENFORCE_LE(intra_node_in_args.size(), 3);
      PADDLE_ENFORCE_GE(intra_node_in_args.size(), 2);
      PADDLE_ENFORCE_EQ(outside_node_in_args.size(), 4);

      auto intra_node_in1 = intra_node->Op()->Input("Out");
      auto intra_node_in2 =
          intra_node->Op()->Input(::paddle::framework::GradVarName("Out"));

      auto outside_node_in1 = outside_node->Op()->Input("X");
      auto outside_node_in2 = outside_node->Op()->Input("Y");
      auto out1 =
          outside_node->Op()->Output(::paddle::framework::GradVarName("X"));
      auto out2 =
          outside_node->Op()->Output(::paddle::framework::GradVarName("Y"));

      op_desc->SetInput("X", outside_node_in1);
      op_desc->SetInput("Y", outside_node_in2);
      op_desc->SetInput("Out", intra_node_in1);
      op_desc->SetInput(::paddle::framework::GradVarName("Out"),
                        intra_node_in2);
      op_desc->SetOutput(::paddle::framework::GradVarName("X"), out1);
      op_desc->SetOutput(::paddle::framework::GradVarName("Y"), out2);
    } else {
      PADDLE_THROW("Not implement.");
    }
  } else {
    op_desc->SetType("fused_elemwise_activation");
    op_desc->SetAttr("functor_list", outside_op_type + "," + intra_op_type);

    if (IsElemwise(outside_op_type)) {
      PADDLE_ENFORCE_EQ(intra_node_in_args.size(), 1);
      PADDLE_ENFORCE_EQ(intra_node_out_args.size(), 1);
      PADDLE_ENFORCE_EQ(outside_node_in_args.size(), 2);

      op_desc->SetInput("Y", intra_node_in_args);

      if (outside_node_in_args[0] == intra_node_out_args[0]) {
        op_desc->SetInput("X", {outside_node_in_args[1]});
      } else if (outside_node_in_args[1] == intra_node_out_args[0]) {
        op_desc->SetInput("X", {outside_node_in_args[0]});
      } else {
        PADDLE_THROW("exception");
      }
    } else {
      op_desc->SetInput("Y", intra_node->Op()->Input("Y"));
      op_desc->SetInput("X", intra_node->Op()->Input("X"));
    }
    // Set output
    op_desc->SetOutput("Out", outside_node->Op()->OutputArgumentNames());
  }

  // Set attrs
  for (auto &n : {intra_node, outside_node}) {
    for (auto &m_ele : n->Op()->GetAttrMap()) {
      op_desc->SetAttr(m_ele.first, m_ele.second);
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(op_fusion_pass, paddle::framework::ir::OpFusionPass);
