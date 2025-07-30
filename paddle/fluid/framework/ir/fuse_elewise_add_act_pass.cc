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

#include "paddle/fluid/framework/ir/fuse_elewise_add_act_pass.h"

#include <string>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir {

void FuseElewiseAddActPass::ApplyImpl(ir::Graph *graph) const {
  std::unordered_set<std::string> act_types = {"relu", "scale", "tanh"};
  graph = FuseActElewiseAdd(graph, act_types);
  graph = FuseElewiseAddAct(graph, act_types);
  // backward
  {
    std::unordered_set<std::string> in_place_act_types = {"relu_grad"};
    graph = FuseElewiseAddActInplaceGrad(graph, in_place_act_types);
    graph = FuseActElewiseAddInplaceGrad(graph, in_place_act_types);
  }

  // Remove the removable intermediate_out.
  RemoveIntermediateOut(graph);
}

// ele_add(x, act(y))
ir::Graph *FuseElewiseAddActPass::FuseElewiseAddAct(
    ir::Graph *graph, const std::unordered_set<std::string> &act_types) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("elewise_add_act", graph);

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode("elewise_add_act/x")
                ->AsInput()
                ->assert_is_op_input("elementwise_add", "X");
  patterns::ElewiseAddAct elewise_add_act_pattern(gpd.mutable_pattern(),
                                                  "elementwise_add");

  elewise_add_act_pattern(x, act_types);

  int found_elewise_add_act_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle FuseElewiseAddAct fuse";
    GET_IR_NODE_FROM_SUBGRAPH(ele_y, ele_y, elewise_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_out, elewise_add_out, elewise_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, elewise_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act, act, elewise_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_add, ele_add, elewise_add_act_pattern);

    std::string ele_x_n = subgraph.at(x)->Name();
    std::string ele_y_n = ele_y->Name();
    std::string ele_out_n = ele_out->Name();
    std::string act_out_n = act_out->Name();

    Node *elewise_add_act_node = CreateFuseElewiseAddActNode(
        g, act, ele_add, ele_x_n, ele_y_n, ele_out_n, act_out_n);

    VLOG(4) << "\n\t " << ele_x_n << " and " << ele_y_n << " -> "
            << ele_add->Name() << " -> " << ele_out_n << "\n"
            << "\t " << ele_out_n << " -> " << act->Name() << " -> "
            << act_out_n;

    ReLinkNodes(g, ele_out, ele_add, act, elewise_add_act_node);
    found_elewise_add_act_count++;
  };

  gpd(graph, handler);

  AddStatis(found_elewise_add_act_count);
  return graph;
}

// act(ele_add(x,y))
ir::Graph *FuseElewiseAddActPass::FuseActElewiseAdd(
    ir::Graph *graph, const std::unordered_set<std::string> &act_types) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("act_elewise_add", graph);

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode("act_elewise_add/x")
                ->AsInput()
                ->assert_is_ops_input(act_types, "X");
  patterns::ActElewiseAdd act_elewise_add_pattern(gpd.mutable_pattern(),
                                                  "act_elewise_add");

  act_elewise_add_pattern(x, act_types);

  int found_elewise_add_act_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle FuseActElewiseAdd fuse";
    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, act_elewise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_x, ele_x, act_elewise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_out, elewise_add_out, act_elewise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act, act, act_elewise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_add, ele_add, act_elewise_add_pattern);

    std::string act_i_n = subgraph.at(x)->Name();
    std::string act_o_n = act_out->Name();
    std::string elewise_add_x_n = ele_x->Name();
    std::string elewise_add_out_n = ele_out->Name();

    Node *elewise_add_act_node = CreateFuseElewiseAddActNode(
        g, ele_add, act, elewise_add_x_n, act_i_n, act_o_n, elewise_add_out_n);

    VLOG(4) << "\n\t " << act_i_n << " -> " << act->Name() << " -> " << act_o_n
            << "\n\t " << act_o_n << " and " << elewise_add_x_n << " -> "
            << ele_add->Name() << " -> " << elewise_add_out_n;

    ReLinkNodes(g, act_out, act, ele_add, elewise_add_act_node);
    found_elewise_add_act_count++;
  };

  gpd(graph, handler);

  AddStatis(found_elewise_add_act_count);
  return graph;
}

// the backward of act(ele_add(x,y))
// act_grad: in["Out", "Out@GRAD"], out["X@GRAD"]
// ele_add_grad: in["Y", "Out@GRAD"], out["X@GRAD", "Y@GRAD"]
ir::Graph *FuseElewiseAddActPass::FuseElewiseAddActInplaceGrad(
    ir::Graph *graph, const std::unordered_set<std::string> &act_types) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("elewise_add_act_grad", graph);

  GraphPatternDetector gpd;
  auto *d_act_out = gpd.mutable_pattern()
                        ->NewNode("elewise_add_act_grad_inplace/x")
                        ->AsInput()
                        ->assert_is_ops_input(act_types, GradVarName("Out"));
  patterns::ElewiseAddActInplaceGrad elewise_add_act_grad_pattern(
      gpd.mutable_pattern(), "elewise_add_act_grad_inplace");
  elewise_add_act_grad_pattern(d_act_out, act_types);

  int found_elewise_add_act_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle FuseElewiseAddActGrad1 fuse";
    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, elewise_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_grad, act_grad, elewise_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        d_intermediate_out, d_intermediate_out, elewise_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_y, ele_y, elewise_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_grad, ele_add_grad, elewise_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(d_ele_x, d_ele_x, elewise_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(d_ele_y, d_ele_y, elewise_add_act_grad_pattern);

    std::string d_act_out_n = subgraph.at(d_act_out)->Name();
    std::string act_out_n = act_out->Name();
    std::string d_intermediate_out_n = d_intermediate_out->Name();
    std::string ele_y_n = ele_y->Name();
    std::string d_ele_x_n = d_ele_x->Name();
    std::string d_ele_y_n = d_ele_y->Name();

    OpDesc desc;
    desc.SetType("fused_elemwise_add_activation_grad");
    desc.SetInput("IntermediateOut", {});
    desc.SetInput("X", {});
    desc.SetInput("Y", std::vector<std::string>({ele_y_n}));
    desc.SetInput("Out", std::vector<std::string>({act_out_n}));
    desc.SetInput(GradVarName("Out"), std::vector<std::string>({d_act_out_n}));
    desc.SetOutput(GradVarName("X"), std::vector<std::string>({d_ele_x_n}));
    desc.SetOutput(GradVarName("Y"), std::vector<std::string>({d_ele_y_n}));
    desc.SetOutput(GradVarName("IntermediateOut"),
                   std::vector<std::string>({d_intermediate_out_n}));

    desc.SetAttr("axis", -1);
    desc.SetAttr("scale", 0.0f);
    desc.SetAttr("save_intermediate_out", false);
    desc.SetAttr("functor_list",
                 std::vector<std::string>(
                     {act_grad->Op()->Type(), ele_add_grad->Op()->Type()}));

    for (auto &n : {act_grad->Op(), ele_add_grad->Op()}) {
      for (auto &m_ele : n->GetAttrMap()) {
        desc.SetAttr(m_ele.first, m_ele.second);
      }
    }

    auto fused_node = g->CreateOpNode(&desc);

    VLOG(4) << "\n\t " << d_act_out_n << " and " << act_out_n << " -> "
            << act_grad->Name() << " -> " << d_intermediate_out_n << "\n\t "
            << d_intermediate_out_n << " and " << act_out_n << " -> "
            << ele_add_grad->Name() << " -> " << d_intermediate_out_n;

    ReLinkNodes(g, d_intermediate_out, act_grad, ele_add_grad, fused_node);
    found_elewise_add_act_count++;
  };

  gpd(graph, handler);

  AddStatis(found_elewise_add_act_count);
  return graph;
}

// the backward of act(ele_add(x,y))
// act_grad: in["Out", "Out@GRAD"], out["X@GRAD"]
// ele_add_grad: in["Y", "Out@GRAD"], out["X@GRAD", "Y@GRAD"]
ir::Graph *FuseElewiseAddActPass::FuseActElewiseAddInplaceGrad(
    ir::Graph *graph, const std::unordered_set<std::string> &act_types) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("act_elewise_add_grad", graph);
  GraphPatternDetector gpd;
  auto *d_out_var =
      gpd.mutable_pattern()
          ->NewNode("act_elewise_add_grad_inplace/d_out_var")
          ->AsInput()
          ->assert_is_ops_input({"elementwise_add_grad"}, GradVarName("Out"));
  patterns::ActElewiseAddInplaceGrad act_elewise_add_grad_pattern(
      gpd.mutable_pattern(), "act_elewise_add_grad_inplace");
  act_elewise_add_grad_pattern(d_out_var, act_types);

  int found_elewise_add_act_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle ActFuseElewiseAddGrad1 fuse";

    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_grad_op, ele_add_grad_op, act_elewise_add_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        act_grad_op, act_grad_op, act_elewise_add_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        intermediate_var, intermediate_var, act_elewise_add_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        d_intermediate_var, d_intermediate_var, act_elewise_add_grad_pattern);

    std::string d_out_var_n = subgraph.at(d_out_var)->Name();
    std::string intermediate_var_n = intermediate_var->Name();
    std::string d_intermediate_var_n = d_intermediate_var->Name();

    OpDesc desc;
    desc.SetType("fused_elemwise_add_activation_grad");
    desc.SetInput("IntermediateOut",
                  std::vector<std::string>({intermediate_var_n}));
    desc.SetInput("X", {});
    desc.SetInput("Y", ele_add_grad_op->Op()->Input("X"));
    desc.SetInput("Out", {});
    desc.SetInput(GradVarName("Out"), std::vector<std::string>({d_out_var_n}));
    desc.SetOutput(GradVarName("X"),
                   act_grad_op->Op()->Output(GradVarName("X")));
    desc.SetOutput(GradVarName("Y"),
                   ele_add_grad_op->Op()->Output(GradVarName("X")));
    desc.SetOutput(GradVarName("IntermediateOut"),
                   std::vector<std::string>({d_intermediate_var_n}));

    desc.SetAttr("axis", -1);
    desc.SetAttr("scale", 0.0f);
    desc.SetAttr("save_intermediate_out", false);
    desc.SetAttr("functor_list",
                 std::vector<std::string>({ele_add_grad_op->Op()->Type(),
                                           act_grad_op->Op()->Type()}));

    for (auto &n : {ele_add_grad_op->Op(), act_grad_op->Op()}) {
      for (auto &m_ele : n->GetAttrMap()) {
        desc.SetAttr(m_ele.first, m_ele.second);
      }
    }

    auto fused_node = g->CreateOpNode(&desc);

    VLOG(4) << "\n\t " << d_out_var_n << " -> " << ele_add_grad_op->Name()
            << " -> " << d_intermediate_var_n << "\n\t " << intermediate_var_n
            << " and " << d_intermediate_var_n << " -> " << act_grad_op->Name();

    ReLinkNodes2(
        g, d_intermediate_var, ele_add_grad_op, act_grad_op, fused_node);
    found_elewise_add_act_count++;
  };

  gpd(graph, handler);

  AddStatis(found_elewise_add_act_count);
  return graph;
}

Node *FuseElewiseAddActPass::CreateFuseElewiseAddActNode(
    Graph *g,
    const Node *op_1,
    const Node *op_2,
    const std::string &ele_x_n,
    const std::string &ele_y_n,
    const std::string &ele_out_n,
    const std::string &act_out_n) const {
  OpDesc desc;
  desc.SetInput("X", std::vector<std::string>({ele_x_n}));
  desc.SetInput("Y", std::vector<std::string>({ele_y_n}));
  desc.SetOutput("Out", std::vector<std::string>({act_out_n}));
  desc.SetOutput("IntermediateOut", std::vector<std::string>({ele_out_n}));
  desc.SetType("fused_elemwise_add_activation");
  desc.SetAttr("axis", -1);
  desc.SetAttr("scale", 0.0f);
  desc.SetAttr("save_intermediate_out", true);
  desc.SetAttr(
      "functor_list",
      std::vector<std::string>({op_1->Op()->Type(), op_2->Op()->Type()}));

  // Set attrs
  for (auto &n : {op_1->Op(), op_2->Op()}) {
    for (auto &m_ele : n->GetAttrMap()) {
      desc.SetAttr(m_ele.first, m_ele.second);
    }
  }

  auto elewise_add_act_node = g->CreateOpNode(&desc);
  return elewise_add_act_node;
}

void FuseElewiseAddActPass::RemoveIntermediateOut(Graph *graph) const {
  std::unordered_set<const Node *> need_removed_nodes;
  for (auto &cur_node : graph->Nodes()) {
    if (cur_node->IsVar()) continue;
    if (cur_node->Name() == "fused_elemwise_add_activation") {
      bool save_intermediate_out = PADDLE_GET_CONST(
          bool, cur_node->Op()->GetAttr("save_intermediate_out"));
      auto intermediate_out_args = cur_node->Op()->Output("IntermediateOut");
      PADDLE_ENFORCE_EQ(
          (save_intermediate_out && !intermediate_out_args.empty()),
          true,
          common::errors::InvalidArgument(
              "The %s should save the intermediate out in the fusing stage.",
              cur_node->Name()));

      // If the intermediate_out's output is empty, it should be removed.
      auto cur_node_outputs = cur_node->outputs;
      for (auto &out : cur_node_outputs) {
        if (out->Name() == intermediate_out_args[0]) {
          if (out->outputs.empty()) {
            cur_node->outputs = this->RemoveNode(out, cur_node->outputs);
            need_removed_nodes.insert(out);
            cur_node->Op()->SetAttr("save_intermediate_out", false);
          }
        }
      }
    } else if (cur_node->Name() == "fused_elemwise_add_activation_grad") {
      auto intermediate_out_grad_args =
          cur_node->Op()->Output(GradVarName("IntermediateOut"));
      PADDLE_ENFORCE_EQ(
          intermediate_out_grad_args.empty(),
          false,
          common::errors::InvalidArgument(
              "The %s should save the intermediate out in the fusing stage.",
              cur_node->Name()));
      auto cur_node_outputs = cur_node->outputs;
      // If the intermediate_out_g's output is empty, it should be removed.
      for (auto &out : cur_node_outputs) {
        if (out->Name() == intermediate_out_grad_args[0] &&
            out->outputs.empty()) {
          cur_node->Op()->SetOutput(GradVarName("IntermediateOut"), {});
          cur_node->outputs = this->RemoveNode(out, cur_node->outputs);
          need_removed_nodes.insert(out);
        }
      }
    }
  }
  details::RemovedVars *saved_removed_nodes = new details::RemovedVars;
  GraphSafeRemoveNodes(graph, need_removed_nodes, saved_removed_nodes);
  if (!saved_removed_nodes->empty()) {
    // TODO(pangyoki): If kRemovedVars exists, merge saved_removed_nodes into
    // RemovedVars.
    PADDLE_ENFORCE_EQ(graph->Has(details::kRemovedVars),
                      false,
                      common::errors::PreconditionNotMet(
                          "Removed nodes are only saved for "
                          "fuse_elewise_add_act_pass in temporary."));
    graph->Set(details::kRemovedVars, saved_removed_nodes);
  }
}

void FuseElewiseAddActPass::ReLinkNodes(Graph *graph,
                                        const Node *intermediate_out,
                                        Node *op_1,
                                        Node *op_2,
                                        Node *fused_op) const {  // delete act
  for (auto &in : op_1->inputs) {
    fused_op->inputs.emplace_back(in);
    in->outputs = this->ReplaceNode(op_1, fused_op, in->outputs);
  }

  std::unordered_set<const Node *> nodes2delete;
  for (auto &out : op_1->outputs) {
    if (out->IsCtrlVar()) {
      auto result_iter = std::find_if(
          op_2->inputs.begin(),
          op_2->inputs.end(),
          [&out](const Node *node) -> bool { return node == out; });

      if (result_iter == op_2->inputs.end()) {
        IR_OP_VAR_LINK(fused_op, out);
      } else {
        nodes2delete.emplace(out);
      }
    } else {
      PADDLE_ENFORCE_EQ(out,
                        intermediate_out,
                        common::errors::InvalidArgument(
                            "Output of op(%s) must be %s, but not %s.",
                            op_1->Name(),
                            intermediate_out->Name(),
                            out->Name()));
      IR_OP_VAR_LINK(fused_op, out);
    }
  }

  for (auto &in : op_2->inputs) {
    if (in == intermediate_out || nodes2delete.count(in)) {
      continue;
    }
    fused_op->inputs.emplace_back(in);
    in->outputs = this->ReplaceNode(op_2, fused_op, in->outputs);
  }

  for (auto &out : op_2->outputs) {
    IR_OP_VAR_LINK(fused_op, out);
  }

  nodes2delete.insert(op_1);
  nodes2delete.insert(op_2);

  GraphSafeRemoveNodes(graph, nodes2delete);
}

void FuseElewiseAddActPass::ReLinkNodes2(Graph *graph,
                                         const Node *intermediate_out,
                                         Node *op_1,
                                         Node *op_2,
                                         Node *fused_op) const {  // delete act
  for (auto &in : op_1->inputs) {
    fused_op->inputs.emplace_back(in);
    in->outputs = this->ReplaceNode(op_1, fused_op, in->outputs);
  }

  std::unordered_set<const Node *> nodes2delete;
  for (auto &out : op_1->outputs) {
    if (out->IsCtrlVar()) {
      auto result_iter = std::find_if(
          op_2->inputs.begin(),
          op_2->inputs.end(),
          [&out](const Node *node) -> bool { return node == out; });

      if (result_iter == op_2->inputs.end()) {
        IR_OP_VAR_LINK(fused_op, out);
      } else {
        nodes2delete.emplace(out);
      }
    } else {
      IR_OP_VAR_LINK(fused_op, out);
    }
  }

  for (auto &in : op_2->inputs) {
    if (in == intermediate_out || nodes2delete.count(in)) {
      continue;
    }
    fused_op->inputs.emplace_back(in);
    in->outputs = this->ReplaceNode(op_2, fused_op, in->outputs);
  }

  for (auto &out : op_2->outputs) {
    IR_OP_VAR_LINK(fused_op, out);
  }

  nodes2delete.insert(op_1);
  nodes2delete.insert(op_2);

  GraphSafeRemoveNodes(graph, nodes2delete);
}

std::vector<Node *> FuseElewiseAddActPass::ReplaceNode(
    Node *cur_node, Node *new_node, const std::vector<Node *> &nodes) const {
  std::vector<Node *> new_list(nodes.size());
  bool has_replaced = false;
  std::transform(
      nodes.begin(), nodes.end(), new_list.begin(), [&](Node *node) -> Node * {
        if (node == cur_node) {
          has_replaced = true;
          return new_node;
        }
        return node;
      });
  PADDLE_ENFORCE_EQ(has_replaced,
                    true,
                    common::errors::NotFound("Not found %s in the node list.",
                                             cur_node->Name()));
  return new_list;
}

std::vector<Node *> FuseElewiseAddActPass::RemoveNode(
    Node *trg_node, const std::vector<Node *> &nodes) const {
  std::vector<Node *> new_list(nodes.size());
  auto end_iter = std::copy_if(
      nodes.begin(), nodes.end(), new_list.begin(), [&](Node *node) -> bool {
        return node != trg_node;
      });
  new_list.resize(
      static_cast<uint64_t>(std::distance(new_list.begin(), end_iter)));
  return new_list;
}
}  // namespace paddle::framework::ir

REGISTER_PASS(fuse_elewise_add_act_pass,
              paddle::framework::ir::FuseElewiseAddActPass);
