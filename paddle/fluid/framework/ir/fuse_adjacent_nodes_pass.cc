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

#include "paddle/fluid/framework/ir/fuse_adjacent_nodes_pass.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
namespace ir {
const char kOpDescs[] = "op_descs";
using OpDescs = std::vector<std::unique_ptr<OpDesc>>;

std::unique_ptr<ir::Graph> FuseAdjacentNodesPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  // The created OpDesc will be stored in graph.
  graph->Set<OpDescs>(kOpDescs, new OpDescs());

  std::vector<NodePtr> topo_order = ir::TopologySortOperations(*graph.get());

  // internal_nodes is used to record the origin node and fused node
  // which has the origin node.
  std::unordered_map<NodePtr, InternalNodePtr> internal_nodes;
  // need_removed_nodes is used to record the unnecessary nodes which should be
  // released.
  std::unordered_set<NodePtr> need_removed_nodes;

  for (auto iter_node = topo_order.begin(); iter_node != topo_order.end();
       ++iter_node) {
    auto cur_op_node = *iter_node;

    // tobe_fused_nodes is used to record the nodes which can be fused with
    // cur_op_node.
    std::unordered_set<NodePtr> tobe_fused_nodes;

    // Whether cur_op_node can fuse with it's adjacent nodes(in-degree)
    if (FindToBeFusedNodes(cur_op_node, internal_nodes, &tobe_fused_nodes)) {
      // fuse cur_op_node and tobe_fused_nodes.
      NodePtr fused_node = FuseNodes(cur_op_node, tobe_fused_nodes,
                                     &need_removed_nodes, graph.get());

      tobe_fused_nodes.emplace(cur_op_node);

      // Add the map between tobe_fused_nodes nodes and the fused node.
      for (auto &sub_node : tobe_fused_nodes) {
        PADDLE_ENFORCE_EQ(internal_nodes.count(sub_node), 0);
        internal_nodes.emplace(sub_node, fused_node);
      }

      // Record these nodes that are no longer useful.
      need_removed_nodes.insert(tobe_fused_nodes.begin(),
                                tobe_fused_nodes.end());
    }
  }

  // Release unnecessary nodes
  for (auto &node : need_removed_nodes) {
    graph->RemoveNode(node);
  }
  return graph;
}

bool FuseAdjacentNodesPass::FindToBeFusedNodes(
    const NodePtr node,
    const std::unordered_map<NodePtr, InternalNodePtr> &internal_nodes,
    std::unordered_set<NodePtr> *tobe_fused_nodes) const {
  PADDLE_ENFORCE(node->IsOp(), "Node should be an operation.");

  NodePtr cur_op_node = node;
  // If the input node has be fused, get the fused_node from
  // internal_nodes, and the fused_node become the cur_op_node.
  if (internal_nodes.count(node)) {
    cur_op_node = internal_nodes.at(node);
  }

  auto no_control_vars = ir::NoControlDepVar(cur_op_node->inputs);
  if (no_control_vars.empty()) return false;

  VLOG(10) << "Current op:" << cur_op_node->Name();

  bool need_fusion = false;
  for (auto &in_var : no_control_vars) {
    if (in_var->inputs.empty()) continue;

    PADDLE_ENFORCE(in_var->IsVar(), "in_var should be a variable.");
    PADDLE_ENFORCE_EQ(in_var->inputs.size(), 1,
                      "in_var's generation op should be only one.");

    auto in_var_gen_op = in_var->inputs[0];

    if (IsFusible(cur_op_node, in_var_gen_op)) {
      need_fusion = true;
      tobe_fused_nodes->insert(in_var_gen_op);
      VLOG(10) << "Fuse: " << in_var_gen_op->Name() << ", "
               << cur_op_node->Name();
    }
  }
  return need_fusion;
}

NodePtr FuseAdjacentNodesPass::FuseNodes(
    const NodePtr cur_op_node,
    const std::unordered_set<NodePtr> &tobe_fused_nodes,
    std::unordered_set<NodePtr> *need_removed_nodes, ir::Graph *graph) const {
  //  Create OpDesc,
  graph->Get<OpDescs>(kOpDescs).emplace_back(new framework::OpDesc());
  auto *fused_op_desc = graph->Get<OpDescs>(kOpDescs).back().get();

  if (tobe_fused_nodes.size() == 1) {
    // Init OpDesc
    if (IsElemwiseAndActivation(cur_op_node, tobe_fused_nodes)) {
      FuseElemwiseAndActivation(cur_op_node, tobe_fused_nodes, fused_op_desc);
    } else {
      PADDLE_THROW(
          "Currently, only support fusing elementwise and activation "
          "operator.");
    }
  } else {
    PADDLE_THROW("Currently only support fusing two operators.");
  }

  // Create Node
  fused_op_desc->Flush();
  auto fused_node = graph->CreateOpNode(fused_op_desc);

  // Adjust the link relationship between nodes

  // Replace cur_node with new_node.
  auto replace_node = [](NodePtr cur_node, NodePtr new_node,
                         std::vector<NodePtr> *nodes) {
    bool has_replaced = false;
    for (auto o : *nodes) {
      if (o == cur_node) {
        o = new_node;
        has_replaced = true;
      }
    }
    PADDLE_ENFORCE(has_replaced, "Not find %s in the node list.",
                   cur_node->Name());
  };

  // Remove cur_node from nodes.
  auto remove_node = [](NodePtr cur_node,
                        std::vector<NodePtr> *nodes) -> std::vector<NodePtr> {
    std::vector<NodePtr> new_list;
    for (auto o : *nodes) {
      if (o != cur_node) {
        new_list.emplace_back(o);
      }
    }
    return new_list;
  };

  // Get the input and output arguments of the fused_node.
  // Node: the in_args may have duplicated name.
  auto in_args = fused_node->Op()->InputArgumentNames();
  std::unordered_set<std::string> in_args_set;
  for (auto &in : in_args) {
    in_args_set.emplace(in);
  }
  auto out_args = fused_node->Op()->InputArgumentNames();
  std::unordered_set<std::string> out_args_set;
  for (auto &out : out_args) {
    out_args_set.emplace(out);
  }

  // link fused_node's input.
  std::unordered_set<NodePtr> has_resolved_nodes;
  std::vector<NodePtr> cur_op_node_ins = cur_op_node->inputs;
  for (auto &var : cur_op_node_ins) {
    PADDLE_ENFORCE(var->IsVar(), "%s should be variable.", var->Name());

    if (has_resolved_nodes.count(var)) continue;
    has_resolved_nodes.emplace(var);

    // If the var's input is empty, or the var's input is not in
    // tobe_fused_nodes,
    // the var should be the input of the fused_op.
    bool is_fused_op_input =
        var->inputs.empty() || !tobe_fused_nodes.count(var->inputs[0]);

    if (is_fused_op_input) {
      // the var only should be in the in_args_set or be a ControlDepVar.
      if (in_args_set.count(var->Name()) || ir::IsControlDepVar(*var)) {
        fused_node->inputs.emplace_back(var);
        replace_node(cur_op_node, fused_node, &(var->outputs));
      } else {
        PADDLE_THROW("%s is not the input of %s, and not a ControlDepVar.",
                     var->Name(), fused_node->Name());
      }
    } else {
      // Otherwise, the var may be removed.
      need_removed_nodes->emplace(var);

      auto &in_var_gen_node = var->inputs[0];
      var->inputs.clear();
      PADDLE_ENFORCE(tobe_fused_nodes.count(in_var_gen_node) == 1, "");

      for (auto &in_var : in_var_gen_node->inputs) {
        PADDLE_ENFORCE(in_var->IsVar(), "%s should be the input of Op(%s)",
                       in_var->Name(), in_var_gen_node->Name());
        fused_node->inputs.emplace_back(in_var);
        replace_node(in_var_gen_node, fused_node, &(in_var->outputs));
      }

      for (auto &out_var : in_var_gen_node->outputs) {
        PADDLE_ENFORCE(out_var->IsVar(), "%s should be the output of Op(%s)",
                       out_var->Name(), in_var_gen_node->Name());
        // the out_var maybe the input of cur_op_node
        has_resolved_nodes.emplace(out_var);

        if (ir::IsControlDepVar(*out_var)) {
          if (out_var->outputs.size() > 0) {
            out_var->outputs = remove_node(cur_op_node, &out_var->outputs);
            out_var->inputs[0] = fused_node;
            if (need_removed_nodes->count(out_var)) {
              need_removed_nodes->erase(out_var);
            }
          } else {
            PADDLE_ENFORCE(out_var->outputs.size() == 1 &&
                           out_var->outputs[0] == cur_op_node);
            cur_op_node->inputs =
                remove_node(cur_op_node, &cur_op_node->inputs);
            out_var->inputs.clear();
            out_var->outputs.clear();
            need_removed_nodes->emplace(out_var);
          }
        } else {
          // If the out_var is in out_args_set, it should be the output of
          // fused_node
          if (out_args_set.count(out_var->Name()) == 1) {
            fused_node->outputs.emplace_back(out_var);
            out_var->inputs[0] = fused_node;
          } else {
            need_removed_nodes->emplace(out_var);
          }
        }
      }
    }
  }

  // link fused_node's output.
  for (auto &cur_output : cur_op_node->outputs) {
    PADDLE_ENFORCE(cur_output->IsVar(), "%s should be the output of Op(%s)",
                   cur_output->Name(), cur_op_node->Name());
    fused_node->outputs.emplace_back(cur_output);
    cur_output->inputs[0] = fused_node;
  }
  return fused_node;
}

bool FuseAdjacentNodesPass::IsBackward(
    const NodePtr node,
    const std::unordered_set<NodePtr> &tobe_fused_nodes) const {
  auto op_role = boost::get<int>(
      node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));

  for (auto &tebe_node : tobe_fused_nodes) {
    PADDLE_ENFORCE_EQ(op_role, boost::get<int>(tebe_node->Op()->GetAttr(
                                   OpProtoAndCheckerMaker::OpRoleAttrName())),
                      "Currently, only support fusing the same role operators");
  }
  return op_role == static_cast<int>(OpRole::kBackward);
}

bool FuseAdjacentNodesPass::IsFusible(const NodePtr cur_op_node,
                                      const NodePtr upstream_op_node) const {
  PADDLE_ENFORCE(cur_op_node->IsOp(), "cur_op_node should be an operation.");
  PADDLE_ENFORCE(upstream_op_node->IsOp(), "n2 should be an operation.");

  // TODO(zcd): hard code
  bool case1 = (cur_op_node->Op()->Type() == "scale" ||
                cur_op_node->Op()->Type() == "relu") &&
               (upstream_op_node->Op()->Type() == "elementwise_add");
  bool case2 = (cur_op_node->Op()->Type() == "elementwise_add") &&
               (upstream_op_node->Op()->Type() == "scale" ||
                upstream_op_node->Op()->Type() == "relu");
  //  bool case3 = (cur_op_node->Op()->Type() == "elementwise_add_grad") &&
  //               (upstream_op_node->Op()->Type() == "scale_grad" ||
  //                upstream_op_node->Op()->Type() == "relu_grad");
  //  bool case4 =
  //    (cur_op_node->Op()->Type() == "scale_grad" || cur_op_node->Op()->Type()
  //    == "relu_grad") &&
  //    (n2->Op()->Type() == "elementwise_add_grad");

  return case1 || case2;
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

bool FuseAdjacentNodesPass::IsElemwiseAndActivation(
    const NodePtr node,
    const std::unordered_set<NodePtr> &tobe_fused_nodes) const {
  PADDLE_ENFORCE_EQ(tobe_fused_nodes.size(), 1);
  auto outside_op_name = node->Op()->Type();
  auto inside_op_name = (*tobe_fused_nodes.begin())->Op()->Type();

  return (IsActivation(outside_op_name) && IsElemwise(inside_op_name)) ||
         (IsActivation(inside_op_name) && IsElemwise(outside_op_name));
}

void FuseAdjacentNodesPass::FuseElemwiseAndActivation(
    const NodePtr cur_op_node,
    const std::unordered_set<NodePtr> &tobe_fused_nodes,
    OpDesc *op_desc) const {
  auto upstream_op_node = *tobe_fused_nodes.begin();
  auto upstream_op_type = upstream_op_node->Op()->Type();
  auto cur_op_type = cur_op_node->Op()->Type();

  auto upstream_op_node_in_args = upstream_op_node->Op()->InputArgumentNames();
  auto upstream_op_node_out_args =
      upstream_op_node->Op()->OutputArgumentNames();
  auto cur_op_node_in_args = cur_op_node->Op()->InputArgumentNames();
  auto cur_op_node_out_args = cur_op_node->Op()->OutputArgumentNames();

  // Set Input
  if (IsBackward(cur_op_node, tobe_fused_nodes)) {
    const std::string op_type = "fused_elemwise_activation_grad";
    op_desc->SetType(op_type);

    // Set attrs
    op_desc->SetAttr("functor_list",
                     std::vector<std::string>({upstream_op_type, cur_op_type}));
    op_desc->SetAttr("recomputation", false);

    auto out_grad = ::paddle::framework::GradVarName("Out");
    auto x_grad = ::paddle::framework::GradVarName("X");
    auto y_grad = ::paddle::framework::GradVarName("Y");

    if (IsElemwise(cur_op_type)) {
      // the backward of  Unary(Binary(X, Y))
      PADDLE_ENFORCE(upstream_op_node_out_args.size() == 1,
                     "The number of output of %s should be 1.",
                     upstream_op_type);
      PADDLE_ENFORCE(cur_op_node_out_args.size() == 2,
                     "The number of output of %s should be 2.", cur_op_type);
      PADDLE_ENFORCE(
          upstream_op_node_in_args.size() == 2 ||
              upstream_op_node_in_args.size() == 3,
          "The number of inputs of %s should be 2 or 3, "
          "if the number is 2, the input is 'Out', 'Out@Grad', "
          "if the number is 3, the input is 'X', 'Out' and 'Out@Grad'.",
          upstream_op_node->Op()->Type());
      PADDLE_ENFORCE(
          cur_op_node_in_args.size() == 2 || cur_op_node_in_args.size() == 4,
          "The number of inputs of %s should be 2 or 4, if the number is 2, "
          "the input variable is `Y`, and `Out@Grad`, if the number is "
          "4, the input variable is `X`, `Y`, `Out`, `Out@Grad`",
          cur_op_type);

      if (cur_op_node_in_args.size() == 4) {
        op_desc->SetInput("X", cur_op_node->Op()->Input("X"));
      } else {
        // for the BinaryFunctor is elementwise_add, the computation
        // of its backward doesn't use 'x' and 'y', but the shape of
        // dy only can be inferred from 'y', so 'y' should be input.
        if (upstream_op_node_in_args.size() == 3) {
          op_desc->SetInput("IntermediateOut",
                            upstream_op_node->Op()->Input("Y"));
        }
      }
      op_desc->SetInput("Y", cur_op_node->Op()->Input("Y"));
      op_desc->SetInput("Out", upstream_op_node->Op()->Input("Out"));
      op_desc->SetInput(out_grad, upstream_op_node->Op()->Input(out_grad));
      op_desc->SetOutput(x_grad, cur_op_node->Op()->Output(x_grad));
      op_desc->SetOutput(y_grad, upstream_op_node->Op()->Output(x_grad));
    } else {
      // the backward of Binary(X, Unary(Y))
      PADDLE_ENFORCE(upstream_op_node_out_args.size() == 2,
                     "The number of output of %s should be 2.",
                     upstream_op_type);
      PADDLE_ENFORCE(cur_op_node_out_args.size() == 1,
                     "The number of output of %s should be 1.", cur_op_type);
      PADDLE_ENFORCE(
          cur_op_node_in_args.size() == 2 || cur_op_node_in_args.size() == 3,
          "The number of inputs of %s should be 2 or 3, "
          "if the number is 2, the input is 'Out', 'Out@Grad', "
          "if the number is 3, the input is 'X', 'Out' and 'Out@Grad'.",
          cur_op_type);
      PADDLE_ENFORCE(
          upstream_op_node_in_args.size() == 2 ||
              upstream_op_node_in_args.size() == 4,
          "The number of inputs of %s should be 2 or 4, if the number is 2, "
          "the input variable is `Y`, and `Out@Grad`, if the number is "
          "4, the input variable is `X`, `Y`, `Out`, `Out@Grad`",
          upstream_op_type);

      if (upstream_op_node_in_args.size() == 4) {
        op_desc->SetInput("X", upstream_op_node->Op()->Input("X"));
      } else {
        // for the BinaryFunctor is elementwise_add, the computation
        // of its backward doesn't use 'x' and 'y', but the shape of
        // dy only can be inferred from 'y', so 'y' should be input.
        if (cur_op_node_in_args.size() == 3) {
          op_desc->SetInput("Y", upstream_op_node->Op()->Input("Y"));
        } else {
        }
      }
      op_desc->SetInput("IntermediateOut", cur_op_node->Op()->Input("Out"));
      op_desc->SetInput("Y", upstream_op_node->Op()->Input("Y"));
      op_desc->SetInput("Out", upstream_op_node->Op()->Input("Out"));
      op_desc->SetInput(out_grad, upstream_op_node->Op()->Input(out_grad));
      op_desc->SetOutput(x_grad, upstream_op_node->Op()->Output(x_grad));
      op_desc->SetOutput(y_grad, cur_op_node->Op()->Output(x_grad));

      // Set the "X"
      auto result_iter = std::find(upstream_op_node_out_args.begin(),
                                   upstream_op_node_out_args.end(),
                                   cur_op_node->Op()->Input(out_grad)[0]);
      if (result_iter == upstream_op_node_out_args.end()) {
        PADDLE_THROW("%s's output is not the input of %s", upstream_op_type,
                     cur_op_type);
      }
    }
  } else {  // The forward of Binary(X, Unary(Y)) or Unary(Binary(X, Y))
    PADDLE_ENFORCE_EQ(upstream_op_node_out_args.size(), 1,
                      "The number of output of UnaryFunctor(BinaryFunctor)[%s] "
                      "should be one.",
                      upstream_op_type);
    PADDLE_ENFORCE_EQ(cur_op_node_out_args.size(), 1,
                      "The number of output of BinaryFunctor(UnaryFunctor)[%s] "
                      "should be one.",
                      cur_op_type);

    op_desc->SetType("fused_elemwise_activation");
    op_desc->SetAttr("functor_list",
                     std::vector<std::string>({cur_op_type, upstream_op_type}));
    op_desc->SetAttr("recomputation", false);

    // The output of compound functor.
    std::vector<std::string> out_args;
    out_args.emplace_back(cur_op_node_out_args[0]);
    bool keep_intermediate_out = false;

    if (IsElemwise(cur_op_type)) {
      // Z = Binary(X, Unary(Y))
      PADDLE_ENFORCE_EQ(upstream_op_node_in_args.size(), 1,
                        "The number of input of UnaryFunctor should be one.");
      PADDLE_ENFORCE_EQ(cur_op_node_in_args.size(), 2,
                        "The number of input of BinaryFunctor should be two.");
      // Set the "Y"
      op_desc->SetInput("Y", upstream_op_node_in_args);

      // Set the "X"
      auto result_iter =
          std::find(cur_op_node_in_args.begin(), cur_op_node_in_args.end(),
                    upstream_op_node_out_args[0]);
      if (result_iter == cur_op_node_in_args.end()) {
        PADDLE_THROW("%s's output is not the input of %s", upstream_op_type,
                     cur_op_type);
      }
      // x_idx is 0 or 1 here.
      int x_idx =
          1 - static_cast<int>(result_iter - cur_op_node_in_args.begin());
      op_desc->SetInput("X", {cur_op_node_in_args[x_idx]});

      if (cur_op_type == "elementwise_add") {
        keep_intermediate_out = true;
      }
    } else {
      // Z = Unary(Binary(X, Y))
      PADDLE_ENFORCE_EQ(cur_op_node_in_args.size(), 1,
                        "The number of input of UnaryFunctor should be one.");
      PADDLE_ENFORCE_EQ(upstream_op_node_in_args.size(), 2,
                        "The number of input of BinaryFunctor should be two.");
      // Set the "Y" and "X"
      op_desc->SetInput("Y", upstream_op_node->Op()->Input("Y"));
      op_desc->SetInput("X", upstream_op_node->Op()->Input("X"));
      // the input of the backward of elementwise_add doesn't include "X",
      // so we must save the intermediate_out here.
      if (cur_op_type == "elementwise_add") {
        keep_intermediate_out = true;
      }
    }

    if (keep_intermediate_out) {
      op_desc->SetAttr("keep_intermediate_value", true);
    }
    op_desc->SetOutput("IntermediateOut", upstream_op_node_out_args);
    op_desc->SetOutput("Out", cur_op_node_out_args);
  }

  // Set attrs
  for (auto &n : {upstream_op_node, cur_op_node}) {
    for (auto &m_ele : n->Op()->GetAttrMap()) {
      op_desc->SetAttr(m_ele.first, m_ele.second);
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(op_fusion_pass, paddle::framework::ir::FuseAdjacentNodesPass);
