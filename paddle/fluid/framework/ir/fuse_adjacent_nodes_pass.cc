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

std::unique_ptr<ir::Graph> FuseAdjacentNodesPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  std::vector<Node *> topo_order = ir::TopologySortOperations(*graph);

  // internal_nodes is used to record the origin node and the fused node
  // which has the origin node.
  std::unordered_map<Node *, Node *> internal_nodes;
  // need_removed_nodes is used to record the useless nodes which should be
  // released.
  std::unordered_set<Node *> need_removed_nodes;

  for (auto iter_node = topo_order.begin(); iter_node != topo_order.end();
       ++iter_node) {
    auto cur_op_node = *iter_node;

    std::unordered_set<Node *> tobe_fused_nodes;

    if (FindToBeFusedNodes(cur_op_node, internal_nodes, &tobe_fused_nodes)) {
      // Generate the fused node.
      Node *fused_node = FuseNodes(cur_op_node, tobe_fused_nodes,
                                   &need_removed_nodes, graph.get());
      if (fused_node == nullptr) {
        continue;
      }

      // Add the map between tobe_fused_nodes nodes and the fused node.
      tobe_fused_nodes.emplace(cur_op_node);
      for (auto &sub_node : tobe_fused_nodes) {
        PADDLE_ENFORCE_EQ(internal_nodes.count(sub_node), 0);
        internal_nodes.emplace(sub_node, fused_node);
      }

      // Record these nodes that are no longer used.
      need_removed_nodes.insert(tobe_fused_nodes.begin(),
                                tobe_fused_nodes.end());
    }
  }

  // Remove the removable intermediate_out.
  RemoveIntermediateOut(graph.get(), &need_removed_nodes);

  // Release unnecessary nodes
  for (auto &node : need_removed_nodes) {
    graph->RemoveNode(node);
  }

  return graph;
}

void FuseAdjacentNodesPass::RemoveIntermediateOut(
    const Graph *graph, std::unordered_set<Node *> *need_removed_nodes) const {
  for (auto upstream_node : graph->Nodes()) {
    if (upstream_node->IsVar()) continue;
    if (upstream_node->Name() == "fused_elemwise_activation") {
      bool save_intermediate_out = boost::get<bool>(
          upstream_node->Op()->GetAttr("save_intermediate_out"));
      auto intermediate_out_args =
          upstream_node->Op()->Output("IntermediateOut");
      PADDLE_ENFORCE(
          save_intermediate_out && !intermediate_out_args.empty(),
          "The %s should save the intermediate_out in the fusing stage.",
          upstream_node->Name());

      // If the intermediate_out's output is only
      // fused_elemwise_activation_grad, but the fused_elemwise_activation_grad
      // doesn't use the intermediate_out.
      bool find_backward = false;
      auto upstream_node_outputs = upstream_node->outputs;
      for (auto out : upstream_node_outputs) {
        if (out->Name() == intermediate_out_args[0]) {
          if (out->outputs.size() == 1 &&
              out->outputs[0]->Name() == "fused_elemwise_activation_grad") {
            auto &backward_node = out->outputs[0];
            bool use_intermediate_out = boost::get<bool>(
                backward_node->Op()->GetAttr("save_intermediate_out"));
            if (!use_intermediate_out) {
              upstream_node->outputs =
                  this->RemoveNode(out, upstream_node->outputs);
              out->inputs.clear();
              backward_node->inputs =
                  this->RemoveNode(out, backward_node->inputs);
              out->outputs.clear();
              need_removed_nodes->emplace(out);
            }
            find_backward = true;
          }
        }
      }
      // If the backward is not found, it is unnecessary to save the
      // intermediate_out
      if (!find_backward) {
        upstream_node->Op()->SetAttr("save_intermediate_out", false);
        auto intermediate_node_iter = std::find_if(
            upstream_node->outputs.begin(), upstream_node->outputs.end(),
            [&intermediate_out_args](const Node *node) -> bool {
              return node->Name() == intermediate_out_args[0];
            });

        PADDLE_ENFORCE(intermediate_node_iter != upstream_node->outputs.end());
        upstream_node->outputs =
            this->RemoveNode(*intermediate_node_iter, upstream_node->outputs);
        need_removed_nodes->emplace(*intermediate_node_iter);
      }
    } else if (upstream_node->Name() == "fused_elemwise_activation_grad") {
      // If the intermediate_out_grad's output is zero.
      auto intermediate_out_arg =
          upstream_node->Op()->Output(GradVarName("IntermediateOut"))[0];
      auto upstream_node_outputs = upstream_node->outputs;
      for (auto &out : upstream_node_outputs) {
        if (out->Name() == intermediate_out_arg && out->outputs.empty()) {
          upstream_node->Op()->SetOutput(GradVarName("IntermediateOut"), {});
          upstream_node->outputs =
              this->RemoveNode(out, upstream_node->outputs);
          need_removed_nodes->emplace(out);
        }
      }
    }
  }
}

bool FuseAdjacentNodesPass::FindToBeFusedNodes(
    Node *node, const std::unordered_map<Node *, Node *> &internal_nodes,
    std::unordered_set<Node *> *tobe_fused_nodes) const {
  PADDLE_ENFORCE(node->IsOp(), "Node should be an operator.");

  Node *cur_op_node = node;

  // If the input node has be fused, get the fused_node from
  // internal_nodes, and the fused_node become the cur_op_node.
  if (internal_nodes.count(node)) {
    cur_op_node = internal_nodes.at(node);
  }

  auto no_control_vars = ir::NoControlDepVar(cur_op_node->inputs);

  if (no_control_vars.empty()) {
    return false;
  }

  VLOG(10) << "Current op:" << cur_op_node->Name();

  bool need_fusion = false;
  for (auto &in_var : no_control_vars) {
    if (in_var->inputs.empty()) continue;

    PADDLE_ENFORCE(in_var->IsVar(), "in_var should be a variable.");
    PADDLE_ENFORCE_EQ(in_var->inputs.size(), 1,
                      "in_var's generation op should be only one.");

    auto upstream_op = in_var->inputs[0];

    if (IsFusible(cur_op_node, upstream_op)) {
      need_fusion = true;
      tobe_fused_nodes->insert(upstream_op);
      VLOG(10) << "Fuse: " << upstream_op->Name() << ", "
               << cur_op_node->Name();
    }
  }
  return need_fusion;
}

Node *FuseAdjacentNodesPass::FuseNodes(
    Node *cur_op_node, const std::unordered_set<Node *> &tobe_fused_nodes,
    std::unordered_set<Node *> *need_removed_nodes, ir::Graph *graph) const {
  framework::OpDesc fused_op_desc;
  // intermediate_outs
  std::unordered_set<Node *> intermediate_outs;
  // Init OpDesc
  if (tobe_fused_nodes.size() == 1) {
    if (IsElemwiseAndActivation(cur_op_node, *tobe_fused_nodes.begin())) {
      FuseElemwiseAndActivation(cur_op_node, *tobe_fused_nodes.begin(),
                                &fused_op_desc, &intermediate_outs);
    } else {
      // Currently, only support fusing elementwise and activation operator.
      return nullptr;
    }
  } else {
    // Currently only support fusing two operators.
    return nullptr;
  }

  // Create Node
  auto fused_node = graph->CreateOpNode(&fused_op_desc);

  // Get the input and output arguments of the fused_node.
  // Node: the in_args may have duplicated name.
  auto in_args = fused_node->Op()->InputArgumentNames();
  std::unordered_set<std::string> in_args_set(in_args.begin(), in_args.end());
  auto out_args = fused_node->Op()->OutputArgumentNames();
  std::unordered_set<std::string> out_args_set(out_args.begin(),
                                               out_args.end());

  std::unordered_set<Node *> visited_nodes;
  std::vector<Node *> cur_op_node_ins = cur_op_node->inputs;

  for (auto &var : cur_op_node_ins) {
    if (visited_nodes.count(var)) continue;
    PADDLE_ENFORCE(var->IsVar(), "%s should be variable.", var->Name());
    visited_nodes.emplace(var);

    // If the var's input is empty, or the var's input is not in
    // tobe_fused_nodes, the var should be the input of the fused_op.
    bool is_fused_op_input =
        var->inputs.empty() || !tobe_fused_nodes.count(var->inputs[0]);

    if (is_fused_op_input) {
      // the var should only be in the in_args_set or be a ControlDepVar.
      if (in_args_set.count(var->Name()) || ir::IsControlDepVar(*var)) {
        fused_node->inputs.emplace_back(var);
        var->outputs = ReplaceNode(cur_op_node, fused_node, var->outputs);
      } else {
        PADDLE_THROW("%s is not the input of %s, and not a ControlDepVar.",
                     var->Name(), fused_node->Name());
      }
    } else {
      auto &in_var_gen_node = var->inputs[0];
      PADDLE_ENFORCE(tobe_fused_nodes.count(in_var_gen_node) == 1,
                     "%s 's generation op(%s) should be in tobe_fused_nodes.",
                     var->Name(), in_var_gen_node->Name());

      // collect inputs
      for (auto &in_var : in_var_gen_node->inputs) {
        PADDLE_ENFORCE(in_var->IsVar(), "%s should be the input of Op(%s)",
                       in_var->Name(), in_var_gen_node->Name());
        fused_node->inputs.emplace_back(in_var);
        in_var->outputs =
            ReplaceNode(in_var_gen_node, fused_node, in_var->outputs);
      }

      // collect outputs
      for (auto &out_var : in_var_gen_node->outputs) {
        PADDLE_ENFORCE(out_var->IsVar(), "%s should be the output of Op(%s)",
                       out_var->Name(), in_var_gen_node->Name());
        // the out_var maybe the input of cur_op_node
        visited_nodes.emplace(out_var);

        // If the out_var is in out_args_set, it should be the output of
        // fused_node
        if (out_args_set.count(out_var->Name()) == 1) {
          fused_node->outputs.emplace_back(out_var);
          out_var->inputs[0] = fused_node;
          cur_op_node->inputs = RemoveNode(cur_op_node, cur_op_node->inputs);
          out_var->outputs = RemoveNode(cur_op_node, out_var->outputs);
        } else if (ir::IsControlDepVar(*out_var)) {
          if (out_var->outputs.size() > 1 ||
              out_var->outputs[0] != cur_op_node) {
            fused_node->outputs.emplace_back(out_var);
            out_var->inputs[0] = fused_node;
            cur_op_node->inputs = RemoveNode(cur_op_node, cur_op_node->inputs);
            out_var->outputs = RemoveNode(cur_op_node, out_var->outputs);
          } else {
            PADDLE_ENFORCE(out_var->outputs.size() == 1);
            PADDLE_ENFORCE(out_var->outputs[0] == cur_op_node,
                           "The two nodes should be the same(%s,%s).",
                           out_var->outputs[0]->Name(), cur_op_node->Name());
            out_var->inputs.clear();
            cur_op_node->inputs = RemoveNode(cur_op_node, cur_op_node->inputs);
            out_var->outputs.clear();
            need_removed_nodes->emplace(out_var);
          }
        } else {
          PADDLE_THROW("%s is not the output of %s, and not a ControlDepVar.",
                       out_var->Name(), fused_node->Name());
        }
      }
    }
  }
  // set fused_node's output.
  for (auto &cur_output : cur_op_node->outputs) {
    PADDLE_ENFORCE(cur_output->IsVar(), "%s should be the output of Op(%s)",
                   cur_output->Name(), cur_op_node->Name());
    fused_node->outputs.emplace_back(cur_output);
    cur_output->inputs[0] = fused_node;
  }

  if (intermediate_outs.size()) {
    for (auto &intermediate_out : intermediate_outs) {
      auto iter = std::find(fused_node->inputs.begin(),
                            fused_node->inputs.end(), intermediate_out);
      if (iter == fused_node->inputs.end()) {
        fused_node->inputs.emplace_back(intermediate_out);
        intermediate_out->outputs.emplace_back(fused_node);
      }
    }
  }
  return fused_node;
}

bool FuseAdjacentNodesPass::IsBackward(Node *node,
                                       Node *tobe_fused_node) const {
  auto op_role = boost::get<int>(
      node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
  PADDLE_ENFORCE_EQ(op_role, boost::get<int>(tobe_fused_node->Op()->GetAttr(
                                 OpProtoAndCheckerMaker::OpRoleAttrName())),
                    "Currently, only support fusing the same role operators");
  return op_role == static_cast<int>(OpRole::kBackward);
}

bool FuseAdjacentNodesPass::IsFusible(Node *cur_op_node,
                                      Node *upstream_op_node) const {
  PADDLE_ENFORCE(cur_op_node->IsOp(), "cur_op_node should be an operation.");
  PADDLE_ENFORCE(upstream_op_node->IsOp(), "n2 should be an operation.");

  static std::unordered_set<std::string> acts({"scale", "relu"});
  static std::unordered_set<std::string> act_grads({"scale_grad", "relu_grad"});
  static std::unordered_set<std::string> elemwises({"elementwise_add"});
  static std::unordered_set<std::string> elemwise_grads(
      {"elementwise_add_grad"});

  bool case1 = (acts.count(cur_op_node->Op()->Type()) == 1) &&
               (elemwises.count(upstream_op_node->Op()->Type()) == 1);
  bool case2 = (elemwises.count(cur_op_node->Op()->Type()) == 1) &&
               (acts.count(upstream_op_node->Op()->Type()) == 1);
  bool case3 = (elemwise_grads.count(cur_op_node->Op()->Type()) == 1) &&
               (act_grads.count(upstream_op_node->Op()->Type()) == 1);
  //  bool case4 =  (act_grads.count(cur_op_node->Op()->Type()) == 1) &&
  //        (elemwise_grads.count(upstream_op_node->Op()->Type()) == 1)

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

bool FuseAdjacentNodesPass::IsElemwiseAndActivation(
    Node *node, Node *tobe_fused_node) const {
  auto outside_op_name = node->Op()->Type();
  auto inside_op_name = tobe_fused_node->Op()->Type();

  return (IsActivation(outside_op_name) && IsElemwise(inside_op_name)) ||
         (IsActivation(inside_op_name) && IsElemwise(outside_op_name));
}

void FuseAdjacentNodesPass::FuseElemwiseAndActivation(
    Node *cur_op_node, Node *upstream_op_node, OpDesc *op_desc,
    std::unordered_set<Node *> *intermediate_out) const {
  auto upstream_op_type = upstream_op_node->Op()->Type();
  auto cur_op_type = cur_op_node->Op()->Type();
  auto upstream_op_node_in_args = upstream_op_node->Op()->InputArgumentNames();
  auto upstream_op_node_out_args =
      upstream_op_node->Op()->OutputArgumentNames();
  auto cur_op_node_in_args = cur_op_node->Op()->InputArgumentNames();
  auto cur_op_node_out_args = cur_op_node->Op()->OutputArgumentNames();

  if (IsBackward(cur_op_node, upstream_op_node)) {
    auto check_unary_backward = [](const std::string &op_type,
                                   const std::vector<std::string> &in_args,
                                   const std::vector<std::string> &out_args) {
      PADDLE_ENFORCE(
          in_args.size() == 2 || in_args.size() == 3,
          "The number of inputs of %s should be 2 or 3, "
          "if the number is 2, the input is 'Out', 'Out@Grad', "
          "if the number is 3, the input is 'X', 'Out' and 'Out@Grad'.",
          op_type);
      PADDLE_ENFORCE(out_args.size() == 1,
                     "The number of output of %s should be 1.", op_type);
    };
    auto check_binary_backward = [](const std::string &op_type,
                                    const std::vector<std::string> &in_args,
                                    const std::vector<std::string> &out_args) {
      PADDLE_ENFORCE(
          in_args.size() == 2 || in_args.size() == 4,
          "The number of inputs of %s should be 2 or 4, if the number is 2, "
          "the input variable is `Y`, and `Out@Grad`, if the number is "
          "4, the input variable is `X`, `Y`, `Out`, `Out@Grad`",
          op_type);
      PADDLE_ENFORCE(out_args.size() == 2,
                     "The number of output of %s should be 2.", op_type);
    };

    auto out_grad = GradVarName("Out");
    auto x_grad = GradVarName("X");
    auto y_grad = GradVarName("Y");
    auto inter_grad = GradVarName("IntermediateOut");

    const std::string op_type = "fused_elemwise_activation_grad";
    op_desc->SetType(op_type);

    // Set attrs
    op_desc->SetAttr("functor_list",
                     std::vector<std::string>({upstream_op_type, cur_op_type}));
    op_desc->SetAttr("recomputation", false);

    if (IsElemwise(cur_op_type)) {
      // the backward of  Unary(Binary(X, Y))
      check_unary_backward(upstream_op_type, upstream_op_node_in_args,
                           upstream_op_node_out_args);
      check_binary_backward(cur_op_type, cur_op_node_in_args,
                            cur_op_node_out_args);
      bool keep_intermediate = false;

      op_desc->SetInput("X", {});
      if (cur_op_node_in_args.size() == 4) {
        op_desc->SetInput("X", cur_op_node->Op()->Input("X"));
      }

      op_desc->SetInput("IntermediateOut", {});
      if (upstream_op_node_in_args.size() == 3) {
        op_desc->SetInput("IntermediateOut",
                          upstream_op_node->Op()->Input("X"));
        keep_intermediate = true;
      }

      op_desc->SetAttr("save_intermediate_out", keep_intermediate);
      op_desc->SetInput("Y", cur_op_node->Op()->Input("Y"));
      op_desc->SetInput("Out", upstream_op_node->Op()->Input("Out"));
      op_desc->SetInput(out_grad, upstream_op_node->Op()->Input(out_grad));
      op_desc->SetOutput(x_grad, cur_op_node->Op()->Output(x_grad));
      op_desc->SetOutput(y_grad, cur_op_node->Op()->Output(y_grad));
      op_desc->SetOutput(inter_grad, upstream_op_node->Op()->Output(x_grad));
      // Get the intermediate_out node.
      // Node(zcd): the name of intermediate_out and out maybe the same.
      for (auto &in : upstream_op_node->inputs) {
        if (in->Name() == upstream_op_node->Op()->Input("Out")[0]) {
          auto upstrem_forward_node = in->inputs[0];
          auto intermediate_out_name =
              upstrem_forward_node->Op()->Output("IntermediateOut")[0];
          for (auto &out : upstrem_forward_node->outputs) {
            if (out->Name() == intermediate_out_name) {
              intermediate_out->insert(out);
            }
          }
          break;
        }
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
    auto check_unary_in_args = [](const std::vector<std::string> &in_args) {
      PADDLE_ENFORCE_EQ(in_args.size(), 1,
                        "The number of input of UnaryFunctor should be one.");
    };
    auto check_binary_in_args = [](const std::vector<std::string> &in_args) {
      PADDLE_ENFORCE_EQ(in_args.size(), 2,
                        "The number of input of BinaryFunctor should be two.");
    };

    op_desc->SetType("fused_elemwise_activation");
    op_desc->SetAttr("functor_list",
                     std::vector<std::string>({cur_op_type, upstream_op_type}));
    op_desc->SetAttr("recomputation", false);

    if (IsElemwise(cur_op_type)) {
      // Z = Binary(X, Unary(Y))
      check_binary_in_args(cur_op_node_in_args);
      check_unary_in_args(upstream_op_node_in_args);
      // NOTE(zcd): If the mem_opt is opened and the Unary is inplace, the
      // name of Y and intermediate_out maybe the same.
      // In this situation, save_intermediate_out should be true,
      // and the backward should use intermediate_out directly but not Y.

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

    } else {
      // Z = Unary(Binary(X, Y))
      check_binary_in_args(upstream_op_node_in_args);
      check_unary_in_args(cur_op_node_in_args);
      // NOTE(zcd): If mem_opt is opened and the Unary is inplace, the name of
      // out and intermediate_out maybe the same.
      // In this situation, the intermediate_out should not be used.

      op_desc->SetInput("Y", upstream_op_node->Op()->Input("Y"));
      op_desc->SetInput("X", upstream_op_node->Op()->Input("X"));
    }

    // Another pass should check whether it is necessary to save intermediate
    // out.
    op_desc->SetAttr("save_intermediate_out", true);
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

std::vector<Node *> FuseAdjacentNodesPass::ReplaceNode(
    Node *cur_node, Node *new_node, const std::vector<Node *> &nodes) const {
  std::vector<Node *> new_list(nodes.size());
  bool has_replaced = false;
  std::transform(nodes.begin(), nodes.end(), new_list.begin(),
                 [&](Node *node) -> Node * {
                   if (node == cur_node) {
                     has_replaced = true;
                     return new_node;
                   }
                   return node;
                 });
  PADDLE_ENFORCE(has_replaced, "Not find %s in the node list.",
                 cur_node->Name());
  return new_list;
}

std::vector<Node *> FuseAdjacentNodesPass::RemoveNode(
    Node *trg_node, const std::vector<Node *> &nodes) const {
  std::vector<Node *> new_list(nodes.size());
  auto end_iter =
      std::copy_if(nodes.begin(), nodes.end(), new_list.begin(),
                   [&](Node *node) -> bool { return node != trg_node; });
  new_list.resize(
      static_cast<uint64_t>(std::distance(new_list.begin(), end_iter)));
  return new_list;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_adjacent_nodes_pass,
              paddle::framework::ir::FuseAdjacentNodesPass);
