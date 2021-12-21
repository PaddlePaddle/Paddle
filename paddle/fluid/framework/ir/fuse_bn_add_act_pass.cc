// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fuse_bn_add_act_pass.h"
#include <string>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void FuseBatchNormAddActPass::ApplyImpl(ir::Graph *graph) const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#if defined(PADDLE_WITH_HIP) || CUDNN_VERSION_MIN(7, 4, 1)
  // forward
  std::unordered_set<std::string> act_types = {"relu"};
  graph = FuseBatchNormAddAct(graph, act_types);
  // backward
  std::unordered_set<std::string> act_grad_types = {"relu_grad"};
  graph = FuseBatchNormAddActGrad(graph, act_grad_types);
#endif
#endif
}

// act(bn(x) + z)
ir::Graph *FuseBatchNormAddActPass::FuseBatchNormAddAct(
    ir::Graph *graph, const std::unordered_set<std::string> &act_types) const {
  PADDLE_ENFORCE_NE(
      graph, nullptr,
      platform::errors::InvalidArgument(
          "The input graph of FuseBatchNormAddAct should not be nullptr."));
  FusePassBase::Init("bn_add_act", graph);

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode("bn_add_act/x")
                ->AsInput()
                ->assert_is_op_input("batch_norm", "X")
                ->assert_var_dtype(proto::VarType::FP16);
  patterns::BatchNormAddAct bn_add_act_pattern(gpd.mutable_pattern(),
                                               "bn_add_act");

  bn_add_act_pattern(x, act_types);

  int found_bn_add_act_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle FuseBatchNormAddAct fuse";
    // BN inputs
    GET_IR_NODE_FROM_SUBGRAPH(bn_scale, bn_scale, bn_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_bias, bn_bias, bn_add_act_pattern);
    // BN outputs
    GET_IR_NODE_FROM_SUBGRAPH(bn_mean_out, bn_mean_out, bn_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_variance_out, bn_variance_out,
                              bn_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_saved_variance, bn_saved_variance,
                              bn_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_saved_mean, bn_saved_mean, bn_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_reserve_space, bn_reserve_space,
                              bn_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_out, bn_out, bn_add_act_pattern);
    // Add outputs
    GET_IR_NODE_FROM_SUBGRAPH(elewise_add_in, elewise_add_in,
                              bn_add_act_pattern);
    // Add outputs
    GET_IR_NODE_FROM_SUBGRAPH(elewise_add_out, elewise_add_out,
                              bn_add_act_pattern);
    // ACT output
    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, bn_add_act_pattern);
    // ops
    GET_IR_NODE_FROM_SUBGRAPH(batch_norm, batch_norm, bn_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elewise_add, elewise_add, bn_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act, act, bn_add_act_pattern);

    std::string bn_x_n = subgraph.at(x)->Name();
    std::string elewise_add_in_n = elewise_add_in->Name();
    std::string bn_scale_n = bn_scale->Name();
    std::string bn_bias_n = bn_bias->Name();
    std::string bn_mean_out_n = bn_mean_out->Name();
    std::string bn_variance_out_n = bn_variance_out->Name();
    std::string bn_saved_variance_n = bn_saved_variance->Name();
    std::string bn_saved_mean_n = bn_saved_mean->Name();
    std::string bn_reserve_space_n = bn_reserve_space->Name();
    std::string bn_out_n = bn_out->Name();
    std::string elewise_add_out_n = elewise_add_out->Name();
    std::string act_out_n = act_out->Name();

    Node *fused_bn_add_act_node = CreateFusedBatchNormAddActNode(
        g, act, elewise_add, batch_norm, bn_x_n, elewise_add_in_n, bn_scale_n,
        bn_bias_n, bn_mean_out_n, bn_variance_out_n, bn_saved_variance_n,
        bn_saved_mean_n, bn_reserve_space_n, act_out_n);

    VLOG(4) << "\n\t " << bn_x_n << ", " << bn_scale_n << ", " << bn_bias_n
            << " -> " << batch_norm->Name() << " -> " << bn_mean_out_n << ", "
            << bn_variance_out_n << ", " << bn_saved_variance_n << ", "
            << bn_saved_mean_n << ", " << bn_reserve_space_n << " and "
            << bn_out_n << "\n"
            << "\t " << bn_out_n << " and " << elewise_add_in_n << " -> "
            << elewise_add->Name() << " -> " << elewise_add_out_n << "\n"
            << "\t " << elewise_add_out_n << " -> " << act->Name() << " -> "
            << act_out_n;

    ReLinkNodes(g, batch_norm, elewise_add, act, fused_bn_add_act_node);
    found_bn_add_act_count++;
  };

  gpd(graph, handler);

  AddStatis(found_bn_add_act_count);
  return graph;
}

Node *FuseBatchNormAddActPass::CreateFusedBatchNormAddActNode(
    Graph *g, const Node *act, const Node *elewise_add, const Node *bn,
    const std::string &bn_x_n, const std::string &elewise_add_in_n,
    const std::string &bn_scale_n, const std::string &bn_bias_n,
    const std::string &bn_mean_out_n, const std::string &bn_variance_out_n,
    const std::string &bn_saved_variance_n, const std::string &bn_saved_mean_n,
    const std::string &bn_reserve_space_n, const std::string &act_out_n) const {
  OpDesc desc;
  desc.SetInput("X", std::vector<std::string>({bn_x_n}));
  desc.SetInput("Z", std::vector<std::string>({elewise_add_in_n}));
  desc.SetInput("Scale", std::vector<std::string>({bn_scale_n}));
  desc.SetInput("Bias", std::vector<std::string>({bn_bias_n}));

  desc.SetOutput("Y", std::vector<std::string>({act_out_n}));
  desc.SetOutput("MeanOut", std::vector<std::string>({bn_mean_out_n}));
  desc.SetOutput("VarianceOut", std::vector<std::string>({bn_variance_out_n}));
  desc.SetOutput("SavedMean", std::vector<std::string>({bn_saved_mean_n}));
  desc.SetOutput("SavedVariance",
                 std::vector<std::string>({bn_saved_variance_n}));
  desc.SetOutput("ReserveSpace",
                 std::vector<std::string>({bn_reserve_space_n}));
  desc.SetType("fused_bn_add_activation");

  desc.SetAttr("act_type", act->Name());
  // Set attrs
  for (auto &n : {act->Op(), elewise_add->Op(), bn->Op()}) {
    for (auto &m : n->GetAttrMap()) {
      desc.SetAttr(m.first, m.second);
    }
  }

  auto fused_bn_add_act_node = g->CreateOpNode(&desc);
  return fused_bn_add_act_node;
}

// the backward of act(bn(x) + z)
ir::Graph *FuseBatchNormAddActPass::FuseBatchNormAddActGrad(
    ir::Graph *graph,
    const std::unordered_set<std::string> &act_grad_types) const {
  PADDLE_ENFORCE_NE(
      graph, nullptr,
      platform::errors::InvalidArgument(
          "The input graph of FuseBatchNormAddActGrad should not be nullptr."));
  FusePassBase::Init("bn_add_act_grad", graph);

  GraphPatternDetector gpd;
  auto *d_act_out =
      gpd.mutable_pattern()
          ->NewNode("bn_add_act_grad/x")
          ->AsInput()
          ->assert_is_ops_input(act_grad_types, GradVarName("Out"))
          ->assert_var_dtype(proto::VarType::FP16);
  patterns::BatchNormAddActGrad bn_add_act_grad_pattern(gpd.mutable_pattern(),
                                                        "bn_add_act_grad");
  bn_add_act_grad_pattern(d_act_out, act_grad_types);

  int found_bn_add_act_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle FuseBatchNormAddActGrad fuse";
    GET_IR_NODE_FROM_SUBGRAPH(act_grad, act_grad, bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elewise_add_grad, elewise_add_grad,
                              bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(batch_norm_grad, batch_norm_grad,
                              bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(d_act_x, d_act_x, bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(d_bn_out, d_bn_out, bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_x, bn_x, bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_scale, bn_scale, bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_bias, bn_bias, bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_saved_mean, bn_saved_mean,
                              bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_saved_variance, bn_saved_variance,
                              bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_reserve_space, bn_reserve_space,
                              bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(d_bn_x, d_bn_x, bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(d_bn_scale, d_bn_scale, bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(d_bn_bias, d_bn_bias, bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(d_elewise_add_in, d_elewise_add_in,
                              bn_add_act_grad_pattern);

    std::string d_act_out_n = subgraph.at(d_act_out)->Name();  // Y@GRAD
    std::string act_out_n = act_out->Name();                   // Y
    std::string d_act_x_n = d_act_x->Name();
    std::string bn_x_n = bn_x->Name();
    std::string bn_scale_n = bn_scale->Name();
    std::string bn_bias_n = bn_bias->Name();
    std::string bn_saved_mean_n = bn_saved_mean->Name();
    std::string bn_saved_variance_n = bn_saved_variance->Name();
    std::string bn_reserve_space_n = bn_reserve_space->Name();
    std::string d_bn_out_n = d_bn_out->Name();
    std::string d_bn_x_n = d_bn_x->Name();
    std::string d_bn_scale_n = d_bn_scale->Name();
    std::string d_bn_bias_n = d_bn_bias->Name();
    std::string d_elewise_add_in_n = d_elewise_add_in->Name();

    OpDesc desc;
    desc.SetType("fused_bn_add_activation_grad");
    desc.SetInput("X", {bn_x_n});
    desc.SetInput("Y", std::vector<std::string>({act_out_n}));
    desc.SetInput(GradVarName("Y"), std::vector<std::string>({d_act_out_n}));
    desc.SetInput("Scale", std::vector<std::string>({bn_scale_n}));
    desc.SetInput("Bias", std::vector<std::string>({bn_bias_n}));
    desc.SetInput("SavedMean", std::vector<std::string>({bn_saved_mean_n}));
    desc.SetInput("SavedVariance",
                  std::vector<std::string>({bn_saved_variance_n}));
    desc.SetInput("ReserveSpace",
                  std::vector<std::string>({bn_reserve_space_n}));
    desc.SetOutput(GradVarName("X"), std::vector<std::string>({d_bn_x_n}));
    desc.SetOutput(GradVarName("Z"),
                   std::vector<std::string>({d_elewise_add_in_n}));
    desc.SetOutput(GradVarName("Scale"),
                   std::vector<std::string>({d_bn_scale_n}));
    desc.SetOutput(GradVarName("Bias"),
                   std::vector<std::string>({d_bn_bias_n}));
    std::string act = act_grad->Name();
    act = act.substr(0, act.length() - 5);  // remove "_grad"
    desc.SetAttr("act_type", act);

    for (auto &n :
         {act_grad->Op(), elewise_add_grad->Op(), batch_norm_grad->Op()}) {
      for (auto &m : n->GetAttrMap()) {
        desc.SetAttr(m.first, m.second);
      }
    }

    auto fused_node = g->CreateOpNode(&desc);

    VLOG(4) << "\n\t " << d_act_out_n << " and " << act_out_n << " -> "
            << act_grad->Name() << " -> " << d_act_x_n << "\n\t ";
    VLOG(4) << d_act_x_n << " -> " << elewise_add_grad->Name() << " -> "
            << d_elewise_add_in_n << "," << d_bn_out_n << "\n\t ";
    VLOG(4) << bn_x_n << ", " << d_bn_out_n << ", " << bn_scale_n << ", "
            << bn_bias_n << ", " << bn_saved_mean_n << ", "
            << bn_saved_variance_n << " and " << bn_reserve_space_n << " -> "
            << batch_norm_grad->Name() << " -> " << d_bn_x_n << ", "
            << d_bn_scale_n << " and " << d_bn_bias_n;

    ReLinkNodes(g, act_grad, elewise_add_grad, batch_norm_grad, fused_node);
    found_bn_add_act_count++;
  };

  gpd(graph, handler);

  AddStatis(found_bn_add_act_count);
  return graph;
}

void FuseBatchNormAddActPass::ReLinkNodes(Graph *graph, Node *op_1, Node *op_2,
                                          Node *op_3,
                                          Node *fused_op) const {  // delete act
  // link inputs of op_1 to fused_op
  for (auto &in : op_1->inputs) {
    fused_op->inputs.emplace_back(in);
    in->outputs = this->ReplaceNode(op_1, fused_op, in->outputs);
  }

  std::unordered_set<const Node *> nodes2delete;

  LinkOutputsToFuseOp(op_1, op_2, fused_op, &nodes2delete);
  LinkOutputsToFuseOp(op_2, op_3, fused_op, &nodes2delete);
  LinkInputsToFuseOp(op_2, fused_op, &nodes2delete);
  LinkInputsToFuseOp(op_3, fused_op, &nodes2delete);

  for (auto &out : op_3->outputs) {
    IR_OP_VAR_LINK(fused_op, out);
  }

  nodes2delete.insert(std::move(op_1));
  nodes2delete.insert(std::move(op_2));
  nodes2delete.insert(std::move(op_3));

  GraphSafeRemoveNodes(graph, nodes2delete);
}

void FuseBatchNormAddActPass::LinkOutputsToFuseOp(
    Node *op_1, Node *op_2, Node *fused_op,
    std::unordered_set<const Node *> *nodes2delete) const {
  // if the outputs of op_1 are inputs of op_2, add the outputs to nodes2delete
  // otherwise link the outputs to fused_op
  for (auto &out : op_1->outputs) {
    auto result_iter =
        std::find_if(op_2->inputs.begin(), op_2->inputs.end(),
                     [&out](const Node *node) -> bool { return node == out; });

    if (result_iter == op_2->inputs.end()) {
      IR_OP_VAR_LINK(fused_op, out);
    } else {
      nodes2delete->emplace(out);
    }
  }
}

void FuseBatchNormAddActPass::LinkInputsToFuseOp(
    Node *op, Node *fused_op,
    std::unordered_set<const Node *> *nodes2delete) const {
  // if the inputs of the op are outputs of previous op, which means
  // these inputs have been added to nodes2delete before, skip the inputs,
  // otherwise link the inputs of the op to fused_op
  for (auto &in : op->inputs) {
    if (nodes2delete->count(in)) {
      continue;
    }
    fused_op->inputs.emplace_back(in);
    in->outputs = this->ReplaceNode(op, fused_op, in->outputs);
  }
}

std::vector<Node *> FuseBatchNormAddActPass::ReplaceNode(
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
  PADDLE_ENFORCE_EQ(has_replaced, true,
                    platform::errors::NotFound("Not found %s in the node list.",
                                               cur_node->Name()));
  return new_list;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_bn_add_act_pass,
              paddle::framework::ir::FuseBatchNormAddActPass);
