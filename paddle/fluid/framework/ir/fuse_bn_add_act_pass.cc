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
#include <algorithm>
#include <string>
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

namespace paddle {
namespace framework {
namespace ir {

void FuseBatchNormAddActPass::ApplyImpl(ir::Graph *graph) const {
#ifdef PADDLE_WITH_CUDA
#if CUDNN_VERSION_MIN(7, 4, 1)
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
    GET_IR_NODE_FROM_SUBGRAPH(bn_variance, bn_variance, bn_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_mean, bn_mean, bn_add_act_pattern);
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
    std::string bn_variance_n = bn_variance->Name();
    std::string bn_mean_n = bn_mean->Name();
    std::string bn_mean_out_n = bn_mean_out->Name();
    std::string bn_variance_out_n = bn_variance_out->Name();
    std::string bn_saved_variance_n = bn_saved_variance->Name();
    std::string bn_saved_mean_n = bn_saved_mean->Name();
    std::string bn_reserve_space_n = bn_reserve_space->Name();
    std::string bn_out_n = bn_out->Name();
    std::string act_out_n = act_out->Name();

    Node *fused_bn_add_act_node = CreateFusedBatchNormAddActNode(
        g, act, elewise_add, batch_norm, bn_x_n, elewise_add_in_n, bn_scale_n,
        bn_bias_n, bn_variance_n, bn_mean_n, bn_mean_out_n, bn_variance_out_n,
        bn_saved_variance_n, bn_saved_mean_n, bn_reserve_space_n, act_out_n);

    ReLinkNodes(g, bn_out, elewise_add_out, batch_norm, elewise_add, act,
                fused_bn_add_act_node);
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
    const std::string &bn_variance_n, const std::string &bn_mean_n,
    const std::string &bn_mean_out_n, const std::string &bn_variance_out_n,
    const std::string &bn_saved_variance_n, const std::string &bn_saved_mean_n,
    const std::string &bn_reserve_space_n, const std::string &act_out_n) const {
  OpDesc desc;
  desc.SetInput("X", std::vector<std::string>({bn_x_n}));
  desc.SetInput("Z", std::vector<std::string>({elewise_add_in_n}));
  desc.SetInput("Scale", std::vector<std::string>({bn_scale_n}));
  desc.SetInput("Bias", std::vector<std::string>({bn_bias_n}));
  desc.SetInput("Mean", std::vector<std::string>({bn_mean_n}));
  desc.SetInput("Variance", std::vector<std::string>({bn_variance_n}));

  desc.SetOutput("Y", std::vector<std::string>({act_out_n}));
  desc.SetOutput("MeanOut", std::vector<std::string>({bn_mean_out_n}));
  desc.SetOutput("VarianceOut", std::vector<std::string>({bn_variance_out_n}));
  desc.SetOutput("SavedMean", std::vector<std::string>({bn_saved_mean_n}));
  desc.SetOutput("SavedVariance",
                 std::vector<std::string>({bn_saved_variance_n}));
  desc.SetOutput("ReserveSpace",
                 std::vector<std::string>({bn_reserve_space_n}));
  desc.SetType("fused_batch_norm_add_act");

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

// the backward of act(bn(x))
// act_grad: in["Out", "Out@GRAD"], out["X@GRAD"]
// bn_grad: in["X", "Y@GRAD", "Scale", "Bias", "SavedMean", "SavedVariance",
// "ReserveSpace"],
// out["X@GRAD", "Scale@GRAD", "Bias@GRAD"]
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
          ->assert_is_ops_input(act_grad_types, GradVarName("Out"));
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
    GET_IR_NODE_FROM_SUBGRAPH(d_itermediate_out1, d_itermediate_out1,
                              bn_add_act_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(d_itermediate_out2, d_itermediate_out2,
                              bn_add_act_grad_pattern);
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
    std::string d_itermediate_out1_n = d_itermediate_out1->Name();
    std::string bn_x_n = bn_x->Name();
    std::string bn_scale_n = bn_scale->Name();
    std::string bn_bias_n = bn_bias->Name();
    std::string bn_saved_mean_n = bn_saved_mean->Name();
    std::string bn_saved_variance_n = bn_saved_variance->Name();
    std::string bn_reserve_space_n = bn_reserve_space->Name();
    std::string d_bn_x_n = d_bn_x->Name();
    std::string d_bn_scale_n = d_bn_scale->Name();
    std::string d_bn_bias_n = d_bn_bias->Name();
    std::string d_elewise_add_in_n = d_elewise_add_in->Name();

    OpDesc desc;
    desc.SetType("fused_batch_norm_add_act_grad");
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

    ReLinkNodes(g, d_itermediate_out1, d_itermediate_out2, act_grad,
                elewise_add_grad, batch_norm_grad, fused_node);
    found_bn_add_act_count++;
  };

  gpd(graph, handler);

  AddStatis(found_bn_add_act_count);
  return graph;
}

void FuseBatchNormAddActPass::ReLinkNodes(Graph *graph,
                                          const Node *intermediate_out1,
                                          const Node *intermediate_out2,
                                          Node *op_1, Node *op_2, Node *op_3,
                                          Node *fused_op) const {  // delete act
  VLOG(3) << "111111111111: op_1=" << op_1->Name();
  for (auto &in : op_1->inputs) {
    VLOG(3) << "in: " << in->Name();
    for (auto &out : in->outputs) {
      VLOG(3) << "out: " << out->Name();
    }
    fused_op->inputs.emplace_back(in);
    in->outputs = this->ReplaceNode(op_1, fused_op, in->outputs);
  }

  std::unordered_set<const Node *> nodes2delete;
  for (auto &out : op_1->outputs) {
    // intermediate_out or ctr_var
    auto result_iter =
        std::find_if(op_2->inputs.begin(), op_2->inputs.end(),
                     [&out](const Node *node) -> bool { return node == out; });

    if (result_iter == op_2->inputs.end()) {
      IR_OP_VAR_LINK(fused_op, out);
    } else {
      nodes2delete.emplace(out);
    }
  }

  // 删除add的输出节点
  for (auto &out : op_2->outputs) {
    nodes2delete.emplace(out);
  }

  VLOG(3) << "111111111111222";
  // 将add的额外输入设置到fuse的输入上
  for (auto &in : op_2->inputs) {
    // intermediate_out是否可以不用？
    if (in == intermediate_out1 || nodes2delete.count(in)) {
      continue;
    }
    fused_op->inputs.emplace_back(in);
    in->outputs = this->ReplaceNode(op_2, fused_op, in->outputs);
  }

  VLOG(3) << "111111111111333";
  // 将act的额外输入设置到fuse的输入上
  for (auto &in : op_3->inputs) {
    // intermediate_out是否可以不用？
    if (in == intermediate_out2 || nodes2delete.count(in)) {
      continue;
    }
    fused_op->inputs.emplace_back(in);
    in->outputs = this->ReplaceNode(op_3, fused_op, in->outputs);
  }

  for (auto &out : op_3->outputs) {
    IR_OP_VAR_LINK(fused_op, out);
  }

  nodes2delete.insert(std::move(op_1));
  nodes2delete.insert(std::move(op_2));
  nodes2delete.insert(std::move(op_3));

  GraphSafeRemoveNodes(graph, nodes2delete);
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
