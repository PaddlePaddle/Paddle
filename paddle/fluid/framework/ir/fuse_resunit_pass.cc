// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2023 NVIDIA Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fuse_resunit_pass.h"

#include <string>
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE_FROM_SUBGRAPH_COND(var, arg, pat, cond) \
  ir::Node *var = nullptr;                                  \
  if (cond) {                                               \
    GET_IR_NODE_FROM_SUBGRAPH(_##var, arg, pat);            \
    var = _##var;                                           \
  }

#define GET_CONV_BN_NODES_COND(idx, pattern_name, cond)                    \
  /* OPERATORS */                                                          \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                          \
      conv##idx##_op, conv##idx##_op, pattern_name, cond);                 \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                          \
      bn##idx##_op, bn##idx##_op, pattern_name, cond);                     \
  /* CONV inputs */                                                        \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                          \
      conv##idx##_w, conv##idx##_w, pattern_name, cond);                   \
  /* CONV outputs */                                                       \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                          \
      conv##idx##_out, conv##idx##_out, pattern_name, cond);               \
  /* BN inputs */                                                          \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                          \
      bn##idx##_scale, bn##idx##_scale, pattern_name, cond);               \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                          \
      bn##idx##_bias, bn##idx##_bias, pattern_name, cond);                 \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                          \
      bn##idx##_mean, bn##idx##_mean, pattern_name, cond);                 \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                          \
      bn##idx##_variance, bn##idx##_variance, pattern_name, cond);         \
  /* BN outputs */                                                         \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                          \
      bn##idx##_out, bn##idx##_out, pattern_name, cond); /* Out */         \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                          \
      bn##idx##_mean_out, bn##idx##_mean_out, pattern_name, cond);         \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                          \
      bn##idx##_variance_out, bn##idx##_variance_out, pattern_name, cond); \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                          \
      bn##idx##_saved_mean, bn##idx##_saved_mean, pattern_name, cond);     \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                          \
      bn##idx##_saved_variance, bn##idx##_saved_variance, pattern_name, cond)

#define GET_CONV_GRAD_NODES(pattern_name)                          \
  /* OPERATORS */                                                  \
  GET_IR_NODE_FROM_SUBGRAPH(conv_grad, conv_grad, pattern_name);   \
  /* dCONV inputs */                                               \
  GET_IR_NODE_FROM_SUBGRAPH(conv_w, conv_w, pattern_name);         \
  GET_IR_NODE_FROM_SUBGRAPH(d_conv_out, d_conv_out, pattern_name); \
  /* dCONV outputs */                                              \
  GET_IR_NODE_FROM_SUBGRAPH(d_conv_x, d_conv_x, pattern_name);     \
  GET_IR_NODE_FROM_SUBGRAPH(d_conv_w, d_conv_w, pattern_name)

#define GET_BN_GRAD_NODES_COND(idx, pattern_name, cond)                        \
  /* OPERATORS */                                                              \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                              \
      batch_norm##idx##_grad, batch_norm##idx##_grad, pattern_name, cond);     \
  /* BN inputs */                                                              \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                              \
      bn##idx##_x, bn##idx##_x, pattern_name, cond);                           \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                              \
      bn##idx##_scale, bn##idx##_scale, pattern_name, cond);                   \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                              \
      bn##idx##_bias, bn##idx##_bias, pattern_name, cond);                     \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                              \
      bn##idx##_saved_mean, bn##idx##_saved_mean, pattern_name, cond);         \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                              \
      bn##idx##_saved_variance, bn##idx##_saved_variance, pattern_name, cond); \
  /* BN outputs */                                                             \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                              \
      d_bn##idx##_x, d_bn##idx##_x, pattern_name, cond);                       \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                              \
      d_bn##idx##_scale, d_bn##idx##_scale, pattern_name, cond);               \
  GET_IR_NODE_FROM_SUBGRAPH_COND(                                              \
      d_bn##idx##_bias, d_bn##idx##_bias, pattern_name, cond)

#define PRINT_DEBUG_INFO(idx)                                                  \
  VLOG(4) << "\n\t " << x##idx->Name() << " and " << conv##idx##_w->Name()     \
          << " -> " << conv##idx##_op->Name() << " -> "                        \
          << conv##idx##_out->Name() << "\n";                                  \
  VLOG(4) << "\n\t " << conv##idx##_out->Name() << ", "                        \
          << bn##idx##_scale->Name() << ", " << bn##idx##_bias->Name() << ", " \
          << " -> " << bn##idx##_op->Name() << " -> "                          \
          << bn##idx##_mean_out->Name() << ", "                                \
          << bn##idx##_variance_out->Name() << ", "                            \
          << bn##idx##_saved_variance->Name() << ", "                          \
          << bn##idx##_saved_mean->Name() << ", " << bn##idx##_out->Name()     \
          << "\n";

#define IR_NODE_LINK_TO_DEBUG(x, y)                       \
  VLOG(4) << "LINK " << x->Name() << " -> " << y->Name(); \
  IR_NODE_LINK_TO(x, y)

ir::Node *CreateVarNode(Graph *g,
                        const std::string scope,
                        const std::string name) {
  VarDesc var_desc(patterns::PDNodeName(scope, name));
  return g->CreateVarNode(&var_desc);
}

ir::Node *CreateConvBNstatsOpNode(Graph *g,
                                  BlockDesc *block,
                                  bool fuse_prologue,
                                  ir::Node *conv_op,
                                  ir::Node *bn_op,
                                  ir::Node *input,
                                  ir::Node *filter,
                                  ir::Node *input_scale,
                                  ir::Node *input_bias,
                                  ir::Node *bn_scale,
                                  ir::Node *bn_bias,
                                  ir::Node *bn_mean,
                                  ir::Node *bn_variance,
                                  ir::Node *conv_out,
                                  ir::Node *bn_mean_out,
                                  ir::Node *bn_variance_out,
                                  ir::Node *bn_saved_mean,
                                  ir::Node *bn_saved_variance,
                                  ir::Node *eq_scale,
                                  ir::Node *eq_bias) {
  OpDesc op_desc(block);
  op_desc.SetType("fused_scale_bias_relu_conv_bn");
  op_desc.SetInput("x", {input->Name()});
  op_desc.SetInput("w", {filter->Name()});
  if (fuse_prologue) {
    op_desc.SetInput("scale", {input_scale->Name()});
    op_desc.SetInput("bias", {input_bias->Name()});
  }
  op_desc.SetInput("bn_scale", {bn_scale->Name()});
  op_desc.SetInput("bn_bias", {bn_bias->Name()});
  op_desc.SetInput("input_running_mean", {bn_mean->Name()});
  op_desc.SetInput("input_running_var", {bn_variance->Name()});
  op_desc.SetOutput("out", {conv_out->Name()});
  op_desc.SetOutput("out_running_mean", {bn_mean_out->Name()});
  op_desc.SetOutput("out_running_var", {bn_variance_out->Name()});
  op_desc.SetOutput("saved_mean", {bn_saved_mean->Name()});
  op_desc.SetOutput("saved_var", {bn_saved_variance->Name()});
  op_desc.SetOutput("eq_scale", {eq_scale->Name()});
  op_desc.SetOutput("eq_bias", {eq_bias->Name()});

  op_desc.SetAttr(
      "paddings",
      PADDLE_GET_CONST(std::vector<int>, conv_op->Op()->GetAttr("paddings")));
  op_desc.SetAttr(
      "dilations",
      PADDLE_GET_CONST(std::vector<int>, conv_op->Op()->GetAttr("dilations")));
  op_desc.SetAttr(
      "strides",
      PADDLE_GET_CONST(std::vector<int>, conv_op->Op()->GetAttr("strides")));
  op_desc.SetAttr(
      "padding_algorithm",
      PADDLE_GET_CONST(std::string,
                       conv_op->Op()->GetAttr("padding_algorithm")));
  op_desc.SetAttr("groups",
                  PADDLE_GET_CONST(int, conv_op->Op()->GetAttr("groups")));
  op_desc.SetAttr(
      "data_format",
      PADDLE_GET_CONST(std::string, conv_op->Op()->GetAttr("data_format")));
  op_desc.SetAttr("momentum",
                  PADDLE_GET_CONST(float, bn_op->Op()->GetAttr("momentum")));
  op_desc.SetAttr("epsilon",
                  PADDLE_GET_CONST(float, bn_op->Op()->GetAttr("epsilon")));
  op_desc.SetAttr("fuse_prologue", fuse_prologue);
  op_desc.SetAttr(
      "exhaustive_search",
      PADDLE_GET_CONST(bool, conv_op->Op()->GetAttr("exhaustive_search")));
  // TODO(tizheng): need to change this for sync_bn
  op_desc.SetAttr("accumulation_count", 0);
  op_desc.SetAttr("op_role", conv_op->Op()->GetAttr("op_role"));
  auto op_node = g->CreateOpNode(&op_desc);
  IR_NODE_LINK_TO(input, op_node);
  IR_NODE_LINK_TO(filter, op_node);
  if (fuse_prologue) {
    IR_NODE_LINK_TO(input_scale, op_node);
    IR_NODE_LINK_TO(input_bias, op_node);
  }
  IR_NODE_LINK_TO(bn_scale, op_node);
  IR_NODE_LINK_TO(bn_bias, op_node);
  IR_NODE_LINK_TO(bn_mean, op_node);
  IR_NODE_LINK_TO(bn_variance, op_node);

  IR_NODE_LINK_TO(op_node, conv_out);
  IR_NODE_LINK_TO(op_node, bn_mean_out);
  IR_NODE_LINK_TO(op_node, bn_variance_out);
  IR_NODE_LINK_TO(op_node, bn_saved_mean);
  IR_NODE_LINK_TO(op_node, bn_saved_variance);
  IR_NODE_LINK_TO(op_node, eq_scale);
  IR_NODE_LINK_TO(op_node, eq_bias);
  return op_node;
}

ir::Node *CreateSBAROpNode(Graph *g,
                           BlockDesc *block,
                           bool fuse_dual,
                           bool exhaustive_search,
                           ir::Node *act_op,
                           ir::Node *x1,
                           ir::Node *scale1,
                           ir::Node *bias1,
                           ir::Node *x2,
                           ir::Node *scale2,
                           ir::Node *bias2,
                           ir::Node *y_out) {
  OpDesc op_desc(block);
  op_desc.SetType("fused_scale_bias_add_relu");
  op_desc.SetInput("x1", {x1->Name()});
  op_desc.SetInput("scale1", {scale1->Name()});
  op_desc.SetInput("bias1", {bias1->Name()});
  op_desc.SetInput("x2", {x2->Name()});
  if (fuse_dual) {
    op_desc.SetInput("scale2", {scale2->Name()});
    op_desc.SetInput("bias2", {bias2->Name()});
  }
  op_desc.SetOutput("out", {y_out->Name()});
  op_desc.SetAttr("fuse_dual", fuse_dual);
  op_desc.SetAttr("exhaustive_search", exhaustive_search);
  op_desc.SetAttr("op_role", act_op->Op()->GetAttr("op_role"));
  auto op_node = g->CreateOpNode(&op_desc);

  IR_NODE_LINK_TO(x1, op_node);
  IR_NODE_LINK_TO(scale1, op_node);
  IR_NODE_LINK_TO(bias1, op_node);
  IR_NODE_LINK_TO(x2, op_node);
  if (fuse_dual) {
    IR_NODE_LINK_TO(scale2, op_node);
    IR_NODE_LINK_TO(bias2, op_node);
  }
  IR_NODE_LINK_TO(op_node, y_out);
  return op_node;
}

ir::Node *CreateDconvDreluDBNOpNode(Graph *g,
                                    BlockDesc *block,
                                    bool fuse_add,
                                    bool fuse_dual,
                                    bool fuse_shortcut,
                                    ir::Node *conv_grad,
                                    ir::Node *bn_grad,
                                    ir::Node *bn2_grad,
                                    ir::Node *d_conv_out,
                                    ir::Node *conv_w,
                                    ir::Node *d_conv_w,
                                    ir::Node *bn_saved_mean,
                                    ir::Node *bn_saved_variance,
                                    ir::Node *bn_scale,
                                    ir::Node *bn_bias,
                                    ir::Node *bn_x,
                                    ir::Node *d_bn_x,
                                    ir::Node *d_bn_scale,
                                    ir::Node *d_bn_bias,
                                    ir::Node *d_relu_out_extra,
                                    ir::Node *elewise_add_input_extra,
                                    ir::Node *d_elewise_add_input_extra,
                                    ir::Node *bn2_saved_mean,
                                    ir::Node *bn2_saved_variance,
                                    ir::Node *bn2_scale,
                                    ir::Node *bn2_bias,
                                    ir::Node *bn2_x,
                                    ir::Node *d_bn2_x,
                                    ir::Node *d_bn2_scale,
                                    ir::Node *d_bn2_bias,
                                    ir::Node *conv_x,
                                    ir::Node *bn_eqscale,
                                    ir::Node *bn_eqbias) {
  OpDesc op_desc(block);
  op_desc.SetType("fused_dconv_drelu_dbn");
  op_desc.SetInput("grad_output", {d_conv_out->Name()});
  op_desc.SetInput("weight", {conv_w->Name()});
  op_desc.SetInput("bn1_mean", {bn_saved_mean->Name()});
  op_desc.SetInput("bn1_inv_std", {bn_saved_variance->Name()});
  op_desc.SetInput("bn1_gamma", {bn_scale->Name()});
  op_desc.SetInput("bn1_beta", {bn_bias->Name()});
  op_desc.SetInput("bn1_input", {bn_x->Name()});
  op_desc.SetOutput("grad_bn1_input", {d_bn_x->Name()});
  op_desc.SetOutput("grad_bn1_gamma", {d_bn_scale->Name()});
  op_desc.SetOutput("grad_bn1_beta", {d_bn_bias->Name()});
  op_desc.SetOutput("grad_weight", {d_conv_w->Name()});
  if (fuse_add) {
    op_desc.SetInput("grad_output_add", {d_relu_out_extra->Name()});
  }
  if (fuse_shortcut) {
    op_desc.SetInput("residual_input", {elewise_add_input_extra->Name()});
    op_desc.SetOutput("grad_bn2_input", {d_elewise_add_input_extra->Name()});
  }
  if (fuse_dual) {
    op_desc.SetInput("bn2_mean", {bn2_saved_mean->Name()});
    op_desc.SetInput("bn2_inv_std", {bn2_saved_variance->Name()});
    op_desc.SetInput("bn2_gamma", {bn2_scale->Name()});
    op_desc.SetInput("bn2_beta", {bn2_bias->Name()});
    op_desc.SetInput("bn2_input", {bn2_x->Name()});
    op_desc.SetOutput("grad_bn2_input", {d_bn2_x->Name()});
    op_desc.SetOutput("grad_bn2_gamma", {d_bn2_scale->Name()});
    op_desc.SetOutput("grad_bn2_beta", {d_bn2_bias->Name()});
  }
  if (fuse_dual || fuse_shortcut) {
    op_desc.SetInput("conv_input", {conv_x->Name()});
  } else {
    op_desc.SetInput("bn1_eqscale", {bn_eqscale->Name()});
    op_desc.SetInput("bn1_eqbias", {bn_eqbias->Name()});
  }

  op_desc.SetAttr(
      "paddings",
      PADDLE_GET_CONST(std::vector<int>, conv_grad->Op()->GetAttr("paddings")));
  op_desc.SetAttr("dilations",
                  PADDLE_GET_CONST(std::vector<int>,
                                   conv_grad->Op()->GetAttr("dilations")));
  op_desc.SetAttr(
      "strides",
      PADDLE_GET_CONST(std::vector<int>, conv_grad->Op()->GetAttr("strides")));
  op_desc.SetAttr(
      "padding_algorithm",
      PADDLE_GET_CONST(std::string,
                       conv_grad->Op()->GetAttr("padding_algorithm")));
  op_desc.SetAttr("groups",
                  PADDLE_GET_CONST(int, conv_grad->Op()->GetAttr("groups")));
  op_desc.SetAttr(
      "data_format",
      PADDLE_GET_CONST(std::string, conv_grad->Op()->GetAttr("data_format")));
  op_desc.SetAttr("fuse_shortcut", fuse_shortcut);
  op_desc.SetAttr("fuse_dual", fuse_dual);
  op_desc.SetAttr("fuse_add", fuse_add);
  op_desc.SetAttr(
      "exhaustive_search",
      PADDLE_GET_CONST(bool, conv_grad->Op()->GetAttr("exhaustive_search")));
  op_desc.SetAttr("op_role", conv_grad->Op()->GetAttr("op_role"));
  auto conv_grad_op_role_val =
      details::GetOpRoleVarsOrEmpty(*(conv_grad->Op()));
  auto bn_grad_op_role_val = details::GetOpRoleVarsOrEmpty(*(bn_grad->Op()));
  std::vector<std::string> fused_op_role_var;
  for (auto i : conv_grad_op_role_val) {
    fused_op_role_var.push_back(i);
  }
  for (auto i : bn_grad_op_role_val) {
    fused_op_role_var.push_back(i);
  }
  if (fuse_dual) {
    auto bn2_grad_op_role_val =
        details::GetOpRoleVarsOrEmpty(*(bn2_grad->Op()));
    for (auto i : bn2_grad_op_role_val) {
      fused_op_role_var.push_back(i);
    }
  }
  op_desc.SetAttr("op_role_var", fused_op_role_var);

  auto op_node = g->CreateOpNode(&op_desc);

  IR_NODE_LINK_TO_DEBUG(d_conv_out, op_node);
  IR_NODE_LINK_TO_DEBUG(conv_w, op_node);
  IR_NODE_LINK_TO_DEBUG(bn_saved_mean, op_node);
  IR_NODE_LINK_TO_DEBUG(bn_saved_variance, op_node);
  IR_NODE_LINK_TO_DEBUG(bn_scale, op_node);
  IR_NODE_LINK_TO_DEBUG(bn_bias, op_node);
  IR_NODE_LINK_TO_DEBUG(bn_x, op_node);

  IR_NODE_LINK_TO_DEBUG(op_node, d_bn_x);
  IR_NODE_LINK_TO_DEBUG(op_node, d_bn_scale);
  IR_NODE_LINK_TO_DEBUG(op_node, d_bn_bias);
  IR_NODE_LINK_TO_DEBUG(op_node, d_conv_w);

  if (fuse_add) {
    IR_NODE_LINK_TO_DEBUG(d_relu_out_extra, op_node);
  }
  if (fuse_shortcut) {
    IR_NODE_LINK_TO_DEBUG(elewise_add_input_extra, op_node);
    IR_NODE_LINK_TO_DEBUG(op_node, d_elewise_add_input_extra);
  }
  if (fuse_dual) {
    IR_NODE_LINK_TO_DEBUG(bn2_saved_mean, op_node);
    IR_NODE_LINK_TO_DEBUG(bn2_saved_variance, op_node);
    IR_NODE_LINK_TO_DEBUG(bn2_scale, op_node);
    IR_NODE_LINK_TO_DEBUG(bn2_bias, op_node);
    IR_NODE_LINK_TO_DEBUG(bn2_x, op_node);

    IR_NODE_LINK_TO_DEBUG(op_node, d_bn2_x);
    IR_NODE_LINK_TO_DEBUG(op_node, d_bn2_scale);
    IR_NODE_LINK_TO_DEBUG(op_node, d_bn2_bias);
  }
  if (fuse_dual || fuse_shortcut) {
    IR_NODE_LINK_TO_DEBUG(conv_x, op_node);
  } else {
    IR_NODE_LINK_TO_DEBUG(bn_eqscale, op_node);
    IR_NODE_LINK_TO_DEBUG(bn_eqbias, op_node);
  }
  return op_node;
}

ir::Node *RetrieveForwardNode(ir::Graph *graph,
                              std::string name,
                              std::string op_type) {
  auto pos = name.find("@GRAD");
  PADDLE_ENFORCE_NE(
      pos,
      std::string::npos,
      common::errors::InvalidArgument("expect @GRAD in name, got (%s)", name));
  std::string fwd_name = name.substr(0, pos);
  for (auto *node : graph->Nodes()) {
    if (node->Name() == fwd_name) {
      for (auto *op : node->outputs) {
        if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type) {
          return node;
        }
      }
    }
  }
  PADDLE_THROW(common::errors::InvalidArgument("The node (%d) does not exist.",
                                               fwd_name));
  return nullptr;
}

void FuseResUnitPass::ApplyImpl(ir::Graph *graph) const {
  // Training
  ResUnitPassCache cache;
  graph = FuseConvBNAddActFwd(graph, {"relu"}, false, true);
  graph = FuseConvBNAddActFwd(graph, {"relu"}, true, true);

  int iteration = 0;
  int found_pattern_count = 0;
  do {
    VLOG(4) << "FuseConvBNActConvBNstats Iteration: " << iteration;
    graph = FuseConvBNActConvBNstats(
        graph, {"relu"}, true, &found_pattern_count, &cache);
    ++iteration;
  } while (found_pattern_count);

  graph = FuseBNAddActConvBwd(graph, {"relu_grad"}, false, true);
  graph = FuseBNAddActConvBwd(graph, {"relu_grad"}, true, true);
  graph = FuseBNAddActConvBwd(graph, {"relu_grad"}, false, false);
  graph = FuseBNAddActConvBwd(graph, {"relu_grad"}, true, false);
  graph = FuseBNActConvBwd(graph, {"relu_grad"}, &cache);

  // Try to fuse evaluation program
  graph = FuseConvBNAddActFwd(graph, {"relu"}, false, false);
  graph = FuseConvBNAddActFwd(graph, {"relu"}, true, false);

  iteration = 0;
  found_pattern_count = 0;
  do {
    VLOG(4) << "FuseConvBNActConvBNstats Iteration: " << iteration;
    graph = FuseConvBNActConvBNstats(
        graph, {"relu"}, false, &found_pattern_count, nullptr);
    ++iteration;
  } while (found_pattern_count);
}

ir::Graph *FuseResUnitPass::FuseConvBNAddActFwd(
    ir::Graph *graph,
    const std::unordered_set<std::string> &act_types,
    bool shortcut,
    bool is_training) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("conv_bn_add_act");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  patterns::ConvBNAddAct conv_bn_add_act_pattern(gpd.mutable_pattern(),
                                                 scope_name);
  conv_bn_add_act_pattern(act_types, shortcut, is_training);

  int found_pattern_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle ConvBNAddAct fuse";

    GET_IR_NODE_FROM_SUBGRAPH(x1, x1, conv_bn_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(x2, x2, conv_bn_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elewise_add_op, elewise_add_op, conv_bn_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_op, act_op, conv_bn_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(add_out, add_out, conv_bn_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, conv_bn_add_act_pattern);
    GET_CONV_BN_NODES_COND(1, conv_bn_add_act_pattern, true);
    GET_CONV_BN_NODES_COND(2, conv_bn_add_act_pattern, (!shortcut));

    auto bn1_eqscale = CreateVarNode(g, scope_name, "bn1_eqscale");
    auto bn1_eqbias = CreateVarNode(g, scope_name, "bn1_eqbias");
    ir::Node *bn2_eqscale = nullptr;
    ir::Node *bn2_eqbias = nullptr;
    if (!shortcut) {
      bn2_eqscale = CreateVarNode(g, scope_name, "bn2_eqscale");
      bn2_eqbias = CreateVarNode(g, scope_name, "bn2_eqbias");
    }

    CreateConvBNstatsOpNode(g,
                            conv1_op->Op()->Block(),
                            false,
                            conv1_op,
                            bn1_op,
                            x1,
                            conv1_w,
                            nullptr,
                            nullptr,
                            bn1_scale,
                            bn1_bias,
                            bn1_mean,
                            bn1_variance,
                            conv1_out,
                            bn1_mean_out,
                            bn1_variance_out,
                            bn1_saved_mean,
                            bn1_saved_variance,
                            bn1_eqscale,
                            bn1_eqbias);

    if (!shortcut) {
      CreateConvBNstatsOpNode(g,
                              conv2_op->Op()->Block(),
                              false,
                              conv2_op,
                              bn2_op,
                              x2,
                              conv2_w,
                              nullptr,
                              nullptr,
                              bn2_scale,
                              bn2_bias,
                              bn2_mean,
                              bn2_variance,
                              conv2_out,
                              bn2_mean_out,
                              bn2_variance_out,
                              bn2_saved_mean,
                              bn2_saved_variance,
                              bn2_eqscale,
                              bn2_eqbias);
    }
    auto *sbar_input2 = shortcut ? x2 : conv2_out;
    bool exhaustive_search =
        PADDLE_GET_CONST(bool, conv1_op->Op()->GetAttr("exhaustive_search"));
    CreateSBAROpNode(g,
                     act_op->Op()->Block(),
                     !shortcut,
                     exhaustive_search,
                     act_op,
                     conv1_out,
                     bn1_eqscale,
                     bn1_eqbias,
                     sbar_input2,
                     bn2_eqscale,
                     bn2_eqbias,
                     act_out);

    PRINT_DEBUG_INFO(1);
    if (shortcut) {
      VLOG(4) << "\n\t " << bn1_out->Name() << ", " << x2->Name() << " -> "
              << elewise_add_op->Name() << " -> " << add_out->Name() << " -> "
              << act_op->Name() << " -> " << act_out->Name();
    } else {
      PRINT_DEBUG_INFO(2);
      VLOG(4) << "\n\t " << bn1_out->Name() << ", " << bn2_out->Name() << " -> "
              << elewise_add_op->Name() << " -> " << add_out->Name() << " -> "
              << act_op->Name() << " -> " << act_out->Name();
    }
    std::unordered_set<const Node *> nodes_to_delete = {
        conv1_op, bn1_op, elewise_add_op, act_op, bn1_out, add_out};

    if (!shortcut) {
      nodes_to_delete.insert(std::move(conv2_op));
      nodes_to_delete.insert(std::move(bn2_op));
      nodes_to_delete.insert(std::move(bn2_out));
    }

    GraphSafeRemoveNodes(g, nodes_to_delete);
    found_pattern_count++;
  };

  gpd(graph, handler);

  AddStatis(found_pattern_count);
  return graph;
}

ir::Graph *FuseResUnitPass::FuseConvBNActConvBNstats(
    ir::Graph *graph,
    const std::unordered_set<std::string> &act_types,
    bool is_training,
    int *found_pattern_count_output,
    ResUnitPassCache *cache) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("conv_bn_act_conv_bnstats");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  patterns::ConvBNActConvBNStats conv_bn_act_conv_bnstats_pattern(
      gpd.mutable_pattern(), scope_name);
  conv_bn_act_conv_bnstats_pattern(act_types, is_training);

  int found_pattern_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle ConvBNActConvBNStats fuse";

    GET_IR_NODE_FROM_SUBGRAPH(conv_x, conv_x, conv_bn_act_conv_bnstats_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_op, act_op, conv_bn_act_conv_bnstats_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        act_out, act_out, conv_bn_act_conv_bnstats_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        conv_bnstats_op, conv_bnstats_op, conv_bn_act_conv_bnstats_pattern);
    GET_CONV_BN_NODES_COND(, conv_bn_act_conv_bnstats_pattern, true);

    auto bn_eqscale = CreateVarNode(g, scope_name, "bn_eqscale");
    auto bn_eqbias = CreateVarNode(g, scope_name, "bn_eqbias");

    CreateConvBNstatsOpNode(g,
                            conv_op->Op()->Block(),
                            false,
                            conv_op,
                            bn_op,
                            conv_x,
                            conv_w,
                            nullptr,
                            nullptr,
                            bn_scale,
                            bn_bias,
                            bn_mean,
                            bn_variance,
                            conv_out,
                            bn_mean_out,
                            bn_variance_out,
                            bn_saved_mean,
                            bn_saved_variance,
                            bn_eqscale,
                            bn_eqbias);

    // Modify the following conv_bnstats_op
    conv_bnstats_op->Op()->SetInput("scale", {bn_eqscale->Name()});
    conv_bnstats_op->Op()->SetInput("bias", {bn_eqbias->Name()});
    conv_bnstats_op->Op()->SetInput("x", {conv_out->Name()});
    conv_bnstats_op->Op()->SetAttr("fuse_prologue", true);

    IR_NODE_LINK_TO(conv_out, conv_bnstats_op);
    IR_NODE_LINK_TO(bn_eqscale, conv_bnstats_op);
    IR_NODE_LINK_TO(bn_eqbias, conv_bnstats_op);

    if (is_training) {
      // link bn_eqscale -> bn_saved_variance
      // bn_eqbias -> bn_saved_mean
      cache->Insert(
          GetCacheKey(bn_saved_variance->Var()->Name(), g->GetBlockId()),
          bn_eqscale);
      cache->Insert(GetCacheKey(bn_saved_mean->Var()->Name(), g->GetBlockId()),
                    bn_eqbias);
    }

    auto *x = conv_x;
    PRINT_DEBUG_INFO();
    VLOG(4) << "\n\t " << bn_out->Name() << " -> " << act_op->Name() << " -> "
            << act_out->Name() << " -> " << conv_bnstats_op->Name();

    std::unordered_set<const Node *> nodes_to_delete = {
        conv_op, bn_op, act_op, bn_out, act_out};

    GraphSafeRemoveNodes(g, nodes_to_delete);
    found_pattern_count++;
  };
  gpd(graph, handler);
  AddStatis(found_pattern_count);
  *found_pattern_count_output = found_pattern_count;
  return graph;
}

ir::Graph *FuseResUnitPass::FuseBNActConvBwd(
    ir::Graph *graph,
    const std::unordered_set<std::string> &act_grad_types,
    ResUnitPassCache *cache) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  VLOG(4) << "Applying FuseBNActConvBwd";
  const std::string scope_name("bn_act_conv_bwd");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  patterns::BNActConvGrad bn_act_conv_grad_pattern(gpd.mutable_pattern(),
                                                   scope_name);
  bn_act_conv_grad_pattern(act_grad_types);

  int found_pattern_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle BNActConvBwd fuse";

    GET_CONV_GRAD_NODES(bn_act_conv_grad_pattern);
    GET_BN_GRAD_NODES_COND(, bn_act_conv_grad_pattern, true);
    GET_IR_NODE_FROM_SUBGRAPH(act_grad, act_grad, bn_act_conv_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(d_act_x, d_act_x, bn_act_conv_grad_pattern);

    auto *bn_eqscale = cache->Get(
        GetCacheKey(bn_saved_variance->Var()->Name(), g->GetBlockId()));
    auto *bn_eqbias =
        cache->Get(GetCacheKey(bn_saved_mean->Var()->Name(), g->GetBlockId()));
    if (bn_eqscale == nullptr || bn_eqbias == nullptr) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The bn_eqscale and bn_eqbias do not exist in the cache. "
          "The forward fusion pass may not be successful."));
    }
    CreateDconvDreluDBNOpNode(g,
                              conv_grad->Op()->Block(),
                              false,
                              false,
                              false,
                              conv_grad,
                              batch_norm_grad,
                              nullptr,
                              d_conv_out,
                              conv_w,
                              d_conv_w,
                              bn_saved_mean,
                              bn_saved_variance,
                              bn_scale,
                              bn_bias,
                              bn_x,
                              d_bn_x,
                              d_bn_scale,
                              d_bn_bias,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              bn_eqscale,
                              bn_eqbias);

    std::unordered_set<const Node *> nodes_to_delete = {
        conv_grad, act_grad, batch_norm_grad, d_act_x, d_conv_x};

    GraphSafeRemoveNodes(g, nodes_to_delete);
    found_pattern_count++;
  };
  gpd(graph, handler);
  AddStatis(found_pattern_count);
  return graph;
}

ir::Graph *FuseResUnitPass::FuseBNAddActConvBwd(
    ir::Graph *graph,
    const std::unordered_set<std::string> &act_grad_types,
    bool shortcut,
    bool with_sum) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  VLOG(4) << "Applying FuseBNAddActConvBwd, shortcut=" << shortcut
          << ", with_sum=" << with_sum;
  const std::string scope_name("bn_add_act_conv_bwd");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  patterns::BNAddActConvGrad bn_add_act_conv_grad_pattern(gpd.mutable_pattern(),
                                                          scope_name);
  bn_add_act_conv_grad_pattern(act_grad_types, shortcut, with_sum);

  int found_pattern_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle BNAddActConvGrad fuse";
    GET_IR_NODE_FROM_SUBGRAPH(conv_x, conv_x, bn_add_act_conv_grad_pattern);
    GET_CONV_GRAD_NODES(bn_add_act_conv_grad_pattern);
    GET_BN_GRAD_NODES_COND(1, bn_add_act_conv_grad_pattern, true);
    GET_BN_GRAD_NODES_COND(2, bn_add_act_conv_grad_pattern, (!shortcut));

    GET_IR_NODE_FROM_SUBGRAPH(act_grad, act_grad, bn_add_act_conv_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elewise_add_grad, elewise_add_grad, bn_add_act_conv_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH_COND(
        sum, sum, bn_add_act_conv_grad_pattern, with_sum);
    GET_IR_NODE_FROM_SUBGRAPH_COND(
        sum_in_extra, sum_in_extra, bn_add_act_conv_grad_pattern, with_sum);
    GET_IR_NODE_FROM_SUBGRAPH_COND(
        sum_out, sum_out, bn_add_act_conv_grad_pattern, with_sum);

    GET_IR_NODE_FROM_SUBGRAPH(d_act_x, d_act_x, bn_add_act_conv_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        d_elewise_add_x, d_elewise_add_x, bn_add_act_conv_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        d_elewise_add_y, d_elewise_add_y, bn_add_act_conv_grad_pattern);

    ir::Node *d_elewise_add_input_extra = shortcut ? d_elewise_add_y : nullptr;
    ir::Node *elewise_add_input_extra = nullptr;
    if (shortcut) {
      elewise_add_input_extra = RetrieveForwardNode(
          graph, d_elewise_add_y->Name(), "elementwise_add_grad");
    }
    CreateDconvDreluDBNOpNode(g,
                              conv_grad->Op()->Block(),
                              with_sum,
                              !shortcut,
                              shortcut,
                              conv_grad,
                              batch_norm1_grad,
                              batch_norm2_grad,
                              d_conv_out,
                              conv_w,
                              d_conv_w,
                              bn1_saved_mean,
                              bn1_saved_variance,
                              bn1_scale,
                              bn1_bias,
                              bn1_x,
                              d_bn1_x,
                              d_bn1_scale,
                              d_bn1_bias,
                              sum_in_extra,
                              elewise_add_input_extra,
                              d_elewise_add_input_extra,
                              bn2_saved_mean,
                              bn2_saved_variance,
                              bn2_scale,
                              bn2_bias,
                              bn2_x,
                              d_bn2_x,
                              d_bn2_scale,
                              d_bn2_bias,
                              conv_x,
                              nullptr,
                              nullptr);

    std::unordered_set<const Node *> nodes_to_delete = {conv_grad,
                                                        act_grad,
                                                        elewise_add_grad,
                                                        batch_norm1_grad,
                                                        d_conv_x,
                                                        d_act_x,
                                                        d_elewise_add_x};
    if (with_sum) {
      nodes_to_delete.insert(std::move(sum));
      nodes_to_delete.insert(std::move(sum_out));
    }
    if (!shortcut) {
      nodes_to_delete.insert(std::move(d_elewise_add_y));
      nodes_to_delete.insert(std::move(batch_norm2_grad));
    }
    GraphSafeRemoveNodes(g, nodes_to_delete);
    found_pattern_count++;
  };
  gpd(graph, handler);
  AddStatis(found_pattern_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_resunit_pass, paddle::framework::ir::FuseResUnitPass);
