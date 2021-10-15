// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/layer_norm_fuse_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace framework {
namespace ir {

// cpplint complaints (wrong!) for not included <string> header in below line.
using string::PrettyLogDetail;  // NOLINT

#define CHECK_TRUE(expr, err_msg) \
  do {                            \
    int e_ = (expr);              \
    if (!e_) {                    \
      VLOG(4) << err_msg;         \
      return;                     \
    }                             \
  } while (0)

#define EXPECT_TRUE(expr, err_msg) \
  do {                             \
    int e_ = (expr);               \
    if (!e_) {                     \
      VLOG(4) << err_msg;          \
      return false;                \
    }                              \
  } while (0)

namespace {

bool validateReduceOpAttrs(const Node* node, const std::string& name) {
  const auto* op = node->Op();
  if (op->HasAttr("dim")) {
    auto dims = BOOST_GET_CONST(std::vector<int>, op->GetAttr("dim"));
    EXPECT_TRUE(
        dims.size() == 1,
        ::paddle::string::Sprintf(
            "The LayerNorm fusion %s reduction must happen only over single "
            "dimension.",
            name));
    EXPECT_TRUE(dims.front() == -1,
                ::paddle::string::Sprintf("The LayerNorm fusion %s reduction "
                                          "must happen over last dimension.",
                                          name));
  }
  if (op->HasAttr("reduce_all")) {
    EXPECT_TRUE(
        !BOOST_GET_CONST(bool, op->GetAttr("reduce_all")),
        ::paddle::string::Sprintf(
            "The LayerNorm fusion %s"
            "reduction must have \'reduce_all\' attribute set to false.",
            name));
  }
  if (op->HasAttr("keep_dim")) {
    EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("keep_dim")),
                ::paddle::string::Sprintf(
                    "The LayerNorm fusion %s"
                    " reduction must have \'keep_dim\' attribute set to true.",
                    name));
  }
  return true;
}

void setIntermediateOut(OpDesc* desc, const std::string& out_name,
                        const std::string& scope_name) {
  std::string new_name = scope_name + "/at." + out_name + ".new";
  desc->SetOutput(out_name, {new_name});
}

void addIntermediateOut(Node* op_node, const std::string& out_name,
                        const std::string& scope_name, Graph* graph) {
  std::string new_name = scope_name + "/at." + out_name + ".new";
  VarDesc out_var(new_name);
  out_var.SetPersistable(false);
  auto* node_var = graph->CreateVarNode(&out_var);
  IR_NODE_LINK_TO(op_node, node_var);
}

}  // namespace

LayerNormFusePass::LayerNormFusePass() {
  AddOpCompat(OpCompat("layer_norm"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddOutput("Mean")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Variance")
      .IsTensor()
      .IsOptional()
      .End()
      .AddAttr("epsilon")
      .IsNumGE(0.0f)
      .IsNumLE(0.001f)
      .End()
      .AddAttr("begin_norm_axis")
      .IsNumGT(0)
      .End();
  AddOpCompat(OpCompat("reduce_mean"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("dim")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("keep_dim")
      .IsBoolEQ(true)
      .End();
  AddOpCompat(OpCompat("sqrt"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
  AddOpCompat(OpCompat("elementwise_sub"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumEQ(1)
      .End();
  AddOpCompat(OpCompat("elementwise_pow"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumEQ(1)
      .End();
  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumEQ(1)
      .End();
  AddOpCompat(OpCompat("elementwise_div"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumEQ(1)
      .End();
  AddOpCompat(OpCompat("elementwise_mul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumEQ(1)
      .End();
}

void LayerNormFusePass::ApplyImpl(Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "The input graph of "
                              "LayerNormFusePass should not be nullptr."));
  FusePassBase::Init(scope_name_, graph);

  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

  GraphPatternDetector gpd;
  patterns::LayerNorm layer_norm_pattern(gpd.mutable_pattern(), scope_name_);
  layer_norm_pattern();

  int found_layer_norm_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    VLOG(4) << "Fuse LayerNorm from subgraph.";
    GET_IR_NODE_FROM_SUBGRAPH(x, x, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(x_mean, x_mean, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(x_mean_out, x_mean_out, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(x_sub_mean, x_sub_mean, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(x_sub_mean_out, x_sub_mean_out,
                              layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(sqr_pow, sqr_pow, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(x_sub_mean_sqr, x_sub_mean_sqr,
                              layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(x_sub_mean_sqr_out, x_sub_mean_sqr_out,
                              layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(std_dev, std_dev, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(std_dev_out, std_dev_out, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eps, eps, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(std_dev_eps, std_dev_eps, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(std_dev_eps_out, std_dev_eps_out,
                              layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(std_dev_eps_sqrt, std_dev_eps_sqrt,
                              layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(std_dev_eps_sqrt_out, std_dev_eps_sqrt_out,
                              layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(division, division, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(division_out, division_out, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(gamma, gamma, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale, scale, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(beta, beta, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(shift, shift, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(shift_out, shift_out, layer_norm_pattern);

    auto* eps_tensor = scope->FindVar(eps->Name())->GetMutable<LoDTensor>();

    // ------------------ subgraph node's validation ---------------------------
    CHECK_TRUE(
        eps_tensor->numel() == 1,
        ::paddle::string::Sprintf(
            "The LayerNorm divisor epsilon value must be one-element tensor, "
            "but has %s elements.",
            eps_tensor->numel()));
    CHECK_TRUE(
        eps_tensor->type() == proto::VarType::FP32,
        ::paddle::string::Sprintf("The LayerNorm divisor epsilon value "
                                  "must be of FP32 data type, but is %s.",
                                  eps_tensor->type()));

    const auto& gamma_shape = gamma->Var()->GetShape();
    const auto& beta_shape = beta->Var()->GetShape();
    const auto& x_shape = x->Var()->GetShape();
    int64_t x_last_dim = x_shape.back();

    CHECK_TRUE(
        gamma_shape.size() == 1,
        ::paddle::string::Sprintf("The LayerNorm gamma (scale) tensor "
                                  "shape must be one-dimensional, but is %s.",
                                  gamma_shape.size()));
    CHECK_TRUE(
        beta_shape.size() == 1,
        ::paddle::string::Sprintf("The LayerNorm beta (shift) tensor "
                                  "shape must be one-dimensional, but is %s.",
                                  beta_shape.size()));
    CHECK_TRUE(beta_shape == gamma_shape,
               ::paddle::string::Sprintf("The LayerNorm beta and gamma tensors "
                                         "shapes' must be equal."));
    CHECK_TRUE(
        gamma_shape.front() == x_last_dim,
        ::paddle::string::Sprintf(
            "The LayerNorm beta and gamma tensors "
            "shapes' must be equal to the last input's dimension size."));

    CHECK_TRUE(validateReduceOpAttrs(x_mean, "input mean"),
               "Validation of input mean node failed.");
    CHECK_TRUE(validateReduceOpAttrs(std_dev, "std_dev mean"),
               "Validation of standard deviation node failed.");

    // ------------------ op creation and placement ---------------------------

    OpDesc ln_op_desc;
    ln_op_desc.SetType("layer_norm");
    ln_op_desc.SetInput("X", {x->Name()});
    ln_op_desc.SetInput("Scale", {gamma->Name()});
    ln_op_desc.SetInput("Bias", {beta->Name()});
    ln_op_desc.SetOutput("Y", {shift_out->Name()});
    setIntermediateOut(&ln_op_desc, "Mean", scope_name_);
    setIntermediateOut(&ln_op_desc, "Variance", scope_name_);
    ln_op_desc.SetAttr("begin_norm_axis", static_cast<int>(x_shape.size() - 1));
    ln_op_desc.SetAttr("epsilon", *(eps_tensor->data<float>()));
    ln_op_desc.SetAttr("is_test", true);

    if (!IsCompat(ln_op_desc)) {
      LOG(WARNING) << "layer norm pass in out layer_norm op compat failed.";
      return;
    }

    Node* ln_op = g->CreateOpNode(&ln_op_desc);

    addIntermediateOut(ln_op, "Mean", scope_name_, g);
    addIntermediateOut(ln_op, "Variance", scope_name_, g);

    IR_NODE_LINK_TO(x, ln_op);
    IR_NODE_LINK_TO(gamma, ln_op);
    IR_NODE_LINK_TO(beta, ln_op);
    IR_OP_VAR_LINK(ln_op, shift_out);
    GraphSafeRemoveNodes(
        g,
        {x_mean, x_mean_out, x_sub_mean, x_sub_mean_out, sqr_pow,
         x_sub_mean_sqr, x_sub_mean_sqr_out, std_dev, std_dev_out, eps,
         std_dev_eps, std_dev_eps_out, std_dev_eps_sqrt, std_dev_eps_sqrt_out,
         division, division_out, scale, scale_out, shift});
    found_layer_norm_count++;
  };

  gpd(graph, handler);
  AddStatis(found_layer_norm_count);
  if (!Has("disable_logs") || !Get<bool>("disable_logs"))
    PrettyLogDetail("---    Fused %d subgraphs into layer_norm op.",
                    found_layer_norm_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

#undef CHECK_TRUE
#undef EXPECT_TRUE

REGISTER_PASS(layer_norm_fuse_pass, paddle::framework::ir::LayerNormFusePass);
REGISTER_PASS_CAPABILITY(layer_norm_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .GE("elementwise_add", 0)
            .LE("elementwise_add", 1)
            .GE("elementwise_div", 0)
            .LE("elementwise_div", 1)
            .GE("elementwise_mul", 0)
            .LE("elementwise_mul", 1)
            .GE("elementwise_pow", 0)
            .LE("elementwise_pow", 1)
            .GE("elementwise_sub", 0)
            .LE("elementwise_sub", 1)
            .EQ("reduce_mean", 0)
            .EQ("sqrt", 0));
