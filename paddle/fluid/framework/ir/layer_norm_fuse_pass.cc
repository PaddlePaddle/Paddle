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

bool validateReduceOpAttrs(const Node* node,
                           const std::vector<int64_t>& x_shape,
                           const std::string& name) {
  const auto* op = node->Op();
  if (op->HasAttr("reduce_all")) {
    EXPECT_TRUE(
        !BOOST_GET_CONST(bool, op->GetAttr("reduce_all")),
        ::paddle::string::Sprintf(
            "The LayerNorm fusion %s"
            "reduction must have \'reduce_all\' attribute set to false.",
            name));
  }
  if (op->HasAttr("dim")) {
    auto dims = BOOST_GET_CONST(std::vector<int>, op->GetAttr("dim"));
    if (dims.size() == x_shape.size()) return false;
    if (1 == dims.size() && -1 == dims.front()) return true;

    if (dims.back() != static_cast<int>(x_shape.size()) - 1) {
      LOG(WARNING) << "The LayerNorm dim of mean must be end of x_input";
      return false;
    }
    for (size_t i = 1; i < dims.size(); ++i) {
      if (1 != dims[i] - dims[i - 1]) {
        LOG(WARNING) << "The LayerNorm dim of mean must be  continuous";
        return false;
      }
    }
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
      .IsIntIn({-1, 0})
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
      .IsIntIn({-1, 0})
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
    const auto& x_shape = x->Var()->GetShape();

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

    CHECK_TRUE(validateReduceOpAttrs(x_mean, x_shape, "input mean"),
               "Validation of input mean node failed.");
    CHECK_TRUE(validateReduceOpAttrs(std_dev, x_shape, "std_dev mean"),
               "Validation of standard deviation node failed.");

    bool keep_dim = BOOST_GET_CONST(bool, x_mean->Op()->GetAttr("keep_dim"));
    std::vector<int> mean_dim =
        BOOST_GET_CONST(std::vector<int>, x_mean->Op()->GetAttr("dim"));
    std::vector<int> std_mean_dim =
        BOOST_GET_CONST(std::vector<int>, std_dev->Op()->GetAttr("dim"));
    if (mean_dim != std_mean_dim) {
      LOG(WARNING) << "The LayerNorm dim of all mean must be same";
      return;
    }
    if (!keep_dim) {
      int sub_axis = BOOST_GET_CONST(int, x_sub_mean->Op()->GetAttr("axis"));
      int div_axis = BOOST_GET_CONST(int, division->Op()->GetAttr("axis"));
      if (sub_axis != 0 || div_axis != 0) return;
    }

    int begin_norm_axis = mean_dim.front();
    if (begin_norm_axis < 0) begin_norm_axis += x_shape.size();
    const auto& gamma_shape = gamma->Var()->GetShape();
    const auto& beta_shape = beta->Var()->GetShape();

    CHECK_TRUE(
        gamma_shape.size() == x_shape.size() - begin_norm_axis,
        ::paddle::string::Sprintf("The LayerNorm gamma (scale) tensor "
                                  "shape must be H(`begin_norm_axis` splits "
                                  "the tensor(`X`) to a matrix [N,H]),"
                                  "but is %s.",
                                  gamma_shape.size()));
    CHECK_TRUE(
        beta_shape.size() == x_shape.size() - begin_norm_axis,
        ::paddle::string::Sprintf("The LayerNorm beta (shift) tensor "
                                  "shape must be H(`begin_norm_axis` splits "
                                  "the tensor(`X`) to a matrix [N,H]),"
                                  "but is %s.",
                                  beta_shape.size()));
    CHECK_TRUE(beta_shape == gamma_shape,
               ::paddle::string::Sprintf("The LayerNorm beta and gamma tensors "
                                         "shapes' must be equal."));
    CHECK_TRUE(
        std::vector<int64_t>(x_shape.begin() + begin_norm_axis,
                             x_shape.end()) == gamma_shape,
        ::paddle::string::Sprintf("The LayerNorm beta and gamma tensors "
                                  "shape must be H(`begin_norm_axis` splits "
                                  "the tensor(`X`) to a matrix [N,H])."));

    // gamma/beta must be a 1-dimensional tensor of size on layer_norm
    auto layer_norm_x_mat_dims = framework::flatten_to_2d(
        framework::make_ddim(x_shape), begin_norm_axis);
    auto* gamma_tensor = scope->FindVar(gamma->Name())->GetMutable<LoDTensor>();
    VarDesc new_gamma_desc(patterns::PDNodeName("layer_norm_fuse", "Scale"));
    new_gamma_desc.SetShape({layer_norm_x_mat_dims[1]});
    new_gamma_desc.SetDataType(gamma_tensor->type());
    new_gamma_desc.SetLoDLevel(gamma->Var()->GetLoDLevel());
    new_gamma_desc.SetPersistable(true);
    auto* new_gamma_node = g->CreateVarNode(&new_gamma_desc);
    auto* new_gamma_tensor =
        scope->Var(new_gamma_node->Name())->GetMutable<LoDTensor>();
    new_gamma_tensor->Resize(framework::make_ddim({layer_norm_x_mat_dims[1]}));
    memcpy(new_gamma_tensor->mutable_data<float>(platform::CPUPlace()),
           gamma_tensor->mutable_data<float>(platform::CPUPlace()),
           layer_norm_x_mat_dims[1] * sizeof(float));

    auto* beta_tensor = scope->FindVar(beta->Name())->GetMutable<LoDTensor>();
    VarDesc new_beta_desc(patterns::PDNodeName("layer_norm_fuse", "Bias"));
    new_beta_desc.SetShape({layer_norm_x_mat_dims[1]});
    new_beta_desc.SetDataType(beta_tensor->type());
    new_beta_desc.SetLoDLevel(beta->Var()->GetLoDLevel());
    new_beta_desc.SetPersistable(true);
    auto* new_beta_node = g->CreateVarNode(&new_beta_desc);
    auto* new_beta_tensor =
        scope->Var(new_beta_node->Name())->GetMutable<LoDTensor>();

    new_beta_tensor->Resize(framework::make_ddim({layer_norm_x_mat_dims[1]}));
    memcpy(new_beta_tensor->mutable_data<float>(platform::CPUPlace()),
           beta_tensor->mutable_data<float>(platform::CPUPlace()),
           layer_norm_x_mat_dims[1] * sizeof(float));

    // ------------------ op creation and placement ---------------------------

    OpDesc ln_op_desc;
    ln_op_desc.SetType("layer_norm");
    ln_op_desc.SetInput("X", {x->Name()});
    ln_op_desc.SetInput("Scale", {new_gamma_node->Name()});
    ln_op_desc.SetInput("Bias", {new_beta_node->Name()});
    ln_op_desc.SetOutput("Y", {shift_out->Name()});
    setIntermediateOut(&ln_op_desc, "Mean", scope_name_);
    setIntermediateOut(&ln_op_desc, "Variance", scope_name_);
    ln_op_desc.SetAttr("begin_norm_axis", begin_norm_axis);
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
    IR_NODE_LINK_TO(new_gamma_node, ln_op);
    IR_NODE_LINK_TO(new_beta_node, ln_op);
    IR_OP_VAR_LINK(ln_op, shift_out);
    GraphSafeRemoveNodes(g, {x_mean,
                             x_mean_out,
                             x_sub_mean,
                             x_sub_mean_out,
                             sqr_pow,
                             x_sub_mean_sqr,
                             x_sub_mean_sqr_out,
                             std_dev,
                             std_dev_out,
                             eps,
                             std_dev_eps,
                             std_dev_eps_out,
                             std_dev_eps_sqrt,
                             std_dev_eps_sqrt_out,
                             division,
                             division_out,
                             scale,
                             scale_out,
                             shift,
                             gamma,
                             beta});
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
