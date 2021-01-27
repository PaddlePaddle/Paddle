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

#include "paddle/fluid/framework/ir/layer_norm_fuse_pass.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

// cpplint complaints (wrong!) for not included <string> header in below line.
using string::PrettyLogDetail;  // NOLINT

void LayerNormFusePass::ApplyImpl(Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "The input graph of "
                              "LayerNormFusePass should not be nullptr."));
  FusePassBase::Init("layer_norm_fuse", graph);

  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

  GraphPatternDetector gpd;
  patterns::LayerNorm layer_norm_pattern(gpd.mutable_pattern(),
                                         "layer_norm_fuse");
  layer_norm_pattern();

  int found_layer_norm_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
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
    PADDLE_ENFORCE_EQ(
        eps_tensor->numel(), 1,
        platform::errors::InvalidArgument(
            "The LayerNorm divisor "
            "epsilon value must be one-element tensor, but has %s "
            "elements.",
            eps_tensor->numel()));
    PADDLE_ENFORCE_EQ(eps_tensor->type(), proto::VarType::FP32,
                      platform::errors::InvalidArgument(
                          "The LayerNorm divisor "
                          "epsilon value must be of FP32 data type, but is %s.",
                          eps_tensor->type()));

    const auto& gamma_shape = gamma->Var()->GetShape();
    const auto& beta_shape = beta->Var()->GetShape();
    const auto& x_shape = x->Var()->GetShape();
    int64_t x_last_dim = x_shape.back();

    PADDLE_ENFORCE_EQ(gamma_shape.size(), 1,
                      platform::errors::InvalidArgument(
                          "The LayerNorm gamma "
                          "(scale) tensor shape must be one-dimensional, "
                          "but is %s.",
                          gamma_shape.size()));
    PADDLE_ENFORCE_EQ(beta_shape.size(), 1,
                      platform::errors::InvalidArgument(
                          "The LayerNorm beta "
                          "(shift) tensor shape must be one-dimensional, "
                          "but is %s.",
                          beta_shape.size()));
    PADDLE_ENFORCE_EQ(beta_shape, gamma_shape,
                      platform::errors::InvalidArgument(
                          "The LayerNorm beta "
                          "and gamma tensors shapes' must be equal."));
    PADDLE_ENFORCE_EQ(gamma_shape.front(), x_last_dim,
                      platform::errors::InvalidArgument(
                          "The LayerNorm beta "
                          "and gamma tensors shapes' must be equal to the last "
                          "input's dimension size."));

    // ------------------ op creation and placement ---------------------------

    OpDesc ln_op_desc;
    ln_op_desc.SetType("layer_norm");
    ln_op_desc.SetInput("X", {x->Name()});
    ln_op_desc.SetInput("Scale", {gamma->Name()});
    ln_op_desc.SetInput("Bias", {beta->Name()});
    ln_op_desc.SetOutput("Y", {shift_out->Name()});
    ln_op_desc.SetAttr("begin_norm_axis", static_cast<int>(x_shape.size() - 1));
    ln_op_desc.SetAttr("epsilon", *(eps_tensor->data<float>()));
    ln_op_desc.SetAttr("is_test", true);
    Node* ln_op = g->CreateOpNode(&ln_op_desc);

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
  PrettyLogDetail("---    Fused %d subgraphs into layer_norm op.",
                  found_layer_norm_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(layer_norm_fuse_pass, paddle::framework::ir::LayerNormFusePass);
REGISTER_PASS_CAPABILITY(layer_norm_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .GE("elementwise_add", 0)
            .GE("elementwise_div", 0)
            .GE("elementwise_mul", 0)
            .GE("elementwise_pow", 0)
            .GE("elementwise_sub", 0)
            .GE("reduce_mean", 0)
            .GE("sqrt", 0));
