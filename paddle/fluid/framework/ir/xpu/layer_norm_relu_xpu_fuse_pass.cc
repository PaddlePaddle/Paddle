// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

/*
fuse ln + activation block in to xpu_ele_fusion op
For example:
graph:
                      X
              Scale   |   Bias
                   \  |  /
                  layer norm
                   /  |  \
                  /   |   \
            variance  |   mean
                      |
                     relu
                      |
                    output
------------------------------------------------------
After the pass is applied:
                      X
              Scale   |   Bias
                   \  |  /
                ln_relu_fusion
                      |
                     Out
*/
struct LayerNormalizeReluXPUPattern : public PatternBase {
  LayerNormalizeReluXPUPattern(PDPattern* pattern,
                               const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(ln);
  PATTERN_DECL_NODE(relu);
  // declare variable node's name
  PATTERN_DECL_NODE(ln_x);
  PATTERN_DECL_NODE(ln_bias);
  PATTERN_DECL_NODE(ln_scale);
  PATTERN_DECL_NODE(ln_y);
  PATTERN_DECL_NODE(ln_mean);
  PATTERN_DECL_NODE(ln_variance);
  PATTERN_DECL_NODE(relu_out);
};

LayerNormalizeReluXPUPattern::LayerNormalizeReluXPUPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto ln = pattern->NewNode(ln_repr())->assert_is_op("layer_norm");
  auto ln_x = pattern->NewNode(ln_x_repr())
                  ->assert_is_op_input("layer_norm", "X")
                  ->AsInput();
  auto ln_bias = pattern->NewNode(ln_bias_repr())
                     ->assert_is_op_input("layer_norm", "Bias")
                     ->assert_is_persistable_var()
                     ->AsInput();
  auto ln_scale = pattern->NewNode(ln_scale_repr())
                      ->assert_is_op_input("layer_norm", "Scale")
                      ->assert_is_persistable_var()
                      ->AsInput();
  auto ln_y = pattern->NewNode(ln_y_repr())
                  ->assert_is_op_output("layer_norm", "Y")
                  ->assert_is_op_input("relu", "X")
                  ->assert_has_n_outputs(1);
  auto ln_mean = pattern->NewNode(ln_mean_repr())
                     ->assert_is_op_output("layer_norm", "Mean")
                     ->assert_has_n_outputs(0);
  auto ln_variance = pattern->NewNode(ln_variance_repr())
                         ->assert_is_op_output("layer_norm", "Variance")
                         ->assert_has_n_outputs(0);
  ln->LinksFrom({ln_x, ln_bias, ln_scale})
      .LinksTo({ln_y, ln_mean, ln_variance});

  auto relu = pattern->NewNode(relu_repr())->assert_is_op("relu");
  auto relu_out = pattern->NewNode(relu_out_repr())
                      ->AsOutput()
                      ->assert_is_op_output("relu", "Out");
  relu->LinksFrom({ln_y}).LinksTo({relu_out});
}

}  // namespace patterns

class LayerNormalizeReluXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void FuseLayerNormalizeRelu(ir::Graph* graph) const;

  const std::string name_scope_{"layer_norm_relu_xpu_fuse_pass"};
};

void LayerNormalizeReluXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  auto* dev_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::XPUPlace()));
  auto version =
      phi::backends::xpu::get_xpu_version(dev_ctx->GetPlace().GetDeviceId());
  if (version == phi::backends::xpu::XPUVersion::XPU2) {
    FuseLayerNormalizeRelu(graph);
  }
}

void LayerNormalizeReluXPUFusePass::FuseLayerNormalizeRelu(
    ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::LayerNormalizeReluXPUPattern pattern(gpd.mutable_pattern(),
                                                 name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle LayerNormalizeReluXPUFusePass fuse";
    // declare operator node's name
    GET_IR_NODE(ln);
    GET_IR_NODE(relu);
    // declare variable node's name
    GET_IR_NODE(ln_x);
    GET_IR_NODE(ln_bias);
    GET_IR_NODE(ln_scale);
    GET_IR_NODE(ln_y);
    GET_IR_NODE(ln_mean);
    GET_IR_NODE(ln_variance);
    GET_IR_NODE(relu_out);

    auto* block = ln->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));
    // delete useless node
    std::unordered_set<const Node*> delete_nodes;

    float eps = PADDLE_GET_CONST(float, ln->Op()->GetAttr("epsilon"));
    int begin_norm_axis =
        PADDLE_GET_CONST(int, ln->Op()->GetAttr("begin_norm_axis"));

    std::string fused_op_out_name;
    fused_op_out_name = relu_out->Name();
    // Generate add_layernorm fused op
    framework::OpDesc fused_op_desc(block);

    fused_op_desc.SetType("layer_norm_relu_xpu");
    // set attrs for fused op
    fused_op_desc.SetAttr("begin_norm_axis", begin_norm_axis);
    fused_op_desc.SetInput("x", {ln_x->Name()});
    fused_op_desc.SetInput("bias", {ln_bias->Name()});
    fused_op_desc.SetInput("scale", {ln_scale->Name()});
    fused_op_desc.SetAttr("epsilon", eps);
    fused_op_desc.SetOutput("out", {fused_op_out_name});
    // relink fused op
    auto* fused_op = graph->CreateOpNode(&fused_op_desc);
    IR_NODE_LINK_TO(ln_x, fused_op);
    IR_NODE_LINK_TO(ln_bias, fused_op);
    IR_NODE_LINK_TO(ln_scale, fused_op);
    IR_NODE_LINK_TO(fused_op, relu_out);

    delete_nodes.insert({ln, relu, ln_y, ln_mean, ln_variance});
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(layer_norm_relu_xpu_fuse_pass,
              paddle::framework::ir::LayerNormalizeReluXPUFusePass);

REGISTER_PASS_CAPABILITY(layer_norm_relu_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "layer_norm_relu_xpu", 0));
