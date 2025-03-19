// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
change layernorm op to fast_layernorm op
For example:
graph:
                      x
                      |
                  layernorm
                      |
                    output
------------------------------------------------------
After the pass is applied:
                      x
                      |
              fast_layernorm_xpu
                      |
                    output
*/
struct FastLayernormXPUPattern : public PatternBase {
  FastLayernormXPUPattern(PDPattern* pattern, const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(l_norm);
  // declare variable node's name
  PATTERN_DECL_NODE(norm_in);
  PATTERN_DECL_NODE(norm_bias);
  PATTERN_DECL_NODE(norm_scale);
  PATTERN_DECL_NODE(norm_mean);
  PATTERN_DECL_NODE(norm_variance);
  PATTERN_DECL_NODE(norm_out);
};

FastLayernormXPUPattern::FastLayernormXPUPattern(PDPattern* pattern,
                                                 const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto l_norm = pattern->NewNode(l_norm_repr())->assert_is_op("layer_norm");
  auto norm_in = pattern->NewNode(norm_in_repr())
                     ->AsInput()
                     ->assert_is_op_input("layer_norm", "X");
  auto norm_bias = pattern->NewNode(norm_bias_repr())
                       ->AsInput()
                       ->assert_is_persistable_var()
                       ->assert_is_op_input("layer_norm", "Bias");
  auto norm_scale = pattern->NewNode(norm_scale_repr())
                        ->AsInput()
                        ->assert_is_persistable_var()
                        ->assert_is_op_input("layer_norm", "Scale");
  auto norm_mean = pattern->NewNode(norm_mean_repr())
                       ->AsOutput()
                       ->assert_is_op_output("layer_norm", "Mean")
                       ->assert_has_n_outputs(0);
  auto norm_variance = pattern->NewNode(norm_variance_repr())
                           ->AsOutput()
                           ->assert_is_op_output("layer_norm", "Variance")
                           ->assert_has_n_outputs(0);
  auto norm_out = pattern->NewNode(norm_out_repr())
                      ->AsOutput()
                      ->assert_is_op_output("layer_norm", "Y");
  l_norm->LinksFrom({norm_in, norm_bias, norm_scale})
      .LinksTo({norm_out, norm_mean, norm_variance});
}

}  // namespace patterns

class FastLayernormXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void FuseFastLayernorm(ir::Graph* graph) const;

  const std::string name_scope_{"fast_layernorm_xpu_fuse_pass"};
};

void FastLayernormXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  FuseFastLayernorm(graph);
}

void FastLayernormXPUFusePass::FuseFastLayernorm(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::FastLayernormXPUPattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FastLayernormXPUFusePass";
    // declare operator node's name
    GET_IR_NODE(l_norm);
    // declare variable node's name
    GET_IR_NODE(norm_in);
    GET_IR_NODE(norm_bias);
    GET_IR_NODE(norm_scale);
    GET_IR_NODE(norm_mean);
    GET_IR_NODE(norm_variance);
    GET_IR_NODE(norm_out);

    auto* block = l_norm->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));

    // delete useless node
    std::unordered_set<const Node*> delete_nodes;

    float eps = PADDLE_GET_CONST(float, l_norm->Op()->GetAttr("epsilon"));
    int begin_norm_axis =
        PADDLE_GET_CONST(int, l_norm->Op()->GetAttr("begin_norm_axis"));

    // Generate fast_layernorm_xpu op
    framework::OpDesc fused_op_desc(block);
    fused_op_desc.SetType("fast_layernorm_xpu");
    fused_op_desc.SetInput("x", {norm_in->Name()});
    fused_op_desc.SetInput("scale", {norm_scale->Name()});
    fused_op_desc.SetInput("bias", {norm_bias->Name()});
    fused_op_desc.SetAttr("epsilon", eps);
    fused_op_desc.SetAttr("begin_norm_axis", begin_norm_axis);
    fused_op_desc.SetOutput("out", {norm_out->Name()});
    auto* fused_op = graph->CreateOpNode(&fused_op_desc);
    IR_NODE_LINK_TO(norm_in, fused_op);
    IR_NODE_LINK_TO(norm_scale, fused_op);
    IR_NODE_LINK_TO(norm_bias, fused_op);
    IR_NODE_LINK_TO(fused_op, norm_out);
    delete_nodes.insert({l_norm, norm_mean, norm_variance});
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fast_layernorm_xpu_fuse_pass,
              paddle::framework::ir::FastLayernormXPUFusePass);

REGISTER_PASS_CAPABILITY(fast_layernorm_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "layer_norm", 0));
