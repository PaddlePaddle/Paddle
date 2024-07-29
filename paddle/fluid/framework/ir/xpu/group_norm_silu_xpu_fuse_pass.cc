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
fuse gn + activation block in to xpu_ele_fusion op
For example:
graph:
                      X
              Scale   |   Bias
                   \  |  /
                  group norm
                   /  |  \
                  /   |   \
            variance  |   mean
                      |
                     silu
                      |
                    output
------------------------------------------------------
After the pass is applied:
                      X
              Scale   |   Bias
                   \  |  /
                gn_silu_fusion
                      |
                     Out
*/
struct GroupNormalizeSiluXPUPattern : public PatternBase {
  GroupNormalizeSiluXPUPattern(PDPattern* pattern,
                               const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(gn);
  PATTERN_DECL_NODE(silu);
  // declare variable node's name
  PATTERN_DECL_NODE(gn_x);
  PATTERN_DECL_NODE(gn_bias);
  PATTERN_DECL_NODE(gn_scale);
  PATTERN_DECL_NODE(gn_y);
  PATTERN_DECL_NODE(gn_mean);
  PATTERN_DECL_NODE(gn_variance);
  PATTERN_DECL_NODE(silu_out);
};

GroupNormalizeSiluXPUPattern::GroupNormalizeSiluXPUPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto gn = pattern->NewNode(gn_repr())->assert_is_op("group_norm");
  auto gn_x = pattern->NewNode(gn_x_repr())
                  ->assert_is_op_input("group_norm", "X")
                  ->AsInput();
  auto gn_bias = pattern->NewNode(gn_bias_repr())
                     ->assert_is_op_input("group_norm", "Bias")
                     ->assert_is_persistable_var()
                     ->AsInput();
  auto gn_scale = pattern->NewNode(gn_scale_repr())
                      ->assert_is_op_input("group_norm", "Scale")
                      ->assert_is_persistable_var()
                      ->AsInput();
  auto gn_y = pattern->NewNode(gn_y_repr())
                  ->assert_is_op_output("group_norm", "Y")
                  ->assert_is_op_input("silu", "X")
                  ->assert_has_n_outputs(1);
  auto gn_mean = pattern->NewNode(gn_mean_repr())
                     ->assert_is_op_output("group_norm", "Mean")
                     ->assert_has_n_outputs(0);
  auto gn_variance = pattern->NewNode(gn_variance_repr())
                         ->assert_is_op_output("group_norm", "Variance")
                         ->assert_has_n_outputs(0);
  gn->LinksFrom({gn_x, gn_bias, gn_scale})
      .LinksTo({gn_y, gn_mean, gn_variance});

  auto silu = pattern->NewNode(silu_repr())->assert_is_op("silu");
  auto silu_out = pattern->NewNode(silu_out_repr())
                      ->AsOutput()
                      ->assert_is_op_output("silu", "Out");
  silu->LinksFrom({gn_y}).LinksTo({silu_out});
}

}  // namespace patterns

class GroupNormalizeSiluXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void FuseGroupNormalizeSilu(ir::Graph* graph) const;

  const std::string name_scope_{"group_norm_silu_xpu_fuse_pass"};
};

void GroupNormalizeSiluXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  FuseGroupNormalizeSilu(graph);
}

void GroupNormalizeSiluXPUFusePass::FuseGroupNormalizeSilu(
    ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::GroupNormalizeSiluXPUPattern pattern(gpd.mutable_pattern(),
                                                 name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle GroupNormalizeSiluXPUFusePass fuse";
    // declare operator node's name
    GET_IR_NODE(gn);
    GET_IR_NODE(silu);
    // declare variable node's name
    GET_IR_NODE(gn_x);
    GET_IR_NODE(gn_bias);
    GET_IR_NODE(gn_scale);
    GET_IR_NODE(gn_y);
    GET_IR_NODE(gn_mean);
    GET_IR_NODE(gn_variance);
    GET_IR_NODE(silu_out);

    auto* block = gn->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));
    // delete useless node
    std::unordered_set<const Node*> delete_nodes;

    float eps = PADDLE_GET_CONST(float, gn->Op()->GetAttr("epsilon"));
    int groups = PADDLE_GET_CONST(int, gn->Op()->GetAttr("groups"));

    std::string fused_op_out_name;
    fused_op_out_name = silu_out->Name();
    // Generate add_layernorm fused op
    framework::OpDesc fused_op_desc(block);

    fused_op_desc.SetType("group_norm_silu_xpu");
    // set attrs for fused op
    fused_op_desc.SetInput("x", {gn_x->Name()});
    fused_op_desc.SetInput("bias", {gn_bias->Name()});
    fused_op_desc.SetInput("scale", {gn_scale->Name()});
    fused_op_desc.SetAttr("epsilon", eps);
    fused_op_desc.SetAttr("groups", groups);
    fused_op_desc.SetOutput("out", {fused_op_out_name});
    // relink fused op
    auto* fused_op = graph->CreateOpNode(&fused_op_desc);
    IR_NODE_LINK_TO(gn_x, fused_op);
    IR_NODE_LINK_TO(gn_bias, fused_op);
    IR_NODE_LINK_TO(gn_scale, fused_op);
    IR_NODE_LINK_TO(fused_op, silu_out);

    delete_nodes.insert({gn, silu, gn_y, gn_mean, gn_variance});
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(group_norm_silu_xpu_fuse_pass,
              paddle::framework::ir::GroupNormalizeSiluXPUFusePass);

REGISTER_PASS_CAPABILITY(group_norm_silu_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "group_norm_silu_xpu", 0));
