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
Fuse original subgraph into __xpu__spatial_transformer_resblock op.
Currently there are 3 different original patterns to match.

Original subgraph (situation 1):

      ------------Input1                     Input2
      |              |                          |
      |          group_norm                    silu
      |              |                          |
      |             silu                      _xpu_fc
      |              |                          |
      |         _xpu_conv2d                  unsqueeze
      |              \                           /
      |               \                         /
      |                \                       /
      |                 \                     /
      |                     elementwise_add
      |                           |
      |                      group_norm
      |                           |
      |                          silu
      |                           |
      |                       _xpu_conv2d
      |                           |
      |____________________elementwise_add
                                  |
                                output

Original subgraph (situation 2):

      ------------Input1
      |              |
      |          group_norm
      |              |
      |             silu
      |              |
      |         _xpu_conv2d
      |              \
      |               \
      |                \
      |                 \
      |                  |
      |              group_norm
      |                  |
      |                 silu
      |                  |
      |              _xpu_conv2d
      |                  |
      |___________elementwise_add
                        |
                      output

Original subgraph (situation 3):

      ------------Input1
      |              |
      |          group_norm
      |              |
      |             silu
      |              |
      |         _xpu_conv2d
      |              \
      |               \
      |                \
      |                 \
      |                  |
      |              group_norm
      |                  |
      |                 silu
      |                  |
      |              _xpu_conv2d
      |                  |
_xpu_conv2d              |
      |                  |
      |                  |
      |                  |
      |___________elementwise_add
                        |
                      output

Fuse to:
(Situation 1):
         Input1     Input2
            \         /
   __xpu__spatial_transformer_resblock
                 |
              output
or:
(Situation 2 and 3):
               Input
                 |
  __xpu__spatial_transformer_resblock
                 |
              output
*/
struct SpatialTransformerResBlockXPUPattern : public PatternBase {
  SpatialTransformerResBlockXPUPattern(PDPattern* pattern,
                                       const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(gn_silu_0);
  PATTERN_DECL_NODE(conv2d_0);
  PATTERN_DECL_NODE(gn_silu_1);
  PATTERN_DECL_NODE(conv2d_1);
  // declare variable node's name
  PATTERN_DECL_NODE(gn_silu_0_x);
  PATTERN_DECL_NODE(gn_silu_0_bias);
  PATTERN_DECL_NODE(gn_silu_0_scale);
  PATTERN_DECL_NODE(gn_silu_0_out);
  PATTERN_DECL_NODE(conv2d_0_bias);
  PATTERN_DECL_NODE(conv2d_0_filter);
  PATTERN_DECL_NODE(conv2d_0_filter_max);
  PATTERN_DECL_NODE(conv2d_0_out);
  PATTERN_DECL_NODE(conv2d_0_out_max);
  PATTERN_DECL_NODE(gn_silu_1_bias);
  PATTERN_DECL_NODE(gn_silu_1_scale);
  PATTERN_DECL_NODE(gn_silu_1_out);
  PATTERN_DECL_NODE(conv2d_1_bias);
  PATTERN_DECL_NODE(conv2d_1_branch_max);
  PATTERN_DECL_NODE(conv2d_1_filter);
  PATTERN_DECL_NODE(conv2d_1_filter_max);
  PATTERN_DECL_NODE(conv2d_1_out);
  PATTERN_DECL_NODE(conv2d_1_out_max);
};

SpatialTransformerResBlockXPUPattern::SpatialTransformerResBlockXPUPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  // gn_silu_0
  auto gn_silu_0 =
      pattern->NewNode(gn_silu_0_repr())->assert_is_op("gn_silu_xpu");
  auto gn_silu_0_x = pattern->NewNode(gn_silu_0_x_repr())
                         ->assert_is_op_input("gn_silu_xpu", "x")
                         ->assert_is_op_input("conv2d_xpu", "branch")
                         ->AsInput();
  auto gn_silu_0_bias = pattern->NewNode(gn_silu_0_bias_repr())
                            ->assert_is_op_input("gn_silu_xpu", "bias")
                            ->assert_is_persistable_var()
                            ->AsInput();
  auto gn_silu_0_scale = pattern->NewNode(gn_silu_0_scale_repr())
                             ->assert_is_op_input("gn_silu_xpu", "scale")
                             ->assert_is_persistable_var()
                             ->AsInput();
  auto gn_silu_0_out = pattern->NewNode(gn_silu_0_out_repr())
                           ->assert_is_op_output("gn_silu_xpu", "out")
                           ->assert_is_op_input("conv2d_xpu", "x")
                           ->assert_has_n_outputs(1);
  gn_silu_0->LinksFrom({gn_silu_0_x, gn_silu_0_bias, gn_silu_0_scale})
      .LinksTo({gn_silu_0_out});

  // conv2d_0
  auto conv2d_0 = pattern->NewNode(conv2d_0_repr())->assert_is_op("conv2d_xpu");
  auto conv2d_0_bias = pattern->NewNode(conv2d_0_bias_repr())
                           ->assert_is_op_input("conv2d_xpu", "bias")
                           ->AsInput();
  auto conv2d_0_filter = pattern->NewNode(conv2d_0_filter_repr())
                             ->assert_is_op_input("conv2d_xpu", "filter")
                             ->AsInput();
  auto conv2d_0_filter_max =
      pattern->NewNode(conv2d_0_filter_max_repr())
          ->assert_is_op_input("conv2d_xpu", "filter_max")
          ->AsInput();
  auto conv2d_0_out = pattern->NewNode(conv2d_0_out_repr())
                          ->assert_is_op_output("conv2d_xpu", "out")
                          ->assert_is_op_input("gn_silu_xpu", "x")
                          ->assert_has_n_outputs(1);
  auto conv2d_0_out_max = pattern->NewNode(conv2d_0_out_max_repr())
                              ->assert_is_op_output("conv2d_xpu", "out_max")
                              ->assert_has_n_outputs(0);
  conv2d_0->LinksFrom(
          {gn_silu_0_out, conv2d_0_bias, conv2d_0_filter, conv2d_0_filter_max})
      .LinksTo({conv2d_0_out, conv2d_0_out_max});

  // gn_silu_1
  auto gn_silu_1 =
      pattern->NewNode(gn_silu_1_repr())->assert_is_op("gn_silu_xpu");
  auto gn_silu_1_bias = pattern->NewNode(gn_silu_1_bias_repr())
                            ->assert_is_op_input("gn_silu_xpu", "bias")
                            ->assert_is_persistable_var()
                            ->AsInput();
  auto gn_silu_1_scale = pattern->NewNode(gn_silu_1_scale_repr())
                             ->assert_is_op_input("gn_silu_xpu", "scale")
                             ->assert_is_persistable_var()
                             ->AsInput();
  auto gn_silu_1_out = pattern->NewNode(gn_silu_1_out_repr())
                           ->assert_is_op_output("gn_silu_xpu", "out")
                           ->assert_is_op_input("conv2d_xpu", "x")
                           ->assert_has_n_outputs(1);
  gn_silu_1->LinksFrom({conv2d_0_out, gn_silu_1_bias, gn_silu_1_scale})
      .LinksTo({gn_silu_1_out});

  // conv2d_1
  auto conv2d_1 = pattern->NewNode(conv2d_1_repr())->assert_is_op("conv2d_xpu");
  auto conv2d_1_bias = pattern->NewNode(conv2d_1_bias_repr())
                           ->assert_is_op_input("conv2d_xpu", "bias")
                           ->AsInput();
  auto conv2d_1_branch_max = pattern->NewNode(conv2d_1_branch_max_repr())
                           ->assert_is_op_input("conv2d_xpu", "branch_max")
                           ->AsInput();
  auto conv2d_1_filter = pattern->NewNode(conv2d_1_filter_repr())
                           ->assert_is_op_input("conv2d_xpu", "filter")
                           ->AsInput();
  auto conv2d_1_filter_max =
      pattern->NewNode(conv2d_1_filter_max_repr())
          ->assert_is_op_input("conv2d_xpu", "filter_max")
          ->AsInput();
  auto conv2d_1_out = pattern->NewNode(conv2d_1_out_repr())
                          ->assert_is_op_output("conv2d_xpu", "out")
                          ->assert_has_n_outputs(1);
  auto conv2d_1_out_max = pattern->NewNode(conv2d_1_out_max_repr())
                              ->assert_is_op_output("conv2d_xpu", "out_max")
                              ->assert_has_n_outputs(0);
  conv2d_1->LinksFrom(
          {gn_silu_1_out, conv2d_1_bias, gn_silu_0_x, conv2d_1_branch_max, conv2d_1_filter, conv2d_1_filter_max})
      .LinksTo({conv2d_1_out, conv2d_1_out_max});
}

}  // namespace patterns

namespace {
void setIntermediateOut(OpDesc* desc,
                        const std::string& out_name,
                        const std::string& scope_name) {
  std::string new_name = scope_name + "/at." + out_name + ".new";
  desc->SetOutput(out_name, {new_name});
}

void addIntermediateOut(Node* op_node,
                        const std::string& out_name,
                        const std::string& scope_name,
                        Graph* graph) {
  std::string new_name = scope_name + "/at." + out_name + ".new";
  VarDesc out_var(new_name);
  out_var.SetPersistable(false);
  auto* node_var = graph->CreateVarNode(&out_var);
  IR_NODE_LINK_TO(op_node, node_var);
}

static std::vector<int> IntVec2DTo1D(const std::vector<std::vector<int>>& vec) {
  std::vector<int> res;
  for (const auto& v : vec) {
    for (const auto& ele : v) {
      res.emplace_back(ele);
    }
  }
  return res;
}

}  // namespace

class SpatialTransformerResBlockXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void FuseSpatialTransformerResBlock(ir::Graph* graph) const;

  const std::string name_scope_{"spatial_transformer_resblock_xpu_fuse_pass"};
};

void SpatialTransformerResBlockXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  FuseSpatialTransformerResBlock(graph);
}

void SpatialTransformerResBlockXPUFusePass::FuseSpatialTransformerResBlock(
    ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::SpatialTransformerResBlockXPUPattern pattern(gpd.mutable_pattern(),
                                                         name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle SpatialTransformerResBlockXPUFusePass fuse";
    // declare operator node's name
    GET_IR_NODE(gn_silu_0);
    GET_IR_NODE(conv2d_0);
    GET_IR_NODE(gn_silu_1);
    GET_IR_NODE(conv2d_1);
    // declare variable node's name
    GET_IR_NODE(gn_silu_0_x);
    GET_IR_NODE(gn_silu_0_bias);
    GET_IR_NODE(gn_silu_0_scale);
    GET_IR_NODE(gn_silu_0_out);
    GET_IR_NODE(conv2d_0_bias);
    GET_IR_NODE(conv2d_0_filter);
    GET_IR_NODE(conv2d_0_filter_max);
    GET_IR_NODE(conv2d_0_out);
    GET_IR_NODE(conv2d_0_out_max);
    GET_IR_NODE(gn_silu_1_bias);
    GET_IR_NODE(gn_silu_1_scale);
    GET_IR_NODE(gn_silu_1_out);
    GET_IR_NODE(conv2d_1_bias);
    GET_IR_NODE(conv2d_1_branch_max);
    GET_IR_NODE(conv2d_1_filter);
    GET_IR_NODE(conv2d_1_filter_max);
    GET_IR_NODE(conv2d_1_out);
    GET_IR_NODE(conv2d_1_out_max);

    auto* block = gn_silu_0->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));
    // delete useless node
    std::unordered_set<const Node*> delete_nodes;

    std::vector<std::vector<int>> strides;
    std::vector<std::vector<int>> paddings;
    std::vector<std::vector<int>> dilations;

    // get attr
    float gn_silu_0_eps = PADDLE_GET_CONST(float, gn_silu_0->Op()->GetAttr("epsilon"));
    int gn_silu_0_groups = PADDLE_GET_CONST(int, gn_silu_0->Op()->GetAttr("groups"));
    float gn_silu_1_eps = PADDLE_GET_CONST(float, gn_silu_1->Op()->GetAttr("epsilon"));
    int gn_silu_1_groups = PADDLE_GET_CONST(int, gn_silu_1->Op()->GetAttr("groups"));

    float conv2d_0_act_param = PADDLE_GET_CONST(float, conv2d_0->Op()->GetAttr("act_param"));
    int conv2d_0_act_type = PADDLE_GET_CONST(int, conv2d_0->Op()->GetAttr("act_type"));  
    auto conv2d_0_dilations = PADDLE_GET_CONST(std::vector<int>, conv2d_0->Op()->GetAttr("dilations"));
    dilations.emplace_back(std::move(conv2d_0_dilations)); 
    int conv2d_0_groups = PADDLE_GET_CONST(int, conv2d_0->Op()->GetAttr("groups"));  
    int conv2d_0_out_dtype = PADDLE_GET_CONST(int, conv2d_0->Op()->GetAttr("out_dtype"));  
    auto conv2d_0_paddings = PADDLE_GET_CONST(std::vector<int>, conv2d_0->Op()->GetAttr("paddings")); 
    paddings.emplace_back(std::move(conv2d_0_paddings)); 
    std::string conv2d_0_padding_algorithm = PADDLE_GET_CONST(std::string, conv2d_0->Op()->GetAttr("padding_algorithm"));
    auto conv2d_0_strides = PADDLE_GET_CONST(std::vector<int>, conv2d_0->Op()->GetAttr("strides"));
    strides.emplace_back(std::move(conv2d_0_strides)); 

    float conv2d_1_act_param = PADDLE_GET_CONST(float, conv2d_1->Op()->GetAttr("act_param"));
    int conv2d_1_act_type = PADDLE_GET_CONST(int, conv2d_1->Op()->GetAttr("act_type"));  
    auto conv2d_1_dilations = PADDLE_GET_CONST(std::vector<int>, conv2d_1->Op()->GetAttr("dilations"));
    dilations.emplace_back(std::move(conv2d_1_dilations));  
    int conv2d_1_groups = PADDLE_GET_CONST(int, conv2d_1->Op()->GetAttr("groups"));  
    int conv2d_1_out_dtype = PADDLE_GET_CONST(int, conv2d_1->Op()->GetAttr("out_dtype"));  
    auto conv2d_1_paddings = PADDLE_GET_CONST(std::vector<int>, conv2d_1->Op()->GetAttr("paddings"));
    paddings.emplace_back(std::move(conv2d_1_paddings));  
    std::string conv2d_1_padding_algorithm = PADDLE_GET_CONST(std::string, conv2d_1->Op()->GetAttr("padding_algorithm"));
    auto conv2d_1_strides = PADDLE_GET_CONST(std::vector<int>, conv2d_1->Op()->GetAttr("strides"));
    strides.emplace_back(std::move(conv2d_1_strides));  


    std::string fused_op_out_name;
    fused_op_out_name = conv2d_1_out->Name();
    // Generate add_layernorm fused op
    framework::OpDesc fused_op_desc(block);

    fused_op_desc.SetType("spatial_transformer_resblock_xpu");
    // set attrs for fused op
    fused_op_desc.SetInput("x", {gn_silu_0_x->Name()});
    fused_op_desc.SetInput("conv_bias", {conv2d_0_bias->Name(), conv2d_1_bias->Name()});
    fused_op_desc.SetInput("conv_filter", {conv2d_0_filter->Name(), conv2d_1_filter->Name()});
    fused_op_desc.SetInput("conv_filter_max", {conv2d_0_filter_max->Name(), conv2d_1_filter_max->Name()});
    fused_op_desc.SetInput("gn_bias", {gn_silu_0_bias->Name(), gn_silu_1_bias->Name()});
    fused_op_desc.SetInput("gn_scale", {gn_silu_0_scale->Name(), gn_silu_1_scale->Name()});

    fused_op_desc.SetAttr("dilations", IntVec2DTo1D(dilations));
    fused_op_desc.SetAttr("paddings", IntVec2DTo1D(paddings));
    fused_op_desc.SetAttr("strides", IntVec2DTo1D(strides));
    fused_op_desc.SetAttr("gn_eps", std::vector<float>{gn_silu_0_eps, gn_silu_1_eps});
    fused_op_desc.SetAttr("gn_groups", std::vector<int>{gn_silu_0_groups, gn_silu_1_groups});
    fused_op_desc.SetAttr("groups", std::vector<int>{conv2d_0_groups, conv2d_1_groups});

    fused_op_desc.SetOutput("out", {fused_op_out_name});
    // relink fused op
    auto* fused_op = graph->CreateOpNode(&fused_op_desc);

    IR_NODE_LINK_TO(gn_silu_0_x, fused_op);
    IR_NODE_LINK_TO(gn_silu_0_bias, fused_op);
    IR_NODE_LINK_TO(gn_silu_0_scale, fused_op);
    IR_NODE_LINK_TO(conv2d_0_bias, fused_op);
    IR_NODE_LINK_TO(conv2d_0_filter, fused_op);
    IR_NODE_LINK_TO(conv2d_0_filter_max, fused_op);
    IR_NODE_LINK_TO(gn_silu_1_bias, fused_op);
    IR_NODE_LINK_TO(gn_silu_1_scale, fused_op);
    IR_NODE_LINK_TO(conv2d_1_bias, fused_op);
    IR_NODE_LINK_TO(conv2d_1_filter, fused_op);
    IR_NODE_LINK_TO(conv2d_1_filter_max, fused_op);
    
    IR_NODE_LINK_TO(fused_op, conv2d_1_out);

    delete_nodes.insert({gn_silu_0, gn_silu_1, conv2d_0, conv2d_1, gn_silu_0_out, 
      conv2d_0_out, conv2d_0_out_max, gn_silu_1_out, 
      conv2d_1_branch_max, conv2d_1_out_max});
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(spatial_transformer_resblock_xpu_fuse_pass,
              paddle::framework::ir::SpatialTransformerResBlockXPUFusePass);

REGISTER_PASS_CAPABILITY(spatial_transformer_resblock_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "spatial_transformer_resblock_xpu", 0));
