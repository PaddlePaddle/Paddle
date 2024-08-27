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
Change layer_norm and act op to layer_norm_act_xpu op
For example:
graph:
                      x
                      |
                  layer_norm
                      |
                  leaky_relu
                      |
                    output
------------------------------------------------------
After the pass is applied:
                      x
                      |
              layer_norm_act_xpu
                      |
                    output
*/

struct LayerNormActXPUPattern : public PatternBase {
  LayerNormActXPUPattern(PDPattern* pattern,
                         const std::string& name_scope,
                         const std::string& act_type);
  // declare operator node's name
  PATTERN_DECL_NODE(layer_norm);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  PATTERN_DECL_NODE(layer_norm_input);
  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_out);
  PATTERN_DECL_NODE(act_out);

 private:
  std::string act_type_;
};

LayerNormActXPUPattern::LayerNormActXPUPattern(PDPattern* pattern,
                                               const std::string& name_scope,
                                               const std::string& act_type)
    : PatternBase(pattern, name_scope, name_scope), act_type_(act_type) {
  auto layer_norm =
      pattern->NewNode(layer_norm_repr())->assert_is_op("layer_norm");
  auto layer_norm_input = pattern->NewNode(layer_norm_input_repr())
                              ->AsInput()
                              ->assert_is_op_input("layer_norm", "X");
  auto layer_norm_bias = pattern->NewNode(layer_norm_bias_repr())
                             ->AsInput()
                             ->assert_is_persistable_var()
                             ->assert_is_op_input("layer_norm", "Bias");
  auto layer_norm_scale = pattern->NewNode(layer_norm_scale_repr())
                              ->AsInput()
                              ->assert_is_persistable_var()
                              ->assert_is_op_input("layer_norm", "Scale");
  auto layer_norm_out = pattern->NewNode(layer_norm_out_repr())
                            ->AsOutput()
                            ->assert_is_op_output("layer_norm", "Y")
                            ->assert_has_n_outputs(1);
  layer_norm_out->assert_is_op_input(act_type_, "X");
  layer_norm->LinksFrom({layer_norm_input, layer_norm_bias, layer_norm_scale})
      .LinksTo({layer_norm_out});

  auto act = pattern->NewNode(act_repr())->assert_is_op(act_type_);
  auto act_out = pattern->NewNode(act_out_repr())
                     ->assert_is_op_output(act_type_, "Out")
                     ->AsOutput();
  act->LinksFrom({layer_norm_out}).LinksTo({act_out});
}

}  // namespace patterns

class LayerNormActXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph, const std::string& act_type) const;

  const std::string name_scope_{"layer_norm_act_xpu_fuse_pass"};
};

void LayerNormActXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  for (auto act_type : {"leaky_relu"}) {
    found_subgraph_count += ApplyImpl(graph, act_type);
  }
  AddStatis(found_subgraph_count);
}

int LayerNormActXPUFusePass::ApplyImpl(ir::Graph* graph,
                                       const std::string& act_type) const {
  GraphPatternDetector gpd;
  patterns::LayerNormActXPUPattern pattern(
      gpd.mutable_pattern(), name_scope_, act_type);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle LayerNormActXPUFusePass fuse";
    // declare operator node's name
    GET_IR_NODE(layer_norm);
    GET_IR_NODE(act);
    // declare variable node's name
    GET_IR_NODE(layer_norm_input);
    GET_IR_NODE(layer_norm_bias);
    GET_IR_NODE(layer_norm_scale);
    GET_IR_NODE(layer_norm_out);
    GET_IR_NODE(act_out);
    auto* block = layer_norm->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));
    // delete useless node
    std::unordered_set<const Node*> delete_nodes;

    float eps = PADDLE_GET_CONST(float, layer_norm->Op()->GetAttr("epsilon"));
    int begin_norm_axis =
        PADDLE_GET_CONST(int, layer_norm->Op()->GetAttr("begin_norm_axis"));

    std::string fused_op_out_name;
    fused_op_out_name = act_out->Name();
    float act_param_ = 0.0f;
    int act_type_ = 0;
    if (!act_type.empty()) {
      if (act_type == "leaky_relu") {
        act_param_ = PADDLE_GET_CONST(float, act->Op()->GetAttr("alpha"));
        act_type_ = static_cast<int>(xpu::Activation_t::LEAKY_RELU);
      }
    }

    // Generate fused op
    framework::OpDesc fused_op_desc(block);
    fused_op_desc.SetType("layer_norm_act_xpu");
    // set attrs for fused op
    fused_op_desc.SetAttr("begin_norm_axis", begin_norm_axis);
    fused_op_desc.SetAttr("epsilon", eps);
    fused_op_desc.SetAttr("act_param", act_param_);
    fused_op_desc.SetAttr("act_type", act_type_);

    fused_op_desc.SetInput("x", {layer_norm_input->Name()});
    fused_op_desc.SetInput("bias", {layer_norm_bias->Name()});
    fused_op_desc.SetInput("scale", {layer_norm_scale->Name()});
    fused_op_desc.SetOutput("out", {fused_op_out_name});
    // relink fused op
    auto* fused_op = graph->CreateOpNode(&fused_op_desc);
    IR_NODE_LINK_TO(layer_norm_input, fused_op);
    IR_NODE_LINK_TO(layer_norm_bias, fused_op);
    IR_NODE_LINK_TO(layer_norm_scale, fused_op);
    IR_NODE_LINK_TO(fused_op, act_out);

    delete_nodes.insert({layer_norm, act, layer_norm_out});
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);

  return found_subgraph_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(layer_norm_act_xpu_fuse_pass,
              paddle::framework::ir::LayerNormActXPUFusePass);

REGISTER_PASS_CAPABILITY(layer_norm_act_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "layer_norm_act_xpu", 0));
