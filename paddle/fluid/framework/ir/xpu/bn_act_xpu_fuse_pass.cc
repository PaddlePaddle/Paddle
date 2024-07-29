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

struct BNActXPUPattern : public PatternBase {
  BNActXPUPattern(PDPattern* pattern,
                  const std::string& name_scope,
                  const std::string& act_type);
  // declare operator node's name
  PATTERN_DECL_NODE(bn);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  PATTERN_DECL_NODE(bn_input);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_mean);
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_var);
  PATTERN_DECL_NODE(bn_out);
  PATTERN_DECL_NODE(act_out);

 private:
  std::string act_type_;
};

BNActXPUPattern::BNActXPUPattern(PDPattern* pattern,
                                 const std::string& name_scope,
                                 const std::string& act_type)
    : PatternBase(pattern, name_scope, name_scope), act_type_(act_type) {
  auto bn = pattern->NewNode(bn_repr())
                ->assert_is_op("batch_norm")
                ->assert_more([](Node* node) {
                  auto is_test = node->Op()->GetAttrIfExists<bool>("is_test");
                  return is_test;
                });
  auto bn_input = pattern->NewNode(bn_input_repr())
                      ->assert_is_op_input("batch_norm", "X")
                      ->assert_var_not_persistable()
                      ->AsInput();
  auto bn_bias = pattern->NewNode(bn_bias_repr())
                     ->assert_is_op_input("batch_norm", "Bias")
                     ->AsInput();
  auto bn_mean = pattern->NewNode(bn_mean_repr())
                     ->assert_is_op_input("batch_norm", "Mean")
                     ->AsInput();
  auto bn_scale = pattern->NewNode(bn_scale_repr())
                      ->assert_is_op_input("batch_norm", "Scale")
                      ->AsInput();
  auto bn_var = pattern->NewNode(bn_var_repr())
                    ->assert_is_op_input("batch_norm", "Variance")
                    ->AsInput();
  auto bn_out = pattern->NewNode(bn_out_repr())
                    ->assert_is_op_output("batch_norm", "Y")
                    ->assert_has_n_outputs(1);

  bn->LinksFrom({bn_input, bn_bias, bn_mean, bn_scale, bn_var})
      .LinksTo({bn_out});
  bn_out->assert_is_op_input(act_type_, "X");
  auto act = pattern->NewNode(act_repr())->assert_is_op(act_type_);
  auto act_out = pattern->NewNode(act_out_repr())
                     ->assert_is_op_output(act_type_, "Out")
                     ->AsOutput();
  act->LinksFrom({bn_out}).LinksTo({act_out});
}

}  // namespace patterns

class BNActXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph, const std::string& act_type) const;

  const std::string name_scope_{"bn_act_xpu_fuse_pass"};
};

void BNActXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  for (auto act_type : {"relu"}) {
    found_subgraph_count += ApplyImpl(graph, act_type);
  }
  AddStatis(found_subgraph_count);
}

int BNActXPUFusePass::ApplyImpl(ir::Graph* graph,
                                const std::string& act_type) const {
  GraphPatternDetector gpd;
  patterns::BNActXPUPattern pattern(
      gpd.mutable_pattern(), name_scope_, act_type);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle BNActXPUFusePass fuse";
    // declare operator node's name
    GET_IR_NODE(bn);
    GET_IR_NODE(act);
    // declare variable node's name
    GET_IR_NODE(bn_input);
    GET_IR_NODE(bn_bias);
    GET_IR_NODE(bn_mean);
    GET_IR_NODE(bn_scale);
    GET_IR_NODE(bn_var);
    GET_IR_NODE(bn_out);
    GET_IR_NODE(act_out);
    auto* block = bn->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));
    std::string fused_op_out_name;
    fused_op_out_name = act_out->Name();
    // Generate fused op
    framework::OpDesc fused_op_desc(block);
    fused_op_desc.SetType("bn_act_xpu");
    // set attrs for fused op
    fused_op_desc.SetAttr("act_type", ConvertActivationType(act_type));
    fused_op_desc.SetAttr("momentum",
                          bn->Op()->GetAttrIfExists<float>("momentum"));
    fused_op_desc.SetAttr("epsilon",
                          bn->Op()->GetAttrIfExists<float>("epsilon"));
    fused_op_desc.SetAttr(
        "data_layout", bn->Op()->GetAttrIfExists<std::string>("data_layout"));
    fused_op_desc.SetInput("x", {bn_input->Name()});
    fused_op_desc.SetInput("bias", {bn_bias->Name()});
    fused_op_desc.SetInput("mean", {bn_mean->Name()});
    fused_op_desc.SetInput("scale", {bn_scale->Name()});
    fused_op_desc.SetInput("variance", {bn_var->Name()});
    fused_op_desc.SetOutput("out", {fused_op_out_name});
    // relink fused op
    auto* fused_op = graph->CreateOpNode(&fused_op_desc);
    IR_NODE_LINK_TO(bn_input, fused_op);
    IR_NODE_LINK_TO(bn_bias, fused_op);
    IR_NODE_LINK_TO(bn_mean, fused_op);
    IR_NODE_LINK_TO(bn_scale, fused_op);
    IR_NODE_LINK_TO(bn_var, fused_op);
    IR_NODE_LINK_TO(fused_op, act_out);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {bn, act, bn_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(bn_act_xpu_fuse_pass, paddle::framework::ir::BNActXPUFusePass);

REGISTER_PASS_CAPABILITY(bn_act_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "bn_act_xpu", 0));
