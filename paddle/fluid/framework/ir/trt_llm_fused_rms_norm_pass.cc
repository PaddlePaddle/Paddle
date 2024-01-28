/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/trt_llm_fused_rms_norm_pass.h"

#include <string>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/helper.h"
#endif

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct RmsNorm : public PatternBase {
  RmsNorm(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "fused_rms_norm") {}
  void operator()();
  // declare node's name
  PATTERN_DECL_NODE(pow_input);
  PATTERN_DECL_NODE(pow);
  PATTERN_DECL_NODE(pow_out);

  PATTERN_DECL_NODE(reduce_mean);
  PATTERN_DECL_NODE(reduce_mean_out);

  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(scale_out);

  PATTERN_DECL_NODE(rsqrt);
  PATTERN_DECL_NODE(rsqrt_out);

  PATTERN_DECL_NODE(elementwise_mul0);
  PATTERN_DECL_NODE(elementwise_mul0_out);

  PATTERN_DECL_NODE(elementwise_mul1);
  PATTERN_DECL_NODE(elementwise_mul1_weight);
  PATTERN_DECL_NODE(elementwise_mul1_out);
};

void RmsNorm::operator()() {
  auto *pow_input =
      pattern->NewNode(pow_input_repr())->assert_is_op_input("pow");
  auto *pow = pattern->NewNode(pow_repr())->assert_is_op("pow");
  auto *pow_out = pattern->NewNode(pow_out_repr())->assert_is_op_output("pow");

  auto *reduce_mean =
      pattern->NewNode(reduce_mean_repr())->assert_is_op("reduce_mean");
  auto *reduce_mean_out = pattern->NewNode(reduce_mean_out_repr())
                              ->assert_is_op_output("reduce_mean");

  auto *scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
  auto *scale_out =
      pattern->NewNode(scale_out_repr())->assert_is_op_output("scale");

  auto *rsqrt = pattern->NewNode(rsqrt_repr())->assert_is_op("rsqrt");
  auto *rsqrt_out =
      pattern->NewNode(rsqrt_out_repr())->assert_is_op_output("rsqrt");

  auto *elementwise_mul0 = pattern->NewNode(elementwise_mul0_repr())
                               ->assert_is_op("elementwise_mul");
  auto *elementwise_mul0_out = pattern->NewNode(elementwise_mul0_out_repr())
                                   ->assert_is_op_output("elementwise_mul");

  auto *elementwise_mul1 = pattern->NewNode(elementwise_mul1_repr())
                               ->assert_is_op("elementwise_mul");
  auto *elementwise_mul1_weight =
      pattern->NewNode(elementwise_mul1_weight_repr())
          ->assert_is_op_input("elementwise_mul", "Y")
          ->assert_is_persistable_var();
  auto *elementwise_mul1_out = pattern->NewNode(elementwise_mul1_out_repr())
                                   ->assert_is_op_output("elementwise_mul")
                                   ->AsOutput();

  // Add links for nodes.
  pow->LinksFrom({pow_input}).LinksTo({pow_out});

  reduce_mean->LinksFrom({pow_out}).LinksTo({reduce_mean_out});

  scale->LinksFrom({reduce_mean_out}).LinksTo({scale_out});

  rsqrt->LinksFrom({scale_out}).LinksTo({rsqrt_out});

  elementwise_mul0->LinksFrom({rsqrt_out, pow_input})
      .LinksTo({elementwise_mul0_out});

  elementwise_mul1->LinksFrom({elementwise_mul0_out, elementwise_mul1_weight})
      .LinksTo({elementwise_mul1_out});
}
}  // namespace patterns

void TrtLLMRmsNormPass::ApplyImpl(ir::Graph *graph) const {
  bool use_tensorrt_llm = Get<bool>("use_tensorrt_llm");
  if (!use_tensorrt_llm) {
    VLOG(3) << "trt_llm_fused_rms_norm_pass need Predictor'Config "
               "EnableTensorRtLLM, skip";
    return;
  }
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("fused_rms_norm_fuse", graph);

  int found_subgraph_count = 0;
  GraphPatternDetector gpd;

  patterns::RmsNorm fused_pattern(gpd.mutable_pattern(), "fused_rms_norm");
  fused_pattern();

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    VLOG(3) << "handle RmsNorm fuse";

    GET_IR_NODE_FROM_SUBGRAPH(pow_input, pow_input, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(pow, pow, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(pow_out, pow_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reduce_mean, reduce_mean, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reduce_mean_out, reduce_mean_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale, scale, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(rsqrt, rsqrt, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(rsqrt_out, rsqrt_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_mul0, elementwise_mul0, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_mul0_out, elementwise_mul0_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_mul1, elementwise_mul1, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_mul1_weight, elementwise_mul1_weight, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_mul1_out, elementwise_mul1_out, fused_pattern);

    // input rank must be 2 or 3.
    if (pow_input->Var()->GetShape().size() != 2 &&
        pow_input->Var()->GetShape().size() != 3) {
      VLOG(3) << "RmsNorm input rank must be 2 or 3. skip";
      return;
    }

    std::unordered_set<const Node *> del_node_set;

    // Create an fused_rms_norm op node
    OpDesc new_desc(pow->Op()->Block());
    new_desc.SetType("fused_rms_norm");

    // inputs
    new_desc.SetInput("x", {pow_input->Name()});
    new_desc.SetInput("scale", {elementwise_mul1_weight->Name()});

    // outputs
    new_desc.SetOutput("y", {elementwise_mul1_out->Name()});

    // attrs
    new_desc.SetAttr("epsilon", scale->Op()->GetAttr("bias"));

    int32_t hidden_size =
        static_cast<int32_t>(pow_input->Var()->GetShape().back());
    new_desc.SetAttr("hidden_size", hidden_size);

    auto fused_node = graph->CreateOpNode(&new_desc);

    del_node_set.insert(pow);
    del_node_set.insert(pow_out);
    del_node_set.insert(reduce_mean);
    del_node_set.insert(reduce_mean_out);
    del_node_set.insert(scale);
    del_node_set.insert(scale_out);
    del_node_set.insert(rsqrt);
    del_node_set.insert(rsqrt_out);
    del_node_set.insert(elementwise_mul0);
    del_node_set.insert(elementwise_mul0_out);
    del_node_set.insert(elementwise_mul1);
    GraphSafeRemoveNodes(graph, del_node_set);

    IR_NODE_LINK_TO(pow_input, fused_node);
    IR_NODE_LINK_TO(elementwise_mul1_weight, fused_node);
    IR_NODE_LINK_TO(fused_node, elementwise_mul1_out);

    found_subgraph_count++;
  };
  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(trt_llm_fused_rms_norm_pass,
              paddle::framework::ir::TrtLLMRmsNormPass);
