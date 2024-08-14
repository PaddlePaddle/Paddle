/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/trt_skip_layernorm_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/helper.h"
#endif

namespace paddle::framework::ir {
class Node;
}  // namespace paddle::framework::ir

namespace paddle::framework::ir::patterns {

struct TrtSkipLayerNorm : public PatternBase {
  TrtSkipLayerNorm(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "skip_layernorm") {}

  PDNode *operator()(PDNode *x, PDNode *y);

  // declare operator node's name
  PATTERN_DECL_NODE(elementwise);
  PATTERN_DECL_NODE(layer_norm);
  // declare variable node's name
  PATTERN_DECL_NODE(
      elementwise_out);  // (elementwise_input_x,elementwise_input_y)
                         // -> elementwise_out
  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_out);
  PATTERN_DECL_NODE(layer_norm_mean);
  PATTERN_DECL_NODE(layer_norm_variance);
};

PDNode *TrtSkipLayerNorm::operator()(PDNode *x, PDNode *y) {
  // Create nodes for elementwise add op.
  x->assert_is_op_input("elementwise_add", "X");
  y->assert_is_op_input("elementwise_add", "Y");
  auto *elementwise =
      pattern->NewNode(elementwise_repr())->assert_is_op("elementwise_add");
  auto *elementwise_out_var =
      pattern->NewNode(elementwise_out_repr())
          ->AsOutput()
          ->assert_is_only_output_of_op("elementwise_add");

  // Add links for elementwise_add op.
  elementwise->LinksFrom({x, y}).LinksTo({elementwise_out_var});

  // Create nodes for layer_norm op.
  elementwise_out_var->AsIntermediate()->assert_is_op_input("layer_norm");
  auto *layer_norm =
      pattern->NewNode(layer_norm_repr())->assert_is_op("layer_norm");
  auto *layer_norm_bias_var = pattern->NewNode(layer_norm_bias_repr())
                                  ->AsInput()
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("layer_norm", "Bias");
  auto *layer_norm_scale_var = pattern->NewNode(layer_norm_scale_repr())
                                   ->AsInput()
                                   ->assert_is_persistable_var()
                                   ->assert_is_op_input("layer_norm", "Scale");

  auto *layer_norm_out_var = pattern->NewNode(layer_norm_out_repr())
                                 ->AsOutput()
                                 ->assert_is_op_output("layer_norm", "Y");
  auto *layer_norm_mean_var = pattern->NewNode(layer_norm_mean_repr())
                                  ->AsOutput()
                                  ->assert_is_op_output("layer_norm", "Mean");
  auto *layer_norm_variance_var =
      pattern->NewNode(layer_norm_variance_repr())
          ->AsOutput()
          ->assert_is_op_output("layer_norm", "Variance");

  // Add links for layer_norm op.
  layer_norm
      ->LinksFrom(
          {elementwise_out_var, layer_norm_bias_var, layer_norm_scale_var})
      .LinksTo(
          {layer_norm_out_var, layer_norm_mean_var, layer_norm_variance_var});
  return layer_norm_out_var;
}

}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

void TrtSkipLayerNormFusePass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("skip_layernorm_fuse", graph);

#ifdef PADDLE_WITH_TENSORRT
  auto trt_version = paddle::inference::tensorrt::GetTrtRuntimeVersion();
  if (std::get<0>(trt_version) * 1000 + std::get<1>(trt_version) * 100 +
          std::get<2>(trt_version) * 10 <
      7200) {
    VLOG(3) << "skip_layernorm oss plugin only available for trt version >= "
               "7.2 Stop this pass";
    return;
  }
#else
  // if no tensorrt, early stop
  return;
#endif
  int found_subgraph_count = 0;

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode("skip_layernorm_fuse/x")
                ->AsInput()
                ->assert_is_op_input("elementwise_add", "X")
                ->assert_var_not_persistable();
  auto *y = gpd.mutable_pattern()
                ->NewNode("skip_layernorm_fuse/y")
                ->AsInput()
                ->assert_is_op_input("elementwise_add", "Y")
                ->assert_var_not_persistable();
  patterns::TrtSkipLayerNorm fused_pattern(gpd.mutable_pattern(),
                                           "skip_layernorm_fuse");
  fused_pattern(x, y);

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(x) <= 0 || subgraph.count(y) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }

    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "skip_layernorm pass in op compat failed.";
      return;
    }

    VLOG(4) << "handle TrtSkipLayerNorm fuse";

    // x and y 's rank must be same
    if (subgraph.at(x)->Var()->GetShape().size() !=
        subgraph.at(y)->Var()->GetShape().size()) {
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(elementwise, elementwise, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_out, elementwise_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm, layer_norm, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_bias, layer_norm_bias, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_scale, layer_norm_scale, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_out, layer_norm_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_mean, layer_norm_mean, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_variance, layer_norm_variance, fused_pattern);

    std::unordered_set<const Node *> del_node_set;

    // Create an TrtSkipLayerNorm op node
    OpDesc new_desc(elementwise->Op()->Block());
    new_desc.SetType("skip_layernorm");

    // inputs
    new_desc.SetInput("X", {subgraph.at(x)->Name()});
    new_desc.SetInput("Y", {subgraph.at(y)->Name()});
    new_desc.SetInput("Scale", {layer_norm_scale->Name()});
    new_desc.SetInput("Bias", {layer_norm_bias->Name()});

    if (layer_norm->Op()->HasAttr("out_threshold")) {
      new_desc.SetAttr("out_threshold",
                       layer_norm->Op()->GetAttr("out_threshold"));
    }
    if (subgraph.at(x)->inputs[0]->Op()->HasAttr("out_threshold")) {
      new_desc.SetAttr(
          "X", subgraph.at(x)->inputs[0]->Op()->GetAttr("out_threshold"));
    }
    if (subgraph.at(y)->inputs[0]->Op()->HasAttr("out_threshold")) {
      new_desc.SetAttr(
          "Y", subgraph.at(y)->inputs[0]->Op()->GetAttr("out_threshold"));
    }

    if (layer_norm->Op()->HasAttr("smooth_scale")) {
      new_desc.SetAttr("smooth_scale",
                       layer_norm->Op()->GetAttr("smooth_scale"));
    }

    // outputs
    new_desc.SetOutput("Out", {layer_norm_out->Name()});

    // attrs
    new_desc.SetAttr("epsilon", layer_norm->Op()->GetAttr("epsilon"));

    if (layer_norm->Op()->HasAttr("begin_norm_axis")) {
      int32_t begin_norm_axis = PADDLE_GET_CONST(
          int32_t, layer_norm->Op()->GetAttr("begin_norm_axis"));
      int32_t input_rank =
          static_cast<int32_t>(elementwise_out->Var()->GetShape().size());
      if ((begin_norm_axis != -1) && (begin_norm_axis != input_rank - 1)) {
        LOG(WARNING) << "skip_layernorm pass only support "
                        "layer_norm'begin_norm_axis == input_rank - 1.";
        return;
      }
      new_desc.SetAttr("begin_norm_axis", begin_norm_axis);
    }
    int32_t hidden_size =
        static_cast<int32_t>(layer_norm_scale->Var()->GetShape()[0]);
    new_desc.SetAttr("hidden_size", hidden_size);

    auto fused_node = graph->CreateOpNode(&new_desc);  // OpDesc will be copied.

    del_node_set.insert(elementwise);
    del_node_set.insert(layer_norm);
    del_node_set.insert(elementwise_out);
    del_node_set.insert(layer_norm_mean);
    del_node_set.insert(layer_norm_variance);
    GraphSafeRemoveNodes(graph, del_node_set);

    IR_NODE_LINK_TO(subgraph.at(x), fused_node);
    IR_NODE_LINK_TO(subgraph.at(y), fused_node);
    IR_NODE_LINK_TO(layer_norm_scale, fused_node);
    IR_NODE_LINK_TO(layer_norm_bias, fused_node);
    IR_NODE_LINK_TO(fused_node, layer_norm_out);

    found_subgraph_count++;
  };

  gpd(graph, handler);
  if (found_subgraph_count > 0) {
    bool use_varseqlen = Get<bool>("use_varseqlen");
    std::string pos_id = Get<std::string>("tensorrt_transformer_posid");
    std::string mask_id = Get<std::string>("tensorrt_transformer_maskid");

    if (use_varseqlen && !pos_id.empty() && !mask_id.empty()) {
      if ((graph->Has(framework::ir::kEmbEltwiseLayernormPass) ||
           graph->Has(framework::ir::kPrelnEmbEltwiseLayernormPass)) &&
          graph->Has(framework::ir::kMultiheadMatmulPass)) {
        VLOG(3) << "start varseqlen trt_skip_layernorm_fuse_pass";
      } else {
        PADDLE_THROW(common::errors::Fatal(
            "Use transformer'varseqlen need "
            "trt_embedding_eltwise_layernorm_fuse_pass, "
            "trt_multihead_matmul_fuse_pass. please use no_varseqlen"));
      }
    } else if (!use_varseqlen && pos_id.empty()) {
      VLOG(3) << "start no_varseqlen trt_skip_layernorm_fuse_pass";
    } else {
      PADDLE_THROW(
          common::errors::Fatal("Use transformer'varseqlen need config: "
                                "use_varseqlen, set pos_id, set "
                                "mask_id. Or not use varseqlen, do not set "
                                "pos_id. Please "
                                "reconfig"));
    }
  }
  AddStatis(found_subgraph_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(trt_skip_layernorm_fuse_pass,
              paddle::framework::ir::TrtSkipLayerNormFusePass);
REGISTER_PASS_CAPABILITY(trt_skip_layernorm_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .EQ("layer_norm", 0));
