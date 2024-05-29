/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/framework/ir/trans_layernorm_fuse_pass.h"
#include <string>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::framework::ir {
class Node;
}  // namespace paddle::framework::ir 

namespace paddle::framework::ir::patterns {
struct TransLayernormPattern : public PatternBase {
  TransLayernormPattern(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "trans_layernorm") {}
  void operator()(PDNode *x);
  PATTERN_DECL_NODE(transpose);
  PATTERN_DECL_NODE(transpose_output);
  PATTERN_DECL_NODE(reshape);
  PATTERN_DECL_NODE(reshape_output);
  PATTERN_DECL_NODE(layernorm);
  PATTERN_DECL_NODE(layernorm_scale);
  PATTERN_DECL_NODE(layernorm_bias);
  PATTERN_DECL_NODE(layernorm_output);
};
void TransLayernormPattern::operator()(PDNode *x) {
  std::unordered_set<std::string> reshape_ops{"reshape2",
                                              "flatten_contiguous_range"};
  auto *transpose =
      pattern->NewNode(transpose_repr())->assert_is_op("transpose2");
  auto *transpose_output = pattern->NewNode(transpose_output_repr())
                               ->assert_is_op_output("transpose2")
                               ->assert_is_ops_input(reshape_ops, "X");
  transpose->LinksFrom({x}).LinksTo({transpose_output});
  auto *reshape = pattern->NewNode(reshape_repr())->assert_is_ops(reshape_ops);
  auto *reshape_output = pattern->NewNode(reshape_output_repr())
                             ->assert_is_ops_output(reshape_ops, "Out")
                             ->assert_is_op_input("layer_norm", "X")
                             ->AsOutput();
  reshape->LinksFrom({transpose_output}).LinksTo({reshape_output});
  auto *layernorm =
      pattern->NewNode(layernorm_repr())->assert_is_op("layer_norm");
  auto *layernorm_scale = pattern->NewNode(layernorm_scale_repr())
                              ->assert_is_op_input("layer_norm", "Scale")
                              ->AsInput();
  auto *layernorm_bias = pattern->NewNode(layernorm_bias_repr())
                             ->assert_is_op_input("layer_norm", "Bias")
                             ->AsInput();
  auto *layernorm_output = pattern->NewNode(layernorm_output_repr())
                               ->assert_is_op_output("layer_norm", "Y")
                               ->AsOutput();
  layernorm->LinksFrom({reshape_output, layernorm_scale, layernorm_bias})
      .LinksTo({layernorm_output});
}
}  // namespace paddle::framework::ir::patterns 
namespace paddle::framework::ir {

// this pass make a fusion as below:
//
//       |
//   transpose(axis= [0,2,3,1])
//       |
//    reshape(n,h*w,c)
//    |    |
//   out  layernorm(begin_norm_axis=2 or -1)
//         |
//        layernorm_out
//
// ->fuse to
//
//     |
//  trans_layernorm
//   |      |
//  out    layernorm_out
//

int TransLayernormFusePass::ApplyConvTransLayernormPattern(
    ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("trans_layernorm_fuse", graph);
  int found_subgraph_count = 0;
  GraphPatternDetector gpd;

  PDNode *x = nullptr;
  x = gpd.mutable_pattern()
          ->NewNode("trans_layernorm_fuse/x")
          ->AsInput()
          ->assert_var_not_persistable()
          ->assert_is_op_input("transpose2", "X");
  patterns::TransLayernormPattern fused_pattern(gpd.mutable_pattern(),
                                                "trans_layernorm_fuse");
  fused_pattern(x);
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(x) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }
    VLOG(4) << "handle transpose layernorm fuse";
    GET_IR_NODE_FROM_SUBGRAPH(transpose, transpose, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose_output, transpose_output, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape, reshape, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_output, reshape_output, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layernorm, layernorm, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layernorm_scale, layernorm_scale, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layernorm_bias, layernorm_bias, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layernorm_output, layernorm_output, fused_pattern);

    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "transpose layernorm pass in op compat failed.";
      return;
    }
    // trans_layernorm is suit for nchw-to-nhwc transpose before layernorm
    // check for it
    std::vector<int> trans_axis =
        PADDLE_GET_CONST(std::vector<int>, transpose->Op()->GetAttr("axis"));
    if (trans_axis != std::vector<int>{0, 2, 3, 1}) {
      VLOG(1) << "transpose layernorm fuse pass, transpose axis check fail, "
                 "stop fusion";
      return;
    }
    if (reshape->Op()->Type() == "flatten_contiguous_range") {
      int start_axis =
          PADDLE_GET_CONST(int, reshape->Op()->GetAttr("start_axis"));
      int stop_axis =
          PADDLE_GET_CONST(int, reshape->Op()->GetAttr("stop_axis"));
      if (!(start_axis == 1 && stop_axis == 2)) {
        VLOG(1) << "transpose layernorm fuse pass, flatten axis check fail, "
                   "stop fusion";
        return;
      }
    } else if (reshape->Op()->Type() == "reshape2") {
      std::vector<int> reshape_shape =
          PADDLE_GET_CONST(std::vector<int>, reshape->Op()->GetAttr("shape"));
      if (reshape_shape.size() != 3) {
        VLOG(1)
            << "transpose layernorm fuse pass, reshape check fail, stop fusion";
        return;
      }
    }
    auto layernorm_begin_norm_axis =
        PADDLE_GET_CONST(int, layernorm->Op()->GetAttr("begin_norm_axis"));
    if (layernorm_begin_norm_axis != 2 && layernorm_begin_norm_axis != -1) {
      VLOG(1) << "transpose layernorm fuse pass, layernorm begin norm axis "
                 "check fail, stop fusion";
      return;
    }
    std::unordered_set<const Node *> del_node_set;
    // Create an preln_groupnorm_act op node
    OpDesc new_desc(*layernorm->Op());
    new_desc.SetType("trans_layernorm");
    new_desc.SetInput("X", {subgraph.at(x)->Name()});
    new_desc.SetOutput("Out_reshape", {reshape_output->Name()});
    new_desc.SetOutput("Out_layernorm", {layernorm_output->Name()});
    new_desc.RemoveOutput("Y");
    new_desc.Flush();
    auto fused_node = graph->CreateOpNode(&new_desc);  // OpDesc will be copied.
    del_node_set.insert(transpose);
    del_node_set.insert(transpose_output);
    del_node_set.insert(reshape);
    del_node_set.insert(layernorm);
    GraphSafeRemoveNodes(graph, del_node_set);

    IR_NODE_LINK_TO(subgraph.at(x), fused_node);
    IR_NODE_LINK_TO(layernorm_scale, fused_node);
    IR_NODE_LINK_TO(layernorm_bias, fused_node);
    IR_NODE_LINK_TO(fused_node, reshape_output);
    IR_NODE_LINK_TO(fused_node, layernorm_output);
    found_subgraph_count++;
  };
  gpd(graph, handler);
  return found_subgraph_count;
}

void TransLayernormFusePass::ApplyImpl(ir::Graph *graph) const {
  FusePassBase::Init("trans_layernorm_fuse_pass", graph);
  int found_subgraph_count = ApplyConvTransLayernormPattern(graph);
  AddStatis(found_subgraph_count);
}

}  // namespace paddle::framework::ir 

REGISTER_PASS(trans_layernorm_fuse_pass,
              paddle::framework::ir::TransLayernormFusePass);
