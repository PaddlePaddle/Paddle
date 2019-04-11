/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/mkldnn/depthwise_conv_mkldnn_pass.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

#define GET_NODE(id, pattern)                               \
  PADDLE_ENFORCE(subgraph.count(pattern.RetrieveNode(#id)), \
                 "pattern has no Node called %s", #id);     \
  auto* id = subgraph.at(pattern.RetrieveNode(#id));        \
  PADDLE_ENFORCE_NOT_NULL(id, "subgraph has no node %s", #id);

void DepthwiseConvMKLDNNPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE(graph);
  FusePassBase::Init("depthwise_conv_mkldnn_pass", graph);
  GraphPatternDetector gpd;

  auto* pattern = gpd.mutable_pattern();
  pattern->NewNode("depthwise_conv")
      ->assert_is_op("depthwise_conv2d")
      ->assert_op_attr("use_mkldnn", true);

  int found_depthwise_conv_mkldnn_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(3) << "handle DepthwiseConvMKLDNN fuse";
    GET_NODE(depthwise_conv, (*pattern));
    depthwise_conv->Op()->SetType("conv2d");
    found_depthwise_conv_mkldnn_count++;
  };

  gpd(graph, handler);
  AddStatis(found_depthwise_conv_mkldnn_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(depthwise_conv_mkldnn_pass,
              paddle::framework::ir::DepthwiseConvMKLDNNPass);
