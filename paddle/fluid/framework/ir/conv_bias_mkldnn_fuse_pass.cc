// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/ir/conv_bias_mkldnn_fuse_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/platform/enforce.h"
namespace paddle {
namespace framework {
namespace ir {
std::unique_ptr<ir::Graph> ConvBiasFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("conv_bias_mkldnn_fuse", graph.get());
  GraphPatternDetector gpd;
  auto* conv_input = gpd.mutable_pattern()
                         ->NewNode("conv_bias_mkldnn_fuse/conv_input")
                         ->AsInput()
                         ->assert_is_op_input("conv2d", "Input");
  patterns::ConvBias conv_bias_pattern(gpd.mutable_pattern(),
                                       "conv_bias_mkldnn_fuse");
  conv_bias_pattern(conv_input);
  int found_conv_bias_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle ConvBias fuse";
    GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight,
                              conv_bias_pattern);                      // Filter
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, conv_bias_pattern);  // tmp
    GET_IR_NODE_FROM_SUBGRAPH(conv, conv, conv_bias_pattern);  // CONV op
    // bias
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_bias, eltwise_bias, conv_bias_pattern);
    // output
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_out, eltwise_out, conv_bias_pattern);
    // elementwise_add op
    GET_IR_NODE_FROM_SUBGRAPH(eltwise, eltwise, conv_bias_pattern);
    // Create an ConvBias Node.
    OpDesc desc;
    std::string conv_bias_i_in = subgraph.at(conv_input)->Name();
    std::string conv_bias_w_in = conv_weight->Name();
    std::string conv_bias_b_in = eltwise_bias->Name();
    std::string conv_bias_out = eltwise_out->Name();
    desc.SetInput("Input", std::vector<std::string>({conv_bias_i_in}));
    desc.SetInput("Filter", std::vector<std::string>({conv_bias_w_in}));
    desc.SetInput("Bias", std::vector<std::string>({conv_bias_b_in}));
    desc.SetOutput("Output", std::vector<std::string>({conv_bias_out}));
    desc.SetType("conv2d");
    for (auto& attr : conv->Op()->GetAttrMap()) {
      desc.SetAttr(attr.first, attr.second);
    }
    auto conv_bias_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
    GraphSafeRemoveNodes(graph.get(), {conv, eltwise, conv_out});
    PADDLE_ENFORCE(subgraph.count(conv_input));
    IR_NODE_LINK_TO(subgraph.at(conv_input), conv_bias_node);
    IR_NODE_LINK_TO(conv_weight, conv_bias_node);
    IR_NODE_LINK_TO(eltwise_bias, conv_bias_node);
    IR_NODE_LINK_TO(conv_bias_node, eltwise_out);
    found_conv_bias_count++;
  };
  gpd(graph.get(), handler);
  AddStatis(found_conv_bias_count);
  return graph;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
REGISTER_PASS(conv_bias_mkldnn_fuse_pass,
              paddle::framework::ir::ConvBiasFusePass);
