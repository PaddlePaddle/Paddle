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

#include "paddle/fluid/framework/ir/conv_relu_mkldnn_fuse_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> ConvReLUFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("conv_relu_mkldnn_fuse", graph.get());

  std::unordered_set<Node*> nodes2delete;

  GraphPatternDetector gpd;
  auto* conv_input = gpd.mutable_pattern()
                         ->NewNode("conv_relu_mkldnn_fuse/conv_input")
                         ->AsInput()
                         ->assert_is_op_input("conv2d", "Input");
  patterns::ConvReLU conv_relu_pattern(gpd.mutable_pattern(),
                                       "conv_relu_mkldnn_fuse");
  conv_relu_pattern(conv_input);

  int found_conv_relu_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle ConvReLU fuse";
    GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight,
                              conv_relu_pattern);  // Filter
    GET_IR_NODE_FROM_SUBGRAPH(conv_bias, conv_bias, conv_relu_pattern);  // Bias
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, conv_relu_pattern);    // tmp
    GET_IR_NODE_FROM_SUBGRAPH(conv, conv, conv_relu_pattern);  // CONV op
    GET_IR_NODE_FROM_SUBGRAPH(relu_out, relu_out, conv_relu_pattern);  // Out
    GET_IR_NODE_FROM_SUBGRAPH(relu, relu, conv_relu_pattern);  // ReLU op

    // Create an ConvReLU Node.
    OpDesc desc;
    std::string conv_relu_i_in = subgraph.at(conv_input)->Name();
    std::string conv_relu_w_in = conv_weight->Name();
    std::string conv_relu_b_in = conv_bias->Name();
    std::string conv_relu_out = relu_out->Name();
    desc.SetInput("Input", std::vector<std::string>({conv_relu_i_in}));
    desc.SetInput("Filter", std::vector<std::string>({conv_relu_w_in}));
    desc.SetInput("Bias", std::vector<std::string>({conv_relu_b_in}));
    desc.SetOutput("Output", std::vector<std::string>({conv_relu_out}));
    desc.SetType("conv2d");
    for (auto& attr : conv->Op()->GetAttrMap()) {
      desc.SetAttr(attr.first, attr.second);
    }
    desc.SetAttr("fuse_relu", true);
    auto conv_relu_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
    GraphSafeRemoveNodes(graph.get(), {conv, relu, conv_out});

    PADDLE_ENFORCE(subgraph.count(conv_input));
    IR_NODE_LINK_TO(subgraph.at(conv_input), conv_relu_node);
    IR_NODE_LINK_TO(conv_weight, conv_relu_node);
    IR_NODE_LINK_TO(conv_bias, conv_relu_node);
    IR_NODE_LINK_TO(conv_relu_node, relu_out);

    found_conv_relu_count++;
  };

  gpd(graph.get(), handler);

  AddStatis(found_conv_relu_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_relu_mkldnn_fuse_pass,
              paddle::framework::ir::ConvReLUFusePass);
