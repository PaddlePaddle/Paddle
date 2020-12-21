// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/conv_activation_mkldnn_fuse_pass.h"

#include <vector>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

class Graph;

void ConvActivationFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("conv_activation_mkldnn_fuse", graph);

  GraphPatternDetector gpd;
  auto* conv_input = gpd.mutable_pattern()
                         ->NewNode("conv_activation_mkldnn_fuse/conv_input")
                         ->AsInput()
                         ->assert_is_op_input(conv_type(), "Input");
  patterns::ConvActivation conv_activation_pattern(
      gpd.mutable_pattern(), "conv_activation_mkldnn_fuse");
  conv_activation_pattern(conv_input, conv_type(), activation_type());

  int found_conv_activation_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle " + conv_type() + "+" + activation_type() + " fuse";
    GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight,
                              conv_activation_pattern);  // Filter
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out,
                              conv_activation_pattern);              // tmp
    GET_IR_NODE_FROM_SUBGRAPH(conv, conv, conv_activation_pattern);  // CONV op
    GET_IR_NODE_FROM_SUBGRAPH(activation_out, activation_out,
                              conv_activation_pattern);  // Out
    GET_IR_NODE_FROM_SUBGRAPH(activation, activation,
                              conv_activation_pattern);  // Activation op

    // Transform Conv node into ConvActivation node.
    OpDesc* desc = conv->Op();
    desc->SetOutput("Output",
                    std::vector<std::string>({activation_out->Name()}));

    desc->SetAttr("fuse_activation", activation_type());

    // MKLDNN ops use alpha and beta as activation parameters but paddle ops are
    // not generalized
    if (activation_type() == "relu6") {
      desc->SetAttr(
          "fuse_alpha",
          BOOST_GET_CONST(float, activation->Op()->GetAttr("threshold")));
    } else if (activation_type() == "swish") {
      // paddle uses beta but mkldnn uses alpha for swish
      desc->SetAttr("fuse_alpha",
                    activation->Op()->GetAttrIfExists<float>("beta"));
    } else {
      desc->SetAttr("fuse_alpha",
                    activation->Op()->GetAttrIfExists<float>("alpha"));
    }
    desc->SetAttr("fuse_beta",
                  activation->Op()->GetAttrIfExists<float>("beta"));

    GraphSafeRemoveNodes(graph, {activation, conv_out});

    PADDLE_ENFORCE_GT(subgraph.count(conv_input), 0UL,
                      platform::errors::InvalidArgument(
                          "Subgraph has to contain conv input node."));
    IR_NODE_LINK_TO(conv, activation_out);
    found_conv_activation_count++;
  };

  gpd(graph, handler);

  AddStatis(found_conv_activation_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_activation_mkldnn_fuse_pass,
              paddle::framework::ir::ConvActivationFusePass);

REGISTER_PASS(conv_relu_mkldnn_fuse_pass,
              paddle::framework::ir::ConvActivationFusePass);
REGISTER_PASS_CAPABILITY(conv_relu_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("relu", 0));

REGISTER_PASS(conv_leaky_relu_mkldnn_fuse_pass,
              paddle::framework::ir::Conv2DLeakyReLUFusePass);
REGISTER_PASS_CAPABILITY(conv_leaky_relu_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .LE("leaky_relu", 1));

REGISTER_PASS(conv_relu6_mkldnn_fuse_pass,
              paddle::framework::ir::Conv2DReLU6FusePass);
REGISTER_PASS_CAPABILITY(conv_relu6_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("relu6", 0));

REGISTER_PASS(conv_swish_mkldnn_fuse_pass,
              paddle::framework::ir::Conv2DSwishFusePass);
REGISTER_PASS_CAPABILITY(conv_swish_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("swish", 0));
