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

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "conv_activation_mkldnn_fuse_pass op compat failed.";
      return;
    }
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

    if (activation_type() == "gelu" &&
        activation->Op()->HasAttr("approximate")) {
      bool approximate =
          BOOST_GET_CONST(bool, activation->Op()->GetAttr("approximate"));
      std::string type = approximate ? "_tanh" : "_erf";
      desc->SetAttr("fuse_activation", "gelu" + type);
    } else {
      desc->SetAttr("fuse_activation", activation_type());
    }

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

    if (activation_type() == "hard_sigmoid") {
      desc->SetAttr("fuse_alpha",
                    activation->Op()->GetAttrIfExists<float>("slope"));
      desc->SetAttr("fuse_beta",
                    activation->Op()->GetAttrIfExists<float>("offset"));
    }

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

ConvActivationFusePass::ConvActivationFusePass() {
  AddOpCompat(OpCompat("conv2d"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsOptional()
      .IsTensor()
      .End()
      .AddInput("ResidualData")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("strides")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("paddings")
      .IsType<std::vector<int>>()
      .End()
      // IsStringIn({"EXPLICIT", "SAME", "VALID"}), MobileNetV2 has no this
      // attribute
      .AddAttr("padding_algorithm")
      .IsOptional()
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .IsType<std::vector<int>>()
      .End()
      // IsStringIn({"NHWC", "NCHW"}) MobileNetV2 has no this attribute
      .AddAttr("data_format")
      .IsOptional()
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();

  AddOpCompat(OpCompat("relu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
}
Conv2DLeakyReLUFusePass::Conv2DLeakyReLUFusePass() {
  AddOpCompat(OpCompat("leaky_relu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      // float, default=0.02
      .AddAttr("alpha")
      .IsType<float>()
      .End();
}
Conv2DReLU6FusePass::Conv2DReLU6FusePass() {
  AddOpCompat(OpCompat("relu6"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      // default = 6.0f
      .AddAttr("threshold")
      .IsType<float>()
      .End();
}
Conv2DSwishFusePass::Conv2DSwishFusePass() {
  AddOpCompat(OpCompat("swish"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("beta")
      .IsType<float>()
      .End();
}
Conv2DHardSwishFusePass::Conv2DHardSwishFusePass() {
  AddOpCompat(OpCompat("hard_swish"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      // float, optional, default=6.0
      .AddAttr("threshold")
      .IsOptional()
      .IsType<float>()
      .End()
      // float, optional, default=6.0
      .AddAttr("scale")
      .IsOptional()
      .IsType<float>()
      .End()
      // float, optional, default=3.0
      .AddAttr("offset")
      .IsOptional()
      .IsType<float>()
      .End();
}
Conv2DMishFusePass::Conv2DMishFusePass() {
  AddOpCompat(OpCompat("mish"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
}
Conv2DHardSigmoidFusePass::Conv2DHardSigmoidFusePass() {
  AddOpCompat(OpCompat("hard_sigmoid"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      // optional, default=0.2
      .AddAttr("slope")
      .IsOptional()
      .IsType<float>()
      .End()
      // optional, default=0.5
      .AddAttr("offset")
      .IsOptional()
      .IsType<float>()
      .End();
}

Conv2DGeluFusePass::Conv2DGeluFusePass() {
  AddOpCompat(OpCompat("gelu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("approximate")
      .IsType<bool>()
      .End();
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

REGISTER_PASS(conv_hard_swish_mkldnn_fuse_pass,
              paddle::framework::ir::Conv2DHardSwishFusePass);
REGISTER_PASS_CAPABILITY(conv_hard_swish_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("hard_swish", 0));

REGISTER_PASS(conv_mish_mkldnn_fuse_pass,
              paddle::framework::ir::Conv2DMishFusePass);
REGISTER_PASS_CAPABILITY(conv_mish_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("mish", 1));

REGISTER_PASS(conv_hard_sigmoid_mkldnn_fuse_pass,
              paddle::framework::ir::Conv2DHardSigmoidFusePass);
REGISTER_PASS_CAPABILITY(conv_hard_sigmoid_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("hard_sigmoid", 0));

REGISTER_PASS(conv_gelu_mkldnn_fuse_pass,
              paddle::framework::ir::Conv2DGeluFusePass);
REGISTER_PASS_CAPABILITY(conv_gelu_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("gelu", 0));
