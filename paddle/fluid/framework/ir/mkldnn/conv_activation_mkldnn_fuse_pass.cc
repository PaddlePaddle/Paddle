// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

ConvActivationMkldnnFusePass::ConvActivationMkldnnFusePass() {
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

  AddOpCompat(OpCompat("leaky_relu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("alpha")
      .IsType<float>()
      .End();

  AddOpCompat(OpCompat("relu6"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("threshold")
      .IsType<float>()
      .End();

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

  AddOpCompat(OpCompat("hard_swish"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("threshold")
      .IsOptional()
      .IsType<float>()
      .End()
      .AddAttr("scale")
      .IsOptional()
      .IsType<float>()
      .End()
      .AddAttr("offset")
      .IsOptional()
      .IsType<float>()
      .End();

  AddOpCompat(OpCompat("mish"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("hard_sigmoid"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("slope")
      .IsOptional()
      .IsType<float>()
      .End()
      .AddAttr("offset")
      .IsOptional()
      .IsType<float>()
      .End();

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

void ConvActivationMkldnnFusePass::ApplyImpl(Graph* graph) const {
  std::vector<std::string> act_types = {
      "relu", "mish",  "swish",   "hard_swish", "hard_sigmoid",
      "gelu", "relu6", "sigmoid", "leaky_relu"};

  std::vector<std::string> conv_types = {"conv2d"};

  for (const auto& conv_type : conv_types)
    for (auto& act_type : act_types) {
      FuseConvAct(graph, conv_type, act_type);
    }
}

void ConvActivationMkldnnFusePass::FuseConvAct(Graph* graph,
                                               const std::string& conv_type,
                                               std::string& act_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(name_scope_, graph);

  GraphPatternDetector gpd;
  auto* conv_input = gpd.mutable_pattern()
                         ->NewNode("conv_activation_mkldnn_fuse/conv_input")
                         ->AsInput()
                         ->assert_is_op_input(conv_type, "Input");
  patterns::ConvActivation conv_act_pattern(gpd.mutable_pattern(), name_scope_);
  conv_act_pattern(conv_input, conv_type, act_type);

  int found_conv_activation_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle " + conv_type + "+" + act_type + " fuse";

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "conv_activation_mkldnn_fuse_pass op compat failed.";
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight, conv_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, conv_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv, conv, conv_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(activation_out, activation_out, conv_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(activation, activation, conv_act_pattern);

    OpDesc* conv_op = conv->Op();
    conv_op->SetOutput("Output", {activation_out->Name()});

    if (act_type == "gelu" && activation->Op()->HasAttr("approximate")) {
      act_type = BOOST_GET_CONST(bool, activation->Op()->GetAttr("approximate"))
                     ? "gelu_tanh"
                     : "gelu_erf";
    }

    conv_op->SetAttr("fuse_activation", act_type);

    // MKLDNN ops use alpha and beta as activation parameters but paddle ops are
    // not generalized
    if (act_type == "relu6") {
      conv_op->SetAttr(
          "fuse_alpha",
          BOOST_GET_CONST(float, activation->Op()->GetAttr("threshold")));
    } else if (act_type == "hard_sigmoid") {
      conv_op->SetAttr("fuse_alpha",
                       activation->Op()->GetAttrIfExists<float>("slope"));
      conv_op->SetAttr("fuse_beta",
                       activation->Op()->GetAttrIfExists<float>("offset"));
    } else if (act_type == "swish") {
      conv_op->SetAttr("fuse_alpha",
                       activation->Op()->GetAttrIfExists<float>("beta"));
      conv_op->SetAttr("fuse_beta",
                       activation->Op()->GetAttrIfExists<float>("alpha"));
    } else {
      conv_op->SetAttr("fuse_alpha",
                       activation->Op()->GetAttrIfExists<float>("alpha"));
      conv_op->SetAttr("fuse_beta",
                       activation->Op()->GetAttrIfExists<float>("beta"));
    }

    IR_NODE_LINK_TO(conv, activation_out);
    GraphSafeRemoveNodes(graph, {activation, conv_out});
    found_conv_activation_count++;
  };

  gpd(graph, handler);
  AddStatis(found_conv_activation_count);
  if (!Has("disable_logs") || !Get<bool>("disable_logs")) {
    PrettyLogDetail("---    fused %d conv with %s activation",
                    found_conv_activation_count, act_type);
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_activation_mkldnn_fuse_pass,
              paddle::framework::ir::ConvActivationMkldnnFusePass);

REGISTER_PASS_CAPABILITY(conv_activation_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("gelu", 0)
            .EQ("hard_sigmoid", 0)
            .LE("hard_swish", 0)
            .LE("leaky_relu", 1)
            .EQ("mish", 1)
            .EQ("relu", 0)
            .EQ("relu6", 0)
            .EQ("sigmoid", 0)
            .EQ("swish", 0));
