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
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void ConvActivationMkldnnFusePass::ApplyImpl(Graph* graph) const {
  auto act_types = phi::funcs::GetSupportedActivations();
  std::vector<std::string> conv_types = {"conv2d", "fused_conv2d"};

  for (auto& act_type : act_types) {
    FuseConvConcatAct(graph, act_type);
    for (const auto& conv_type : conv_types) {
      FuseConvAct(graph, conv_type, act_type);
    }
  }
}

void ConvActivationMkldnnFusePass::FuseConvAct(Graph* graph,
                                               const std::string& conv_type,
                                               std::string& act_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(conv_type + "_" + act_type + "_mkldnn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::OperatorActivation conv_act_pattern(gpd.mutable_pattern(),
                                                "conv_activation_mkldnn_fuse");
  conv_act_pattern(conv_type, act_type);

  int found_conv_activation_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "conv_activation_mkldnn_fuse_pass op compat failed.";
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(conv, preceding_op, conv_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, preceding_op_out, conv_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(activation, activation, conv_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(activation_out, activation_out, conv_act_pattern);

    OpDesc* conv_op = conv->Op();
    OpDesc* act_op = activation->Op();

    auto attr_map = phi::funcs::GetAttributeMap(act_type);
    for (const auto& attrs : attr_map) {
      if (act_op->HasAttr(attrs.first)) {
        conv_op->SetAttr(attrs.second, act_op->GetAttr(attrs.first));
      }
    }

    if (act_type == "gelu" && activation->Op()->HasAttr("approximate")) {
      act_type =
          PADDLE_GET_CONST(bool, activation->Op()->GetAttr("approximate"))
              ? "gelu_tanh"
              : "gelu_erf";
      conv_op->SetAttr("fuse_alpha", 0.0f);
      conv_op->SetAttr("fuse_beta", 0.0f);
    }
    conv_op->SetAttr("fuse_activation", act_type);
    conv_op->SetOutput("Output", {activation_out->Name()});

    IR_NODE_LINK_TO(conv, activation_out);
    GraphSafeRemoveNodes(graph, {activation, conv_out});
    found_conv_activation_count++;
  };

  gpd(graph, handler);
  AddStatis(found_conv_activation_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      found_conv_activation_count > 0) {
    PrettyLogDetail("---    fused %d conv with %s activation",
                    found_conv_activation_count,
                    act_type);
  }
}

void ConvActivationMkldnnFusePass::FuseConvConcatAct(
    Graph* graph, std::string& act_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("conv2d_concat_" + act_type + "_mkldnn_fuse_pass", graph);

  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::OperatorActivation conv_concat_act(
      pattern, "conv2d_concat_" + act_type + "_mkldnn_fuse_pass");
  conv_concat_act("concat", act_type);

  int found_conv_concat_activation_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "conv_concat_activation_mkldnn_fuse_pass op compat failed.";
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(concat_op, preceding_op, conv_concat_act);
    GET_IR_NODE_FROM_SUBGRAPH(concat_out, preceding_op_out, conv_concat_act);
    GET_IR_NODE_FROM_SUBGRAPH(activation_op, activation, conv_concat_act);
    GET_IR_NODE_FROM_SUBGRAPH(activation_out, activation_out, conv_concat_act);

    auto concat_inputs = concat_op->inputs;
    for (auto node : concat_inputs) {
      auto prev_op_nodes = node->inputs;
      if (prev_op_nodes.size() != 1) {
        LOG(WARNING)
            << "Operator connected to concat can have only one output.";
        return;
      }

      bool is_not_conv_mkldnn =
          !(prev_op_nodes[0]->Op()->GetAttrIfExists<bool>("use_mkldnn"));
      if (prev_op_nodes[0]->Op()->Type() != "conv2d" || is_not_conv_mkldnn) {
        LOG(WARNING)
            << "This fuse pass supports only conv2d (mkldnn) + activation.";
        return;
      }
    }

    for (auto node : concat_inputs) {
      OpDesc* conv_op = node->inputs[0]->Op();
      OpDesc* act_op = activation_op->Op();

      auto attr_map = phi::funcs::GetAttributeMap(act_type);
      for (const auto& attrs : attr_map) {
        if (act_op->HasAttr(attrs.first)) {
          conv_op->SetAttr(attrs.second, act_op->GetAttr(attrs.first));
        }
      }

      if (act_type == "gelu" && act_op->HasAttr("approximate")) {
        act_type = PADDLE_GET_CONST(bool, act_op->GetAttr("approximate"))
                       ? "gelu_tanh"
                       : "gelu_erf";
        conv_op->SetAttr("fuse_alpha", 0.0f);
        conv_op->SetAttr("fuse_beta", 0.0f);
      }
      conv_op->SetAttr("fuse_activation", act_type);
    }

    concat_op->Op()->SetOutput("Out", {activation_out->Name()});
    GraphSafeRemoveNodes(graph, {activation_op, concat_out});
    IR_NODE_LINK_TO(concat_op, activation_out);

    found_conv_concat_activation_count++;
  };
  gpd(graph, handler);
  AddStatis(found_conv_concat_activation_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      found_conv_concat_activation_count > 0) {
    PrettyLogDetail("---    fused %d conv_concat with %s activation",
                    found_conv_concat_activation_count,
                    act_type);
  }
}

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
      .AddAttr("data_format")
      .IsOptional()
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();

  AddOpCompat(OpCompat("fused_conv2d"))
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
      .AddAttr("data_format")
      .IsOptional()
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();

  AddOpCompat(OpCompat("concat"))
      .AddInput("X")
      .End()
      .AddInput("AxisTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumGE(0)
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

  AddOpCompat(OpCompat("tanh"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("sigmoid"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("sqrt"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("abs"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
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
            .EQ("concat", 0)
            .EQ("abs", 0)
            .LE("clip", 1)
            .EQ("gelu", 0)
            .EQ("hard_sigmoid", 0)
            .LE("hard_swish", 0)
            .LE("leaky_relu", 1)
            .LE("mish", 1)
            .EQ("relu", 0)
            .EQ("relu6", 0)
            .EQ("sigmoid", 0)
            .EQ("sqrt", 0)
            .EQ("swish", 0)
            .EQ("tanh", 0));
