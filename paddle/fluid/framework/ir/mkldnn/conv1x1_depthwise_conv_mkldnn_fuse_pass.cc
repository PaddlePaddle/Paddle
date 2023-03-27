// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/conv1x1_depthwise_conv_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

Conv1x1DepthwiseConvOneDNNFusePass::Conv1x1DepthwiseConvOneDNNFusePass() {
  AddOpCompat(OpCompat("conv2d"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .IsOptional()
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
      .IsStringIn({"NHWC", "NCHW", "AnyLayout"})
      .End();
}

static bool isValidForFusing(phi::DenseTensor* conv_filter,
                             phi::DenseTensor* dw_conv_filter,
                             OpDesc* dw_conv_op) {
  auto conv_filter_dims = phi::vectorize(conv_filter->dims());
  auto dw_conv_filter_dims = phi::vectorize(dw_conv_filter->dims());

  auto dw_conv_paddings =
      dw_conv_op->HasAttr("paddings")
          ? PADDLE_GET_CONST(std::vector<int>, dw_conv_op->GetAttr("paddings"))
          : std::vector<int>({0, 0});
  auto dw_conv_strides =
      dw_conv_op->HasAttr("strides")
          ? PADDLE_GET_CONST(std::vector<int>, dw_conv_op->GetAttr("strides"))
          : std::vector<int>({0, 0});
  int dw_groups = dw_conv_op->HasAttr("groups")
                      ? PADDLE_GET_CONST(int, dw_conv_op->GetAttr("groups"))
                      : 0;

  if (conv_filter_dims[2] != 1 || conv_filter_dims[3] != 1) {
    VLOG(4) << "Skipping because first conv filter is not 1x1";
    return false;
  }

  if (dw_conv_filter_dims[1] != 1) {
    VLOG(4) << "Skippping because second conv is not depthwise";
    return false;
  }

  if (dw_conv_filter_dims[2] != 3 || dw_conv_filter_dims[3] != 3) {
    VLOG(4) << "Skipping because depthwise filter is not 3x3";
    return false;
  }

  if (dw_conv_strides[0] > 2 || dw_conv_strides[1] > 2 ||
      dw_conv_strides[0] != dw_conv_strides[1]) {
    VLOG(4) << "Skipping because depthwise strides are not equal or in [1, 2]";
    return false;
  }

  if (dw_conv_paddings[0] != 1 || dw_conv_paddings[0] != 1) {
    VLOG(4) << "Skipping because depthwise paddings are not 1";
    return false;
  }

  if (dw_groups != dw_conv_filter_dims[0]) {
    VLOG(4) << "Skipping because depthwise groups are not equal to IC";
    return false;
  }

  return true;
}

void Conv1x1DepthwiseConvOneDNNFusePass::FuseConvDepthWise(bool with_bias,
                                                           Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::ConvDepthwiseConv conv_depthwise_conv_pattern(gpd.mutable_pattern(),
                                                          name_scope_);
  conv_depthwise_conv_pattern(with_bias);
  int found_conv_depthwise_conv_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Fuse conv with depthwise conv";
    GET_IR_NODE_FROM_SUBGRAPH(
        conv_weights, conv_weights, conv_depthwise_conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, conv_depthwise_conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(depthwise_conv_weights,
                              depthwise_conv_weights,
                              conv_depthwise_conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        depthwise_conv_out, depthwise_conv_out, conv_depthwise_conv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(conv, conv, conv_depthwise_conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        depthwise_conv, depthwise_conv, conv_depthwise_conv_pattern);

    auto* conv_op = conv->Op();
    auto* depthwise_conv_op = depthwise_conv->Op();

    if (!conv_op->HasAttr("use_mkldnn") ||
        !(PADDLE_GET_CONST(bool, conv_op->GetAttr("use_mkldnn")))) {
      VLOG(4) << "The conv1x1 + depthwise_conv fusion may happen only when "
                 "oneDNN library is used.";
      return;
    }

    if (conv_op->HasAttr("mkldnn_data_type") &&
        PADDLE_GET_CONST(std::string, conv_op->GetAttr("mkldnn_data_type")) !=
            "float32") {
      VLOG(4) << "The conv1x1 + depthwise_conv fusion is implemented only on "
                 "float32.";
      return;
    }

    auto* scope = param_scope();
    auto* conv_weights_tensor =
        scope->FindVar(conv_weights->Name())->GetMutable<phi::DenseTensor>();
    auto* depthwise_conv_weights_tensor =
        scope->FindVar(depthwise_conv_weights->Name())
            ->GetMutable<phi::DenseTensor>();

    if (!isValidForFusing(conv_weights_tensor,
                          depthwise_conv_weights_tensor,
                          depthwise_conv_op)) {
      return;
    }

    conv_op->SetInput("FilterDW", {depthwise_conv_weights->Name()});

    int stride =
        depthwise_conv_op->HasAttr("strides")
            ? PADDLE_GET_CONST(std::vector<int>,
                               depthwise_conv_op->GetAttr("strides"))[0]
            : 0;
    const std::string depthwise_type = "k3s" + std::to_string(stride) + "p1";

    conv_op->SetAttr("depthwise_type", depthwise_type);

    const auto fuse_activation =
        depthwise_conv_op->HasAttr("fuse_activation")
            ? PADDLE_GET_CONST(std::string,
                               depthwise_conv_op->GetAttr("fuse_activation"))
            : "";
    const auto fuse_alpha =
        depthwise_conv_op->HasAttr("fuse_alpha")
            ? PADDLE_GET_CONST(float, depthwise_conv_op->GetAttr("fuse_alpha"))
            : 0;
    const auto fuse_beta =
        depthwise_conv_op->HasAttr("fuse_beta")
            ? PADDLE_GET_CONST(float, depthwise_conv_op->GetAttr("fuse_beta"))
            : 0;

    conv_op->SetAttr("fuse_activation_dw", fuse_activation);
    conv_op->SetAttr("fuse_alpha_dw", fuse_alpha);
    conv_op->SetAttr("fuse_beta_dw", fuse_beta);
    conv_op->SetOutput("Output", {depthwise_conv_out->Name()});

    if (with_bias) {
      GET_IR_NODE_FROM_SUBGRAPH(depthwise_conv_bias,
                                depthwise_conv_bias,
                                conv_depthwise_conv_pattern);
      conv_op->SetInput("BiasDW", {depthwise_conv_bias->Name()});
      IR_NODE_LINK_TO(depthwise_conv_bias, conv);
    }

    IR_NODE_LINK_TO(depthwise_conv_weights, conv);
    IR_OP_VAR_LINK(conv, depthwise_conv_out);
    GraphSafeRemoveNodes(g, {depthwise_conv, conv_out});
    found_conv_depthwise_conv_count++;
  };

  gpd(graph, handler);

  AddStatis(found_conv_depthwise_conv_count);
  if (!Has("disable_logs") || !Get<bool>("disable_logs"))
    PrettyLogDetail("---    fused %d conv1x1 with depthwise conv",
                    found_conv_depthwise_conv_count);
}

void Conv1x1DepthwiseConvOneDNNFusePass::ApplyImpl(Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));

  FuseConvDepthWise(/*with bias*/ true, graph);
  FuseConvDepthWise(/*without bias*/ false, graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv1x1_depthwise_conv_mkldnn_fuse_pass,
              paddle::framework::ir::Conv1x1DepthwiseConvOneDNNFusePass);
REGISTER_PASS_CAPABILITY(conv1x1_depthwise_conv_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().LE(
            "conv2d", 1));
