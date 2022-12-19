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

#include "paddle/fluid/framework/ir/mkldnn/int8_scale_calculation_mkldnn_pass.h"

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

Int8ScaleCalculationMkldnnPass::Int8ScaleCalculationMkldnnPass() {
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
      .IsStringIn({"NCHW", "AnyLayout"})
      .End();
}

void Int8ScaleCalculationMkldnnPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  FusePassBase::Init("int8_scale_calculation_mkldnn_pass", graph);
  GraphPatternDetector gpd;
  patterns::Conv conv_pattern(gpd.mutable_pattern(),
                              "int8_scale_calculation_mkldnn_pass");
  conv_pattern();

  int found_int8_scales_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);

    if (!platform::HasOpINT8DataType(conv_op->Op()) ||
        conv_op->Op()->HasAttr("Sum_scale")) {
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);

    auto input_names = conv_op->Op()->InputNames();
    bool has_bias = std::find(input_names.begin(), input_names.end(), "Bias") !=
                    input_names.end();
    std::vector<int64_t> weights_tz = conv_filter->Var()->GetShape();
    const int groups =
        std::max(conv_op->Op()->GetAttrIfExists<int>("groups"), 1);

    const auto& scale_weights_data =
        conv_op->Op()->GetAttrIfExists<std::vector<float>>("Scale_weights");
    const auto& scale_in_data =
        conv_op->Op()->GetAttrIfExists<float>("Scale_in");

    bool is_multi_channel = scale_weights_data.size() > 1;

    int count = 1;
    if (is_multi_channel) {
      count *= weights_tz[0];
      if (groups > 1) {
        count *= weights_tz[1];
      }
    }

    if (has_bias && conv_op->Op()->Input("Bias").size() > 0) {
      auto bias_scales = std::vector<float>(count);
      for (int i = 0; i < count; i++) {
        bias_scales[i] = scale_in_data * scale_weights_data[i];
      }
      conv_op->Op()->SetAttr("Bias_scales", bias_scales);
    }

    const bool& force_fp32_output =
        conv_op->Op()->GetAttrIfExists<bool>("force_fp32_output");
    const bool& fuse_residual_conn =
        conv_op->Op()->GetAttrIfExists<bool>("fuse_residual_connection");
    const auto& scale_in_eltwise_data =
        conv_op->Op()->GetAttrIfExists<float>("Scale_in_eltwise");
    bool has_activation =
        !conv_op->Op()->GetAttrIfExists<std::string>("fuse_activation").empty();
    float activation_scale =
        force_fp32_output ? 1.0f
        : has_activation  ? conv_op->Op()->GetAttrIfExists<float>("Scale_out")
                          : 1.0f;
    auto scale_out_data =
        force_fp32_output ? 1.0f
        : has_activation  ? 1.0f
                          : conv_op->Op()->GetAttrIfExists<float>("Scale_out");
    float sum_scale =
        fuse_residual_conn ? scale_out_data / scale_in_eltwise_data : 1.0f;

    std::vector<float> output_shift_scale(count);

#pragma omp parallel for if (count > 50)
    for (int i = 0; i < count; i++) {
      if (scale_weights_data[i] == 0.0)
        // weights data will contain 0 in some models, then weights
        // scale couldn't be calculated
        output_shift_scale[i] = scale_out_data;
      else
        output_shift_scale[i] =
            static_cast<float>(static_cast<double>(scale_out_data) /
                               (static_cast<double>(scale_in_data) *
                                static_cast<double>(scale_weights_data[i])));
    }

    conv_op->Op()->SetAttr("Sum_scale", sum_scale);
    conv_op->Op()->SetAttr("Output_shift_scale", output_shift_scale);
    conv_op->Op()->SetAttr("Activation_scale", activation_scale);
    found_int8_scales_count++;
  };
  gpd(graph, handler);
  AddStatis(found_int8_scales_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(int8_scale_calculation_mkldnn_pass,
              paddle::framework::ir::Int8ScaleCalculationMkldnnPass);
REGISTER_PASS_CAPABILITY(int8_scale_calculation_mkldnn_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().LE(
            "conv2d", 1));
