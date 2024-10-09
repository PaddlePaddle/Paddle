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

#include "paddle/fluid/framework/ir/onednn/params_quantization_onednn_pass.h"

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/onednn_helper.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle::framework::ir {

namespace {

template <typename T_out>
void QuantizeParams(phi::DenseTensor* param_tensor,
                    const std::vector<float>& scales) {
  std::vector<T_out> tmp_data;
  tmp_data.reserve(param_tensor->numel());

  auto length = param_tensor->numel() / scales.size();

  const float* param_data = param_tensor->data<float>();
  for (int64_t i = 0; i < param_tensor->numel(); ++i) {
    tmp_data[i] =
        static_cast<T_out>(std::round(param_data[i] * scales[i / length]));
  }

  auto dims = param_tensor->dims();
  param_tensor->clear();
  param_tensor->Resize(dims);

  auto int_param_data = param_tensor->mutable_data<T_out>(CPUPlace());
  std::copy_n(tmp_data.data(), param_tensor->numel(), int_param_data);
}

bool HasBias(ir::Node* conv_op) {
  auto input_names = conv_op->Op()->InputNames();
  return std::find(input_names.begin(), input_names.end(), "Bias") !=
             input_names.end() &&
         !conv_op->Op()->Input("Bias").empty();
}

template <typename T>
void QuantizeConvInput(Scope* scope,
                       ir::Graph* g,
                       ir::Node* conv_op,
                       const std::string& input_name,
                       const std::string& scales_attr_name) {
  auto var = scope->GetVar(input_name);
  if (var->Get<phi::DenseTensor>().dtype() != phi::DataType::FLOAT32) {
    VLOG(0) << "Skipping convolution filter: " << input_name
            << " because it is detected again.";
    conv_op->Op()->SetAttr(scales_attr_name, std::vector<float>(1, 1));
  } else {
    const auto scales =
        conv_op->Op()->GetAttrIfExists<std::vector<float>>(scales_attr_name);

    auto* tensor = scope->GetVar(input_name)->GetMutable<phi::DenseTensor>();
    QuantizeParams<T>(tensor, scales);
    conv_op->Op()->SetAttr(scales_attr_name, std::vector<float>(1, 1));
  }
}

}  // namespace

ParamsQuantizationMkldnnPass::ParamsQuantizationMkldnnPass() {  // NOLINT
  AddOpCompat(OpCompat("fused_conv2d"))
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

void ParamsQuantizationMkldnnPass::QuantizeConv(ir::Graph* graph,
                                                const std::string& conv_type,
                                                bool with_residual_data) const {
  GraphPatternDetector gpd;
  patterns::ConvResidual conv_pattern(gpd.mutable_pattern(), name_scope_);
  conv_pattern(conv_type, with_residual_data);

  int params_to_int8_conv_found = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    VLOG(4) << "handle convolution in params_quantization_onednn_pass";

    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);

    // get scope to interact with tensors
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));

    // If not a quantized OP
    if (!platform::HasOpINT8DataType(conv_op->Op())) {
      return;
    }

    QuantizeConvInput<int8_t>(
        scope, g, conv_op, conv_filter->Name(), "Scale_weights");

    if (HasBias(conv_op)) {
      QuantizeConvInput<int32_t>(
          scope, g, conv_op, conv_op->Op()->Input("Bias")[0], "Bias_scales");
    }
    params_to_int8_conv_found++;
  };
  gpd(graph, handler);
  AddStatis(params_to_int8_conv_found);

  std::stringstream msg_ss;
  msg_ss << "Quantized params of " << params_to_int8_conv_found << " "
         << conv_type << " ops";
  if (with_residual_data) msg_ss << " with residual connection";
  paddle::string::PrettyLogDetail(msg_ss.str().c_str());
}

void ParamsQuantizationMkldnnPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          common::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  FusePassBase::Init(name_scope_, graph);
  QuantizeConv(graph, "fused_conv2d", true /*with_residual_data*/);
  QuantizeConv(graph, "fused_conv2d", false /*with_residual_data*/);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(params_quantization_onednn_pass,
              paddle::framework::ir::ParamsQuantizationMkldnnPass);
REGISTER_PASS_CAPABILITY(params_quantization_onednn_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().LE(
            "conv2d", 1));
