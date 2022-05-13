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

#include "paddle/fluid/framework/ir/delete_weight_dequant_linear_op_pass.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                                 \
  GET_IR_NODE(weight_dequantize_linear_op_x);     \
  GET_IR_NODE(weight_dequantize_linear_op_scale); \
  GET_IR_NODE(weight_dequantize_linear_op);       \
  GET_IR_NODE(weight_dequantize_linear_op_out);   \
  GET_IR_NODE(any_op2);

DeleteWeightQuantDequantLinearOpPass::DeleteWeightQuantDequantLinearOpPass() {
  AddOpCompat(OpCompat("quantize_linear"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("ZeroPoint")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddAttr("bit_length")
      .IsType<int>()
      .End()
      .AddAttr("quant_axis")
      .IsType<int>()
      .End();
  AddOpCompat(OpCompat("dequantize_linear"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("ZeroPoint")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddAttr("bit_length")
      .IsType<int>()
      .End()
      .AddAttr("quant_axis")
      .IsType<int>()
      .End();
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
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();
  AddOpCompat(OpCompat("depthwise_conv2d"))
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
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();
  AddOpCompat(OpCompat("mul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("x_num_col_dims")
      .IsNumGE(1)
      .End()
      .AddAttr("y_num_col_dims")
      .IsNumEQ(1)
      .End();
  AddOpCompat(OpCompat("matmul_v2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("trans_x")
      .IsBoolEQ(false)
      .End()
      .AddAttr("trans_y")
      .IsBoolEQ(false)
      .End();
  AddOpCompat(OpCompat("matmul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("alpha")
      .IsNumGE(0.99f)
      .IsNumLE(1.01f)
      .End()
      .AddAttr("transpose_X")
      .IsBoolEQ(false)
      .End()
      .AddAttr("transpose_Y")
      .IsBoolEQ(false)
      .End();
  AddOpCompat(OpCompat("fc"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("W")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("in_num_col_dims")
      .IsNumGE(1)
      .End()
      .AddAttr("activation_type")
      .IsStringIn({"relu", ""})
      .End();
  AddOpCompat(OpCompat("conv2d_transpose"))
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
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("output_padding")
      .IsType<std::vector<int>>()
      .IsOptional()
      .End()
      .AddAttr("output_size")
      .IsType<std::vector<int>>()
      .IsOptional()
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .IsType<std::vector<int>>()
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
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();
}
// Delete dequantize_linear_op, then dequantize weight
void DeleteWeightQuantDequantLinearOpPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name =
      "delete_weight_quantdequant_linear_op_pattern";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::InvalidArgument(
          "Scope in DeleteWeightQuantDequantLinearOpPass should not be null."));
  // Create pattern
  patterns::DeleteWeightQuantDequantLinearOpPattern pattern(
      gpd.mutable_pattern(), pattern_name);
  pattern();
  int found_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;
    /*
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "delete_weight_dequant_linear_op_pass "
                      "compat check failed.";
      return;
    }
    */
    std::unordered_set<const Node*> nodes2rm = {};
    int bit_length = BOOST_GET_CONST(
        int, weight_dequantize_linear_op->Op()->GetAttr("bit_length"));
    int range = ((1 << (bit_length - 1)) - 1);

    auto* any_op2_desc = any_op2->Op();

    // get weight tensor
    auto* weight_tensor = scope->GetVar(weight_dequantize_linear_op_x->Name())
                              ->GetMutable<LoDTensor>();
    int8_t* quantized_weight_data =
        weight_tensor->mutable_data<int8_t>(platform::CPUPlace());
    auto w_dims = weight_tensor->dims();

    // Get weight scale
    std::vector<float> weight_scale;
    auto* weight_scale_tensor =
        scope->GetVar(weight_dequantize_linear_op_scale->Name())
            ->GetMutable<LoDTensor>();
    float* weight_scale_data =
        weight_scale_tensor->mutable_data<float>(platform::CPUPlace());

    auto weight_scale_nums = weight_scale_tensor->numel();
    for (int i = 0; i < weight_scale_nums; i++) {
      weight_scale.push_back(weight_scale_data[i] / range);
    }

    // dequant weight
    std::vector<float> weight_data_tmp;
    weight_data_tmp.reserve(weight_tensor->numel());

    int quant_axis = BOOST_GET_CONST(
        int, weight_dequantize_linear_op->Op()->GetAttr("quant_axis"));
    if (quant_axis == -1) {  // per_layer quant_dequant: all OP
      PADDLE_ENFORCE_EQ(weight_scale_nums, 1,
                        platform::errors::InvalidArgument(
                            "When quant_axis == -1 means use per_layer "
                            "quant_dequant, weight_scale'number should be 1."));

      //  float(weight) * scale
      for (int i = 0; i < weight_tensor->numel(); i++) {
        weight_data_tmp[i] =
            static_cast<float>(quantized_weight_data[i]) * weight_scale[0];
      }
    } else if (quant_axis == 0) {  // per_channel quant_dequant: conv2d,
                                   // depthwise_conv2d, conv2d_fusion
      PADDLE_ENFORCE_EQ(
          weight_scale_nums, w_dims[quant_axis],
          platform::errors::InvalidArgument(
              "When quant_axis == 0 means use per_channel quant_dequant, "
              "weight_scale'numbers should be equal channels."));
      PADDLE_ENFORCE_EQ(w_dims.size(), 4,
                        platform::errors::InvalidArgument(
                            "When quant_axis == 0 means use per_channel "
                            "quant_dequant, (conv2d, depthwise_conv2d, "
                            "conv2d_fusion)'s weight dims should be 4."));

      for (int i = 0; i < weight_tensor->numel(); i++) {
        int inner_size = w_dims[1] * w_dims[2] * w_dims[3];
        weight_data_tmp[i] = static_cast<float>(quantized_weight_data[i]) *
                             weight_scale[i / inner_size];
      }
    } else if (quant_axis == 1) {
      PADDLE_ENFORCE_EQ(
          weight_scale_nums, w_dims[quant_axis],
          platform::errors::InvalidArgument(
              "When quant_axis == 1 means use per_channel quant_dequant, "
              "weight_scale'numbers should be equal channels."));

      if (w_dims.size() == 4) {  // conv2d_transpose
        std::string quantized_op_type = any_op2->Op()->Type();
        PADDLE_ENFORCE_EQ(
            quantized_op_type, "conv2d_transpose",
            platform::errors::InvalidArgument(
                "When quant_axis == 1 means use per_channel quant_dequant, "
                "only conv2d_transpose weight dims equal 4."));
        for (int i = 0; i < weight_tensor->numel(); i++) {
          int inner_size = w_dims[2] * w_dims[3];
          weight_data_tmp[i] = static_cast<float>(quantized_weight_data[i]) *
                               weight_scale[(i / inner_size) % w_dims[1]];
        }
      } else if (w_dims.size() == 2) {
        for (int i = 0; i < weight_tensor->numel(); i++) {
          weight_data_tmp[i] = static_cast<float>(quantized_weight_data[i]) *
                               weight_scale[i % w_dims[1]];
        }
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "When quant_axis == 1 , weight dims should be 2 or 4, please check "
            "your model "));
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "quant_axis should be -1 or 0 or 1, please check your model "
          "OP'attribute "));
    }
    weight_tensor->clear();  // clear int weight
    weight_tensor->Resize(phi::make_ddim(phi::vectorize(w_dims)));
    float* new_quantized_weight_data =
        weight_tensor->mutable_data<float>(platform::CPUPlace());
    memcpy(new_quantized_weight_data, weight_data_tmp.data(),
           weight_tensor->numel() * sizeof(float));

    nodes2rm.insert(weight_dequantize_linear_op_scale);
    nodes2rm.insert(weight_dequantize_linear_op);
    nodes2rm.insert(weight_dequantize_linear_op_out);

    // relink weight to any_op2
    any_op2_desc->RenameInput(weight_dequantize_linear_op_out->Var()->Name(),
                              weight_dequantize_linear_op_x->Var()->Name());
    any_op2_desc->Flush();
    IR_NODE_LINK_TO(weight_dequantize_linear_op_x, any_op2);
    GraphSafeRemoveNodes(graph, nodes2rm);
    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_weight_dequant_linear_op_pass,
              paddle::framework::ir::DeleteWeightQuantDequantLinearOpPass);
