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

#include "paddle/fluid/framework/ir/delete_weight_dequant_linear_op_encoder_pass.h"

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

DeleteWeightDequantLinearOpEncoderPass::
    DeleteWeightDequantLinearOpEncoderPass() {
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
      .End()
      .AddAttr("round_type")
      .IsOptional()
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
      .End()
      .AddAttr("round_type")
      .IsOptional()
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
void DeleteWeightDequantLinearOpEncoderPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name =
      "delete_weight_dequant_linear_op_encoder_pattern";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(scope,
                          platform::errors::InvalidArgument(
                              "Scope in DeleteWeightDequantLinearOpEncoderPass "
                              "should not be null."));
  // Create pattern
  patterns::DeleteWeightDequantLinearOpEncoderPattern pattern(
      gpd.mutable_pattern(), pattern_name);
  pattern();
  int found_count = 0;
  bool is_int8 = false;

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
    is_int8 = true;
    std::unordered_set<const Node*> nodes2rm = {};

    auto* any_op2_desc = any_op2->Op();

    // Get weight scale
    std::vector<float> weight_scale;
    auto* weight_scale_tensor =
        scope->GetVar(weight_dequantize_linear_op_scale->Name())
            ->GetMutable<phi::DenseTensor>();
    auto weight_scale_nums = weight_scale_tensor->numel();

    if (weight_scale_tensor->dtype() ==
        paddle::experimental::DataType::FLOAT32) {
      float* weight_scale_data = weight_scale_tensor->data<float>();
      for (int i = 0; i < weight_scale_nums; i++) {
        weight_scale.push_back(weight_scale_data[i]);
      }
    } else if (weight_scale_tensor->dtype() ==
               paddle::experimental::DataType::FLOAT16) {
      phi::dtype::float16* weight_scale_data =
          weight_scale_tensor->data<phi::dtype::float16>();
      for (int i = 0; i < weight_scale_nums; i++) {
        weight_scale.push_back(static_cast<float>(weight_scale_data[i]));
      }
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "%d is not supported.", weight_scale_tensor->dtype()));
    }

    int quant_axis = PADDLE_GET_CONST(
        int, weight_dequantize_linear_op->Op()->GetAttr("quant_axis"));
    if (quant_axis == -1) {  // per_layer quant_dequant: all OP
      PADDLE_ENFORCE_EQ(weight_scale_nums,
                        1,
                        platform::errors::InvalidArgument(
                            "When quant_axis == -1 means use per_layer "
                            "quant_dequant, weight_scale'number should be 1."));

      // Add attr to anyop 2
      any_op2_desc->SetAttr("weight_scale", weight_scale[0]);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Delete Weight Dequant Linear Op Encoder Pass is not supported for "
          "per-channel quantization"));
    }

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
  graph->Set("enable_int8", new bool(is_int8));
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_weight_dequant_linear_op_encoder_pass,
              paddle::framework::ir::DeleteWeightDequantLinearOpEncoderPass);
