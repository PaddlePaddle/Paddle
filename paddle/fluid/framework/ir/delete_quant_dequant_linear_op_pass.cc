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

#include "paddle/fluid/framework/ir/delete_quant_dequant_linear_op_pass.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                        \
  GET_IR_NODE(quantize_linear_op_x);     \
  GET_IR_NODE(quantize_linear_op_scale); \
  GET_IR_NODE(quantize_linear_op);       \
  GET_IR_NODE(quantize_linear_op_out);   \
  GET_IR_NODE(dequantize_linear_op);     \
  GET_IR_NODE(dequantize_linear_op_out);

DeleteQuantDequantLinearOpPass::DeleteQuantDequantLinearOpPass() {
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
}
// Delete quantize_linear_op dequantize_linear_op, then add input_scales
void DeleteQuantDequantLinearOpPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "delete_quantdequant_linear_op_pattern";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::InvalidArgument(
          "Scope in DeleteQuantDequantLinearOpPass should not be null."));
  // Create pattern
  patterns::DeleteQuantDequantLinearOpPattern pattern(gpd.mutable_pattern(),
                                                      pattern_name);
  pattern();
  int found_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;
    /*
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "delete_quant_dequant_linear_op_pass "
                      "compat check failed.";
      return;
    }
    */
    std::unordered_set<const Node*> nodes2rm = {};

    // Get input scale from tensor
    const phi::DenseTensor& input_scale_tensor =
        scope->GetVar(quantize_linear_op_scale->Name())
            ->Get<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_cpu_place(input_scale_tensor.place()),
        true,
        platform::errors::InvalidArgument(
            "Input scale tensor's place should be CPU."));

    float input_scale;
    if (input_scale_tensor.dtype() == paddle::experimental::DataType::FLOAT32) {
      const float* input_scale_data = input_scale_tensor.data<float>();
      input_scale = input_scale_data[0];
    } else if (input_scale_tensor.dtype() ==
               paddle::experimental::DataType::FLOAT16) {
      const phi::dtype::float16* input_scale_data =
          input_scale_tensor.data<phi::dtype::float16>();
      input_scale = static_cast<float>(input_scale_data[0]);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented("%d is not supported.",
                                                   input_scale_tensor.dtype()));
    }

    int nums_any_ops = dequantize_linear_op_out->outputs.size();
    for (int i = 0; i < nums_any_ops; ++i) {
      auto* any_op_desc = dequantize_linear_op_out->outputs[i]->Op();
      // if (any_op_desc->Type() == "batch_norm" || any_op_desc->Type() ==
      // "relu"  || any_op_desc->Type() == "hard_swish" || any_op_desc->Type()
      // == "elementwise_add"    || any_op_desc->Type() == "elementwise_mul" ||
      // any_op_desc->Type() == "pool2d")
      if (quantize_linear_op_x->Var()->Name() != "batch_norm_0.tmp_2" &&
          quantize_linear_op_x->Var()->Name() != "conv2d_73.tmp_1" &&
          quantize_linear_op_x->Var()->Name() != "relu_7.tmp_0" &&
          quantize_linear_op_x->Var()->Name() != "hard_sigmoid_0.tmp_0" &&
          quantize_linear_op_x->Var()->Name() != "batch_norm_6.tmp_2" &&
          quantize_linear_op_x->Var()->Name() != "hard_swish_0.tmp_0" &&
          any_op_desc->Type() != "conv2d" &&
          any_op_desc->Type() != "depthwise_conv2d" &&
          any_op_desc->Type() != "conv2d_transpose" &&
          any_op_desc->Type() != "conv2d_fusion" &&
          any_op_desc->Type() != "depthwise_conv2d_transpose") {
        std::cout << "remove q dq with  :" << any_op_desc->Type() << std::endl;
        std::cout << "var name   :" << quantize_linear_op_x->Var()->Name()
                  << std::endl;

        any_op_desc->SetAttr(
            "Input_scale_" + quantize_linear_op_x->Var()->Name(), input_scale);

        // link x to any_op2
        any_op_desc->RenameInput(dequantize_linear_op_out->Var()->Name(),
                                 quantize_linear_op_x->Var()->Name());
        any_op_desc->Flush();
        IR_NODE_LINK_TO(quantize_linear_op_x,
                        dequantize_linear_op_out->outputs[i]);
      }
    }

    // Forbid removing weight tensor when weight is shared between ops
    // if (dequantize_linear_op_out->outputs[0]->Op()->Type() == "batch_norm" ||
    // dequantize_linear_op_out->outputs[0]->Op()->Type() == "relu" ||
    // dequantize_linear_op_out->outputs[0]->Op()->Type() == "hard_swish"  ||
    // dequantize_linear_op_out->outputs[0]->Op()->Type() ==  "elementwise_add"
    // || dequantize_linear_op_out->outputs[0]->Op()->Type() ==
    // "elementwise_mul" || dequantize_linear_op_out->outputs[0]->Op()->Type()
    // =="pool2d")
    auto type_test = dequantize_linear_op_out->outputs[0]->Op()->Type();
    if (quantize_linear_op_x->Var()->Name() != "batch_norm_0.tmp_2" &&
        quantize_linear_op_x->Var()->Name() != "conv2d_73.tmp_1" &&
        quantize_linear_op_x->Var()->Name() != "relu_7.tmp_0" &&
        quantize_linear_op_x->Var()->Name() != "hard_sigmoid_0.tmp_0" &&
        quantize_linear_op_x->Var()->Name() != "batch_norm_6.tmp_2" &&
        quantize_linear_op_x->Var()->Name() != "hard_swish_0.tmp_0" &&
        type_test != "conv2d" && type_test != "depthwise_conv2d" &&
        type_test != "conv2d_transpose" && type_test != "conv2d_fusion" &&
        type_test != "depthwise_conv2d_transpose") {
      if (quantize_linear_op_scale->outputs.size() <= 1UL)
        nodes2rm.insert(quantize_linear_op_scale);
      nodes2rm.insert(quantize_linear_op);
      nodes2rm.insert(quantize_linear_op_out);
      nodes2rm.insert(dequantize_linear_op);
      nodes2rm.insert(dequantize_linear_op_out);
      GraphSafeRemoveNodes(graph, nodes2rm);
      found_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_quant_dequant_linear_op_pass,
              paddle::framework::ir::DeleteQuantDequantLinearOpPass);
