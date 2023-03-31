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

    const std::vector<std::string> retain_op_type{"conv2d",
                                                  "depthwise_conv2d",
                                                  "conv2d_transpose",
                                                  "conv2d_fusion",
                                                  "depthwise_conv2d_transpose"};
    // for example, modify according to your needs, save d/dq in the front
    // const std::vector<std::string> retain_op_name{"batch_norm_0.tmp_2"};
    const std::vector<std::string> retain_op_name;

    // Conclusion: qdq befor bn or relu must be delete.
    // {"batch_norm", "relu"};

    auto quantize_x_name = quantize_linear_op_x->Var()->Name();

    auto is_retain_op = [&](const std::vector<std::string>& retain_op_vector,
                            const std::string& type_or_name) {
      return std::find(retain_op_vector.begin(),
                       retain_op_vector.end(),
                       type_or_name) != retain_op_vector.end();
    };

    bool is_retain_op_name = is_retain_op(retain_op_name, quantize_x_name);

    int nums_any_ops = dequantize_linear_op_out->outputs.size();
    for (int i = 0; i < nums_any_ops; ++i) {
      auto* any_op_desc = dequantize_linear_op_out->outputs[i]->Op();

      bool is_retain_op_type =
          is_retain_op(retain_op_type, any_op_desc->Type());
      if (!is_retain_op_type && !is_retain_op_name) {
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
    if (!is_retain_op(retain_op_type,
                      dequantize_linear_op_out->outputs[0]->Op()->Type()) &&
        !is_retain_op_name) {
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
