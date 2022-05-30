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
  GET_IR_NODE(dequantize_linear_op_out); \
  GET_IR_NODE(any_op2);

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
    int bit_length =
        BOOST_GET_CONST(int, quantize_linear_op->Op()->GetAttr("bit_length"));
    int range = ((1 << (bit_length - 1)) - 1);

    // Get input scale from tensor
    const LoDTensor& input_scale_tensor =
        scope->GetVar(quantize_linear_op_scale->Name())->Get<LoDTensor>();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_cpu_place(input_scale_tensor.place()), true,
        platform::errors::InvalidArgument(
            "Input scale tensor's place should be CPU."));
    const float* input_scale_data = input_scale_tensor.data<float>();
    float input_scale = input_scale_data[0] / range;

    auto* any_op2_desc = any_op2->Op();
    any_op2_desc->SetAttr("Input_scale_" + quantize_linear_op_x->Var()->Name(),
                          input_scale);

    nodes2rm.insert(quantize_linear_op_scale);
    nodes2rm.insert(quantize_linear_op);
    nodes2rm.insert(quantize_linear_op_out);
    nodes2rm.insert(dequantize_linear_op);
    nodes2rm.insert(dequantize_linear_op_out);

    // link x to any_op2
    any_op2_desc->RenameInput(dequantize_linear_op_out->Var()->Name(),
                              quantize_linear_op_x->Var()->Name());
    any_op2_desc->Flush();
    IR_NODE_LINK_TO(quantize_linear_op_x, any_op2);
    GraphSafeRemoveNodes(graph, nodes2rm);
    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_quant_dequant_linear_op_pass,
              paddle::framework::ir::DeleteQuantDequantLinearOpPass);
