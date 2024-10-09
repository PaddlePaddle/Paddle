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
#include "paddle/fluid/framework/ir/quantize_helper.h"

namespace paddle::framework::ir {

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
  const std::string pattern_name = "delete_quant_dequant_linear_op_pattern";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      common::errors::InvalidArgument(
          "Scope in DeleteQuantDequantLinearOpPass should not be null."));

  VLOG(3) << "Running delete_quant_dequant_linear_op_pass.";
  if (graph->IsMainGraph()) {
    VLOG(3) << "The ID of block running delete_quant_dequant_linear_op_pass "
               "is: 0(main_graph)";
  } else {
    VLOG(3)
        << "The ID of block running delete_quant_dequant_linear_op_pass is: "
        << graph->GetBlockId();
  }

  std::unordered_map<std::string, std::vector<float>> var_quant_scales{};

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
    // Scale and ZeroPoint tensor should be removed in save_optimized_model_pass
    std::vector<std::string> vars2rm = {};
    vars2rm.emplace_back(quantize_linear_op->Op()->Input("Scale")[0]);
    vars2rm.emplace_back(quantize_linear_op->Op()->Input("ZeroPoint")[0]);
    vars2rm.emplace_back(dequantize_linear_op->Op()->Input("Scale")[0]);
    vars2rm.emplace_back(dequantize_linear_op->Op()->Input("ZeroPoint")[0]);
    auto& scale_and_zero_point_param = g->GetOrInit<std::vector<std::string>>(
        framework::ir::kScaleAndZeroPointParamAttr);
    scale_and_zero_point_param.insert(
        scale_and_zero_point_param.end(), vars2rm.begin(), vars2rm.end());

    std::unordered_set<const Node*> nodes2rm = {};

    // Get input scale from tensor
    const phi::DenseTensor& input_scale_tensor =
        scope->GetVar(quantize_linear_op_scale->Name())
            ->Get<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(phi::is_cpu_place(input_scale_tensor.place()),
                      true,
                      common::errors::InvalidArgument(
                          "Input scale tensor's place should be CPU."));

    float input_scale = NAN;
    if (input_scale_tensor.dtype() == phi::DataType::FLOAT32) {
      const float* input_scale_data = input_scale_tensor.data<float>();
      input_scale = input_scale_data[0];
    } else if (input_scale_tensor.dtype() == phi::DataType::FLOAT16) {
      const phi::dtype::float16* input_scale_data =
          input_scale_tensor.data<phi::dtype::float16>();
      input_scale = static_cast<float>(input_scale_data[0]);
    } else {
      PADDLE_THROW(common::errors::Unimplemented("%d is not supported.",
                                                 input_scale_tensor.dtype()));
    }

    int nums_any_ops =
        static_cast<int>(dequantize_linear_op_out->outputs.size());
    for (int i = 0; i < nums_any_ops; ++i) {
      auto* any_op_desc = dequantize_linear_op_out->outputs[i]->Op();
      any_op_desc->SetAttr("Input_scale_" + quantize_linear_op_x->Var()->Name(),
                           input_scale);
      if (!var_quant_scales.count(quantize_linear_op_x->Var()->Name())) {
        var_quant_scales.insert(
            std::make_pair(quantize_linear_op_x->Var()->Name(),
                           std::vector<float>({input_scale})));
      }
      // link x to any_op2
      any_op_desc->RenameInput(dequantize_linear_op_out->Var()->Name(),
                               quantize_linear_op_x->Var()->Name());
      any_op_desc->Flush();
      IR_NODE_LINK_TO(quantize_linear_op_x,
                      dequantize_linear_op_out->outputs[i]);
    }
    // Forbid removing weight tensor when weight is shared between ops
    if (quantize_linear_op_scale->outputs.size() <= 1UL)
      nodes2rm.insert(quantize_linear_op_scale);
    nodes2rm.insert(quantize_linear_op);
    nodes2rm.insert(quantize_linear_op_out);
    nodes2rm.insert(dequantize_linear_op);
    nodes2rm.insert(dequantize_linear_op_out);
    GraphSafeRemoveNodes(graph, nodes2rm);
    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);

  SaveQuantInfoInTheGraph(
      graph, "has_quant_info", "var_quant_scales", var_quant_scales);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(delete_quant_dequant_linear_op_pass,
              paddle::framework::ir::DeleteQuantDequantLinearOpPass);
