// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/delete_quant_dequant_op_pass.h"

#include <string>

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                         \
  GET_IR_NODE(quant_dequant_op_inscale);  \
  GET_IR_NODE(quant_dequant_op);          \
  GET_IR_NODE(quant_dequant_op_outscale); \
  GET_IR_NODE(quant_dequant_op_out);

void DeleteQuantDequantOpPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "delete_quantdequant_op_pattern";
  FusePassBase::Init(pattern_name, graph);
  GraphPatternDetector gpd;

  std::string quantdequant_types =
      "fake_quantize_dequantize_moving_average_abs_max";

  auto* input_node = gpd.mutable_pattern()
                         ->NewNode("input_node")
                         ->assert_is_op_input(quantdequant_types, "X")
                         ->AsInput();

  patterns::DeleteQuantDequantOpPattern pattern(gpd.mutable_pattern(),
                                                pattern_name);
  pattern(input_node, quantdequant_types);
  auto* scope = param_scope();
  int found_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    PADDLE_ENFORCE_EQ(
        subgraph.count(input_node), true,
        platform::errors::NotFound(
            "Input act node(%s) not found in QuantDequantFuse pass.",
            input_node->name()));
    Node* input = subgraph.at(input_node);
    GET_NODES;
    int bit_length =
        BOOST_GET_CONST(int, quant_dequant_op->Op()->GetAttr("bit_length"));
    int range = ((1 << (bit_length - 1)) - 1);

    // Get input scale from tensor
    std::string input_scale_var_name =
        quant_dequant_op->Op()->Input("InScale").front();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument(
                   "Scope in DeleteQuantDequantOpPass should not be null."));
    const LoDTensor& input_scale_tensor =
        scope->FindVar(input_scale_var_name)->Get<LoDTensor>();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_cpu_place(input_scale_tensor.place()), true,
        platform::errors::InvalidArgument(
            "Input scale tensor's place should be CPU."));
    const float* input_scale_data = input_scale_tensor.data<float>();
    float input_scale = input_scale_data[0] / range;

    // Set input scale in attr, and relink nodes
    std::string input_name = input->Var()->Name();
    std::string quant_dequant_output_name = quant_dequant_op_out->Var()->Name();
    auto outlinks = quant_dequant_op_out->outputs;
    for (auto* quantized_node : outlinks) {
      auto op_desc = quantized_node->Op();
      std::string quantized_op_type = op_desc->Type();
      if (quantized_op_type == "mul" || quantized_op_type == "matmul" ||
          quantized_op_type == "matmul_v2") {
        op_desc->SetAttr("X_scale", input_scale);
      } else {
        op_desc->SetAttr("Input_scale", input_scale);
      }
      op_desc->SetAttr("bit_length", bit_length);
      op_desc->RenameInput(quant_dequant_output_name, input_name);
      op_desc->Flush();
      IR_NODE_LINK_TO(input, quantized_node);
    }

    // Delete the unneeded nodes.
    GraphSafeRemoveNodes(graph,
                         {quant_dequant_op_inscale, quant_dequant_op,
                          quant_dequant_op_outscale, quant_dequant_op_out});
    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_quant_dequant_op_pass,
              paddle::framework::ir::DeleteQuantDequantOpPass);
