/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/delete_weight_dequant_linear_op_pass.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/quantize_helper.h"

#include "glog/logging.h"

namespace paddle::framework::ir {

class Graph;

void DeleteWeightDequantLinearOpPass::ApplyImpl(ir::Graph* graph) const {
  std::unordered_set<std::string> op_list = {"matrix_multiply",
                                             "matmul_v2",
                                             "matmul",
                                             "mul",
                                             "depthwise_conv2d",
                                             "conv2d",
                                             "conv2d_transpose"};
  PADDLE_ENFORCE_EQ(graph->Has(kParamScopeAttr),
                    true,
                    common::errors::InvalidArgument(
                        "Graph must have kParamScopeAttr attribute."));

  VLOG(3) << "Running delete_weight_dequant_linear_op_pass.";
  if (graph->IsMainGraph()) {
    VLOG(3) << "The ID of block running delete_weight_dequant_linear_op_pass "
               "is: 0(main_graph)";
  } else {
    VLOG(3)
        << "The ID of block running delete_weight_dequant_linear_op_pass is: "
        << graph->GetBlockId();
  }

  auto& scope = graph->Get<framework::Scope>(kParamScopeAttr);
  bool is_int8 = false;

  std::unordered_set<const Node*> nodes2rm;
  std::unordered_map<std::string, std::vector<float>> var_quant_scales{};

  for (const Node* n : graph->Nodes()) {
    if (n->IsOp()) {
      auto* op = n->Op();
      if (op->Type() == "dequantize_linear") {
        Node* weight_var_node = nullptr;
        Node* calcu_op_node = nullptr;
        Node* while_op_node = nullptr;
        Node *dequantized_weight_var_node = nullptr, *scale_var_node = nullptr;
        // 1. Judge whether for dequant weight and find
        // weight_var_node/scale_var_node
        for (auto* input_node : n->inputs) {
          if (input_node->IsVar() && input_node->Var()->Persistable()) {
            is_int8 = true;
            if (input_node->Var()->Name() == op->Input("X")[0]) {
              weight_var_node = input_node;
            } else if (input_node->Var()->Name() == op->Input("Scale")[0]) {
              scale_var_node = input_node;
            }
          } else {
            break;
          }
        }
        if (weight_var_node == nullptr || scale_var_node == nullptr) {
          continue;
        }
        // 2. Find next_op_node
        // For while op: delete its input which is related to dequantized
        // For calculation op: set weight scale as their attributes
        for (auto* output_node : n->outputs) {
          if (output_node->IsVar() &&
              output_node->Var()->Name() == op->Output("Y")[0]) {
            dequantized_weight_var_node = output_node;
            for (auto* next_op_node : output_node->outputs) {
              if (next_op_node->IsOp()) {
                if (next_op_node->Op()->Type() == "while") {
                  while_op_node = next_op_node;
                  auto while_op_desc = while_op_node->Op();
                  auto while_Xs = while_op_desc->Input("X");
                  while_Xs.erase(std::remove(std::begin(while_Xs),
                                             std::end(while_Xs),
                                             output_node->Var()->Name()),
                                 std::end(while_Xs));
                  while_op_node->Op()->SetInput("X", while_Xs);
                } else if (op_list.count(next_op_node->Op()->Type()) != 0) {
                  calcu_op_node = next_op_node;
                  auto* calcu_op_desc = calcu_op_node->Op();

                  std::vector<float> weight_scale;
                  auto* weight_scale_tensor =
                      scope.GetVar(scale_var_node->Name())
                          ->GetMutable<phi::DenseTensor>();
                  auto weight_scale_nums = weight_scale_tensor->numel();

                  if (weight_scale_tensor->dtype() == phi::DataType::FLOAT32) {
                    float* weight_scale_data =
                        weight_scale_tensor->data<float>();
                    for (int i = 0; i < weight_scale_nums; i++) {
                      weight_scale.push_back(weight_scale_data[i]);
                    }
                  } else if (weight_scale_tensor->dtype() ==
                             phi::DataType::FLOAT16) {
                    phi::dtype::float16* weight_scale_data =
                        weight_scale_tensor->data<phi::dtype::float16>();
                    for (int i = 0; i < weight_scale_nums; i++) {
                      weight_scale.push_back(
                          static_cast<float>(weight_scale_data[i]));
                    }
                  } else {
                    PADDLE_THROW(common::errors::Unimplemented(
                        "The dtype of quantization scale must be FP32/FP16, "
                        "but received %d, which is not supported.",
                        weight_scale_tensor->dtype()));
                  }

                  int quant_axis =
                      PADDLE_GET_CONST(int, op->GetAttr("quant_axis"));
                  if (quant_axis == -1) {  // per_layer quant_dequant: all OP
                    PADDLE_ENFORCE_EQ(
                        weight_scale_nums,
                        1,
                        common::errors::InvalidArgument(
                            "When quant_axis == -1, it means using per_layer "
                            "dequantization. In this situation, the number of "
                            "weight_scale should be 1, but received %d.",
                            weight_scale_nums));

                    calcu_op_desc->SetAttr("weight_scale", weight_scale[0]);
                  } else {
                    std::vector<int64_t> weights_shape =
                        weight_var_node->Var()->GetShape();
                    quant_axis = quant_axis >= 0
                                     ? quant_axis
                                     : quant_axis + weights_shape.size();
                    PADDLE_ENFORCE_EQ(
                        weight_scale_nums,
                        weights_shape[quant_axis],
                        common::errors::InvalidArgument(
                            "When quant_axis != -1, it means using per_channel "
                            "dequantization. In this situation, the number of "
                            "weight_scale should be equal with "
                            "weights_shape[quant_axis=%d]=%ld , but received "
                            "%d.",
                            quant_axis,
                            weights_shape[quant_axis],
                            weight_scale_nums));
                    calcu_op_desc->SetAttr("weight_scale", weight_scale);
                  }
                  if (!var_quant_scales.count(weight_var_node->Var()->Name())) {
                    var_quant_scales.insert(std::make_pair(
                        weight_var_node->Var()->Name(), weight_scale));
                  }

                  calcu_op_desc->RenameInput(
                      dequantized_weight_var_node->Var()->Name(),
                      weight_var_node->Var()->Name());
                  calcu_op_desc->Flush();
                }
              }
            }
          }
        }

        // 3. Delete dequant op
        IR_NODE_LINK_TO(weight_var_node, calcu_op_node);
        std::vector<const Node*> nodes2rm_local{
            dequantized_weight_var_node, scale_var_node, n};
        for (auto* node2rm : nodes2rm_local) {
          if (node2rm) {
            nodes2rm.insert(node2rm);
          }
        }
      }
    }
  }

  GraphSafeRemoveNodes(graph, nodes2rm);
  SaveQuantInfoInTheGraph(
      graph, "has_quant_info", "var_quant_scales", var_quant_scales);
  graph->Set("enable_int8", new bool(is_int8));
}
}  // namespace paddle::framework::ir

REGISTER_PASS(delete_weight_dequant_linear_op_pass,
              paddle::framework::ir::DeleteWeightDequantLinearOpPass);
