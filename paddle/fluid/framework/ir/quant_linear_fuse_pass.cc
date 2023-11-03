// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/quant_linear_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"

namespace {
template <typename T1, typename T2>
void ConvertTensorType(phi::DenseTensor* tensor) {
  phi::DenseTensor tmp_tensor;
  tmp_tensor.set_type(phi::CppTypeToDataType<T2>::Type());
  tmp_tensor.Resize(tensor->dims());
  auto* tmp_data = tmp_tensor.mutable_data<T2>(paddle::platform::CPUPlace());
  auto* data = tensor->mutable_data<T1>(paddle::platform::CPUPlace());
  for (int i = 0; i < tensor->numel(); i++) {
    tmp_data[i] = static_cast<T2>(data[i]);
  }
  tensor->clear();
  paddle::framework::TensorCopySync(
      tmp_tensor, paddle::platform::CPUPlace(), tensor);
}
}  // namespace

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


QuantLinearFusePass::QuantLinearFusePass() {
  AddOpCompat(OpCompat("matmul_v2"))
      .AddInput("X")  // the shape shoule be (B, S, N*H)
      .IsTensor()
      .End()
      .AddInput("Y")  // the shape shoule be (N*H, N*H)
      .IsTensor()
      .End()
      .AddOutput("Out")  // the shape shoule be (B, S, N*H)
      .IsTensor()
      .End()
      .AddAttr("trans_x")
      .IsType<bool>()
      .End()
      .AddAttr("trans_y")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumMatch<int>([](int axis) -> bool {
        if (axis == -1 || axis >= 1) {
          return true;
        }
        return false;
      })
      .End();


}

// delete the quant and dequant op and weight dequant op,
// then fuse the matmul_v2 and elementwise_add op to a quant_linear op
void QuantLinearFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("quant_linear_fuse", graph);

  int found_count = 0;
  ApplyDeleteQuantDequantPattern(graph);
  ApplyDeleteWeightDequantPattern(graph);
  found_count = ApplyQuantLinearFusePattern(graph);
  AddStatis(found_count);
}

int QuantLinearFusePass::ApplyQuantLinearFusePattern(Graph* graph) const {
  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("quant_linear_fuse/x")
                ->AsInput()
                ->assert_is_op_input("matmul_v2", "X");

  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::InvalidArgument(
          "Scope in QuantLinearFusePass should not be "
          "null."));
 
  patterns::QuantLinearFusePattern pattern(gpd.mutable_pattern(), "quant_linear_fuse");
  pattern(x, true /*with bias*/);

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (subgraph.count(x) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }

    VLOG(4) << "handle quant_linear fuse";
    GET_IR_NODE_FROM_SUBGRAPH(w, w, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bias, bias, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_add_out, elementwise_add_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul, mul, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add, elementwise_add, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul_out, mul_out, pattern);

    // convrt weight fp32 --> int8
    auto* weight_tensor =
        scope->FindVar(w->Name())->GetMutable<phi::DenseTensor>();
    ConvertTensorType<float, int8_t>(weight_tensor);


    // Create an QuantLinear Node.
    OpDesc desc(mul->Op()->Block());
    desc.SetType("quant_linear");

    // Set inputs of quant_linear
    desc.SetInput("x", {subgraph.at(x)->Name()});
    desc.SetInput("w", {w->Name()});
    desc.SetInput("bias", {bias->Name()});

    // Set output of quant_linear
    std::string quant_linear_out_name = elementwise_add_out->Name();
    desc.SetOutput("out", std::vector<std::string>({quant_linear_out_name}));

    auto* mul_op_desc = mul->Op();
       
    desc.SetAttr("scale_weight", mul_op_desc->GetAttr("weight_scale"));

    desc.Flush();

    auto quant_linear_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
    GraphSafeRemoveNodes(graph, {mul, elementwise_add, mul_out});

    IR_NODE_LINK_TO(subgraph.at(x), quant_linear_node);

    IR_NODE_LINK_TO(w, quant_linear_node);

    IR_NODE_LINK_TO(bias, quant_linear_node);

    IR_NODE_LINK_TO(quant_linear_node, elementwise_add_out);

    found_count++;
  };
  gpd(graph, handler);
  return found_count;
}


// copied from delete_weight_dequant_linear_op_pass
void QuantLinearFusePass::ApplyDeleteWeightDequantPattern(Graph* graph) const {
  std::unordered_set<std::string> op_list = {"matrix_multiply",
                                            "matmul_v2",
                                            "matmul",
                                            "mul",
                                            "depthwise_conv2d",
                                            "conv2d",
                                            "conv2d_transpose"};
  PADDLE_ENFORCE_EQ(graph->Has(kParamScopeAttr),
                    true,
                    platform::errors::InvalidArgument(
                        "Graph must have kParamScopeAttr attribute."));

  auto& scope = graph->Get<framework::Scope>(kParamScopeAttr);
  bool is_int8 = false;

  std::unordered_set<const Node*> nodes2rm;

  for (const Node* n : graph->Nodes()) {
    if (n->IsOp()) {
      auto* op = n->Op();
      if (op->Type() == "dequantize_linear") {
        Node *weight_var_node = nullptr, *calcu_op_node = nullptr,
             *while_op_node = nullptr;
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
            return;
          }
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
                    PADDLE_THROW(platform::errors::Unimplemented(
                        "The dtype of quantization scale must be FP32/16, "
                        "but received %d, which is not supported.",
                        weight_scale_tensor->dtype()));
                  }

                  int quant_axis =
                      PADDLE_GET_CONST(int, op->GetAttr("quant_axis"));
                  if (quant_axis == -1) {  // per_layer quant_dequant: all OP
                    PADDLE_ENFORCE_EQ(
                        weight_scale_nums,
                        1,
                        platform::errors::InvalidArgument(
                            "When quant_axis == -1, it means using per_layer "
                            "dequantization. In this situation, the number of "
                            "weight_scale should be 1, but received %d.",
                            weight_scale_nums));

                    calcu_op_desc->SetAttr("weight_scale", weight_scale[0]);
                  } else {
                    PADDLE_THROW(platform::errors::Unimplemented(
                        "Delete Weight Dequant Linear Op Pass is not supported "
                        "for "
                        "per-channel quantization"));
                  }
                  calcu_op_desc->RenameInput(
                      dequantized_weight_var_node->Var()->Name(),
                      weight_var_node->Var()->Name());
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
  graph->Set("enable_int8", new bool(is_int8));
}


// copied from delete_quant_dequant_linear_op_pass
void QuantLinearFusePass::ApplyDeleteQuantDequantPattern(Graph* graph) const {
 GraphPatternDetector gpd;
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::InvalidArgument(
          "Scope in DeleteQuantDequantLinearOpPass should not be null."));
  // Create pattern
  patterns::DeleteQuantDequantLinearOpPattern pattern(gpd.mutable_pattern(),
                                                      "quant_linear_fuse");
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

    float input_scale = NAN;
    if (input_scale_tensor.dtype() == phi::DataType::FLOAT32) {
      const float* input_scale_data = input_scale_tensor.data<float>();
      input_scale = input_scale_data[0];
    } else if (input_scale_tensor.dtype() == phi::DataType::FLOAT16) {
      const phi::dtype::float16* input_scale_data =
          input_scale_tensor.data<phi::dtype::float16>();
      input_scale = static_cast<float>(input_scale_data[0]);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented("%d is not supported.",
                                                   input_scale_tensor.dtype()));
    }

    int nums_any_ops =
        static_cast<int>(dequantize_linear_op_out->outputs.size());
    for (int i = 0; i < nums_any_ops; ++i) {
      auto* any_op_desc = dequantize_linear_op_out->outputs[i]->Op();
      any_op_desc->SetAttr("Input_scale_" + quantize_linear_op_x->Var()->Name(),
                           input_scale);

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
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(quant_linear_fuse_pass, paddle::framework::ir::QuantLinearFusePass);