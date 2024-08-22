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

#include "paddle/fluid/framework/ir/xpu/quant_dequant_xpu_pass.h"

#include <string>

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/onednn/onednn_pass_util.h"
#include "paddle/fluid/framework/ir/quantize_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

void QuantDequantXPUPass::CollectWeightScalesInfoFromDequantize(
    ir::Graph* graph,
    Scope* scope,
    const std::unordered_set<std::string>& fake_dequantize_types,
    std::unordered_map<std::string, std::vector<float>>* weight_thresholds)
    const {
  VLOG(3) << "gather weight_thresholds from fake dequantized ops";
  for (auto* op_node :
       ir::TopologyVariantSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    if (fake_dequantize_types.count(op_node->Name())) {
      auto* op_desc = op_node->Op();
      auto x_var_name = op_desc->Input("X")[0];
      // find the var name of weight node to be dequantized
      std::string weight_var_name;
      for (auto* in : op_node->inputs) {
        if (in->Name() == x_var_name) {
          auto cal_op_node = in->inputs[0];
          if (cal_op_node->Name() == "conv2d" ||
              cal_op_node->Name() == "depthwise_conv2d" ||
              cal_op_node->Name() == "fused_conv2d") {
            weight_var_name = cal_op_node->Op()->Input("Filter")[0];
          } else if (cal_op_node->Name() == "mul" ||
                     cal_op_node->Name() == "matmul" ||
                     cal_op_node->Name() == "matmul_v2") {
            weight_var_name = cal_op_node->Op()->Input("Y")[0];
          }
        }
      }

      if (op_desc->HasAttr("max_range")) {
        const float max_range =
            PADDLE_GET_CONST(float, op_desc->GetAttr("max_range"));
        std::vector<float> thresholds = {127 * 127 / max_range};
        weight_thresholds->insert(std::make_pair(weight_var_name, thresholds));
      } else {
        auto scale_name = op_desc->Input("Scales")[0];
        auto* var = scope->FindVar(scale_name);
        PADDLE_ENFORCE_NOT_NULL(
            var,
            common::errors::NotFound(
                "The Scales variable [%s] of dequantize op is not found.",
                var));

        auto* scale_tensor = var->GetMutable<phi::DenseTensor>();
        auto* scale_data = scale_tensor->data<float>();
        std::vector<float> thresholds{};
        for (int i = 0; i < scale_tensor->numel(); i++) {
          thresholds.push_back(scale_data[i]);
        }
        weight_thresholds->insert(std::make_pair(weight_var_name, thresholds));
      }
    }
  }
}

void QuantDequantXPUPass::CollectWeightScalesInfoFromONNXFormatDequantize(
    ir::Graph* graph,
    Scope* scope,
    std::unordered_map<std::string, std::vector<float>>* weight_thresholds,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
    const {
  VLOG(3) << "gather weight_thresholds from onnx format dequantized ops";
  for (auto* op_node :
       ir::TopologyVariantSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    if (op_node->Name() == "dequantize_linear") {
      auto* op_desc = op_node->Op();

      auto scale_name = op_desc->Input("Scale")[0];
      auto* var = scope->FindVar(scale_name);
      PADDLE_ENFORCE_NOT_NULL(
          var,
          common::errors::NotFound(
              "The Scales variable [%s] of dequantize op is not found.", var));

      auto* scale_tensor = var->GetMutable<phi::DenseTensor>();
      auto* scale_data = scale_tensor->data<float>();

      auto x_var_name = op_desc->Input("X")[0];
      auto* weight_var = scope->FindVar(x_var_name);
      if (!weight_var) {
        auto out_var_name = op_desc->Output("Y")[0];
        float scale = scale_data[0];
        if (std::isinf(scale) || std::isnan(scale)) {
          scale = 0.0;
        }
        std::vector<float> scale_v = {scale};
        if (!var_quant_scales->count(out_var_name)) {
          var_quant_scales->insert(std::make_pair(out_var_name, scale_v));
        }
        if (!var_quant_scales->count(x_var_name)) {
          var_quant_scales->insert(std::make_pair(x_var_name, scale_v));
        }
      } else {
        std::vector<float> thresholds(scale_data,
                                      scale_data + scale_tensor->numel());
        weight_thresholds->insert(std::make_pair(x_var_name, thresholds));
      }
    }
  }
}

void QuantDequantXPUPass::CollectInputScalesFromQuantize(
    ir::Graph* graph,
    Scope* scope,
    const std::unordered_set<std::string>& fake_quantize_types,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
    const {
  VLOG(3) << "gather input scales from fake quantized ops";
  for (auto* op_node :
       ir::TopologyVariantSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    if (op_node->Name() == "fake_quantize_dequantize_moving_average_abs_max" ||
        op_node->Name() == "quantize_linear" ||
        fake_quantize_types.count(op_node->Name())) {
      auto* op_desc = op_node->Op();
      const int bit_length =
          PADDLE_GET_CONST(int, op_desc->GetAttr("bit_length"));
      PADDLE_ENFORCE_EQ(
          bit_length,
          8,
          common::errors::InvalidArgument("Unsupported number quantization "
                                          "bits: %d, only 8 is supported now.",
                                          bit_length));

      std::string scale_name = "InScale";
      std::string out_name = "Out";
      if (op_node->Name() == "quantize_linear") {
        scale_name = "Scale";
        out_name = "Y";
      }
      auto x_var_name = op_desc->Input("X")[0];
      auto scale_var_name = op_desc->Input(scale_name)[0];
      auto out_var_name = op_desc->Output(out_name)[0];

      auto* var = scope->FindVar(scale_var_name);
      PADDLE_ENFORCE_NOT_NULL(
          var,
          common::errors::NotFound(
              "The InScale variable [%s] of quantize op is not found.", var));

      auto* scale_tensor = var->GetMutable<phi::DenseTensor>();
      auto* scale_data = scale_tensor->data<float>();
      float scale = scale_data[0];
      if (std::isinf(scale) || std::isnan(scale)) {
        continue;
      }

      if (!var_quant_scales->count(x_var_name)) {
        std::vector<float> scale_v = {scale};
        var_quant_scales->insert(std::make_pair(x_var_name, scale_v));
      }

      if (!var_quant_scales->count(out_var_name)) {
        std::vector<float> scale_v = {scale};
        var_quant_scales->insert(std::make_pair(out_var_name, scale_v));
      }

      for (auto* out : op_node->outputs) {
        if (out->Name() == out_var_name) {
          for (auto* var : out->outputs) {
            auto op_desc = var->Op();
            op_desc->SetAttr("enable_int8", true);
            op_desc->Flush();
          }
        }
      }
    }
  }
}

void QuantDequantXPUPass::CollectOutputScalesFromAttr(
    ir::Graph* graph,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
    const {
  VLOG(3) << "gather output scales from op's attr";
  for (auto* op_node :
       ir::TopologyVariantSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    auto* op_desc = op_node->Op();
    if (op_desc->HasAttr("out_threshold")) {
      const float attr_scale =
          PADDLE_GET_CONST(float, op_desc->GetAttr("out_threshold"));
      if (attr_scale == 0.0) continue;
      float scale = attr_scale;
      std::vector<float> scale_v = {scale};

      auto var_name_map = op_desc->Outputs();
      for (auto& item : var_name_map) {
        for (auto const& var_name : item.second) {
          var_quant_scales->insert(std::make_pair(var_name, scale_v));
        }
      }
    }
  }
}

void QuantDequantXPUPass::CollectFakeQuantizeOps(
    ir::Graph* graph,
    Node* op_node,
    std::unordered_set<const Node*>* nodes2rm) const {
  auto* op_desc = op_node->Op();
  auto x_var_name = op_desc->Input("X")[0];
  auto in_scale_name = op_desc->Input("InScale")[0];
  auto out_var_name = op_desc->Output("Out")[0];
  auto out_scale_name = op_desc->Output("OutScale")[0];

  Node* fake_quant_in = nullptr;
  Node* fake_quant_in_scale = nullptr;
  for (auto* node_input : op_node->inputs) {
    if (node_input->Name() == x_var_name) {
      fake_quant_in = node_input;
    } else if (node_input->Name() == in_scale_name) {
      fake_quant_in_scale = node_input;
    }
  }

  Node* fake_quant_out = nullptr;
  Node* fake_quant_out_scale = nullptr;
  for (auto* node_output : op_node->outputs) {
    if (node_output->Name() == out_var_name) {
      fake_quant_out = node_output;
    } else if (node_output->Name() == out_scale_name) {
      fake_quant_out_scale = node_output;
    }
  }

  PADDLE_ENFORCE_NOT_NULL(
      fake_quant_in,
      common::errors::NotFound(
          "The input var [%s] of quantize op is not found.", x_var_name));
  PADDLE_ENFORCE_NOT_NULL(
      fake_quant_out,
      common::errors::NotFound(
          "The output var [%s] of quantize op is not found.", out_var_name));

  std::string input_act_name = fake_quant_in->Var()->Name();
  std::string output_act_name = fake_quant_out->Var()->Name();
  auto outlinks = fake_quant_out->outputs;
  for (auto* next_node : outlinks) {
    if (!next_node->IsOp()) continue;
    next_node->Op()->RenameInput(output_act_name, input_act_name);
    IR_NODE_LINK_TO(fake_quant_in, next_node);
  }

  nodes2rm->insert(op_node);
  nodes2rm->insert(fake_quant_in_scale);
  nodes2rm->insert(fake_quant_out);
  nodes2rm->insert(fake_quant_out_scale);
}

void QuantDequantXPUPass::CollectFakeDequantizeOps(
    ir::Graph* graph,
    Node* op_node,
    std::unordered_set<const Node*>* nodes2rm) const {
  auto* op_desc = op_node->Op();
  auto x_var_name = op_desc->Input("X")[0];
  auto out_var_name = op_desc->Output("Out")[0];

  Node* fake_dequant_in = nullptr;
  for (auto* node_input : op_node->inputs) {
    if (node_input->Name() == x_var_name) {
      fake_dequant_in = node_input;
      break;
    }
  }

  Node* fake_dequant_out = nullptr;
  for (auto* node_output : op_node->outputs) {
    if (node_output->Name() == out_var_name) {
      fake_dequant_out = node_output;
      break;
    }
  }

  PADDLE_ENFORCE_NOT_NULL(
      fake_dequant_in,
      common::errors::NotFound(
          "The input var [%s] of dequantize op is not found.", x_var_name));
  PADDLE_ENFORCE_NOT_NULL(
      fake_dequant_out,
      common::errors::NotFound(
          "The output var [%s] of dequantize op is not found.", out_var_name));

  std::string input_act_name = fake_dequant_in->Var()->Name();
  std::string output_act_name = fake_dequant_out->Var()->Name();
  auto outlinks = fake_dequant_out->outputs;
  for (auto* next_node : outlinks) {
    next_node->Op()->RenameInput(output_act_name, input_act_name);
    IR_NODE_LINK_TO(fake_dequant_in, next_node);
  }

  nodes2rm->insert(op_node);
  nodes2rm->insert(fake_dequant_out);
}

void QuantDequantXPUPass::CollectQuantizeDequantizeOpsFromONNXFormat(
    ir::Graph* graph,
    Node* op_node,
    std::unordered_set<const Node*>* nodes2rm) const {
  auto* op_desc = op_node->Op();
  auto x_var_name = op_desc->Input("X")[0];
  auto in_scale_name = op_desc->Input("Scale")[0];
  auto in_zero_name = op_desc->Input("ZeroPoint")[0];
  auto out_var_name = op_desc->Output("Y")[0];

  Node* fake_quant_in = nullptr;
  Node* fake_quant_in_scale = nullptr;
  for (auto* node_input : op_node->inputs) {
    if (node_input->Name() == x_var_name) {
      fake_quant_in = node_input;
    } else if (node_input->Name() == in_scale_name) {
      fake_quant_in_scale = node_input;
    }
  }

  Node* fake_quant_out = nullptr;
  for (auto* node_output : op_node->outputs) {
    if (node_output->Name() == out_var_name) {
      fake_quant_out = node_output;
    }
  }

  PADDLE_ENFORCE_NOT_NULL(
      fake_quant_in,
      common::errors::NotFound(
          "The input var [%s] of quantize op is not found.", x_var_name));
  PADDLE_ENFORCE_NOT_NULL(
      fake_quant_in_scale,
      common::errors::NotFound(
          "The scale var [%s] of quantize op is not found.", in_scale_name));
  PADDLE_ENFORCE_NOT_NULL(
      fake_quant_out,
      common::errors::NotFound(
          "The output var [%s] of quantize op is not found.", out_var_name));

  std::string input_act_name = fake_quant_in->Var()->Name();
  std::string output_act_name = fake_quant_out->Var()->Name();
  for (auto* next_node : fake_quant_out->outputs) {
    if (!next_node->IsOp()) continue;
    next_node->Op()->RenameInput(output_act_name, input_act_name);
    IR_NODE_LINK_TO(fake_quant_in, next_node);
  }

  nodes2rm->insert(op_node);
  nodes2rm->insert(fake_quant_in_scale);
  nodes2rm->insert(fake_quant_out);
}

void QuantDequantXPUPass::RemoveFakeOps(
    ir::Graph* graph,
    const std::unordered_set<std::string>& fake_quantize_types,
    const std::unordered_set<std::string>& fake_dequantize_types,
    const std::unordered_set<std::string>& fake_quantize_dequantize_types,
    const std::unordered_set<std::string>&
        onnx_format_quantize_dequantize_types) const {
  VLOG(3) << "remove fake quantize and dequantize ops";

  std::unordered_set<const Node*> nodes2rm = {};
  for (auto* op_node :
       ir::TopologyVariantSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    if (fake_quantize_types.count(op_node->Name())) {
      CollectFakeQuantizeOps(graph, op_node, &nodes2rm);
    } else if (fake_dequantize_types.count(op_node->Name()) ||
               fake_quantize_dequantize_types.count(op_node->Name())) {
      CollectFakeDequantizeOps(graph, op_node, &nodes2rm);
    } else if (onnx_format_quantize_dequantize_types.count(op_node->Name())) {
      CollectQuantizeDequantizeOpsFromONNXFormat(graph, op_node, &nodes2rm);
    }
  }

  GraphSafeRemoveNodes(graph, nodes2rm);
}

void QuantDequantXPUPass::RestoreWeightsToInt8(
    Scope* scope,
    const std::unordered_map<std::string, std::vector<float>>&
        weight_thresholds) const {
  std::vector<float> scales;
  for (auto it : weight_thresholds) {
    auto weight_var_name = it.first;
    auto* var = scope->FindVar(weight_var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::NotFound(
            "The input persistable [%s] var of [%s] op is not found.",
            weight_var_name));
    auto* weight_tensor = var->GetMutable<phi::DenseTensor>();
    float* fp32_weight_data = weight_tensor->data<float>();
    std::vector<int8_t> weight_data;
    weight_data.resize(weight_tensor->numel());
    for (int i = 0; i < weight_tensor->numel(); i++) {
      weight_data[i] = static_cast<int8_t>(fp32_weight_data[i]);
    }
    const auto weight_dims = weight_tensor->dims();
    weight_tensor->clear();  // clear int weight
    weight_tensor->set_type(phi::DataType::INT8);
    weight_tensor->Resize(common::make_ddim(common::vectorize(weight_dims)));
    auto* cpu_ctx = static_cast<phi::CPUContext*>(
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
    auto* new_weight_data = cpu_ctx->Alloc<int8_t>(weight_tensor);
    memcpy(new_weight_data,
           weight_data.data(),
           weight_tensor->numel() * sizeof(int8_t));
  }
}

void QuantDequantXPUPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Convert (old) paddle slim quantized model to quantized model.";
  const std::string pattern_name = "quant_dequant_xpu_pass";
  FusePassBase::Init(pattern_name, graph);

  const std::unordered_set<std::string> skip_ops = {"conv2d",
                                                    "depthwise_conv2d",
                                                    "fused_conv2d",
                                                    "mul",
                                                    "matmul",
                                                    "matmul_v2"};

  const std::unordered_set<std::string> fake_quantize_types = {
      "fake_quantize_moving_average_abs_max", "fake_quantize_range_abs_max"};

  const std::unordered_set<std::string> fake_dequantize_types = {
      "fake_dequantize_max_abs", "fake_channel_wise_dequantize_max_abs"};

  const std::unordered_set<std::string> fake_quantize_dequantize_types = {
      "fake_quantize_dequantize_abs_max",
      "fake_quantize_dequantize_moving_average_abs_max",
      "fake_channel_wise_quantize_dequantize_abs_max"};

  const std::unordered_set<std::string> onnx_format_quantize_dequantize_types =
      {"quantize_linear", "dequantize_linear"};

  std::unordered_map<std::string, std::vector<float>> weight_thresholds{};
  std::unordered_map<std::string, std::vector<float>> var_quant_scales{};
  auto* scope = param_scope();
  CollectWeightScalesInfoFromDequantize(
      graph, scope, fake_dequantize_types, &weight_thresholds);
  CollectWeightScalesInfoFromONNXFormatDequantize(
      graph, scope, &weight_thresholds, &var_quant_scales);
  CollectInputScalesFromQuantize(
      graph, scope, fake_quantize_types, &var_quant_scales);
  CollectOutputScalesFromAttr(graph, &var_quant_scales);
  RemoveFakeOps(graph,
                fake_quantize_types,
                fake_dequantize_types,
                fake_quantize_dequantize_types,
                onnx_format_quantize_dequantize_types);
  RestoreWeightsToInt8(scope, weight_thresholds);

  SaveQuantInfoInTheGraph(
      graph, "has_quant_info", "var_quant_scales", weight_thresholds);
  SaveQuantInfoInTheGraph(
      graph, "has_quant_info", "var_quant_scales", var_quant_scales);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(quant_dequant_xpu_pass,
              paddle::framework::ir::QuantDequantXPUPass);

REGISTER_PASS_CAPABILITY(quant_dequant_xpu_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("fc", 0)
            .LE("conv2d_transpose", 2)
            .EQ("fake_quantize_abs_max", 0)
            .EQ("fake_quantize_range_abs_max", 0)
            .EQ("fake_quantize_moving_average_abs_max", 0)
            .LE("fake_channel_wise_quantize_abs_max", 1)
            .EQ("fake_dequantize_max_abs", 0));
