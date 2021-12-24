// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/quant_dequant_mkldnn_fuse_pass.h"
#include <string>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void QuantDequantMkldnnFusePass::MarkSkipQuantizedOps(
    ir::Graph* graph, std::unordered_set<std::string> skip_ops) const {
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    if (skip_ops.count(op_node->Name())) {
      auto* op_desc = op_node->Op();
      if (!op_desc->HasAttr("quantization_type")) {
        bool is_quantized_op = true;
        for (auto* node_input : op_node->inputs) {
          for (auto* node_input_input : node_input->inputs) {
            if (!node_input_input->IsOp()) continue;
            if (op_node->Name().find("quantize_dequantize") ==
                std::string::npos) {
              is_quantized_op = false;
              break;
            }
          }
          if (!is_quantized_op) break;
        }

        if (!is_quantized_op) {
          op_node->Op()->SetAttr("skip_quant", true);
        }
      }
    }
  }
}

void QuantDequantMkldnnFusePass::GatherInfoFromFake(
    ir::Graph* graph, Scope* scope,
    std::unordered_set<std::string> fake_dequantize_types,
    std::unordered_map<std::string, std::vector<float>> weight_thresholds)
    const {
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    if (fake_dequantize_types.count(op_node->Name())) {
      auto* op_desc = op_node->Op();
      auto x_var_name = op_desc->Input("X")[0];
      if (op_desc->HasAttr("max_range")) {
        float max_range = BOOST_GET_CONST(float, op_desc->GetAttr("max_range"));
        weight_thresholds[x_var_name].push_back(127 * 127 / max_range);
      } else {
        auto scale_name = op_desc->Input("Scales")[0];
        // scope->FindVar(scale_name)判空？
        const LoDTensor& scale_tensor =
            scope->FindVar(scale_name)->Get<LoDTensor>();
        const float* scale_data = scale_tensor.data<float>();
        for (int i = 0; i < scale_tensor.numel(); i++) {
          weight_thresholds[x_var_name].push_back(scale_data[i]);
        }
      }
    }
  }
}

void QuantDequantMkldnnFusePass::GatherInputScalesFromFake(
    ir::Graph* graph, Scope* scope,
    std::unordered_set<std::string> fake_quantize_types,
    std::unordered_map<std::string, std::pair<int, std::vector<float>>>
        var_quant_scales) const {
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    if (op_node->Name() == "fake_quantize_dequantize_moving_average_abs_max" ||
        fake_quantize_types.count(op_node->Name())) {
      auto* op_desc = op_node->Op();
      int bit_length = BOOST_GET_CONST(int, op_desc->GetAttr("bit_length"));
      PADDLE_ENFORCE_EQ(bit_length, 8, platform::errors::InvalidArgument(
                                           "Unsupported number quantization "
                                           "bits: %d, only 8 is supported now.",
                                           bit_length));

      auto x_var_name = op_desc->Input("X")[0];
      auto scale_name = op_desc->Input("InScale")[0];
      auto out_var_name = op_desc->Output("Out")[0];
      auto* scale_tensor = scope->FindVar(scale_name)->GetMutable<LoDTensor>();
      auto scale_data = scale_tensor->mutable_data<float>(platform::CPUPlace());
      float scale = 1.0 / scale_data[0];

      auto iter_in = var_quant_scales.find(x_var_name);
      if (iter_in == var_quant_scales.end()) {
        std::vector<float> scale_vector = {scale};
        var_quant_scales[x_var_name] == std::make_pair(0, scale_vector);
      }

      auto iter_out = var_quant_scales.find(out_var_name);
      if (iter_out == var_quant_scales.end()) {
        std::vector<float> scale_vector = {scale};
        var_quant_scales[out_var_name] == std::make_pair(0, scale_vector);
      }
    }
  }
}

void QuantDequantMkldnnFusePass::GatherOutputScalesFromAttr(
    ir::Graph* graph,
    std::unordered_map<std::string, std::pair<int, std::vector<float>>>
        var_quant_scales) const {
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    auto* op_desc = op_node->Op();
    if (op_desc->HasAttr("out_threshold")) {
      float attr_scale =
          BOOST_GET_CONST(float, op_desc->GetAttr("out_threshold"));
      if (attr_scale == 0.0) continue;
      float scale = 1.0 / attr_scale;

      auto var_name_map = op_desc->Outputs();
      for (auto iter = var_name_map.begin(); iter != var_name_map.end();
           ++iter) {
        for (auto var_name : iter->second) {
          std::vector<float> scale_vector = {scale};
          var_quant_scales[var_name] == std::make_pair(0, scale_vector);
        }
      }
    }
  }
}

void QuantDequantMkldnnFusePass::RemoveFakeOps(
    ir::Graph* graph, std::unordered_set<std::string> fake_quantize_types,
    std::unordered_set<std::string> fake_dequantize_types,
    std::unordered_set<std::string> fake_quantize_dequantize_types) const {
  auto collect_fake_quantize = [&](ir::Graph* graph, Node* op_node,
                                   std::unordered_set<const Node*>& nodes2rm) {
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

    std::string input_act_name = fake_quant_in->Var()->Name();
    std::string output_act_name = fake_quant_out->Var()->Name();
    auto outlinks = fake_quant_out->outputs;
    for (auto* next_node : outlinks) {
      next_node->Op()->RenameInput(output_act_name, input_act_name);
      IR_NODE_LINK_TO(fake_quant_in, next_node);
    }

    nodes2rm.insert(op_node);
    nodes2rm.insert(fake_quant_in_scale);
    nodes2rm.insert(fake_quant_out);
    nodes2rm.insert(fake_quant_out_scale);
  };

  auto collect_fake_dequantize = [&](
      ir::Graph* graph, Node* op_node,
      std::unordered_set<const Node*>& nodes2rm) {
    auto* op_desc = op_node->Op();
    auto x_var_name = op_desc->Input("X")[0];
    auto out_var_name = op_desc->Output("Out")[0];

    Node* fake_dequant_in = nullptr;
    for (auto* node_input : op_node->inputs) {
      if (node_input->Name() == x_var_name) {
        fake_dequant_in = node_input;
      }
    }

    Node* fake_dequant_out = nullptr;
    for (auto* node_output : op_node->outputs) {
      if (node_output->Name() == out_var_name) {
        fake_dequant_out = node_output;
      }
    }

    std::string input_act_name = fake_dequant_in->Var()->Name();
    std::string output_act_name = fake_dequant_out->Var()->Name();
    auto outlinks = fake_dequant_out->outputs;
    for (auto* next_node : outlinks) {
      next_node->Op()->RenameInput(output_act_name, input_act_name);
      IR_NODE_LINK_TO(fake_dequant_in, next_node);
    }

    nodes2rm.insert(op_node);
    nodes2rm.insert(fake_dequant_out);
  };

  std::unordered_set<const Node*> nodes2rm = {};
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    if (fake_quantize_types.count(op_node->Name())) {
      collect_fake_quantize(graph, op_node, nodes2rm);
    } else if (fake_dequantize_types.count(op_node->Name())) {
      collect_fake_dequantize(graph, op_node, nodes2rm);
    } else if (fake_quantize_dequantize_types.count(op_node->Name())) {
      collect_fake_dequantize(graph, op_node, nodes2rm);
    }
  }

  GraphSafeRemoveNodes(graph, nodes2rm);
}

void QuantDequantMkldnnFusePass::DequantizeWeights(
    ir::Graph* graph, Scope* scope,
    std::unordered_map<std::string, std::vector<float>> weight_thresholds)
    const {
  auto is_int8_weights = [&](Node* op_node, Scope* scope,
                             std::string weight_name) -> bool {
    auto* op_desc = op_node->Op();
    auto var_name = op_desc->Input(weight_name)[0];
    std::cout << "var_name: " << var_name << std::endl;
    if (scope->FindVar(var_name) == nullptr) {
      std::cout << "eeeeeeeeeeeeeee" << std::endl;
      return false;
    }
    auto* weight_tensor = scope->FindVar(var_name)->GetMutable<LoDTensor>();
    return weight_tensor->type() == framework::proto::VarType::INT8;
  };

  auto dequantize_op_weights = [&](Node* op_node, Scope* scope,
                                   std::string weight_name,
                                   std::string output_name) {
    auto* op_desc = op_node->Op();
    auto weight_var_name = op_desc->Input(weight_name)[0];
    auto output_var_name = op_desc->Output(output_name)[0];
    std::vector<float> scales = weight_thresholds[output_var_name];
    auto* weight_tensor =
        scope->FindVar(weight_var_name)->GetMutable<LoDTensor>();

    int size = scales.size();
    if (size == 1 || size == weight_tensor->dims()[0]) {
      auto weight_data =
          weight_tensor->mutable_data<int8_t>(platform::CPUPlace());
      for (int i = 0; i < weight_tensor->numel(); i++) {
        weight_data[i] /= 127;
      }
      // } else if (weight_tensor->dims().size() > 1 && scales.size() ==
      // weight_tensor->dims()[1]) {

    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The size of weight scales vector (%d) does not "
          "match the dimensions (%d) of the weights tensor %s.",
          size, weight_tensor->dims().size(), weight_var_name));
    }
  };

  std::cout << "scope11111: " << static_cast<void*>(scope) << std::endl;
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;
    std::cout << "8888888888" << std::endl;
    if (op_node->Name() == "conv2d" || op_node->Name() == "depthwise_conv2d") {
      std::cout << "9999999999" << std::endl;
      if (is_int8_weights(op_node, scope, "Filter")) {
        std::cout << "95555555555" << std::endl;
        dequantize_op_weights(op_node, scope, "Filter", "Output");
      }
    } else if (op_node->Name() == "mul" || op_node->Name() == "matmul" ||
               op_node->Name() == "matmul_v2") {
      std::cout << "qqqqqqqqq" << std::endl;
      if (is_int8_weights(op_node, scope, "Y")) {
        dequantize_op_weights(op_node, scope, "Y", "Out");
      }
    }
  }
}

void QuantDequantMkldnnFusePass::UpdateActivations(ir::Graph* graph) const {
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    if (op_node->Name() == "conv2d" || op_node->Name() == "depthwise_conv2d") {
      auto* op_desc = op_node->Op();
      if (!op_desc->HasAttr("fuse_activation")) {
        std::string activation;
        if (op_desc->HasAttr("fuse_relu")) {
          bool fuse_relu = BOOST_GET_CONST(bool, op_desc->GetAttr("fuse_relu"));
          if (fuse_relu) activation = "relu";
        } else if (op_desc->HasAttr("fuse_brelu")) {
          bool fuse_brelu =
              BOOST_GET_CONST(bool, op_desc->GetAttr("fuse_relu"));
          if (fuse_brelu) {
            activation = "relu6";
            float alpha = 6.0;
            if (op_desc->HasAttr("fuse_brelu_threshold")) {
              alpha = BOOST_GET_CONST(float,
                                      op_desc->GetAttr("fuse_brelu_threshold"));
            }
            op_node->Op()->SetAttr("fuse_alpha", alpha);
          }
        }
        op_node->Op()->SetAttr("fuse_activation", activation);
      }
    }
  }
}

void QuantDequantMkldnnFusePass::RemoveCtrlVars(ir::Graph* graph) const {
  std::unordered_set<const Node*> nodes2rm = {};
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (op_node->IsCtrlVar()) {
      nodes2rm.insert(op_node);
    }
  }

  GraphSafeRemoveNodes(graph, nodes2rm);
}

void QuantDequantMkldnnFusePass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "quant_dequant_mkldnn_fuse_pass";
  FusePassBase::Init(pattern_name, graph);

  std::unordered_set<std::string> skip_ops = {"conv2d", "depthwise_conv2d",
                                              "mul", "matmul", "matmul_v2"};
  std::unordered_set<std::string> fake_quantize_types = {
      "fake_quantize_moving_average_abs_max", "fake_quantize_range_abs_max"};
  std::unordered_set<std::string> fake_dequantize_types = {
      "fake_dequantize_max_abs", "fake_channel_wise_dequantize_max_abs"};
  std::unordered_set<std::string> fake_quantize_dequantize_types = {
      "fake_quantize_dequantize_abs_max",
      "fake_quantize_dequantize_moving_average_abs_max",
      "fake_channel_wise_quantize_dequantize_abs_max"};

  std::unordered_map<std::string, std::vector<float>> weight_thresholds;
  std::unordered_map<std::string, std::pair<int, std::vector<float>>>
      var_quant_scales;

  auto* scope = param_scope();
  std::cout << "scope: " << static_cast<void*>(scope) << std::endl;

  MarkSkipQuantizedOps(graph, skip_ops);
  std::cout << "11111111" << std::endl;
  GatherInfoFromFake(graph, scope, fake_dequantize_types, weight_thresholds);
  std::cout << "2222222" << std::endl;
  GatherInputScalesFromFake(graph, scope, fake_quantize_types,
                            var_quant_scales);
  std::cout << "333333333" << std::endl;
  GatherOutputScalesFromAttr(graph, var_quant_scales);
  std::cout << "444444444" << std::endl;
  RemoveFakeOps(graph, fake_quantize_types, fake_dequantize_types,
                fake_quantize_dequantize_types);
  std::cout << "555555555" << std::endl;
  DequantizeWeights(graph, scope, weight_thresholds);
  std::cout << "666666666" << std::endl;
  UpdateActivations(graph);
  std::cout << "77777777" << std::endl;
  RemoveCtrlVars(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(quant_dequant_mkldnn_fuse_pass,
              paddle::framework::ir::QuantDequantMkldnnFusePass);

REGISTER_PASS_CAPABILITY(quant_dequant_mkldnn_fuse_pass)
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
