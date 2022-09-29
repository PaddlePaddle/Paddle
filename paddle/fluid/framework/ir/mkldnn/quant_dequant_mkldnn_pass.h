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

#pragma once

#include <string>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {

class QuantDequantMkldnnPass : public FusePassBase {
 public:
  QuantDequantMkldnnPass() = default;
  virtual ~QuantDequantMkldnnPass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void MarkSkipQuantizedOps(
      ir::Graph* graph, const std::unordered_set<std::string>& skip_ops) const;

  void MarkSkipQuantizedPool2d(ir::Graph* graph) const;

  void CollectInfoFromFake(
      ir::Graph* graph,
      Scope* scope,
      const std::unordered_set<std::string>& fake_dequantize_types,
      std::unordered_map<std::string, std::vector<float>>* weight_thresholds)
      const;

  ///
  /// \brief collect scale info for weight from onnx_format dequantize_linear op
  /// onnx_format_dequantize_types: the onnx_format dequantize op type
  /// weight_thresholds: scale info for weight
  /// var_quant_scales: scale info for act
  /// onnx_format_quantize_model: recorder if the quantize model is a
  /// onnx_format quantize model
  ///
  void CollectWeightScalesInfoFromONNXFormatDequantize(
      ir::Graph* graph,
      Scope* scope,
      std::unordered_map<std::string, std::vector<float>>* weight_thresholds,
      std::unordered_map<std::string, std::vector<float>>* var_quant_scales,
      bool* onnx_format_quantize_model) const;

  void CollectInputScalesFromQuantize(
      ir::Graph* graph,
      Scope* scope,
      const std::unordered_set<std::string>& fake_quantize_types,
      std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
      const;

  void ConvertFromINT8ToFP32(const std::vector<float>& scales,
                             phi::DenseTensor* weight_tensor,
                             int8_t* int8_weight_data,
                             float* fp32_weight_data,
                             const std::string& weight_var_name) const;

  void CollectOutputScalesFromAttr(
      ir::Graph* graph,
      std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
      const;

  void CollectFakeQuantizeOps(ir::Graph* graph,
                              Node* op_node,
                              std::unordered_set<const Node*>* nodes2rm) const;

  void CollectFakeDequantizeOps(
      ir::Graph* graph,
      Node* op_node,
      std::unordered_set<const Node*>* nodes2rm) const;

  ///
  /// \brief collect all the onnx_format quantize related ops to remove
  /// nodes2rm: record all quantize related ops to remove
  ///
  void CollectQuantizeDequantizeOpsFromONNXFormat(
      ir::Graph* graph,
      Node* op_node,
      std::unordered_set<const Node*>* nodes2rm) const;

  void RemoveFakeOps(
      ir::Graph* graph,
      const std::unordered_set<std::string>& fake_quantize_types,
      const std::unordered_set<std::string>& fake_dequantize_types,
      const std::unordered_set<std::string>& fake_quantize_dequantize_types,
      const std::unordered_set<std::string>&
          onnx_format_quantize_dequantize_types) const;

  bool IsInt8Weight(Node* op_node,
                    Scope* scope,
                    const std::string& weight_name) const;

  void TransposeWeight(phi::DenseTensor* input) const;

  void DequantizeOpWeights(
      Node* op_node,
      Scope* scope,
      const std::string& weight_name,
      const std::string& output_name,
      const std::unordered_map<std::string, std::vector<float>>&
          weight_thresholds) const;

  ///
  /// \brief Dequantize weight in conv or matmul
  /// weight_thresholds: recorded scale info for weight
  ///
  void DequantizeOpWeightsFromONNXFormat(
      Node* op_node,
      Scope* scope,
      const std::string& weight_name,
      const std::unordered_map<std::string, std::vector<float>>&
          weight_thresholds) const;

  void DequantizeWeights(
      ir::Graph* graph,
      Scope* scope,
      const std::unordered_map<std::string, std::vector<float>>&
          weight_thresholds,
      const bool& onnx_format_quantize_model) const;

  void UpdateActivations(ir::Graph* graph) const;

  void RemoveCtrlVars(ir::Graph* graph) const;
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
