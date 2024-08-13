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

class QuantDequantXPUPass : public FusePassBase {
 public:
  QuantDequantXPUPass() = default;
  virtual ~QuantDequantXPUPass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void CollectWeightScalesInfoFromDequantize(
      ir::Graph* graph,
      Scope* scope,
      const std::unordered_set<std::string>& fake_dequantize_types,
      std::unordered_map<std::string, std::vector<float>>* weight_thresholds)
      const;

  void CollectWeightScalesInfoFromONNXFormatDequantize(
      ir::Graph* graph,
      Scope* scope,
      std::unordered_map<std::string, std::vector<float>>* weight_thresholds,
      std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
      const;

  void CollectInputScalesFromQuantize(
      ir::Graph* graph,
      Scope* scope,
      const std::unordered_set<std::string>& fake_quantize_types,
      std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
      const;

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

  void RestoreWeightsToInt8(
      Scope* scope,
      const std::unordered_map<std::string, std::vector<float>>&
          weight_thresholds) const;
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
