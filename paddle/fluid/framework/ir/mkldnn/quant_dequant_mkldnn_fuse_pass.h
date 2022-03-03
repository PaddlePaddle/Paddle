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

class QuantDequantMkldnnFusePass : public FusePassBase {
 public:
  QuantDequantMkldnnFusePass() = default;
  virtual ~QuantDequantMkldnnFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void MarkSkipQuantizedOps(ir::Graph* graph,
                            std::unordered_set<std::string> skip_ops) const;

  void GatherInfoFromFake(ir::Graph* graph, Scope* scope,
                          std::unordered_set<std::string> fake_dequantize_types,
                          std::unordered_map<std::string, std::vector<float>>&
                              weight_thresholds) const;

  void GatherInputScalesFromFake(
      ir::Graph* graph, Scope* scope,
      std::unordered_set<std::string> fake_quantize_types,
      std::unordered_map<std::string, std::vector<float>>&
          var_quant_scales)  // NOLINT
      const;

  void GatherOutputScalesFromAttr(
      ir::Graph* graph, std::unordered_map<std::string, std::vector<float>>&
                            var_quant_scales)  // NOLINT
      const;

  void RemoveFakeOps(
      ir::Graph* graph, std::unordered_set<std::string> fake_quantize_types,
      std::unordered_set<std::string> fake_dequantize_types,
      std::unordered_set<std::string> fake_quantize_dequantize_types) const;

  void DequantizeWeights(ir::Graph* graph, Scope* scope,
                         std::unordered_map<std::string, std::vector<float>>&
                             weight_thresholds) const;

  void UpdateActivations(ir::Graph* graph) const;

  void RemoveCtrlVars(ir::Graph* graph) const;

  void SaveQuantInfo(ir::Graph* graph,
                     std::unordered_map<std::string, std::vector<float>>&
                         weight_thresholds,  // NOLINT
                     std::unordered_map<std::string, std::vector<float>>&
                         var_quant_scales)  // NOLINT
      const;

 private:
  void TransposeWeight(Tensor* input) const;
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
