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
#include <vector>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {

class QuantDequantMkldnnV2FusePass : public FusePassBase {
 public:
  QuantDequantMkldnnV2FusePass() = default;
  virtual ~QuantDequantMkldnnV2FusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  //   void MarkSkipQuantizedOps(ir::Graph* graph,
  //                             std::unordered_set<std::string> skip_ops)
  //                             const;

  void GatherInputWeightsScalesFromFake(
      ir::Graph* graph, Scope* scope,
      std::unordered_set<std::string> quantize_linear_types,
      std::unordered_set<std::string> dequantize_linear_types,
      std::unordered_map<std::string, std::vector<float>>* weight_thresholds,
      std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
      const;

  void RemoveQuantDequantLinearOps(
      ir::Graph* graph, Scope* scope) const;
    
  void RemoveDequantLinearOps(
      ir::Graph* graph, Scope* scope) const;


  const std::string qdq_name_scope_{"quantize_dequantize_linear_fuse"};
  const std::string dq_name_scope_{"dequantize_linear_fuse"};
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
