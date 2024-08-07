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
#include "paddle/fluid/framework/ir/onednn/onednn_pass_util.h"

namespace paddle {
namespace framework {
namespace ir {

class ComputePropagateScalesMkldnnPass : public FusePassBase {
 public:
  ComputePropagateScalesMkldnnPass() = default;
  virtual ~ComputePropagateScalesMkldnnPass() {}

#ifdef PADDLE_WITH_TESTING
  friend class ComputePropagateScalesMkldnnPassTest;
#endif

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void GetTensorFromVector(const std::vector<float>& data_v,
                           phi::DenseTensor* tensor) const;

  void GetQuantInfo(ir::Graph* graph, StringPairMap* var_quant_scales) const;

  std::vector<float> GetScales(phi::DenseTensor* tensor, int axis) const;

  void ComputeVarScales(ir::Graph* graph,
                        Scope* scope,
                        const std::unordered_set<std::string>& ops,
                        const std::string& weight_name,
                        const int axis,
                        StringPairMap* var_quant_scales) const;

  void ComputeSingleGruWeightScales(Scope* scope,
                                    const std::string& wx_var_name,
                                    const std::string& wh_var_name,
                                    phi::DenseTensor* tensor) const;

  void ComputeGruWeightScales(ir::Graph* graph,
                              Scope* scope,
                              const std::string& wx_name,
                              const std::string& wh_name,
                              StringPairMap* var_quant_scales) const;

  void ComputeSingleLstmWeightScales(Scope* scope,
                                     const std::string& wx_var_name,
                                     const std::string& wh_var_name,
                                     phi::DenseTensor* tensor) const;

  void ComputeLstmWeightScales(ir::Graph* graph,
                               Scope* scope,
                               const std::string& wx_name,
                               const std::string& wh_name,
                               StringPairMap* var_quant_scales) const;

  void ComputeWeightScales(ir::Graph* graph,
                           Scope* scope,
                           StringPairMap* var_quant_scales) const;

  void UpdateReluOutputScales(ir::Graph* graph,
                              StringPairMap* var_quant_scales) const;

  void UpdateScaleOpInOutScales(Node* op_node,
                                const std::string& input_name,
                                const std::string& output_name,
                                StringPairMap* var_quant_scales) const;

  std::unordered_set<std::string> UpdateScales(
      ir::Graph* graph,
      StringPairMap* var_quant_scales,
      const std::unordered_set<std::string>& scale_immutable_ops) const;

  void PropagateScales(
      ir::Graph* graph,
      StringPairMap* var_quant_scales,
      const std::unordered_set<std::string>& scale_immutable_ops) const;
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
