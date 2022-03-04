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

using StringTensorPtrMap = std::unordered_map<std::string, Tensor*>;
using StringPairMap = std::unordered_map<std::string, std::pair<bool, Tensor*>>;

class RequantMkldnnFusePass : public FusePassBase {
 public:
  RequantMkldnnFusePass() = default;
  virtual ~RequantMkldnnFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void GetTensorFromVector(const std::vector<float>& data_v,
                           Tensor* tensor) const;

  void GetQuantInfo(ir::Graph* graph, Scope* scope,
                    StringTensorPtrMap& weight_thresholds,  // NOLINT
                    StringPairMap& var_quant_scales)        // NOLINT
      const;

  void ComputeWeightScales(ir::Graph* graph, Scope* scope,
                           StringPairMap& var_quant_scales)  // NOLINT
      const;
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
