/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/common.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

TensorDistAttr GetInferedDistAttr(
    const TensorDistAttr& origin_dist_attr,
    const std::vector<int64_t>& shape,
    const std::string& tensor_axes,
    const std::unordered_map<std::string, int64_t>& axis_to_dim_map,
    const bool trans_axis);

void FillMatmulOperandNotation(const int x_ndim,
                               const int y_ndim,
                               std::string* x_axes,
                               std::string* y_axes,
                               std::string* out_axes);

class MatmulSPMDRule : public SPMDRuleBase {
 public:
  std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
  InferForward(const std::vector<DistTensorSpec>& input_specs,
               const paddle::framework::AttributeMap& attrs) override;

  std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
  InferBackward(const std::vector<DistTensorSpec>& input_specs,
                const std::vector<DistTensorSpec>& output_specs,
                const paddle::framework::AttributeMap& attrs) override;
};
}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
