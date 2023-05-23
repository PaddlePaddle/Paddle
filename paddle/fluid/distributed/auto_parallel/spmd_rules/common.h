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

#include "paddle/fluid/framework/type_defs.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

class SPMDRuleBase {
 public:
  virtual ~SPMDRuleBase() {}

  virtual std::vector<DistTensorSpec> InferForward(
      const std::vector<DistTensorSpec>& input_specs,
      const paddle::framework::AttributeMap& attrs);

  virtual std::vector<DistTensorSpec> InferBackward(
      const std::vector<DistTensorSpec>& output_specs,
      const paddle::framework::AttributeMap& attrs);

  template <typename T>
  inline const T& ExtractAttr(
      const std::string& name,
      const paddle::framework::AttributeMap& attrs) const {
    return PADDLE_GET_CONST(T, GetAttr(name, attrs));
  }

  virtual const Attribute& GetAttr(
      const std::string& name,
      const paddle::framework::AttributeMap& attrs) const {
    auto iter = attrs.find(name);
    PADDLE_ENFORCE_NE(
        iter,
        attrs.end(),
        platform::errors::NotFound("(%s) is not found in AttributeMap."));
    return iter->second;
  }
};

std::unordered_map<std::string, int64_t> ShardingMergeForTensors(
    const std::vector<std::pair<const std::string, const std::vector<int64_t>>>&
        tensor_notation_to_dim_pairs);

// Rule1: A repicated dimension could be merged by any sharded dimension.
// Rule2: A tensor axis could at most be sharded by one mesh dimension.
// (TODO trigger heuristics cost model and reshard to handle axis sharded by
// multiple dimension case.)
int64_t ShardingMergeForAxis(const std::string axis,
                             const int64_t mesh_dim1,
                             const int64_t mesh_dim2);

TensorDistAttr CopyTensorDistAttrForOutput(const TensorDistAttr& src_dist_attr);

std::vector<int64_t> ResoluteOutputPartialDimension(
    const std::unordered_map<std::string, int64_t>& in_axis_to_dim_map,
    const std::string& out_axis);

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
