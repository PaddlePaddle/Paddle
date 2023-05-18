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
      const paddle::framework::AttributeMap& attrs) {
    PADDLE_THROW(
        phi::errors::Unimplemented("InferForward should be called from a "
                                   "derived class of SPMDRuleBase !"));
  }

  virtual std::vector<DistTensorSpec> InferBackward(
      const std::vector<DistTensorSpec>& output_specs,
      const paddle::framework::AttributeMap& attrs) {
    PADDLE_THROW(
        phi::errors::Unimplemented("InferBackward should be called from a "
                                   "derived class of SPMDRuleBase !"));
  }

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
}

std::unordered_map<std::string, int64_t>
ShardingMergeForTensors(
    const std::vector<std::pair<std::string, const std::vector<int64_t>>>&
        tensor_notation_to_dim_pairs) {
  std::unordered_map<std::string, int64_t> axis_to_dim_map;
  std::unordered_map<int64_t, std::string> dim_to_axis_map;

  for (auto& pair : tensor_notation_to_dim_pairs) {
    for (int i = 0; i < pair.second.size(); i++) {
      auto tensor_axis = pair.first[i];
      auto mesh_dim = pair.second[i];

      if (axis_to_dim_map.count(tensor_axis) == 0) {
        axis_to_dim_map.insert({tensor_axis, mesh_dim});

      } else {
        int64_t merge_dim = ShardingMergeForAxis(
            tensor_axis, mesh_dim, axis_to_dim_map[tensor_axis]);
        axis_to_dim_map.insert({tensor_axis, merge_dim});
      }

      if (dim_to_axis_map.count(mesh_dim) == 0) {
        dim_to_axis_map.insert({tensor_axis, mesh_dim});
      }
    }
  }
}

// Rule1: A repicated dimension could be merged by any sharded dimension.
// Rule2: A tensor axis could at most be sharded by one mesh dimension.
// (TODO trigger heuristics cost model and reshard to handle axis sharded by
// multiple dimension case.)
int64_t ShardingMergeForAxis(const std::string axis,
                             const int64_t mesh_dim1,
                             const int64_t mesh_dim2) {
  if (mesh_dim1 != mesh_dim2) {
    if (mesh_dim1 == -1) {
      return mesh_dim2;
    } else if (mesh_dim2 == -1) {
      return mesh_dim1;
    } else {
      PADDLE_THROW(
          phi::errors::Unimplemented("Tensor Axis[%s] is Sharded by two "
                                     "different mesh dimension [%d] and [%d].",
                                     axis,
                                     mesh_dim1,
                                     mesh_dim2));
    }

  } else {
    return mesh_dim1;
  }
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
