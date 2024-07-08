/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/add_n.h"
#include "paddle/phi/infermeta/spmd_rules/elementwise.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {
SpmdInfo AddNInferSpmd(
    const std::vector<phi::distributed::DistMetaTensor>& inputs) {
  auto N = inputs.size();
  PADDLE_ENFORCE_GT(
      N,
      0,
      phi::errors::InvalidArgument(
          "The inputs tensor's size of AddNOp must greater than 0."));
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::vector<std::pair<std::string, std::vector<int64_t>>>
      tensor_axes_to_dim_pairs;
  auto ndim = common::vectorize(inputs[0].dims()).size();
  auto axes = alphabet.substr(0, ndim);
  for (const auto& input : inputs) {
    auto input_shape = common::vectorize(input.dims());
    auto input_ndim = input_shape.size();
    TensorDistAttr input_dist_attr_src = input.dist_attr();
    std::vector<int64_t> input_dims_mapping =
        input_dist_attr_src.dims_mapping();
    PADDLE_ENFORCE_EQ(
        input_ndim,
        ndim,
        common::errors::InvalidArgument("AddNInferSpmd, The all input's rank "
                                        "shoule be the same as first input."));
    PADDLE_ENFORCE_EQ(ndim,
                      input_dims_mapping.size(),
                      common::errors::InvalidArgument(
                          "AddNInferSpmd, The all input's dimmapping size "
                          "shoule be the same as first input."));
    tensor_axes_to_dim_pairs.push_back(
        std::make_pair(axes, input_dims_mapping));
  }

  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(tensor_axes_to_dim_pairs);

  // Step2.2: Infer output dims mapping from merged input dims mapping
  std::vector<int64_t> dims_mapping =
      GetDimsMappingForAxes(axes, axis_to_dim_map);

  std::vector<TensorDistAttr> inputs_spmd_info;
  for (const auto& input : inputs) {
    TensorDistAttr dist_attr_dst =
        CopyTensorDistAttrForOutput(input.dist_attr());
    dist_attr_dst.set_dims_mapping(dims_mapping);
    inputs_spmd_info.push_back(dist_attr_dst);
  }
  // Handle input tensor partial(TODO)

  return {{inputs_spmd_info}, {inputs_spmd_info[0]}};
}

}  // namespace phi::distributed
