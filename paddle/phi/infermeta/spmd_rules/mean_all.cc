/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/mean_all.h"

#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

SpmdInfo MeanAllInferSpmd(const DistMetaTensor& x) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  // Setp1: Build einsum notation
  // get einsum notation for input
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string x_axes = alphabet.substr(0, x_ndim);

  // get einsum notation for output
  std::string out_axes = "";

  // Step2: Sharding Propagation
  // Step2.1: Merge input shardings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping_src}});

  // Step2.2: Infer output dims mapping from merged input dims mapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);

  TensorDistAttr out_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  // Step3: handle output partial
  std::vector<int64_t> partial_on_dims =
      ResoluteOutputPartialDimension(axis_to_dim_map, out_axes);
  out_dist_attr_dst.set_partial_status(partial_on_dims, ReduceType::kRedAvg);

  return {{x_dist_attr_src}, {out_dist_attr_dst}};
}

SpmdInfo MeanAllGradInferSpmd(const DistMetaTensor& x,
                              const DistMetaTensor& out_grad) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  // Clean partial status of out_grad
  TensorDistAttr out_grad_dist_attr_dst = out_grad.dist_attr();
  out_grad_dist_attr_dst.clean_partial_status();

  // Build einsum notation for x_grad and x
  std::string x_axes(x_ndim, '1');

  // Infer x and x_grad dims mapping
  auto x_dims_mapping_dst = GetDimsMappingForAxes(x_axes, {});
  auto x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  auto x_grad_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_grad_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);

  return {{x_dist_attr_dst, out_grad_dist_attr_dst}, {x_grad_dist_attr_dst}};
}

}  // namespace distributed
}  // namespace phi
