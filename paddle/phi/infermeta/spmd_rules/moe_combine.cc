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

#include "paddle/phi/infermeta/spmd_rules/moe_combine.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

SpmdInfo MoECombineFwdInferSpmd(const DistMetaTensor& x,
                                const DistMetaTensor& combine_weights,
                                const DistMetaTensor& scatter_index) {
  /* kernel logic:
  y is [seqlen, hidden_size]
  for kk in k:
    y[i][j] += x[scatter_index[i][kk]][j] * combine_weights[i][kk]
  */

  // Step 0: validity check
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(x);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(combine_weights);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(scatter_index);

  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      2,
      errors::InvalidArgument(
          "x should be a 2-D tensor, but got x_shape.size() == %d",
          x_shape.size()));
  PADDLE_ENFORCE_EQ(
      combine_weights_shape.size(),
      2,
      errors::InvalidArgument("combine_weights should be a 2-D tensor, but got "
                              "combine_weights_shape.size() == %d",
                              combine_weights.size()));
  PADDLE_ENFORCE_EQ(
      scatter_index_shape.size(),
      2,
      errors::InvalidArgument("scatter_index should be a 2-D tensor, but got "
                              "scatter_index_shape.size() == %d",
                              scatter_index.size()));

  // Step 1: infer sharding
  std::string x_axes = "sh", combine_weights_axes = "sk",
              scatter_index_axes = "sk", out_axes = "sh";
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(
          {{x_axes, x_dims_mapping_src},
           {combine_weights_axes, combine_weights_dims_mapping_src},
           {scatter_index_axes, scatter_index_dims_mapping_src}});

  if (axis_to_dim_map["k"] != -1) {
    axis_to_dim_map["h"] =
        -1;  // Not allowed that k-dim and h-dim both be sharded
  }

  std::vector<int64_t> y_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr combine_weights_dist_attr_dst =
      CopyTensorDistAttrForOutput(combine_weights_dist_attr_src);
  TensorDistAttr scatter_index_dist_attr_dst =
      CopyTensorDistAttrForOutput(scatter_index_dist_attr_src);
  TensorDistAttr y_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);

  x_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(x_axes, axis_to_dim_map));
  combine_weights_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(combine_weights_axes, axis_to_dim_map));
  scatter_index_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(scatter_index_axes, axis_to_dim_map));
  y_dist_attr_dst.set_dims_mapping(y_dims_mapping);

  // Step 2: infer partial, the output h-dim is partial when k is sharded
  if (axis_to_dim_map["k"] != -1) {
    y_dist_attr_dst.set_partial_status(std::vector<int64_t>({1}));
  }

  // Step 3: Log messages
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(combine_weights);
  LOG_SPMD_INPUT(scatter_index);
  LOG_SPMD_OUTPUT(y_dist_attr_dst);

  return {{x_dist_attr_dst,
           combine_weights_dist_attr_dst,
           scatter_index_dist_attr_dst},
          {y_dist_attr_dst}};
}

SpmdInfo MoECombineBwdInferSpmd(const DistMetaTensor& x,
                                const DistMetaTensor& combine_weights,
                                const DistMetaTensor& scatter_index,
                                const DistMetaTensor& grad_y) {
  /* kernel logic:
  for(int i = 0; i < s; ++i) {
      for(int j = 0; j < h; ++j) {
          for(int ki = 0; ki < k; ++ki) {
              grad_x[scatter_index[i][ki]][j] = grad_y[i][j] *
  combine_weights[i][ki]; grad_combine_weights_helper[i][ki][j] = grad_y[i][j] *
  x[scatter_index[i][ki]][j];
          }
      }
  }
  */

  // step 0 : validity check
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(x);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(combine_weights);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(scatter_index);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(grad_y);

  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      2,
      errors::InvalidArgument(
          "x should be a 2-D tensor, but got x_shape.size() == %d",
          x_shape.size()));

  PADDLE_ENFORCE_EQ(
      combine_weights_shape.size(),
      2,
      errors::InvalidArgument("combine_weights should be a 2-D tensor, but got "
                              "combine_weights_shape.size() == %d",
                              combine_weights_shape.size()));
  PADDLE_ENFORCE_EQ(
      scatter_index_shape.size(),
      2,
      errors::InvalidArgument("scatter_index should be a 2-D tensor, but got "
                              "scatter_index_shape.size() == %d",
                              scatter_index_shape.size()));
  PADDLE_ENFORCE_EQ(
      grad_y_shape.size(),
      2,
      errors::InvalidArgument(
          "grad_y should be a 2-D tensor, but got grad_y_shape.size() == %d",
          grad_y_shape.size()));

  // step 1 : infer sharding
  std::string x_axes = "sh", combine_weights_axes = "sk",
              scatter_index_axes = "sk", grad_y_axes = "sh", grad_x_axes = "sh",
              grad_combine_weights_axes = "sk", grad_scatter_index_axes = "sk";
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(
          {{x_axes, x_dims_mapping_src},
           {combine_weights_axes, combine_weights_dims_mapping_src},
           {scatter_index_axes, scatter_index_dims_mapping_src},
           {grad_y_axes, grad_y_dims_mapping_src}});

  // k-dim should be replicated
  axis_to_dim_map["k"] = -1;

  std::vector<int64_t> grad_x_dims_mapping =
      GetDimsMappingForAxes(grad_x_axes, axis_to_dim_map);
  std::vector<int64_t> grad_combine_weights_dims_mapping =
      GetDimsMappingForAxes(grad_combine_weights_axes, axis_to_dim_map);
  std::vector<int64_t> grad_scatter_index_dims_mapping =
      GetDimsMappingForAxes(grad_scatter_index_axes, axis_to_dim_map);

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr combine_weights_dist_attr_dst =
      CopyTensorDistAttrForOutput(combine_weights_dist_attr_src);
  TensorDistAttr scatter_index_dist_attr_dst =
      CopyTensorDistAttrForOutput(scatter_index_dist_attr_src);
  TensorDistAttr grad_y_dist_attr_dst =
      CopyTensorDistAttrForOutput(grad_y_dist_attr_src);
  TensorDistAttr grad_x_dist_attr_dst =
      CopyTensorDistAttrForOutput(grad_y_dist_attr_src);
  TensorDistAttr grad_combine_weights_dist_attr_dst =
      CopyTensorDistAttrForOutput(grad_y_dist_attr_src);
  TensorDistAttr grad_scatter_index_dist_attr_dst =
      CopyTensorDistAttrForOutput(grad_y_dist_attr_src);

  x_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(x_axes, axis_to_dim_map));
  combine_weights_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(combine_weights_axes, axis_to_dim_map));
  scatter_index_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(scatter_index_axes, axis_to_dim_map));
  grad_y_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(grad_y_axes, axis_to_dim_map));
  grad_x_dist_attr_dst.set_dims_mapping(grad_x_dims_mapping);
  grad_combine_weights_dist_attr_dst.set_dims_mapping(
      grad_combine_weights_dims_mapping);
  grad_scatter_index_dist_attr_dst.set_dims_mapping(
      grad_scatter_index_dims_mapping);

  // Step 2: Log messages
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(combine_weights);
  LOG_SPMD_INPUT(scatter_index);
  LOG_SPMD_INPUT(grad_y);
  LOG_SPMD_OUTPUT(grad_x_dist_attr_dst);
  LOG_SPMD_OUTPUT(grad_combine_weights_dist_attr_dst);

  return {{x_dist_attr_dst,
           combine_weights_dist_attr_dst,
           scatter_index_dist_attr_dst,
           grad_y_dist_attr_dst},
          {grad_x_dist_attr_dst,
           grad_combine_weights_dist_attr_dst,
           grad_scatter_index_dist_attr_dst}};
}

}  // namespace distributed
}  // namespace phi
