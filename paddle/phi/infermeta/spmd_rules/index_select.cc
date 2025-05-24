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

#include "paddle/phi/infermeta/spmd_rules/index_select.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

SpmdInfo IndexSelectInferSpmd(const DistMetaTensor& x,
                              const DistMetaTensor& index,
                              int axis) {
  // Step0: Verify Input
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(index);
  axis = axis < 0 ? x_ndim + axis : axis;
  PADDLE_ENFORCE_EQ(
      0 <= axis && axis < x_ndim,
      true,
      phi::errors::InvalidArgument(
          "The axis of index_select should be in range [0, %d), but got %d.",
          x_ndim,
          axis));

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijlmnopqrstuvwxyz";
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string index_axes = "k";
  std::string out_axes = x_axes;
  out_axes[axis] = 'k';

  // Step2: Sharding Propagation
  // Step2.1: Merge input shardings
  std::vector<int64_t> x_dims_mapping_dst(x_dims_mapping_src);
  x_dims_mapping_dst[axis] = -1;
  std::vector<int64_t> index_dims_mapping_dst(index_dims_mapping_src);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(
          {{x_axes, x_dims_mapping_dst}, {index_axes, index_dims_mapping_dst}});

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);

  TensorDistAttr index_dist_attr_dst =
      CopyTensorDistAttrForOutput(index_dist_attr_src);
  index_dist_attr_dst.set_dims_mapping(index_dims_mapping_dst);

  // Step2.2: Infer output dims mapping
  std::vector<int64_t> out_dims_mapping_dst =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);
  TensorDistAttr out_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping_dst);

  VLOG(4) << "IndexSelectInferSpmd: Done.";
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(index);
  LOG_SPMD_OUTPUT(out_dist_attr_dst);
  return {{x_dist_attr_dst, index_dist_attr_dst}, {out_dist_attr_dst}};
}

SpmdInfo IndexSelectGradInferSpmd(const DistMetaTensor& x,
                                  const DistMetaTensor& index,
                                  const DistMetaTensor& out_grad,
                                  int axis) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(index);
  EXTRACT_SHAPE_AND_DIST_ATTR(out_grad);
  axis = axis < 0 ? x_ndim + axis : axis;
  PADDLE_ENFORCE_EQ(
      0 <= axis && axis < x_ndim,
      true,
      phi::errors::InvalidArgument(
          "The axis of index_select should be in range [0, %d), but got %d.",
          x_ndim,
          axis));
  PADDLE_ENFORCE_EQ(x_ndim,
                    out_grad_ndim,
                    common::errors::InvalidArgument(
                        "IndexSelectGrad: The rank of x [%d] and outgrad [%d] "
                        "must be the same.",
                        x_ndim,
                        out_grad_ndim));

  std::string alphabet = "abcdefghijlmnopqrstuvwxyz";
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string index_axes = "k";
  std::string out_grad_axes = x_axes;
  out_grad_axes[axis] = 'k';

  std::vector<int64_t> x_dims_mapping_dst(x_dims_mapping_src);
  x_dims_mapping_dst[axis] = -1;
  std::vector<int64_t> index_dims_mapping_dst(index_dims_mapping_src);
  std::vector<int64_t> out_grad_dims_mapping_dst(out_grad_dims_mapping_src);

  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping_dst},
                               {index_axes, index_dims_mapping_dst},
                               {out_grad_axes, out_grad_dims_mapping_dst}});

  x_dims_mapping_dst = GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  out_grad_dims_mapping_dst =
      GetDimsMappingForAxes(out_grad_axes, axis_to_dim_map);
  index_dims_mapping_dst = GetDimsMappingForAxes(index_axes, axis_to_dim_map);

  TensorDistAttr x_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr index_dist_attr_dst =
      CopyTensorDistAttrForOutput(index_dist_attr_src);
  TensorDistAttr out_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_grad_dist_attr_src);

  x_grad_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  index_dist_attr_dst.set_dims_mapping(index_dims_mapping_dst);
  out_grad_dist_attr_dst.set_dims_mapping(out_grad_dims_mapping_dst);

  // Handle partial if index and out_grad[axis] are sharded.
  if (index_dims_mapping_dst[0] != -1) {
    std::vector<int64_t> partial_dims(1, index_dims_mapping_dst[0]);
    x_grad_dist_attr_dst.set_partial_status(partial_dims);
  }

  VLOG(4) << "IndexSelectGradInferSpmd: Done.";
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(index);
  LOG_SPMD_INPUT(out_grad);
  LOG_SPMD_OUTPUT(x_grad_dist_attr_dst);
  return {{x_dist_attr_dst, index_dist_attr_dst, out_grad_dist_attr_dst},
          {x_grad_dist_attr_dst}};
}

}  // namespace phi::distributed
