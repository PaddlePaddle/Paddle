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

#include "paddle/phi/infermeta/spmd_rules/topk.h"
#include "glog/logging.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

SpmdInfo TopkInferSpmd(
    const DistMetaTensor& x, int k, int axis, bool largest, bool sorted) {
  // Verify input args
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  axis = axis < 0 ? x_ndim + axis : axis;
  PADDLE_ENFORCE_EQ(
      0 <= axis && axis < x_ndim,
      true,
      common::errors::InvalidArgument(
          "The axis of topk should be in range [0, %d), but got %d.",
          x_ndim,
          axis));

  // Create destination dist attrs
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr out_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr indices_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);

  // Infer dims_mapping
  std::vector<int64_t> x_dims_mapping_dst = x_dims_mapping_src;
  x_dims_mapping_dst[axis] = -1;
  std::vector<int64_t> out_dims_mapping_dst = x_dims_mapping_dst;
  std::vector<int64_t> indices_dims_mapping_dst = x_dims_mapping_dst;

  // Set the dims mapping for outputs
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping_dst);
  indices_dist_attr_dst.set_dims_mapping(indices_dims_mapping_dst);

  // Update the dims mapping for inputs
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  VLOG(4) << "TopkInferSpmd: Done.";
  LOG_SPMD_INPUT(x);
  LOG_SPMD_OUTPUT(out_dist_attr_dst);
  LOG_SPMD_OUTPUT(indices_dist_attr_dst);

  return {{x_dist_attr_dst}, {out_dist_attr_dst, indices_dist_attr_dst}};
}

SpmdInfo TopkGradInferSpmd(const DistMetaTensor& x,
                           const DistMetaTensor& indices,
                           const DistMetaTensor& out_grad,
                           int k,
                           int axis,
                           bool largest,
                           bool sorted) {
  // Verify input args
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(indices);
  EXTRACT_SHAPE_AND_DIST_ATTR(out_grad);
  PADDLE_ENFORCE_EQ(indices_ndim,
                    out_grad_ndim,
                    common::errors::InvalidArgument(
                        "TopKGrad: The rank of Indices [%d] and OutGrad [%d] "
                        "must be the same.",
                        indices_ndim,
                        out_grad_ndim));
  PADDLE_ENFORCE_EQ(x_ndim,
                    indices_ndim,
                    common::errors::InvalidArgument(
                        "TopKGrad: The rank of Input [%d] and Indices [%d] "
                        "must be the same.",
                        x_ndim,
                        indices_ndim));
  axis = axis < 0 ? x_ndim + axis : axis;
  PADDLE_ENFORCE_EQ(
      0 <= axis && axis < x_ndim,
      true,
      common::errors::InvalidArgument(
          "The axis of topk_grad should be in range [0, %d), but got %d.",
          x_ndim,
          axis));
  // Build einsum notation
  std::string alphabet = "abcdefghijlopqrstuvwxyz";
  std::string x_axes = alphabet.substr(0, x_ndim - 1);
  std::string indices_axes = x_axes;
  std::string out_grad_axes = x_axes;

  // Merge sharding
  std::pair<std::string, std::vector<int64_t>> indices_pair(
      indices_axes, indices_dims_mapping_src);
  std::pair<std::string, std::vector<int64_t>> out_grad_pair(
      out_grad_axes, out_grad_dims_mapping_src);
  std::pair<std::string, std::vector<int64_t>> x_pair(x_axes,
                                                      x_dims_mapping_src);
  auto axis_to_dim_map =
      ShardingMergeForTensors({x_pair, indices_pair, out_grad_pair});

  // Infer dims mapping
  std::vector<int64_t> x_grad_dims_mapping_dst =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  x_grad_dims_mapping_dst.insert(x_grad_dims_mapping_dst.begin() + axis, -1);
  std::vector<int64_t> x_dims_mapping_dst = x_grad_dims_mapping_dst;
  std::vector<int64_t> indices_dims_mapping_dst = x_grad_dims_mapping_dst;
  std::vector<int64_t> out_grad_dims_mapping_dst = x_grad_dims_mapping_dst;

  // Set the dims mapping
  TensorDistAttr x_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_grad_dist_attr_src);
  TensorDistAttr x_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_grad_dist_attr_src);
  TensorDistAttr indices_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_grad_dist_attr_src);
  TensorDistAttr out_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_grad_dist_attr_src);

  x_grad_dist_attr_dst.set_dims_mapping(x_grad_dims_mapping_dst);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  indices_dist_attr_dst.set_dims_mapping(indices_dims_mapping_dst);
  out_grad_dist_attr_dst.set_dims_mapping(out_grad_dims_mapping_dst);

  VLOG(4) << "TopkGradInferSpmd: Done.";
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(indices);
  LOG_SPMD_INPUT(out_grad);
  LOG_SPMD_OUTPUT(x_grad_dist_attr_dst);

  return {{x_dist_attr_dst, indices_dist_attr_dst, out_grad_dist_attr_dst},
          {x_grad_dist_attr_dst}};
}
SpmdInfo TopkInferSpmdDynamic(const DistMetaTensor& x,
                              const Scalar& k,
                              int axis,
                              bool largest,
                              bool sorted) {
  return TopkInferSpmd(x, k.to<int>(), axis, largest, sorted);
}

SpmdInfo TopkGradInferSpmdDynamic(const DistMetaTensor& x,
                                  const DistMetaTensor& indices,
                                  const DistMetaTensor& out_grad,
                                  const Scalar& k,
                                  int axis,
                                  bool largest,
                                  bool sorted) {
  return TopkGradInferSpmd(
      x, indices, out_grad, k.to<int>(), axis, largest, sorted);
}

}  // namespace distributed
}  // namespace phi
