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

#include "paddle/phi/infermeta/spmd_rules/take_along_axis.h"

#include "glog/logging.h"

#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {
SpmdInfo TakeAlongAxisInferSpmd(const DistMetaTensor& x,
                                const DistMetaTensor& index,
                                int axis) {
  /*
    gather computation formula:

    out[i][j][k] = x[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = x[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = x[i][j][index[i][j][k]]  # if dim == 2
  */

  // Deduced spmd rule:
  // x: cannot be sharded on `axis` dim;
  // index: the `axis` dim could be either sharded or not, other dimension
  // should be the same as x out: same as index

  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(index);
  PADDLE_ENFORCE_EQ(x_ndim,
                    index_ndim,
                    common::errors::InvalidArgument(
                        "x and index must have the same number of dimensions "
                        "but received x_ndim [%d], index_ndim [%d]",
                        x_ndim,
                        index_ndim));

  // Step1: Build Einsum Notation
  // e.g. axis=1, x: azc, index: abc, out: abc
  std::string alphabet = "abcdefghijklmnopqrstuvwxy";
  std::string index_axes = GetBroadcastAxes(index_ndim, index_ndim, alphabet);
  std::string x_axes = index_axes;
  x_axes.replace(axis, 1, "z");
  std::string out_axes = index_axes;

  // Step2: Sharding Propagation
  // Step2.1: Merge input shardings
  std::vector<int64_t> x_dims_mapping(x_dims_mapping_src);
  if (x_dims_mapping[axis] != -1) x_dims_mapping[axis] = -1;

  std::vector<int64_t> index_dims_mapping(index_dims_mapping_src);
  for (int i = 0; i < index_ndim; ++i) {
    if (i != axis && index_dims_mapping[i] != x_dims_mapping[i])
      index_dims_mapping[i] = x_dims_mapping[i];
  }

  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(
          {{x_axes, x_dims_mapping}, {index_axes, index_dims_mapping}});

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  TensorDistAttr index_dist_attr_dst =
      CopyTensorDistAttrForOutput(index_dist_attr_src);
  index_dist_attr_dst.set_dims_mapping(index_dims_mapping);

  // Step2.2: Infer output dims mapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);
  TensorDistAttr out_dist_attr =
      CopyTensorDistAttrForOutput(index_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  VLOG(4) << "x_axes: " << x_axes << " index_axes: " << index_axes
          << " out_axes: " << out_axes;
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(index);
  VLOG(4) << "out";
  VLOG(4) << "dist_attr: [" << out_dist_attr.to_string() << "]";
  return {{x_dist_attr_dst, index_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo TakeAlongAxisGradInferSpmd(const DistMetaTensor& x,
                                    const DistMetaTensor& index,
                                    const DistMetaTensor& out_grad,
                                    int axis) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(index);
  EXTRACT_SHAPE_AND_DIST_ATTR(out_grad);

  // Step1: Build Einsum Notation
  // e.g. axis=1, out_grad: abc -> x: azc, index: abc, x_grad: azc
  std::string alphabet = "abcdefghijklmnopqrstuvwxy";
  std::string out_grad_axes =
      GetBroadcastAxes(out_grad_ndim, out_grad_ndim, alphabet);
  std::string index_axes = out_grad_axes;

  // Step2: Sharding Propagation
  // Step2.1: Merge input shardings
  std::vector<int64_t> out_grad_dims_mapping(out_grad_dims_mapping_src);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{out_grad_axes, out_grad_dims_mapping}});

  // step2.2: Infer input dims mapping from merged input dims mapping
  std::vector<int64_t> index_dims_mapping =
      GetDimsMappingForAxes(index_axes, axis_to_dim_map);
  auto index_dist_attr_dst = CopyTensorDistAttrForOutput(index_dist_attr_src);
  index_dist_attr_dst.set_dims_mapping(index_dims_mapping);
  auto out_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_grad_dist_attr_src);
  out_grad_dist_attr_dst.set_dims_mapping(index_dims_mapping);

  std::vector<int64_t> x_dims_mapping(index_dims_mapping);
  if (x_dims_mapping[axis] != -1) x_dims_mapping[axis] = -1;
  auto x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  auto x_grad_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_grad_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  VLOG(4) << "out_grad";
  VLOG(4) << "dist_attr: [" << out_grad_dist_attr_dst.to_string() << "]";
  VLOG(4) << "index";
  VLOG(4) << "dist_attr: [" << index_dist_attr_dst.to_string() << "]";
  VLOG(4) << "x";
  VLOG(4) << "dist_attr: [" << x_dist_attr_dst.to_string() << "]";
  VLOG(4) << "x_grad";
  VLOG(4) << "dist_attr: [" << x_grad_dist_attr_dst.to_string() << "]";

  return {{x_dist_attr_dst, index_dist_attr_dst, out_grad_dist_attr_dst},
          {x_grad_dist_attr_dst}};
}
}  // namespace phi::distributed
