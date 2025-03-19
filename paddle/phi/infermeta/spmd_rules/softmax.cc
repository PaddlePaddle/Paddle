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

#include "paddle/phi/infermeta/spmd_rules/softmax.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo SoftmaxInferSpmd(const DistMetaTensor& x, int axis) {
  // Step0: Verify input args based on softmax logic
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  auto x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor X's rank [%d] and X's "
                                      "dims_mapping size [%d] are not matched.",
                                      x_ndim,
                                      x_dims_mapping.size()));

  VLOG(6) << "SoftmaxInferSpmd Inputs: "
          << "X shape: [" << str_join(x_shape) << "], x_dims_mapping: ["
          << str_join(x_dims_mapping) << "]; axis: "
          << "[" << axis << "]; ";

  // normalize axis
  if (axis < 0) {
    axis = x_ndim + axis;
  }

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string out_axes = x_axes;

  // Step2: Sharding Propagation
  // naive support for sharding on softmax_axis
  // softmax_axis should be resharded as replicated (TODO: support sharding on
  // softmax_axis efficiently)
  if (x_dims_mapping[axis] >= 0) {
    x_dims_mapping[axis] = -1;
    VLOG(6) << "SoftmaxSPMDRule InferForward: softmax axis is reshard to be "
               "replicated: "
            << "original dims_mapping["
            << str_join(x_dist_attr_src.dims_mapping()) << "], "
            << "resharded dims_mapping[" << str_join(x_dims_mapping) << "].";
  }

  // Avoid multiple tensor axes sharded by same mesh dimension
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}}, false);

  // Step3: Infer Output's Dims Mapping.
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  // Update x's dist_attr
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  VLOG(4) << "SoftmaxInferSpmd:\n"
          << "Einsum notation: [" << x_axes << " --> " << out_axes << "].\n"
          << "Input shape: [" << str_join(x_shape) << "], src_dims_mapping: ["
          << str_join(x_dist_attr_src.dims_mapping())
          << "], dst_dims_mapping: [" << str_join(x_dims_mapping) << "]\n"
          << "Output dims_mapping: [" << str_join(out_dims_mapping) << "]\n\n";

  return {{x_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo SoftmaxInferSpmdReverse(const DistMetaTensor& x,
                                 const DistMetaTensor& out,
                                 int axis) {
  // Step0: verify input args based on softmax logic
  auto x_shape = common::vectorize(x.dims());
  auto out_shape = common::vectorize(out.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  int out_ndim = static_cast<int>(out_shape.size());
  auto out_dist_attr_src = out.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor Out's rank [%d] and Out's "
                                      "dims_mapping size [%d] are not matched.",
                                      out_ndim,
                                      out_dims_mapping.size()));

  // Step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string out_axes = x_axes;

  // normalize axis
  if (axis < 0) {
    axis = x_ndim + axis;
  }

  // sharding on softmax_axis is not supported now,
  // so set its dim mapping to -1
  out_dims_mapping[axis] = -1;

  // Step2: Sharding Propagation
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{out_axes, out_dims_mapping}});

  // infer input's dims mapping.
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  TensorDistAttr x_dist_attr = CopyTensorDistAttrForOutput(x.dist_attr());
  x_dist_attr.set_dims_mapping(x_dims_mapping);

  // update output's dims mapping.
  TensorDistAttr out_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  VLOG(4) << "SoftmaxInferSpmdReverse:\n"
          << "softmax_axis: " << axis << std::endl
          << "Einsum notation: [" << x_axes << " --> " << out_axes << "].\n"
          << "Output shape: [" << str_join(out_shape)
          << "], src_dims_mapping: ["
          << str_join(out_dist_attr_src.dims_mapping())
          << "], dst_dims_mapping: [" << str_join(out_dims_mapping) << "]\n"
          << "Input dims_mapping: [" << str_join(x_dims_mapping) << "]\n\n";

  return {{x_dist_attr}, {out_dist_attr_dst}};
}

SpmdInfo SoftmaxGradInferSpmd(const DistMetaTensor& out,
                              const DistMetaTensor& out_grad,
                              int axis) {
  axis = axis < 0 ? out.dims().size() + axis : axis;

  PADDLE_ENFORCE_EQ(out_grad.dims().size(),
                    out_grad.dist_attr().dims_mapping().size(),
                    common::errors::InvalidArgument(
                        "The Tensor out_grad's rank [%d] and out_grad's "
                        "dims_mapping size [%d] are not matched.",
                        out_grad.dims().size(),
                        out_grad.dist_attr().dims_mapping().size()));

  PADDLE_ENFORCE_GE(out_grad.dist_attr().dims_mapping().size(),
                    axis,
                    common::errors::InvalidArgument(
                        "The Tensor out_grad's rank [%d] must be "
                        "greater than axis [%d].",
                        out_grad.dist_attr().dims_mapping().size(),
                        axis));

  // To keeping consistent with forward propagation, sharding on softmax_axis
  // is not supported now, the axis should be resharded as replicated.
  auto out_grad_dims_mapping = out_grad.dist_attr().dims_mapping();
  if (out_grad_dims_mapping[axis] >= 0) {
    out_grad_dims_mapping[axis] = -1;
    VLOG(6) << "SoftmaxGradInferSpmd: The out_grad's softmax_axis is reshard "
               "to be replicated: "
            << "original dims_mapping["
            << str_join(out_grad.dist_attr().dims_mapping()) << "], "
            << "resharded dims_mapping[" << str_join(out_grad_dims_mapping)
            << "].";
  }
  auto out_dims_mapping = out.dist_attr().dims_mapping();
  if (out_dims_mapping[axis] >= 0) {
    out_dims_mapping[axis] = -1;
    VLOG(6) << "SoftmaxGradInferSpmd: The out's softmax_axis is reshard "
               "to be replicated: "
            << "original dims_mapping["
            << str_join(out.dist_attr().dims_mapping()) << "], "
            << "resharded dims_mapping[" << str_join(out_dims_mapping) << "].";
  }

  auto out_dist_attr = CopyTensorDistAttrForOutput(out.dist_attr());
  out_dist_attr.set_dims_mapping(out_dims_mapping);
  auto out_grad_dist_attr = CopyTensorDistAttrForOutput(out_grad.dist_attr());
  out_grad_dist_attr.set_dims_mapping(out_grad_dims_mapping);

  return ElementwiseBinaryInferSpmd(
      DistMetaTensor(out.dims(), out_dist_attr),
      DistMetaTensor(out_grad.dims(), out_grad_dist_attr));
}

}  // namespace phi::distributed
