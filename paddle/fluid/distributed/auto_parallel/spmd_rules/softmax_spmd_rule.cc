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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/softmax_spmd_rule.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

using phi::distributed::auto_parallel::str_join;

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
SoftmaxSPMDRule::InferForward(const std::vector<DistTensorSpec>& input_specs,
                              const paddle::framework::AttributeMap& attrs) {
  // step0: verify input args based on softmax logic
  auto input_specs_size = input_specs.size();
  PADDLE_ENFORCE_EQ(
      input_specs_size,
      1,
      phi::errors::InvalidArgument(
          "The size of InputSpec of softmax should be 1, but got [%d].",
          input_specs_size));

  auto x_shape = input_specs[0].shape();
  int x_ndim = x_shape.size();
  auto x_dist_attr_src = input_specs[0].dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();

  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      phi::errors::InvalidArgument(
          "Mismatch of X's tensor size: [%d] and X's dims_mapping size [%d].",
          x_ndim,
          x_dims_mapping.size()));

  int axis = ExtractAttr<int>("axis", attrs);

  VLOG(6) << "SoftmaxSPMDRule InferForward Inputs: "
          << "X shape: [" << str_join(x_shape) << "], x_dims_mapping: ["
          << str_join(x_dims_mapping) << "]; axis: "
          << "[" << axis << "]; ";

  // normalize axis
  if (axis < 0) {
    axis = x_ndim + axis;
  }

  // step1: build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string out_axes = x_axes;

  // step2: Sharding Propogation

  // naive support for sharding on softmax_axis
  // softmax_axis should be resharded as replicated (TODO: support sharding on
  // softmax_axis effeciently)
  if (x_dims_mapping[axis] >= 0) {
    x_dims_mapping[axis] = -1;
    VLOG(6) << "SoftmaxSPMDRule InferForward: softmax axis is reshard to be "
               "replicated: "
            << "original dims_mapping["
            << str_join(x_dist_attr_src.dims_mapping()) << "], "
            << "resharded dims_mapping[" << str_join(x_dims_mapping) << "].";
  }

  // Avoid multiple tensor axes sharded by same mesh deminsion
  auto axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}}, false);

  // step3: Infer Output's Dims Mapping.
  TensorDistAttr output_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  std::vector<int64_t> out_dims_mapping;
  out_dims_mapping.reserve(out_axes.size());
  for (size_t i = 0; i < out_axes.size(); ++i) {
    out_dims_mapping.push_back(axis_to_dim_map[out_axes.substr(i, 1)]);
  }
  output_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  // Update x's dist_attr
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  VLOG(4) << "SoftmaxSPMDRule InferForward: "
          << "Einsum notation: [" << x_axes << " --> " << out_axes << "]. "
          << std::endl
          << "X shape: [" << str_join(x_shape) << "], src_dims_mapping: ["
          << str_join(x_dist_attr_src.dims_mapping())
          << "], dst_dims_mapping: [" << str_join(x_dims_mapping)
          << "]; Output dims_mapping: [" << str_join(out_dims_mapping) << "]";

  return {{x_dist_attr_dst}, {output_dist_attr_dst}};
}

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
SoftmaxSPMDRule::InferBackward(const std::vector<DistTensorSpec>& input_specs,
                               const std::vector<DistTensorSpec>& output_specs,
                               const paddle::framework::AttributeMap& attrs) {
  // step0: verify input args based on softmax logic
  int64_t input_specs_size = input_specs.size();
  int64_t output_specs_size = output_specs.size();
  PADDLE_ENFORCE_EQ(
      input_specs_size,
      1,
      phi::errors::InvalidArgument(
          "The size of InputSpec of softmax should be 1, but got [%d].",
          input_specs_size));
  PADDLE_ENFORCE_EQ(
      output_specs_size,
      1,
      phi::errors::InvalidArgument(
          "The size of InputSpec of softmax should be 1, but got [%d].",
          output_specs_size));
  VerifySpecs(output_specs, "softmax_backward");

  // step1: build Einsum Notation
  std::vector<int64_t> x_shape = input_specs[0].shape();
  int64_t x_ndim = input_specs[0].shape().size();

  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string x_axes = GetBroadcastAxes(x_ndim, x_ndim, alphabet);
  std::string out_axes = x_axes;

  int axis = ExtractAttr<int>("axis", attrs);
  // normalize axis
  if (axis < 0) {
    axis = x_ndim + axis;
  }

  // sharding on softmax_axis is not supported now, so set
  // the notation on softmax_axis to '1' so that we can set
  // its dim mapping to -1
  x_axes[axis] = '1';

  // step2: Sharding Propogation
  std::vector<int64_t> out_dims_mapping = output_specs[0].dims_mapping();
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{out_axes, out_dims_mapping}});

  // infer input's dims mapping.
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  TensorDistAttr input_dist_attr(input_specs[0].dist_attr());
  input_dist_attr.set_dims_mapping(x_dims_mapping);

  // update output's dims mapping.
  out_dims_mapping[axis] = -1;
  TensorDistAttr output_dist_attr(output_specs[0].dist_attr());
  output_dist_attr.set_dims_mapping(out_dims_mapping);

  VLOG(4) << "SoftmaxSPMDRule InferBackward: "
          << "softmax_axis: " << axis << std::endl
          << "Einsum notation: [" << x_axes << " --> " << out_axes << "]. "
          << std::endl
          << "Output shape: [" << str_join(output_specs[0].shape())
          << "], src_dims_mapping: ["
          << str_join(output_specs[0].dims_mapping())
          << "], dst_dims_mapping: [" << str_join(out_dims_mapping)
          << "]; Input dims_mapping: [" << str_join(x_dims_mapping) << "]";

  return {{input_dist_attr}, {output_dist_attr}};
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
