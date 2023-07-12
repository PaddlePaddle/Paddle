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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/cross_entropy_with_softmax_spmd_rule.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

using phi::distributed::auto_parallel::str_join;

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
CrossEntropyWithSoftmaxSPMDRule::InferForward(
    const std::vector<DistTensorSpec>& input_specs,
    const paddle::framework::AttributeMap& attrs) {
  // step0: verify input args based on cross_entropy_with_softmax logic
  auto input_specs_size = input_specs.size();
  PADDLE_ENFORCE_EQ(
      input_specs_size,
      2,
      phi::errors::InvalidArgument("The size of InputSpec of cross entropy "
                                   "with softmax should be 1, but got [%d].",
                                   input_specs_size));

  auto x_shape = input_specs[0].shape();
  int x_ndim = x_shape.size();
  auto x_dist_attr_src = input_specs[0].dist_attr();
  std::vector<int64_t> x_dims_mapping_src = x_dist_attr_src.dims_mapping();

  auto label_shape = input_specs[1].shape();
  int label_ndim = label_shape.size();
  auto label_dist_attr_src = input_specs[1].dist_attr();
  std::vector<int64_t> label_dims_mapping_src =
      label_dist_attr_src.dims_mapping();

  int axis = ExtractAttr<int>("axis", attrs);
  int ignore_index = ExtractAttr<int>("ignore_index", attrs);
  bool numeric_stable_mode = ExtractAttr<bool>("numeric_stable_mode", attrs);
  bool use_softmax = ExtractAttr<bool>("use_softmax", attrs);
  bool soft_label = ExtractAttr<bool>("soft_label", attrs);

  VLOG(6) << "CrossEntropyWithSoftmaxSPMDRule InferForward Inputs: "
          << "X shape: [" << str_join(x_shape) << "], x_dims_mapping_src: ["
          << str_join(x_dims_mapping_src) << "]; Label shape: ["
          << str_join(label_shape) << "], Label dims mapping: ["
          << str_join(label_dims_mapping_src) << "]; axis: "
          << "[" << axis << "], ignore_index: [" << ignore_index
          << "], numeric_stable_mode: [" << numeric_stable_mode
          << "], use_softmax: [" << use_softmax << "], soft_label: ["
          << soft_label << "].";

  // normalize axis
  if (axis < 0) {
    axis = x_ndim + axis;
  }

  // trying to shard the normal axis of softmax, BUT
  // c_softmax_with_entropy kernel not support:
  // 1. soft label
  // 2. axis != -1
  // support above two features in future.
  if (x_dims_mapping_src[axis] > -1) {
    PADDLE_ENFORCE_EQ(
        soft_label,
        false,
        phi::errors::InvalidArgument(
            "Trying to shard the softmax_normalize axis of the input tensor, "
            "but the soft_label is set as True, which is not supported yet!"));

    PADDLE_ENFORCE_EQ(
        axis,
        x_ndim - 1,
        phi::errors::InvalidArgument(
            "Trying to shard the softmax_normalize axis of the input tensor, "
            "but the softmax_normalize axis is not the last axis, which is not "
            "supported yet! The softmax_normalize is [%d].",
            axis));

    PADDLE_ENFORCE_EQ(use_softmax,
                      true,
                      phi::errors::InvalidArgument(
                          "Trying to shard the softmax_normalize axis of the "
                          "input tensor, use_softmax must be set to True !"));
  }

  // step1: build Einsum Notation
  std::string alphabet =
      "abcdefghijlmnopqrstuvwxyz";  // k for softmax_normalize axis
  std::string broadcast_axes =
      GetBroadcastAxes(x_ndim - 1, x_ndim - 1, alphabet);
  std::string x_axes = broadcast_axes;
  x_axes.insert(axis, "k");
  if (soft_label) {
    std::string label_axes = x_axes;
  } else {
    std::string label_axes = broadcast_axes;
    label_axes.insert(axis, "1");
  }
  std::string loss_axes = broadcast_axes;
  loss_axes.insert(axis, "1");
  // optional output
  if (use_softmax) {
    std::string softmax_out_axes = x_axes;
  } else {
    std::string softmax_out_axes = "";
  }

  // step2: Sharding Propogation
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  axes_sharding_info =
      GetAxesDimsMappingPair({x_axes, label_axes}, input_specs);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  // step3: Infer dst Dims Mapping.
  TensorDistAttr loss_dist_attr_dst =
      CopyTensorDistAttrForOutput(label_dist_attr_src);
  loss_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(loss_axes, axis_to_dim_map));
  TensorDistAttr softmax_out_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  softmax_out_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(softmax_out_axes, axis_to_dim_map));

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(x_axes, axis_to_dim_map));
  TensorDistAttr label_dist_attr_dst =
      CopyTensorDistAttrForOutput(label_dist_attr_src);
  label_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(label_axes, axis_to_dim_map));

  VLOG(4) << "CrossEntropyWithSoftmaxSPMDRule InferForward Inputs: "
          << "X shape: [" << str_join(x_shape) << "], x_dims_mapping_src: ["
          << str_join(x_dims_mapping_src) << "], x_dims_mapping_dst: ["
          << str_join(x_dist_attr_dst.dims_mapping()) << "]; Label shape: ["
          << str_join(label_shape) << "], label_dims_mapping_src: ["
          << str_join(label_dims_mapping_src) << "], label_dims_mapping_dst: ["
          << str_join(label_dist_attr_dst.dims_mapping())
          << "]; loss_dims_mapping: ["
          << str_join(loss_dist_attr_dst.dims_mapping())
          << "], softmax_out_dims_mapping_src: ["
          << str_join(softmax_out_dist_attr_dst.dims_mapping()) << "]; axis: "
          << "[" << axis << "], ignore_index: [" << ignore_index
          << "], numeric_stable_mode: [" << numeric_stable_mode
          << "], use_softmax: [" << use_softmax << "], soft_label: ["
          << soft_label << "].";

  // todo if softmax_normalize axis is sharded, notify downstream phi api to
  // select c_softmax_with_entropy_kernel.
  std::vector<TensorDistAttr> output_vec = {loss_dist_attr_dst};
  if (use_softmax) {
    output_vec.emplace_back(softmax_out_dist_attr_dst);
  }
  return {{x_dist_attr_dst, label_dist_attr_dst}, output_vec};
}

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
CrossEntropyWithSoftmaxSPMDRule::InferBackward(
    const std::vector<DistTensorSpec>& input_specs,
    const paddle::framework::AttributeMap& attrs) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "InferBackward of CrossEntropyWithSoftmaxSPMDRule is NOT implemented "
      "yet."));
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
