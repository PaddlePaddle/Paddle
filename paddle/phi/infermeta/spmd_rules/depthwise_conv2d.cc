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

#include "paddle/phi/infermeta/spmd_rules/depthwise_conv2d.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo DepthwiseConv2dInferSpmd(const DistMetaTensor& input,
                                  const DistMetaTensor& filter,
                                  const std::vector<int>& strides,
                                  const std::vector<int>& paddings,
                                  const std::string& padding_algorithm,
                                  int groups,
                                  const std::vector<int>& dilations,
                                  const std::string& data_format) {
  // Step0: verify input args based on depthwise_conv2d logic
  // input_dim: NCHinWin, filter_dim: M1HfWf, C = groups, M % groups == 0
  // output_dim: NMHoutWout
  VLOG(4) << "step 0: verify input args based on depthwise_conv2d logic";
  auto original_input_shape = common::vectorize(input.dims());
  auto original_filter_shape = common::vectorize(filter.dims());
  int input_ndim = static_cast<int>(original_input_shape.size());
  int filter_ndim = static_cast<int>(original_filter_shape.size());
  const auto& input_dist_attr_src = input.dist_attr();
  const auto& filter_dist_attr_src = filter.dist_attr();
  std::vector<int64_t> input_dims_mapping = input_dist_attr_src.dims_mapping();
  std::vector<int64_t> filter_dims_mapping =
      filter_dist_attr_src.dims_mapping();

  PADDLE_ENFORCE_EQ(input_ndim,
                    4,
                    common::errors::InvalidArgument(
                        "The Tensor Input's rank must be 4 in "
                        "depthwise_conv2d, for NCHW or NHWC format."
                        "But now it's [%d]",
                        input_ndim));

  PADDLE_ENFORCE_EQ(
      filter_ndim,
      4,
      common::errors::InvalidArgument("The Tensor Filter's rank must be 4 in "
                                      "depthwise_conv2d, for MCHW format."
                                      "But now it's [%d]",
                                      filter_ndim));

  PADDLE_ENFORCE_EQ(input_ndim,
                    input_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "The Tensor Input's rank [%d] and Input's "
                        "dims_mapping size [%d] are not matched.",
                        input_ndim,
                        input_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(filter_ndim,
                    filter_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "The Tensor Filter's rank [%d] and Filter's "
                        "dims_mapping size [%d] are not matched.",
                        filter_ndim,
                        filter_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(filter_dims_mapping[1],
                    -1,
                    common::errors::InvalidArgument(
                        "The Tensor Filter's dims_mapping on channel dim "
                        "should always be -1.",
                        "But now it's [%d]",
                        filter_dims_mapping[1]));

  VLOG(6) << "DepthwiseConv2D InferForward Inputs: "
          << "Input shape: [" << str_join(original_input_shape)
          << "], input_dims_mapping: [" << str_join(input_dims_mapping)
          << "]; Filter shape: [" << str_join(original_filter_shape)
          << "], filter_dims_mapping: [" << str_join(filter_dims_mapping)
          << "]; ";

  // Step1: build Einsum Notation
  // todo: check output notation, how to deal with the "Input HW, Filter HW and
  // Output HW"...
  // todo: if shard channel_dim, attribute group should also be changed on each
  // device, which is not supported, so channel_dim currently should not be
  // sharded.
  VLOG(4) << "step 1: build Einsum Notation";
  std::string input_axes = (data_format == "NCHW") ? "n1hw" : "nhw1";
  std::string filter_axes = "m1hw";
  std::string output_axes = "nmhw";

  if (data_format == "NCHW")
    input_dims_mapping[1] = -1;
  else
    input_dims_mapping[3] = -1;

  // Step2: sharding propagation
  VLOG(4) << "step 2: sharding propagation";
  // Step2.1: merge input sharding
  std::pair<std::string, std::vector<int64_t>> input_pair(input_axes,
                                                          input_dims_mapping);
  std::pair<std::string, std::vector<int64_t>> filter_pair(filter_axes,
                                                           filter_dims_mapping);
  auto axis_to_dim_map = ShardingMergeForTensors({input_pair, filter_pair});
  // Step2.2: infer output dims mapping
  TensorDistAttr output_dist_attr_dst =
      CopyTensorDistAttrForOutput(input_dist_attr_src);
  output_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(output_axes, axis_to_dim_map));

  // Step2.3: update input dims mapping
  TensorDistAttr input_dist_attr_dst =
      CopyTensorDistAttrForOutput(input_dist_attr_src);
  TensorDistAttr filter_dist_attr_dst =
      CopyTensorDistAttrForOutput(filter_dist_attr_src);
  input_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(input_axes, axis_to_dim_map));
  filter_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(filter_axes, axis_to_dim_map));

  // Step3: Handle Partial
  VLOG(4) << "DepthwiseConv2DSPMDRule InferForward: "
          << "Einsum notation: [" << input_axes << "," << filter_axes << " --> "
          << output_axes << "]. " << std::endl;
  LogInputDistAttr(
      "Input", original_input_shape, input_dist_attr_src, input_dist_attr_dst);
  LogInputDistAttr("Filter",
                   original_filter_shape,
                   filter_dist_attr_src,
                   filter_dist_attr_dst);
  LogOutputDistAttr("Output", output_dist_attr_dst);
  VLOG(4) << std::endl;

  return {{input_dist_attr_dst, filter_dist_attr_dst}, {output_dist_attr_dst}};
}

SpmdInfo DepthwiseConv2dGradInferSpmd(const DistMetaTensor& input,
                                      const DistMetaTensor& filter,
                                      const DistMetaTensor& output_grad,
                                      const std::vector<int>& strides,
                                      const std::vector<int>& paddings,
                                      const std::string& padding_algorithm,
                                      int groups,
                                      const std::vector<int>& dilations,
                                      const std::string& data_format) {
  auto input_dist_attr_src = input.dist_attr();
  auto filter_dist_attr_src = filter.dist_attr();
  auto output_grad_dist_attr_src = output_grad.dist_attr();

  std::string input_axes = (data_format == "NCHW") ? "n1hw" : "nhw1";
  std::string filter_axes = "m1hw";
  std::string output_axes = "nmhw";

  std::pair<std::string, std::vector<int64_t>> input_pair(
      input_axes, input_dist_attr_src.dims_mapping());
  std::pair<std::string, std::vector<int64_t>> filter_pair(
      filter_axes, filter_dist_attr_src.dims_mapping());
  std::pair<std::string, std::vector<int64_t>> output_grad_pair(
      output_axes, output_grad_dist_attr_src.dims_mapping());

  // input_grad_dist, copy n_dim and merge m_dim
  auto axis_to_dim_map_1 =
      ShardingMergeForTensors({filter_pair, output_grad_pair});
  TensorDistAttr input_grad_dist_attr_dst =
      GetReplicatedDistAttr(input_dist_attr_src);
  input_grad_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(input_axes, axis_to_dim_map_1));
  TensorDistAttr filter_dist_attr_dst =
      CopyTensorDistAttrForOutput(filter_dist_attr_src);
  filter_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(filter_axes, axis_to_dim_map_1));

  // filter_grad_dist, copy m_dim and merge n_dim
  auto axis_to_dim_map_2 =
      ShardingMergeForTensors({input_pair, output_grad_pair});
  TensorDistAttr filter_grad_dist_attr_dst =
      GetReplicatedDistAttr(filter_dist_attr_src);
  filter_grad_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(filter_axes, axis_to_dim_map_2));
  TensorDistAttr input_dist_attr_dst =
      CopyTensorDistAttrForOutput(input_dist_attr_src);
  input_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(input_axes, axis_to_dim_map_2));

  // output_grad
  auto axis_to_dim_map_3 =
      ShardingMergeForTensors({input_pair, filter_pair, output_grad_pair});
  TensorDistAttr output_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(output_grad_dist_attr_src);
  output_grad_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(output_axes, axis_to_dim_map_3));

  return {
      {input_dist_attr_dst, filter_dist_attr_dst, output_grad_dist_attr_dst},
      {input_grad_dist_attr_dst, filter_grad_dist_attr_dst}};
}

}  // namespace distributed
}  // namespace phi
