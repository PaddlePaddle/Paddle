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

#include "paddle/phi/infermeta/spmd_rules/conv2d.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo Conv2dInferSpmdBase(const DistMetaTensor& input,
                             const DistMetaTensor& filter) {
  // Step0: verify input args based on conv2d logic
  VLOG(4) << "step 0: verify input args based on conv2d logic";
  auto original_input_shape = common::vectorize(input.dims());
  auto original_filter_shape = common::vectorize(filter.dims());
  int input_ndim = static_cast<int>(original_input_shape.size());
  int filter_ndim = static_cast<int>(original_filter_shape.size());
  const auto& input_dist_attr_src = input.dist_attr();
  const auto& filter_dist_attr_src = filter.dist_attr();
  std::vector<int64_t> input_dims_mapping = input_dist_attr_src.dims_mapping();
  std::vector<int64_t> filter_dims_mapping =
      filter_dist_attr_src.dims_mapping();

  PADDLE_ENFORCE_EQ(
      input_ndim,
      4,
      common::errors::InvalidArgument("The Tensor Input's rank must be 4 in "
                                      "conv2d, for NCHW or NHWC format."
                                      "But now it's [%d]",
                                      input_ndim));

  PADDLE_ENFORCE_EQ(
      filter_ndim,
      4,
      common::errors::InvalidArgument(
          "The Tensor Filter's rank must be 4 in conv2d, for MCHW format."
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
  // todo: NCHW or NHWC check, check channel logic, input's channel dims_mapping
  // must be equal to filter's channel dims_mapping if NCHW
  int channel_dim = 1;
  PADDLE_ENFORCE_EQ(input_dims_mapping[channel_dim],
                    filter_dims_mapping[channel_dim],
                    common::errors::InvalidArgument(
                        "The Input channel's dims mapping must be equal to "
                        "filter channel's dims mapping in conv2d. "
                        "When shard channel dim to a mesh (multiple cards), "
                        "each card will compute partial output, ",
                        "otherwise, mark channel dim as replicate, each card "
                        "will compute complete output.",
                        "But now the Input channel's dims mapping is [%d], and "
                        "the filter channel's dims mapping is [%d].",
                        input_dims_mapping[channel_dim],
                        filter_dims_mapping[channel_dim]));

  VLOG(6) << "Conv2D InferForward Inputs: "
          << "Input shape: [" << str_join(original_input_shape)
          << "], input_dims_mapping: [" << str_join(input_dims_mapping)
          << "]; Filter shape: [" << str_join(original_filter_shape)
          << "], filter_dims_mapping: [" << str_join(filter_dims_mapping)
          << "]; ";

  // Step1: build Einsum Notation
  // todo: check output notation, how to deal with the "Input HW, Filter HW and
  // Output HW"...
  VLOG(4) << "step 1: build Einsum Notation";
  std::string input_axes = "nchw";
  std::string filter_axes = "mchw";
  std::string output_axes = "nmhw";

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

  // Step2.4: Handle Partial
  // Step2.4.1 Output Partial
  std::vector<int64_t> partial_on_dims =
      ResoluteOutputPartialDimension(axis_to_dim_map, output_axes);
  output_dist_attr_dst.set_partial_status(partial_on_dims);

  VLOG(4) << "Conv2DSPMDRule InferForward: "
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

SpmdInfo Conv2dInferSpmdReverseBase(const DistMetaTensor& input,
                                    const DistMetaTensor& filter,
                                    const DistMetaTensor& output) {
  // Step0: verify input args based on conv2d logic
  VLOG(4) << "step 0: verify args based on conv2d logic";
  auto original_output_shape = common::vectorize(output.dims());
  int output_ndim = static_cast<int>(original_output_shape.size());
  PADDLE_ENFORCE_EQ(
      output_ndim,
      4,
      common::errors::InvalidArgument(
          "The Tensor Output's rank must be 4 in conv2d, for NCoutHW format."
          "But now it's [%d]",
          output_ndim));

  auto output_dist_attr_src = output.dist_attr();
  std::vector<int64_t> output_dims_mapping =
      output_dist_attr_src.dims_mapping();

  // Step1: build Einsum Notation
  // todo: check output notation, how to deal with the "Input HW, Filter HW and
  // Output HW"...
  VLOG(4) << "step 1: build Einsum Notation";
  std::string input_axes = "nchw";
  std::string filter_axes = "mchw";
  std::string output_axes = "nmhw";

  // Step2: sharding propagation
  auto axis_to_dim_map =
      ShardingMergeForTensors({{output_axes, output_dims_mapping}}, false);
  TensorDistAttr input_dist_attr_dst =
      CopyTensorDistAttrForOutput(input.dist_attr());
  TensorDistAttr filter_dist_attr_dst =
      CopyTensorDistAttrForOutput(filter.dist_attr());
  input_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(input_axes, axis_to_dim_map, true));
  filter_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(filter_axes, axis_to_dim_map, true));

  VLOG(4) << "Conv2dReverseSPMDRule InferBackward: "
          << "Einsum notation: [" << input_axes << "," << filter_axes << " --> "
          << output_axes << "]. " << std::endl;
  LogInputDistAttr("Output",
                   original_output_shape,
                   output_dist_attr_src,
                   output_dist_attr_src);
  LogOutputDistAttr("Input", input_dist_attr_dst);
  LogOutputDistAttr("Filter", filter_dist_attr_dst);
  VLOG(4) << std::endl;

  return {{input_dist_attr_dst, filter_dist_attr_dst}, {output_dist_attr_src}};
}

SpmdInfo Conv2dGradInferSpmdBase(const DistMetaTensor& input,
                                 const DistMetaTensor& filter,
                                 const DistMetaTensor& output_grad) {
  auto check_channel_dist_attr =
      [&](const phi::distributed::TensorDistAttr& input_dist_attr,
          const phi::distributed::TensorDistAttr& filter_dist_attr,
          const phi::distributed::TensorDistAttr& output_grad_dist_attr) {
        int channel_dim = 1;  // NCHW support
        if (output_grad_dist_attr.is_partial()) {
          std::set<int64_t> partial_dims = output_grad_dist_attr.partial_dims();
          PADDLE_ENFORCE_EQ(
              partial_dims.size(),
              1,
              common::errors::InvalidArgument(
                  "For conv2d output, only support partial on channel_dim for "
                  "output, "
                  "which means shard on channel_dim for input and filter."
                  "But now the output is partial on [%d] dims.",
                  partial_dims.size()));

          int64_t partial_dim = *partial_dims.begin();
          auto input_dims_mapping = input_dist_attr.dims_mapping();
          auto filter_dims_mapping = filter_dist_attr.dims_mapping();
          if (input_dims_mapping[channel_dim] == partial_dim &&
              filter_dims_mapping[channel_dim] == partial_dim) {
            return true;
          }
        }

        return false;
      };

  auto input_dist_attr_src = input.dist_attr();
  auto filter_dist_attr_src = filter.dist_attr();
  auto output_grad_dist_attr_src = output_grad.dist_attr();

  std::string input_axes = "nchw";
  std::string filter_axes = "mchw";
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
  // handle partial for input_grad
  std::vector<int64_t> partial_on_m_dim =
      ResoluteOutputPartialDimension(axis_to_dim_map_1, input_axes);
  input_grad_dist_attr_dst.set_partial_status(partial_on_m_dim);
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
  // handle partial for filter_grad
  std::vector<int64_t> partial_on_n_dim =
      ResoluteOutputPartialDimension(axis_to_dim_map_2, filter_axes);
  filter_grad_dist_attr_dst.set_partial_status(partial_on_n_dim);
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

  // process channel_dim, handle partial
  int channel_dim = 1;  // NCHW format
  if (check_channel_dist_attr(input_dist_attr_src,
                              filter_dist_attr_src,
                              output_grad_dist_attr_src)) {
    int partial_mesh_dim = *output_grad_dist_attr_src.partial_dims().begin();
    std::vector<int64_t> input_grad_dims_mapping_dst =
        input_grad_dist_attr_dst.dims_mapping();
    input_grad_dims_mapping_dst[channel_dim] = partial_mesh_dim;
    input_grad_dist_attr_dst.set_dims_mapping(input_grad_dims_mapping_dst);
    std::vector<int64_t> filter_grad_dims_mapping_dst =
        filter_grad_dist_attr_dst.dims_mapping();
    filter_grad_dims_mapping_dst[channel_dim] = partial_mesh_dim;
    filter_grad_dist_attr_dst.set_dims_mapping(filter_grad_dims_mapping_dst);
    output_grad_dist_attr_dst.set_partial_status(
        std::vector<int64_t>({partial_mesh_dim}));
  }

  return {
      {input_dist_attr_dst, filter_dist_attr_dst, output_grad_dist_attr_dst},
      {input_grad_dist_attr_dst, filter_grad_dist_attr_dst}};
}

SpmdInfo Conv2dInferSpmd(const DistMetaTensor& input,
                         const DistMetaTensor& filter,
                         const std::vector<int>& strides,
                         const std::vector<int>& paddings,
                         const std::string& padding_algorithm,
                         const std::vector<int>& dilations,
                         int groups,
                         const std::string& data_format) {
  return Conv2dInferSpmdBase(input, filter);
}

SpmdInfo Conv2dInferSpmdReverse(const DistMetaTensor& input,
                                const DistMetaTensor& filter,
                                const DistMetaTensor& output,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                const std::string& padding_algorithm,
                                const std::vector<int>& dilations,
                                int groups,
                                const std::string& data_format) {
  return Conv2dInferSpmdReverseBase(input, filter, output);
}

SpmdInfo Conv2dGradInferSpmd(const DistMetaTensor& input,
                             const DistMetaTensor& filter,
                             const DistMetaTensor& output_grad,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::string& padding_algorithm,
                             const std::vector<int>& dilations,
                             int groups,
                             const std::string& data_format) {
  return Conv2dGradInferSpmdBase(input, filter, output_grad);
}

}  // namespace distributed
}  // namespace phi
