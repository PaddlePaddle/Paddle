/* Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/linear_v2.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

SpmdInfo LinearV2InferSpmdBase(const DistMetaTensor& input,
                               const DistMetaTensor& weight,
                               const DistMetaTensor& bias,
                               bool transpose_weight) {
  PADDLE_ENFORCE_EQ(transpose_weight,
                    false,
                    common::errors::InvalidArgument(
                        "When in SPMD mode, the transpose_weight in linear_v2 "
                        "should be false, but got [%d].",
                        transpose_weight));
  // Step0: verify input args based on matmul logic
  auto ori_input_shape = vectorize(input.dims());
  auto ori_weight_shape = vectorize(weight.dims());
  auto ori_bias_shape = vectorize(bias.dims());
  int input_ndim = static_cast<int>(ori_input_shape.size());
  int weight_ndim = static_cast<int>(ori_weight_shape.size());
  int bias_ndim = static_cast<int>(ori_bias_shape.size());
  const auto& input_dist_attr_src = input.dist_attr();
  const auto& weight_dist_attr_src = weight.dist_attr();
  const auto& bias_dist_attr_src = bias.dist_attr();
  std::vector<int64_t> input_dims_mapping = input_dist_attr_src.dims_mapping();
  std::vector<int64_t> weight_dims_mapping =
      weight_dist_attr_src.dims_mapping();
  std::vector<int64_t> bias_dims_mapping = bias_dist_attr_src.dims_mapping();

  PADDLE_ENFORCE_EQ(input_ndim,
                    input_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "LinearV2, The Tensor input's rank [%d] and input's "
                        "dims_mapping size [%d] are not matched.",
                        input_ndim,
                        input_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(weight_ndim,
                    weight_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "LinearV2, The Tensor weight's rank [%d] and weight's "
                        "dims_mapping size [%d] are not matched.",
                        weight_ndim,
                        weight_dims_mapping.size()));

  PADDLE_ENFORCE_EQ(
      bias_ndim,
      1,
      common::errors::InvalidArgument(
          "LinearV2, The ndim of bias should be 1, but got [%d].", bias_ndim));

  VLOG(4) << "LinearV2SPMDRule InferForward Inputs: ";
  VLOG(4) << "input shape: [" << str_join(ori_input_shape)
          << "], input_dims_mapping: [" << str_join(input_dims_mapping) << "];";
  VLOG(4) << "weight shape: [" << str_join(ori_weight_shape)
          << "], weight_dims_mapping: [" << str_join(weight_dims_mapping)
          << "];";
  VLOG(4) << "bias shape: [" << str_join(ori_bias_shape)
          << "], bias_dims_mapping: [" << str_join(bias_dims_mapping) << "];";
  // Step1: build Einsum Notation
  std::string input_axes;
  std::string weight_axes;
  std::string out_axes;
  FillMatmulPartOperandNotation(
      input_ndim, weight_ndim, &input_axes, &weight_axes, &out_axes);

  // Step2.1: Sharding Merge
  std::pair<std::string, std::vector<int64_t>> x_pair(input_axes,
                                                      input_dims_mapping);
  std::pair<std::string, std::vector<int64_t>> y_pair(weight_axes,
                                                      weight_dims_mapping);
  auto axis_to_dim_map = ShardingMergeForTensors({x_pair, y_pair});

  // Step2.2: Infer Output's Dims Mapping.
  TensorDistAttr output_dist_attr_dst =
      CopyTensorDistAttrForOutput(input_dist_attr_src);
  std::vector<int64_t> out_dims_mapping;
  out_dims_mapping.reserve(out_axes.size());
  for (size_t i = 0; i < out_axes.size(); ++i) {
    out_dims_mapping.push_back(axis_to_dim_map[out_axes.substr(i, 1)]);
  }
  output_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  // Step2.3: Merge and get Inputs' New Dims Mapping.
  auto x_shape = vectorize(input.dims());
  auto y_shape = vectorize(weight.dims());

  TensorDistAttr x_dist_attr_dst = GetMatmulPartInferredDistAttr(
      input_dist_attr_src, x_shape, input_axes, axis_to_dim_map, false);
  TensorDistAttr y_dist_attr_dst = GetMatmulPartInferredDistAttr(
      weight_dist_attr_src, y_shape, weight_axes, axis_to_dim_map, false);
  TensorDistAttr bias_dist_attr_dst =
      CopyTensorDistAttrForOutput(bias_dist_attr_src);
  bias_dist_attr_dst.set_dims_mapping(
      std::vector<int64_t>{output_dist_attr_dst.dims_mapping().back()});

  // Step2.3: Handle Partial
  // Step2.3.1 Output Partial
  std::vector<int64_t> partial_on_dims =
      ResoluteOutputPartialDimension(axis_to_dim_map, out_axes);
  output_dist_attr_dst.set_partial_status(partial_on_dims);

  if (output_dist_attr_dst.is_partial()) {
    // NOTE(Pan Zhaowu): linear_v2, as a fused matmul+elew op, which is
    // different from legacy hacked behaviour, so disabled partial distribution
    // strategy for now.
    output_dist_attr_dst.clean_partial_status();
    SetTensorDistAttrReplicated(&x_dist_attr_dst, input_ndim);
    SetTensorDistAttrReplicated(&y_dist_attr_dst, weight_ndim);
    SetTensorDistAttrReplicated(&bias_dist_attr_dst, bias_ndim);
    SetTensorDistAttrReplicated(&output_dist_attr_dst, out_axes.size());
  }
  TensorDistAttr output_reserve_dist_attr_dst =
      CopyTensorDistAttrForOutput(output_dist_attr_dst);
  VLOG(4) << "LinearV2SPMDRule InferForward: "
          << "Einsum notation: [" << input_axes << "," << weight_axes << " --> "
          << out_axes << "+" << out_axes.back() << "]. " << std::endl;
  LogInputDistAttr(
      "input", ori_input_shape, input_dist_attr_src, x_dist_attr_dst);
  LogInputDistAttr(
      "weight", ori_weight_shape, weight_dist_attr_src, y_dist_attr_dst);
  LogInputDistAttr(
      "Bias", ori_bias_shape, bias_dist_attr_src, bias_dist_attr_dst);
  LogOutputDistAttr("Output", output_dist_attr_dst);

  return {{x_dist_attr_dst, y_dist_attr_dst, bias_dist_attr_dst},
          {output_dist_attr_dst, output_reserve_dist_attr_dst}};
}
SpmdInfo LinearV2InferSpmd(const DistMetaTensor& input,
                           const DistMetaTensor& weight,
                           const DistMetaTensor& bias,
                           bool transpose_weight) {
  return LinearV2InferSpmdBase(input, weight, bias, transpose_weight);
}
}  // namespace phi::distributed
