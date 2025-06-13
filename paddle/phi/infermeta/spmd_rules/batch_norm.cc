// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/infermeta/spmd_rules/batch_norm.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

SpmdInfo BatchNormInferSpmd(const DistMetaTensor& x,
                            const DistMetaTensor& mean,
                            const DistMetaTensor& variance,
                            const DistMetaTensor& scale,
                            const DistMetaTensor& bias,
                            const bool is_test,
                            const float momentum,
                            const float epsilon,
                            const std::string& data_format,
                            const bool use_global_stats,
                            const bool trainable_statistics) {
  // Step0: verify input args based on batch_norm logic
  auto x_shape = common::vectorize(x.dims());
  auto mean_shape = common::vectorize(mean.dims());
  auto variance_shape = common::vectorize(variance.dims());
  auto scale_shape = common::vectorize(scale.dims());
  auto bias_shape = common::vectorize(bias.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  int mean_ndim = static_cast<int>(mean_shape.size());
  int variance_ndim = static_cast<int>(variance_shape.size());
  int scale_ndim = static_cast<int>(scale_shape.size());
  int bias_ndim = static_cast<int>(bias_shape.size());
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  TensorDistAttr mean_dist_attr_src = mean.dist_attr();
  TensorDistAttr variance_dist_attr_src = variance.dist_attr();
  TensorDistAttr scale_dist_attr_src = scale.dist_attr();
  TensorDistAttr bias_dist_attr_src = bias.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> mean_dims_mapping = mean.dist_attr().dims_mapping();
  std::vector<int64_t> variance_dims_mapping =
      variance.dist_attr().dims_mapping();
  std::vector<int64_t> scale_dims_mapping = scale.dist_attr().dims_mapping();
  std::vector<int64_t> bias_dims_mapping = bias.dist_attr().dims_mapping();

  PADDLE_ENFORCE_GE(
      x_ndim,
      2,
      common::errors::InvalidArgument(
          "The ndim of x in batch_norm should be greater than 1, but got [%d].",
          x_ndim));
  PADDLE_ENFORCE_LE(
      x_ndim,
      5,
      common::errors::InvalidArgument(
          "The ndim of x in batch_norm should be less than 6, but got [%d].",
          x_ndim));

  PADDLE_ENFORCE_EQ(
      mean_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of mean in batch_norm should be 1, but got [%d].",
          mean_ndim));

  PADDLE_ENFORCE_EQ(
      variance_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of variance in batch_norm should be 1, but got [%d].",
          variance_ndim));

  PADDLE_ENFORCE_EQ(
      scale_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of scale in batch_norm should be 1, but got [%d].",
          scale_ndim));

  PADDLE_ENFORCE_EQ(
      bias_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of bias in batch_norm should be 1, but got [%d].",
          bias_ndim));

  // Step1: Build Einsum Notation

  std::string alphabet = "ijklmnopqrstuvwxyz";
  // get input notation
  // The mean and variance was flatten at C axis
  std::string x_axes(x_ndim, '1');
  for (int i = 0; i < x_ndim; ++i) {
    x_axes[i] = alphabet[i];
  }
  int c_index = data_format[1] == 'C' ? 1 : x_ndim - 1;
  std::string mean_axes(1, x_axes[c_index]);
  std::string variance_axes(1, x_axes[c_index]);
  std::string scale_axes(1, x_axes[c_index]);
  std::string bias_axes(1, x_axes[c_index]);

  // get output notation
  std::string out_axes = x_axes;

  // Step2: Sharding Propagation
  // Step2.1: merge input sharding
  // Only C axis can be shard.
  auto c_dim =
      x_dims_mapping[c_index];  // type: "NC"、"NCL"、"NLC"、"NCHW"、"NHWC"" and
                                // "NCDHW"

  for (int i = 0; i < x_ndim; ++i) {
    x_dims_mapping[i] = -1;
  }
  x_dims_mapping[c_index] = c_dim;
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}});

  // Step2.2: infer output dims mapping
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr mean_dist_attr = CopyTensorDistAttrForOutput(mean.dist_attr());
  TensorDistAttr variance_dist_attr =
      CopyTensorDistAttrForOutput(variance.dist_attr());
  TensorDistAttr saved_mean_dist_attr =
      CopyTensorDistAttrForOutput(mean.dist_attr());
  TensorDistAttr saved_variance_dist_attr =
      CopyTensorDistAttrForOutput(variance.dist_attr());
  TensorDistAttr reserve_space_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(out_axes, axis_to_dim_map));
  mean_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(mean_axes, axis_to_dim_map));
  variance_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(variance_axes, axis_to_dim_map));
  saved_mean_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(mean_axes, axis_to_dim_map));
  saved_variance_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(variance_axes, axis_to_dim_map));
  reserve_space_dist_attr.set_dims_mapping({-1});

  // Step2.3: update input dims mapping
  // mean, variance, mean_out, variance_out and
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr scale_dist_attr_dst =
      CopyTensorDistAttrForOutput(scale.dist_attr());
  TensorDistAttr bias_dist_attr_dst =
      CopyTensorDistAttrForOutput(bias.dist_attr());
  TensorDistAttr mean_dist_attr_dst =
      CopyTensorDistAttrForOutput(mean.dist_attr());
  TensorDistAttr variance_dist_attr_dst =
      CopyTensorDistAttrForOutput(variance.dist_attr());
  scale_dist_attr_dst.set_dims_mapping({-1});
  bias_dist_attr_dst.set_dims_mapping({-1});
  variance_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(variance_axes, axis_to_dim_map));
  mean_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(mean_axes, axis_to_dim_map));

  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(mean);
  LOG_SPMD_INPUT(variance);
  LOG_SPMD_INPUT(scale);
  LOG_SPMD_INPUT(bias);
  LOG_SPMD_OUTPUT(out_dist_attr);
  LOG_SPMD_OUTPUT(mean_dist_attr);
  LOG_SPMD_OUTPUT(variance_dist_attr);
  LOG_SPMD_OUTPUT(saved_mean_dist_attr);
  LOG_SPMD_OUTPUT(saved_variance_dist_attr);
  LOG_SPMD_OUTPUT(reserve_space_dist_attr);
  return {{x_dist_attr_dst,
           mean_dist_attr_dst,
           variance_dist_attr_dst,
           scale_dist_attr_dst,
           bias_dist_attr_dst},
          {out_dist_attr,
           mean_dist_attr,
           variance_dist_attr,
           saved_mean_dist_attr,
           saved_variance_dist_attr,
           reserve_space_dist_attr}};
}
SpmdInfo BatchNormInferSpmdStatic(const DistMetaTensor& x,
                                  const DistMetaTensor& mean,
                                  const DistMetaTensor& variance,
                                  const DistMetaTensor& scale,
                                  const DistMetaTensor& bias) {
  return BatchNormInferSpmd(x, mean, variance, scale, bias);
}
SpmdInfo BatchNormGradInferSpmd(const DistMetaTensor& x,
                                const DistMetaTensor& scale,
                                const DistMetaTensor& bias,
                                const DistMetaTensor& mean_out,
                                const DistMetaTensor& variance_out,
                                const DistMetaTensor& saved_mean,
                                const DistMetaTensor& saved_variance,
                                const DistMetaTensor& reserve_space,
                                const DistMetaTensor& out_grad,
                                const float momentum,
                                const float epsilon,
                                const std::string& data_format,
                                const bool is_test,
                                const bool use_global_stats,
                                const bool trainable_statistics) {
  auto x_shape = common::vectorize(x.dims());
  auto scale_shape = common::vectorize(scale.dims());
  auto bias_shape = common::vectorize(bias.dims());
  auto mean_out_shape = common::vectorize(mean_out.dims());
  auto variance_out_shape = common::vectorize(variance_out.dims());
  auto saved_mean_shape = common::vectorize(saved_mean.dims());
  auto saved_variance_shape = common::vectorize(saved_variance.dims());
  auto reserve_space_shape = common::vectorize(reserve_space.dims());
  auto out_grad_shape = common::vectorize(out_grad.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  int scale_ndim = static_cast<int>(scale_shape.size());
  int bias_ndim = static_cast<int>(bias_shape.size());
  int mean_out_ndim = static_cast<int>(mean_out_shape.size());
  int variance_out_ndim = static_cast<int>(variance_out_shape.size());
  int saved_mean_ndim = static_cast<int>(saved_mean_shape.size());
  int saved_variance_ndim = static_cast<int>(saved_variance_shape.size());
  int reserve_space_ndim = static_cast<int>(reserve_space_shape.size());
  int out_grad_ndim = static_cast<int>(out_grad_shape.size());
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  TensorDistAttr scale_dist_attr_src = scale.dist_attr();
  TensorDistAttr bias_dist_attr_src = bias.dist_attr();
  TensorDistAttr mean_out_dist_attr_src = mean_out.dist_attr();
  TensorDistAttr variance_out_dist_attr_src = variance_out.dist_attr();
  TensorDistAttr saved_mean_dist_attr_src = saved_mean.dist_attr();
  TensorDistAttr saved_variance_dist_attr_src = saved_variance.dist_attr();
  TensorDistAttr reserve_space_dist_attr_src = reserve_space.dist_attr();
  TensorDistAttr out_grad_dist_attr_src = out_grad.dist_attr();
  PADDLE_ENFORCE_GE(
      x_ndim,
      2,
      common::errors::InvalidArgument(
          "The ndim of x in batch_norm should be greater than 1, but got [%d].",
          x_ndim));
  PADDLE_ENFORCE_LE(
      x_ndim,
      5,
      common::errors::InvalidArgument(
          "The ndim of x in batch_norm should be less than 6, but got [%d].",
          x_ndim));
  PADDLE_ENFORCE_EQ(out_grad_ndim,
                    x_ndim,
                    common::errors::InvalidArgument(
                        "The ndim of out_grad in batch_norm should be equal "
                        "with x, but got out_grad:[%d] and x:[%d].",
                        out_grad_ndim,
                        x_ndim));
  PADDLE_ENFORCE_EQ(
      mean_out_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of mean_out in batch_norm should be 1, but got [%d].",
          mean_out_ndim));

  PADDLE_ENFORCE_EQ(
      variance_out_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of variance_out in batch_norm should be 1, but got [%d].",
          variance_out_ndim));

  PADDLE_ENFORCE_EQ(
      scale_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of scale in batch_norm should be 1, but got [%d].",
          scale_ndim));

  PADDLE_ENFORCE_EQ(
      bias_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of bias in batch_norm should be 1, but got [%d].",
          bias_ndim));
  PADDLE_ENFORCE_EQ(
      saved_mean_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of saved_mean in batch_norm should be 1, but got [%d].",
          saved_mean_ndim));

  PADDLE_ENFORCE_EQ(
      saved_variance_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of saved_variance in batch_norm should be 1, but got [%d].",
          saved_variance_ndim));

  PADDLE_ENFORCE_EQ(
      reserve_space_ndim,
      1,
      common::errors::InvalidArgument("The ndim of reserve_space_ndim in "
                                      "batch_norm should be 1, but got [%d].",
                                      reserve_space_ndim));

  std::string alphabet = "ijklmnopqrstuvwxyz";
  // get input notation
  // The mean and variance was flatten at C axis
  std::string x_axes(x_ndim, '1');
  std::string out_grad_axes(out_grad_ndim, '1');

  for (int i = 0; i < x_ndim; ++i) {
    x_axes[i] = alphabet[i];
    out_grad_axes[i] = alphabet[i];
  }
  int c_index = data_format[1] == 'C' ? 1 : x_ndim - 1;
  std::string mean_out_axes(1, x_axes[c_index]);
  std::string variance_out_axes(1, x_axes[c_index]);
  std::string scale_axes(1, x_axes[c_index]);
  std::string bias_axes(1, x_axes[c_index]);
  std::string saved_mean_axes(1, x_axes[c_index]);
  std::string saved_variance_axes(1, x_axes[c_index]);
  std::string reserve_space_axes(1, x_axes[c_index]);

  auto c_dim =
      x_dims_mapping[c_index];  // Only C axis can be sharded. ndim Type:
                                // type: "NC"、"NCL"、"NLC"、"NCHW"、"NHWC"" and
                                // "NCDHW"

  for (int i = 0; i < x_ndim; ++i) {
    x_dims_mapping[i] = -1;
  }
  x_dims_mapping[c_index] = c_dim;

  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}});
  // infer output spmdinfo
  TensorDistAttr x_grad_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_grad_dist_attr.set_dims_mapping(x_dims_mapping);
  TensorDistAttr scale_grad_dist_attr =
      CopyTensorDistAttrForOutput(scale.dist_attr());
  scale_grad_dist_attr.set_dims_mapping({-1});
  TensorDistAttr bias_grad_dist_attr =
      CopyTensorDistAttrForOutput(bias.dist_attr());
  bias_grad_dist_attr.set_dims_mapping({-1});
  // infer input spmdinfo
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  TensorDistAttr mean_out_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  mean_out_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(mean_out_axes, axis_to_dim_map));
  TensorDistAttr variance_out_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  variance_out_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(variance_out_axes, axis_to_dim_map));
  TensorDistAttr scale_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  scale_dist_attr_dst.set_dims_mapping({-1});
  TensorDistAttr bias_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  bias_dist_attr_dst.set_dims_mapping({-1});
  TensorDistAttr saved_mean_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  saved_mean_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(saved_mean_axes, axis_to_dim_map));
  TensorDistAttr saved_variance_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  saved_variance_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(saved_variance_axes, axis_to_dim_map));
  TensorDistAttr reserve_space_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  reserve_space_dist_attr_dst.set_dims_mapping({-1});
  TensorDistAttr out_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_grad_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(out_grad_axes, axis_to_dim_map));

  // partial grad dim
  std::vector<int64_t> partial_on_dims;
  for (int i = 0; i < x_ndim; ++i) {
    auto mapping = x_dims_mapping[i];
    if (mapping != -1) {
      partial_on_dims.push_back(mapping);
    }
  }
  scale_grad_dist_attr.set_partial_status(partial_on_dims);
  bias_grad_dist_attr.set_partial_status(partial_on_dims);

  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(scale);
  LOG_SPMD_INPUT(bias);
  LOG_SPMD_INPUT(mean_out);
  LOG_SPMD_INPUT(variance_out);
  LOG_SPMD_INPUT(saved_mean);
  LOG_SPMD_INPUT(saved_variance);
  LOG_SPMD_INPUT(reserve_space);
  LOG_SPMD_INPUT(out_grad);
  LOG_SPMD_OUTPUT(x_grad_dist_attr);
  LOG_SPMD_OUTPUT(scale_grad_dist_attr);
  LOG_SPMD_OUTPUT(bias_grad_dist_attr);

  return {{x_dist_attr_dst,
           scale_dist_attr_dst,
           bias_dist_attr_dst,
           mean_out_dist_attr_dst,
           variance_out_dist_attr_dst,
           saved_mean_dist_attr_dst,
           saved_variance_dist_attr_dst,
           reserve_space_dist_attr_dst,
           out_grad_dist_attr_dst},
          {x_grad_dist_attr, scale_grad_dist_attr, bias_grad_dist_attr}};
}

}  // namespace phi::distributed
