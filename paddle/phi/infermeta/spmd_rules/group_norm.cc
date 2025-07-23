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

#include "paddle/phi/infermeta/spmd_rules/group_norm.h"

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;
// Tensor x support  "NCL", "NCHW", "NCDHW", "NLC", "NHWC", "NDHWC".
// default:"NCHW"

SpmdInfo GroupNormInferSpmdBase(const DistMetaTensor& x,
                                const DistMetaTensor& scale,
                                const DistMetaTensor& bias) {
  // Step0: verify input args based on group_norm logic
  auto x_shape = common::vectorize(x.dims());
  auto scale_shape = common::vectorize(scale.dims());
  auto bias_shape = common::vectorize(bias.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  int scale_ndim = static_cast<int>(scale_shape.size());
  int bias_ndim = static_cast<int>(bias_shape.size());
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  TensorDistAttr scale_dist_attr_src = scale.dist_attr();
  TensorDistAttr bias_dist_attr_src = bias.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> scale_dims_mapping = scale_dist_attr_src.dims_mapping();
  std::vector<int64_t> bias_dims_mapping = bias_dist_attr_src.dims_mapping();

  PADDLE_ENFORCE_GE(
      x_ndim,
      3,
      common::errors::InvalidArgument(
          "The ndim of x in group_norm should grater than 2, but got [%d].",
          x_ndim));

  PADDLE_ENFORCE_LE(
      x_ndim,
      5,
      common::errors::InvalidArgument(
          "The ndim of x in group_norm should be less than 6 , but got [%d].",
          x_ndim));
  PADDLE_ENFORCE_EQ(
      scale_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of scale in group_norm should be 1, but got [%d].",
          scale_ndim));

  PADDLE_ENFORCE_EQ(
      bias_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of bias in group_norm should be 1, but got [%d].",
          bias_ndim));
  // Step1: Build Einsum Notation
  // Only N axis can be sharded.
  std::string alphabet = "ijklmnopqrstuvwxyz";
  std::string x_axes(x_ndim, '1');
  for (int i = 0; i < x_ndim; ++i) {
    x_axes[i] = alphabet[i];
  }
  std::string mean_axes(2, '1');
  std::string variance_axes(2, '1');

  for (int i = 0; i < 2; ++i) {
    mean_axes[i] = x_axes[i];
    variance_axes[i] = x_axes[i];
  }

  // x_axes[0] = alphabet[0];
  std::string scale_axes(1, x_axes[0]);
  std::string bias_axes(1, x_axes[0]);
  // get output notation
  std::string out_axes = x_axes;

  // Step2: Sharding Propagation
  // Step2.1: merge input sharding
  for (int i = 1; i < x_ndim; ++i) {
    x_dims_mapping[i] = -1;
  }
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}});

  // Step2.2: infer output dims mapping
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr mean_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr variance_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(out_axes, axis_to_dim_map));
  mean_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(mean_axes, axis_to_dim_map));
  variance_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(variance_axes, axis_to_dim_map));

  // Step2.3: update input dims mapping
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr scale_dist_attr_dst =
      CopyTensorDistAttrForOutput(scale.dist_attr());
  TensorDistAttr bias_dist_attr_dst =
      CopyTensorDistAttrForOutput(bias.dist_attr());
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  scale_dist_attr_dst.set_dims_mapping(std::vector<int64_t>{-1});
  bias_dist_attr_dst.set_dims_mapping(std::vector<int64_t>{-1});

  // Step2.4.  handle input and out tensor partial
  // GroupNorm not support
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(scale);
  LOG_SPMD_INPUT(bias);
  LOG_SPMD_OUTPUT(out_dist_attr);
  LOG_SPMD_OUTPUT(mean_dist_attr);
  LOG_SPMD_OUTPUT(variance_dist_attr);

  return {{x_dist_attr_dst, scale_dist_attr_dst, bias_dist_attr_dst},
          {out_dist_attr, mean_dist_attr, variance_dist_attr}};
}
SpmdInfo GroupNormInferSpmd(const DistMetaTensor& x,
                            const DistMetaTensor& scale,
                            const DistMetaTensor& bias,
                            float epsilon,
                            int groups,
                            const std::string& data_format) {
  return GroupNormInferSpmdBase(x, scale, bias);
}

SpmdInfo GroupNormGradInferSpmdBase(const DistMetaTensor& x,
                                    const DistMetaTensor& scale,
                                    const DistMetaTensor& bias,
                                    const DistMetaTensor& y,
                                    const DistMetaTensor& mean,
                                    const DistMetaTensor& variance,
                                    const DistMetaTensor y_grad) {
  // Step0: verify input args based on group_norm logic
  auto x_shape = common::vectorize(x.dims());
  auto scale_shape = common::vectorize(scale.dims());
  auto bias_shape = common::vectorize(bias.dims());
  auto y_shape = common::vectorize(y.dims());
  auto mean_shape = common::vectorize(mean.dims());
  auto variance_shape = common::vectorize(variance.dims());
  auto y_grad_shape = common::vectorize(y_grad.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  int scale_ndim = static_cast<int>(scale_shape.size());
  int bias_ndim = static_cast<int>(bias_shape.size());
  int y_ndim = static_cast<int>(y_shape.size());
  int mean_ndim = static_cast<int>(mean_shape.size());
  int variance_ndim = static_cast<int>(variance_shape.size());
  int y_grad_ndim = static_cast<int>(y_grad_shape.size());
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  TensorDistAttr scale_dist_attr_src = scale.dist_attr();
  TensorDistAttr bias_dist_attr_src = bias.dist_attr();
  TensorDistAttr y_dist_attr_src = y.dist_attr();
  TensorDistAttr mean_dist_attr_src = mean.dist_attr();
  TensorDistAttr variance_dist_attr_src = variance.dist_attr();
  TensorDistAttr y_grad_dist_attr_src = mean.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> scale_dims_mapping = scale.dist_attr().dims_mapping();
  std::vector<int64_t> bias_dims_mapping = bias.dist_attr().dims_mapping();
  std::vector<int64_t> y_dims_mapping = scale.dist_attr().dims_mapping();
  std::vector<int64_t> mean_dims_mapping = bias.dist_attr().dims_mapping();
  std::vector<int64_t> variance_dims_mapping = scale.dist_attr().dims_mapping();
  std::vector<int64_t> y_grad_dims_mapping = bias.dist_attr().dims_mapping();

  PADDLE_ENFORCE_GE(
      x_ndim,
      3,
      common::errors::InvalidArgument(
          "The ndim of x in group_norm should grater than 2, but got [%d].",
          x_ndim));

  PADDLE_ENFORCE_LE(
      x_ndim,
      5,
      common::errors::InvalidArgument(
          "The ndim of x in group_norm should be less than 6 , but got [%d].",
          x_ndim));
  PADDLE_ENFORCE_EQ(x_ndim,
                    y_ndim,
                    common::errors::InvalidArgument(
                        "The ndim of x and y in group_norm should be equal, "
                        "but got x:[%d] and y[%d] .",
                        x_ndim,
                        y_ndim));
  PADDLE_ENFORCE_EQ(
      x_ndim,
      y_grad_ndim,
      common::errors::InvalidArgument(
          "The ndim of x and y_grad in group_norm should be equal, "
          "but got x:[%d] and y_grad[%d] .",
          x_ndim,
          y_grad_ndim));
  PADDLE_ENFORCE_EQ(
      scale_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of scale in group_norm should be 1, but got [%d].",
          scale_ndim));

  PADDLE_ENFORCE_EQ(
      bias_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of bias in group_norm should be 1, but got [%d].",
          bias_ndim));
  PADDLE_ENFORCE_EQ(
      mean_ndim,
      2,
      common::errors::InvalidArgument(
          "The ndim of bias in group_norm should be 2, but got [%d].",
          bias_ndim));
  PADDLE_ENFORCE_EQ(
      variance_ndim,
      2,
      common::errors::InvalidArgument(
          "The ndim of bias in group_norm should be 2, but got [%d].",
          bias_ndim));

  // Step1: Build Einsum Notation
  // Only N axis can be sharded.
  std::string alphabet = "ijklmnopqrstuvwxyz";
  // input
  std::string x_axes(x_ndim, '1');
  std::string y_axes(y_ndim, '1');
  std::string y_grad_axes(y_grad_ndim, '1');

  for (int i = 0; i < x_ndim; ++i) {
    x_axes[i] = alphabet[i];
    y_axes[i] = alphabet[i];
    y_grad_axes[i] = alphabet[i];
  }
  std::string scale_axes(1, x_axes[0]);
  std::string bias_axes(1, x_axes[0]);
  std::string mean_axes(2, '1');
  std::string variance_axes(2, '1');

  for (int i = 0; i < 2; ++i) {
    mean_axes[i] = x_axes[i];
    variance_axes[i] = x_axes[i];
  }
  // output
  std::string x_grad_axes = x_axes;
  std::string scale_grad_axes(1, x_axes[0]);  // C axis
  std::string bias_grad_axes(1, x_axes[0]);
  // Step2: Sharding Propagation
  // Step2.1: merge input sharding
  for (int i = 1; i < x_ndim; ++i) {
    x_dims_mapping[i] = -1;
  }
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}});

  // Step2.2: infer output dims mapping
  TensorDistAttr x_grad_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr scale_grad_dist_attr =
      CopyTensorDistAttrForOutput(scale.dist_attr());
  TensorDistAttr bias_grad_dist_attr =
      CopyTensorDistAttrForOutput(bias.dist_attr());
  x_grad_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(x_grad_axes, axis_to_dim_map));
  scale_grad_dist_attr.set_dims_mapping(std::vector<int64_t>{-1});
  bias_grad_dist_attr.set_dims_mapping(std::vector<int64_t>{-1});

  // Step2.3: update input dims mapping
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr scale_dist_attr_dst =
      CopyTensorDistAttrForOutput(scale.dist_attr());
  TensorDistAttr bias_dist_attr_dst =
      CopyTensorDistAttrForOutput(bias.dist_attr());
  TensorDistAttr y_dist_attr_dst = CopyTensorDistAttrForOutput(y.dist_attr());
  TensorDistAttr mean_dist_attr_dst =
      CopyTensorDistAttrForOutput(mean.dist_attr());
  TensorDistAttr variance_dist_attr_dst =
      CopyTensorDistAttrForOutput(variance.dist_attr());
  TensorDistAttr y_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(y_grad.dist_attr());
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  y_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  y_grad_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  mean_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(mean_axes, axis_to_dim_map));
  variance_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(variance_axes, axis_to_dim_map));
  scale_dist_attr_dst.set_dims_mapping(std::vector<int64_t>{-1});
  bias_dist_attr_dst.set_dims_mapping(std::vector<int64_t>{-1});

  std::vector<int64_t> partial_on_dims;
  const auto& dim_mapping = x_dims_mapping;
  for (int i = 0; i < x_ndim; ++i) {
    auto mapping = dim_mapping[i];
    if (mapping != -1) {
      partial_on_dims.push_back(mapping);
    }
  }
  scale_grad_dist_attr.set_partial_status(partial_on_dims);
  bias_grad_dist_attr.set_partial_status(partial_on_dims);

  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(scale);
  LOG_SPMD_INPUT(bias);
  LOG_SPMD_INPUT(y);
  LOG_SPMD_INPUT(mean);
  LOG_SPMD_INPUT(variance);
  LOG_SPMD_INPUT(y_grad);
  LOG_SPMD_OUTPUT(x_grad_dist_attr);
  LOG_SPMD_OUTPUT(scale_grad_dist_attr);
  LOG_SPMD_OUTPUT(bias_grad_dist_attr);

  return {{x_dist_attr_dst,
           scale_dist_attr_dst,
           bias_dist_attr_dst,
           y_dist_attr_dst,
           mean_dist_attr_dst,
           variance_dist_attr_dst,
           y_grad_dist_attr_dst},
          {x_grad_dist_attr, scale_grad_dist_attr, bias_grad_dist_attr}};
}
SpmdInfo GroupNormGradInferSpmd(const DistMetaTensor& x,
                                const DistMetaTensor& scale,
                                const DistMetaTensor& bias,
                                const DistMetaTensor& y,
                                const DistMetaTensor& mean,
                                const DistMetaTensor& variance,
                                const DistMetaTensor y_grad,
                                float epsilon,
                                int groups,
                                const std::string& data_format) {
  return GroupNormGradInferSpmdBase(x, scale, bias, y, mean, variance, y_grad);
}
}  // namespace phi::distributed
