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

#include "paddle/phi/infermeta/spmd_rules/instance_norm.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {
using phi::distributed::auto_parallel::str_join;
// The input tensor shape is [N,C,H,W], the shape of scale and bias is [C]
//  only N,C axis can be sharded.
SpmdInfo InstanceNormInferSpmd(const DistMetaTensor& x,
                               const DistMetaTensor& scale,
                               const DistMetaTensor& bias,
                               float epsilon = 1e-5) {
  // Step0: verify input args based on layer_norm logic
  auto x_shape = common::vectorize(x.dims());
  auto scale_shape = common::vectorize(scale.dims());
  auto bias_shape = common::vectorize(bias.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  int scale_ndim = static_cast<int>(scale_shape.size());
  int bias_ndim = static_cast<int>(bias_shape.size());
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> scale_dims_mapping = scale.dist_attr().dims_mapping();
  std::vector<int64_t> bias_dims_mapping = bias.dist_attr().dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      4,
      common::errors::InvalidArgument(
          "The ndim of x in instance_norm should be 4, but got [%d].", x_ndim));
  PADDLE_ENFORCE_EQ(
      scale_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of scale in instance_norm should be 1, but got [%d].",
          scale_ndim));

  PADDLE_ENFORCE_EQ(
      bias_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of bias in instance_norm should be 1, but got [%d].",
          bias_ndim));

  // Step1: Build Einsum Notation
  std::string alphabet = "ijklmnopqrstuvwxyz";
  std::string x_axes(x_ndim, '1');
  std::string saved_mean_axes(2, '1');
  std::string saved_variance_axes(2, '1');
  for (int i = 0; i < x_ndim; ++i) {
    x_axes[i] = alphabet[i];
  }
  for (int i = 0; i < 2; ++i) {
    saved_mean_axes[i] = alphabet[i];
    saved_variance_axes[i] = alphabet[i];
  }
  std::string scale_axes(1, x_axes[1]);  // C axis
  std::string bias_axes(1, x_axes[1]);
  std::string y_axes = x_axes;

  // Step2: Sharding Propagation
  // Step2.1: merge input sharding
  // The [H,W] can not be sharded.
  x_dims_mapping[2] = -1;
  x_dims_mapping[3] = -1;
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}});

  // Step2.2: infer output dims mapping
  TensorDistAttr y_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr saved_mean_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr saved_variance_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  y_dist_attr.set_dims_mapping(GetDimsMappingForAxes(y_axes, axis_to_dim_map));
  saved_mean_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(saved_mean_axes, axis_to_dim_map));
  saved_variance_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(saved_variance_axes, axis_to_dim_map));

  // Step2.3: update input dims mapping
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr scale_dist_attr_dst =
      CopyTensorDistAttrForOutput(scale.dist_attr());
  TensorDistAttr bias_dist_attr_dst =
      CopyTensorDistAttrForOutput(bias.dist_attr());
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  scale_dist_attr_dst.set_dims_mapping({-1});
  bias_dist_attr_dst.set_dims_mapping({-1});

  VLOG(4) << "InstanceNormInferSpmd:";
  VLOG(4) << "data type: [N,C,H,W]";
  VLOG(4) << "Einsum Notation: " << x_axes << "," << scale_axes << ","
          << bias_axes << "-->" << y_axes << "," << saved_mean_axes << ","
          << saved_variance_axes;
  VLOG(4) << "X"
          << " shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(x_dims_mapping) << "]";
  VLOG(4) << "Scale"
          << " shape: [" << str_join(scale_shape) << "] "
          << "src_dims_mapping: [" << str_join(scale_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(scale_dist_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "Bias"
          << " shape: [" << str_join(bias_shape) << "] "
          << "src_dims_mapping: [" << str_join(bias_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(bias_dist_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "Out dims mapping: [" << str_join(y_dist_attr.dims_mapping())
          << "]";
  VLOG(4) << "Saved_Mean dims mapping: ["
          << str_join(saved_mean_dist_attr.dims_mapping()) << "]";
  VLOG(4) << "Saved_Variance dims mapping: ["
          << str_join(saved_variance_dist_attr.dims_mapping()) << "]";
  VLOG(4) << std::endl;

  return {{x_dist_attr_dst, scale_dist_attr_dst, bias_dist_attr_dst},
          {y_dist_attr, saved_mean_dist_attr, saved_variance_dist_attr}};
}

SpmdInfo InstanceNormGradInferSpmd(const DistMetaTensor& x,
                                   const DistMetaTensor& scale,
                                   const DistMetaTensor& saved_mean,
                                   const DistMetaTensor& saved_variance,
                                   const DistMetaTensor& y_grad,
                                   float epsilon = 1e-5) {
  // Step0: verify input args based on layer_norm logic
  auto x_shape = common::vectorize(x.dims());
  auto scale_shape = common::vectorize(scale.dims());
  auto saved_mean_shape = common::vectorize(saved_mean.dims());
  auto saved_variance_shape = common::vectorize(saved_variance.dims());
  auto y_grad_shape = common::vectorize(y_grad.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  int scale_ndim = static_cast<int>(scale_shape.size());
  int saved_mean_ndim = static_cast<int>(saved_mean_shape.size());
  int saved_variance_ndim = static_cast<int>(saved_variance_shape.size());
  int y_grad_ndim = static_cast<int>(y_grad_shape.size());
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> scale_dims_mapping = scale.dist_attr().dims_mapping();
  std::vector<int64_t> saved_mean_dims_mapping =
      saved_mean.dist_attr().dims_mapping();
  std::vector<int64_t> saved_variance_dims_mapping =
      saved_variance.dist_attr().dims_mapping();
  std::vector<int64_t> y_grad_dims_mapping = y_grad.dist_attr().dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      4,
      common::errors::InvalidArgument(
          "The ndim of x in instance_norm should be 4, but got [%d].", x_ndim));
  PADDLE_ENFORCE_EQ(
      y_grad_ndim,
      4,
      common::errors::InvalidArgument(
          "The ndim of y_grad in instance_norm should be 4, but got [%d].",
          y_grad));
  PADDLE_ENFORCE_EQ(
      scale_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of scale in instance_norm should be 1, but got [%d].",
          scale_ndim));

  PADDLE_ENFORCE_EQ(
      saved_mean_ndim,
      2,
      common::errors::InvalidArgument(
          "The ndim of saved_mean in instance_norm should be 2, but got [%d].",
          saved_mean_ndim));
  PADDLE_ENFORCE_EQ(saved_variance_ndim,
                    2,
                    common::errors::InvalidArgument(
                        "The ndim of saved_variance in instance_norm should be "
                        "2, but got [%d].",
                        saved_variance_ndim));

  // Step1: Build Einsum Notation
  std::string alphabet = "ijklmnopqrstuvwxyz";
  std::string x_axes(x_ndim, '1');
  std::string y_grad_axes(x_ndim, '1');
  std::string saved_mean_axes(2, '1');
  std::string saved_variance_axes(2, '1');
  for (int i = 0; i < x_ndim; ++i) {
    x_axes[i] = alphabet[i];
    y_grad_axes[i] = alphabet[i];
  }
  for (int i = 0; i < 2; ++i) {
    saved_mean_axes[i] = alphabet[i];
    saved_variance_axes[i] = alphabet[i];
  }
  std::string scale_axes(1, x_axes[1]);  // C axis
  std::string scale_grad_axes(1, x_axes[1]);
  std::string bias_grad_axes(1, x_axes[1]);
  std::string x_grad_axes = x_axes;

  // Step2: Sharding Propagation
  // Step2.1: merge input sharding
  // The [H,W] can not be sharded.
  x_dims_mapping[2] = -1;
  x_dims_mapping[3] = -1;
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}});

  // Step2.2: infer output dims mapping
  TensorDistAttr x_grad_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_grad_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(x_grad_axes, axis_to_dim_map));
  TensorDistAttr scale_grad_dist_attr =
      CopyTensorDistAttrForOutput(scale.dist_attr());
  scale_grad_dist_attr_dst.set_dims_mapping({-1});
  TensorDistAttr bias_grad_dist_attr =
      CopyTensorDistAttrForOutput(scale.dist_attr());
  bias_grad_dist_attr_dst.set_dims_mapping({-1});

  // Step2.3: update input dims mapping
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  TensorDistAttr y_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(y_grad.dist_attr());
  y_grad_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  TensorDistAttr saved_mean_dist_attr =
      CopyTensorDistAttrForOutput(saved_mean.dist_attr());
  saved_mean_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(saved_mean_axes, axis_to_dim_map));
  TensorDistAttr saved_variance_dist_attr =
      CopyTensorDistAttrForOutput(saved_variance.dist_attr());
  saved_variance_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(saved_variance_axes, axis_to_dim_map));
  TensorDistAttr scale_dist_attr_dst =
      CopyTensorDistAttrForOutput(scale.dist_attr());
  scale_dist_attr_dst.set_dims_mapping({-1});

  VLOG(4) << "InstanceNormGradInferSpmd:";
  VLOG(4) << "data type: [N,C,H,W]";
  VLOG(4) << "Einsum Notation: " << x_axes << "," << scale_axes << ","
          << saved_mean_axes << "," << saved_variance_axes << "," << y_grad_axes
          << "-->" << x_grad_axes << "," << scale_grad_axes << ","
          << bias_grad_axes;
  VLOG(4) << "X"
          << " shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(x_dims_mapping) << "]";
  VLOG(4) << "Scale"
          << " shape: [" << str_join(scale_shape) << "] "
          << "src_dims_mapping: [" << str_join(scale_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(scale_dist_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "Saved_mean"
          << " shape: [" << str_join(saved_mean_shape) << "] "
          << "src_dims_mapping: [" << str_join(saved_mean_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(saved_mean_dist_attr.dims_mapping()) << "]";
  VLOG(4) << "Saved_variance"
          << " shape: [" << str_join(saved_variance_shape) << "] "
          << "src_dims_mapping: [" << str_join(saved_variance_dims_mapping)
          << "] "
          << "dst_dims_mapping: ["
          << str_join(saved_variance_dist_attr.dims_mapping()) << "]";
  VLOG(4) << "Y_grad"
          << " shape: [" << str_join(y_grad_shape) << "] "
          << "src_dims_mapping: [" << str_join(y_grad_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(y_grad_dist_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "x_grad dims mapping: ["
          << str_join(x_grad_dist_attr.dims_mapping()) << "]";
  VLOG(4) << "Scale_grad dims mapping: ["
          << str_join(scale_grad_dist_attr.dims_mapping()) << "]";
  VLOG(4) << "Bias_grad dims mapping: ["
          << str_join(bias_grad_dist_attr.dims_mapping()) << "]";
  VLOG(4) << std::endl;

  return {{x_dist_attr_dst,
           scale_dist_attr_dst,
           saved_mean_dist_attr,
           saved_variance_dist_attr,
           y_grad_dist_attr_dst},
          {x_grad_dist_attr, scale_grad_dist_attr, bias_grad_dist_attr}};
}
}  // namespace phi::distributed
