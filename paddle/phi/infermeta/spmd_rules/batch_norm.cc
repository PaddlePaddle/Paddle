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
                            const std::string data_format,
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
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> mean_dims_mapping = mean.dist_attr().dims_mapping();
  std::vector<int64_t> variance_dims_mapping =
      variance.dist_attr().dims_mapping();
  std::vector<int64_t> scale_dims_mapping = scale.dist_attr().dims_mapping();
  std::vector<int64_t> bias_dims_mapping = bias.dist_attr().dims_mapping();

  PADDLE_ENFORCE_EQ(
      x_ndim,
      4,
      common::errors::InvalidArgument(
          "The ndim of x in batch_norm should be 4, but got [%d].", x_ndim));

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
  std::string mean_axes(1, '1');
  std::string variance_axes(1, '1');
  std::string scale_axes(1, '1');
  std::string bias_axes(1, '1');

  //  allow axis before begin_norm_axis be sharded
  for (int i = 0; i < x_ndim; ++i) {
    x_axes[i] = alphabet[i];
  }
  if (data_format == "NHWC") {
    mean_axes[0] = x_axes[3];
    variance_axes[0] = x_axes[3];
    scale_axes[0] = x_axes[3];
    bias_axes[0] = x_axes[3];
  } else {  // NCHW
    mean_axes[0] = x_axes[1];
    variance_axes[0] = x_axes[1];
    scale_axes[0] = x_axes[1];
    bias_axes[0] = x_axes[1];
  }

  // get output notation
  std::string out_axes = x_axes;

  // Step2: Sharding Propagation
  // Step2.1: merge input sharding
  // Only C axis can be shard.
  if (data_format == "NHWC") {
    for (int i = 0; i < x_ndim - 1; ++i) {
      x_dims_mapping[i] = -1;
    }

  } else {  // NCHW
    x_dims_mapping[0] = -1;
    x_dims_mapping[2] = -1;
    x_dims_mapping[3] = -1;
  }
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}});

  // Step2.2: infer output dims mapping
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr mean_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr variance_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr saved_mean_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr saved_variance_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
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
  std::vector<int64_t> reserve_space_dims_mapping(1);
  reserve_space_dims_mapping[0] = -1;
  reserve_space_dist_attr.set_dims_mapping(reserve_space_dims_mapping);

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
  scale_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(scale_axes, axis_to_dim_map));
  bias_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(bias_axes, axis_to_dim_map));
  variance_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(variance_axes, axis_to_dim_map));
  mean_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(mean_axes, axis_to_dim_map));

  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  VLOG(4) << "BatchNormInferSpmd:";
  VLOG(4) << "Einsum Notation: " << x_axes << "," << mean_axes << ","
          << variance_axes << "," << scale_axes << "," << bias_axes << "-->"
          << out_axes << "," << mean_axes << "," << variance_axes;
  VLOG(4) << "X"
          << " shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(x_dims_mapping) << "]";
  VLOG(4) << "Mean"
          << " shape: [" << str_join(mean_shape) << "] "
          << "src_dims_mapping: [" << str_join(mean_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(mean_dist_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "Variance"
          << " shape: [" << str_join(variance_shape) << "] "
          << "src_dims_mapping: [" << str_join(variance_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(variance_dist_attr_dst.dims_mapping()) << "]";
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
  VLOG(4) << "Out dims mapping: [" << str_join(out_dist_attr.dims_mapping())
          << "]";
  VLOG(4) << "Mean_out dims mapping: ["
          << str_join(mean_dist_attr.dims_mapping()) << "]";
  VLOG(4) << "Variance_out dims mapping: ["
          << str_join(variance_dist_attr.dims_mapping()) << "]";
  VLOG(4) << "Saved_mean dims mapping: ["
          << str_join(mean_dist_attr.dims_mapping()) << "]";
  VLOG(4) << "Saved_variance dims mapping: ["
          << str_join(variance_dist_attr.dims_mapping()) << "]";
  VLOG(4) << "Reserve_space dims mapping: ["
          << str_join(reserve_space_dist_attr.dims_mapping()) << "]";
  VLOG(4) << std::endl;

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
                                const std::string data_format,
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
  PADDLE_ENFORCE_EQ(
      x_ndim,
      4,
      common::errors::InvalidArgument(
          "The ndim of x in batch_norm should be 4, but got [%d].", x_ndim));
  PADDLE_ENFORCE_EQ(
      out_grad_ndim,
      4,
      common::errors::InvalidArgument(
          "The ndim of out_grad in batch_norm should be 4, but got [%d].",
          out_grad_ndim));
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
  std::string mean_out_axes(1, '1');
  std::string variance_out_axes(1, '1');
  std::string scale_axes(1, '1');
  std::string bias_axes(1, '1');
  std::string saved_mean_axes(1, '1');
  std::string saved_variance_axes(1, '1');
  std::string reserve_space_axes(1, '1');
  std::string out_grad_axes(out_grad_ndim, '1');

  //  allow axis before begin_norm_axis be sharded
  for (int i = 0; i < x_ndim; ++i) {
    x_axes[i] = alphabet[i];
    out_grad_axes[i] = alphabet[i];
  }
  if (data_format == "NHWC") {
    mean_out_axes[0] = x_axes[3];
    variance_out_axes[0] = x_axes[3];
    scale_axes[0] = x_axes[3];
    bias_axes[0] = x_axes[3];
    saved_mean_axes[0] = x_axes[3];
    saved_variance_axes[0] = x_axes[3];
    reserve_space_axes[0] = x_axes[3];
  } else {  // NCHW
    mean_out_axes[0] = x_axes[1];
    variance_out_axes[0] = x_axes[1];
    scale_axes[0] = x_axes[1];
    bias_axes[0] = x_axes[1];
    saved_mean_axes[0] = x_axes[1];
    saved_variance_axes[0] = x_axes[1];
    reserve_space_axes[0] = x_axes[1];
  }

  if (data_format == "NHWC") {
    for (int i = 0; i < x_ndim - 1; ++i) {
      x_dims_mapping[i] = -1;
    }
  } else {  // NCHW
    x_dims_mapping[0] = -1;
    x_dims_mapping[2] = -1;
    x_dims_mapping[3] = -1;
  }
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}});
  // infer output spmdinfo
  TensorDistAttr x_grad_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_grad_dist_attr.set_dims_mapping(x_dims_mapping);
  TensorDistAttr scale_grad_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  scale_grad_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(scale_axes, axis_to_dim_map));
  TensorDistAttr bias_grad_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  bias_grad_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(bias_axes, axis_to_dim_map));
  // infer input spmdinfo
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  TensorDistAttr mean_out_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  mean_out_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(mean_out_axes, axis_to_dim_map));
  TensorDistAttr variance_out_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  variance_out_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(variance_out_axes, axis_to_dim_map));
  TensorDistAttr scale_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  scale_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(scale_axes, axis_to_dim_map));
  TensorDistAttr bias_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  bias_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(bias_axes, axis_to_dim_map));
  TensorDistAttr saved_mean_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  saved_mean_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(saved_mean_axes, axis_to_dim_map));
  TensorDistAttr saved_variance_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  saved_variance_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(saved_variance_axes, axis_to_dim_map));
  TensorDistAttr reserve_space_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  std::vector<int64_t> reserve_space_dims_mapping = {-1};
  reserve_space_attr_dst.set_dims_mapping(reserve_space_dims_mapping);
  TensorDistAttr out_grad_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_grad_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(out_grad_axes, axis_to_dim_map));

  VLOG(4) << "BatchNormGradInferSpmd:";
  VLOG(4) << "Einsum Notation: " << x_axes << scale_axes << "," << bias_axes
          << "," << mean_out_axes << "," << variance_out_axes << ","
          << saved_mean_axes << "," << saved_variance_axes << ","
          << "-->" << reserve_space_axes << "," << out_grad_axes;
  VLOG(4) << "X"
          << " shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(x_dims_mapping) << "]";
  VLOG(4) << "Mean_out"
          << " shape: [" << str_join(mean_out_shape) << "] "
          << "src_dims_mapping: ["
          << str_join(mean_out.dist_attr().dims_mapping()) << "] "
          << "dst_dims_mapping: [" << str_join(mean_out_attr_dst.dims_mapping())
          << "]";
  VLOG(4) << "Variance_out"
          << " shape: [" << str_join(variance_out_shape) << "] "
          << "src_dims_mapping: ["
          << str_join(variance_out.dist_attr().dims_mapping()) << "] "
          << "dst_dims_mapping: ["
          << str_join(variance_out_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "Scale"
          << " shape: [" << str_join(scale_shape) << "] "
          << "src_dims_mapping: [" << str_join(scale.dist_attr().dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(scale_attr_dst.dims_mapping())
          << "]";
  VLOG(4) << "Bias"
          << " shape: [" << str_join(bias_shape) << "] "
          << "src_dims_mapping: [" << str_join(bias.dist_attr().dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(bias_attr_dst.dims_mapping())
          << "]";
  VLOG(4) << "Saved_mean"
          << " shape: [" << str_join(saved_mean_shape) << "] "
          << "src_dims_mapping: ["
          << str_join(saved_mean.dist_attr().dims_mapping()) << "] "
          << "dst_dims_mapping: ["
          << str_join(saved_mean_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "Saved_variance"
          << " shape: [" << str_join(saved_variance_shape) << "] "
          << "src_dims_mapping: ["
          << str_join(saved_variance.dist_attr().dims_mapping()) << "] "
          << "dst_dims_mapping: ["
          << str_join(saved_variance_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "Reserve_space"
          << " shape: [" << str_join(reserve_space_shape) << "] "
          << "src_dims_mapping: ["
          << str_join(reserve_space.dist_attr().dims_mapping()) << "] "
          << "dst_dims_mapping: ["
          << str_join(reserve_space_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "Out_grad"
          << " shape: [" << str_join(out_grad_shape) << "] "
          << "src_dims_mapping: ["
          << str_join(out_grad.dist_attr().dims_mapping()) << "] "
          << "dst_dims_mapping: [" << str_join(out_grad_attr_dst.dims_mapping())
          << "]";

  VLOG(4) << "Out dims mapping: [" << str_join(x_grad_dist_attr.dims_mapping())
          << "]";
  VLOG(4) << "Scale_grad dims mapping: ["
          << str_join(scale_grad_dist_attr.dims_mapping()) << "]";
  VLOG(4) << "Bias_grad dims mapping: ["
          << str_join(bias_grad_dist_attr.dims_mapping()) << "]";
  VLOG(4) << std::endl;

  return {{x_dist_attr_dst,
           scale_attr_dst,
           bias_attr_dst,
           mean_out_attr_dst,
           variance_out_attr_dst,
           saved_mean_attr_dst,
           saved_variance_attr_dst,
           reserve_space_attr_dst,
           out_grad_attr_dst},
          {x_grad_dist_attr, scale_grad_dist_attr, bias_grad_dist_attr}};
}
}  // namespace phi::distributed
