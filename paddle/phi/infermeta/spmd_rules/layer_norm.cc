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

#include "paddle/phi/infermeta/spmd_rules/layer_norm.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo LayerNormInferSpmd(const DistMetaTensor& x,
                            const DistMetaTensor& scale,
                            const DistMetaTensor& bias,
                            float epsilon,
                            int begin_norm_axis) {
  // Step0: verify input args based on layer_norm logic
  auto x_shape = phi::vectorize(x.dims());
  auto scale_shape = phi::vectorize(scale.dims());
  auto bias_shape = phi::vectorize(bias.dims());
  int x_ndim = x_shape.size();
  int scale_ndim = scale_shape.size();
  int bias_ndim = bias_shape.size();
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> scale_dims_mapping = scale.dist_attr().dims_mapping();
  std::vector<int64_t> bias_dims_mapping = bias.dist_attr().dims_mapping();

  PADDLE_ENFORCE_EQ(
      scale_ndim,
      1,
      phi::errors::InvalidArgument(
          "The ndim of scale in layer_norm should be 1, but got [%d].",
          scale_ndim));

  PADDLE_ENFORCE_EQ(
      bias_ndim,
      1,
      phi::errors::InvalidArgument(
          "The ndim of bias in layer_norm should be 1, but got [%d].",
          bias_ndim));

  // Step1: Build Einsum Notation
  // ijk,k,k->ijk,z,z (x,scale,bias->out,mean,variance, begin_norm_axis=2, z=ij)
  // ijkl,y(kl),y(kl)->ijkl,z(ij),z(ij) (x,scale,bias->out,mean,variance,
  // begin_norm_axis=2, z=ij, y=kl)
  std::string alphabet = "ijklmnopqrstuvwxyz";
  // get input notation
  // Because the mean and variance is 'flattened' from
  // x[0:begin_norm_axis], only the first axis of x can
  // be sharded
  std::string x_axes(x_ndim, '1');
  x_axes[0] = alphabet[0];
  std::string scale_axes(1, x_axes[x_ndim - 1]);
  std::string bias_axes(1, x_axes[x_ndim - 1]);

  // get output notation
  std::string out_axes = x_axes;
  std::string mean_axes(1, '1'), variance_axes(1, '1');
  if (begin_norm_axis > 0) {
    mean_axes[0] = out_axes[0];
    variance_axes[0] = out_axes[0];
  }

  // Step2: Sharding Propogation
  // Step2.1: merge input sharding
  // As the mean and variance in outputs are `flattened` from
  // x[0:begin_norm_axis], only the first axis can be sharded,
  // the axes 1 to begin_norm_axis-1 are set to be replicated.
  std::fill(x_dims_mapping.begin() + 1, x_dims_mapping.end(), -1);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}});

  // Step2.2: infer output dims mapping
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr mean_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr varience_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(out_axes, axis_to_dim_map));
  mean_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(mean_axes, axis_to_dim_map));
  varience_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(variance_axes, axis_to_dim_map));

  // Step2.3: update input dims mapping
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr scale_dist_attr_dst =
      CopyTensorDistAttrForOutput(scale.dist_attr());
  TensorDistAttr bias_dist_attr_dst =
      CopyTensorDistAttrForOutput(bias.dist_attr());
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  // TODO(zhiqiu): support shardding on scale and bias
  // Now, apply replicating.
  scale_dist_attr_dst.set_dims_mapping({-1});
  bias_dist_attr_dst.set_dims_mapping({-1});

  // Step2.4.  handle input and out tensor partial
  // LayerNorm not support
  VLOG(4) << "LayerNormInferSpmd:";
  VLOG(4) << "begin_norm_axis: " << begin_norm_axis;
  VLOG(4) << "Einsum Notation: " << x_axes << "," << scale_axes << ","
          << bias_axes << "-->" << out_axes << "," << mean_axes << ","
          << variance_axes;
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
  VLOG(4) << "Out dims mapping: [" << str_join(out_dist_attr.dims_mapping())
          << "]";
  VLOG(4) << "Mean dims mapping: [" << str_join(mean_dist_attr.dims_mapping())
          << "]";
  VLOG(4) << "Variance dims mapping: ["
          << str_join(varience_dist_attr.dims_mapping()) << "]";
  VLOG(4) << std::endl;

  return {{x_dist_attr_dst, scale_dist_attr_dst, bias_dist_attr_dst},
          {out_dist_attr, mean_dist_attr, varience_dist_attr}};
}

SpmdInfo LayerNormInferSpmdReverse(const DistMetaTensor& x,
                                   const DistMetaTensor& scale,
                                   const DistMetaTensor& bias,
                                   const DistMetaTensor& out,
                                   const DistMetaTensor& mean,
                                   const DistMetaTensor& variance,
                                   float epsilon,
                                   int begin_norm_axis) {
  // Step0: Verify input args based on layer_norm logic
  auto x_shape = phi::vectorize(x.dims());
  auto out_shape = phi::vectorize(out.dims());
  auto mean_shape = phi::vectorize(mean.dims());
  auto variance_shape = phi::vectorize(variance.dims());
  int x_ndim = x_shape.size();
  int out_ndim = out_shape.size();
  int mean_ndim = mean_shape.size();
  int variance_ndim = variance_shape.size();
  auto out_dist_attr_src = out.dist_attr();
  auto mean_dist_attr_src = mean.dist_attr();
  auto variance_dist_attr_src = variance.dist_attr();
  std::vector<int64_t> out_dims_mapping = out_dist_attr_src.dims_mapping();
  std::vector<int64_t> mean_dims_mapping = mean_dist_attr_src.dims_mapping();
  std::vector<int64_t> variance_dims_mapping =
      variance_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor Out's rank [%d] and Out's "
                                   "dims_mapping size [%d] are not matched.",
                                   out_ndim,
                                   out_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(
      mean_ndim,
      mean_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor Mean's rank [%d] and Mean's "
                                   "dims_mapping size [%d] are not matched.",
                                   mean_ndim,
                                   mean_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(variance_ndim,
                    variance_dims_mapping.size(),
                    phi::errors::InvalidArgument(
                        "The Tensor Variance's rank [%d] and Variance's "
                        "dims_mapping size [%d] are not matched.",
                        variance_ndim,
                        variance_dims_mapping.size()));
  // Step1: Build Einsum Notation
  // ijk,k,k->ijk,z,z (x,scale,bias->out,mean,variance, begin_norm_axis=2, z=ij)
  // ijkl,y(kl),y(kl)->ijkl,z(ij),z(ij) (x,scale,bias->out,mean,variance,
  // begin_norm_axis=2, z=ij, y=kl)
  std::string alphabet = "ijklmnopqrstuvwxyz";
  // the axes after norm_axis should be replicated,
  // so set their notation to '1'.
  std::string x_axes(x_ndim, '1');
  x_axes[0] = alphabet[0];
  std::string scale_axes(1, x_axes[x_ndim - 1]);
  std::string bias_axes(1, x_axes[x_ndim - 1]);

  std::string out_axes = x_axes;
  std::string mean_axes(1, '1'), variance_axes(1, '1');
  if (begin_norm_axis > 0) {
    mean_axes[0] = out_axes[0];
    variance_axes[0] = out_axes[0];
  }

  // Step2: Sharding Propogation
  // For the axes after norm_axis in both input and output tensors,
  // set their dims mappings to -1. For the other axes, set input
  // tensor's dims mapping the same as output tensor's dims mapping.
  // step2.1 merge dims mappings of output, mean, variance.
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  axes_sharding_info.emplace_back(std::make_pair(out_axes, out_dims_mapping));
  axes_sharding_info.emplace_back(std::make_pair(mean_axes, mean_dims_mapping));
  axes_sharding_info.emplace_back(
      std::make_pair(variance_axes, variance_dims_mapping));
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  // Step2.2 infer input dims mapping
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  std::vector<TensorDistAttr> input_dist_attrs;
  input_dist_attrs.emplace_back(x.dist_attr());
  input_dist_attrs.emplace_back(scale.dist_attr());
  input_dist_attrs.emplace_back(bias.dist_attr());

  input_dist_attrs[0].set_dims_mapping(x_dims_mapping);
  // set bias and scale to be replicated
  input_dist_attrs[1].set_dims_mapping({-1});
  input_dist_attrs[2].set_dims_mapping({-1});

  // Step2.3 Update output dims mappings with merged one
  std::vector<TensorDistAttr> output_dist_attrs;
  output_dist_attrs.emplace_back(out_dist_attr_src);
  output_dist_attrs.emplace_back(mean_dist_attr_src);
  output_dist_attrs.emplace_back(variance_dist_attr_src);
  output_dist_attrs[0].set_dims_mapping(
      GetDimsMappingForAxes(out_axes, axis_to_dim_map));
  output_dist_attrs[1].set_dims_mapping(
      GetDimsMappingForAxes(mean_axes, axis_to_dim_map));
  output_dist_attrs[2].set_dims_mapping(
      GetDimsMappingForAxes(variance_axes, axis_to_dim_map));

  VLOG(4) << "LayerNormInferSpmdReverse:";
  VLOG(4) << "begin_norm_axis: " << begin_norm_axis;
  VLOG(4) << "Einsum Notation: " << x_axes << "," << scale_axes << ","
          << bias_axes << "-->" << out_axes << "," << mean_axes << ","
          << variance_axes;
  VLOG(4) << "Out"
          << " shape: [" << str_join(out_shape) << "] "
          << " src_dims_mapping: [" << str_join(out_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(output_dist_attrs[0].dims_mapping()) << "]";
  VLOG(4) << "Mean"
          << " shape: [" << str_join(mean_shape) << "] "
          << " src_dims_mapping: [" << str_join(mean_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(output_dist_attrs[1].dims_mapping()) << "]";
  VLOG(4) << "Variance"
          << " shape: [" << str_join(variance_shape) << "] "
          << " src_dims_mapping: [" << str_join(variance_dims_mapping) << "] "
          << "dst_dims_mapping: ["
          << str_join(output_dist_attrs[2].dims_mapping()) << "]";

  for (int i = 0, n = input_dist_attrs.size(); i < n; i++) {
    VLOG(4) << "Input" << std::to_string(i) << " dims_mapping: ["
            << str_join(input_dist_attrs[i].dims_mapping()) << "]";
  }
  VLOG(4) << std::endl;

  return {input_dist_attrs, output_dist_attrs};
}

}  // namespace distributed
}  // namespace phi
