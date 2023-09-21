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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/layer_norm_spmd_rule.h"

#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {
using phi::distributed::auto_parallel::str_join;
std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
LayerNormSPMDRule::InferForward(const std::vector<DistTensorSpec>& input_specs,
                                const paddle::framework::AttributeMap& attrs) {
  // step0: verify input args based on layer_norm logic
  auto input_specs_size = input_specs.size();
  PADDLE_ENFORCE_EQ(
      input_specs_size,
      3,
      phi::errors::InvalidArgument(
          "The size of InputSpec of layer_norm should be 3, but got [%d].",
          input_specs_size));
  auto x_shape = input_specs[0].shape();
  auto scale_shape = input_specs[1].shape();
  auto bias_shape = input_specs[2].shape();
  int x_ndim = static_cast<int>(x_shape.size());
  int scale_ndim = static_cast<int>(scale_shape.size());
  int bias_ndim = static_cast<int>(bias_shape.size());

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

  auto x_dims_mapping = input_specs[0].dist_attr().dims_mapping();
  auto scale_dims_mapping = input_specs[1].dist_attr().dims_mapping();
  auto bias_dims_mapping = input_specs[2].dist_attr().dims_mapping();

  auto x_dist_attr_src = input_specs[0].dist_attr();

  std::vector<TensorDistAttr> input_dist_attrs;
  input_dist_attrs.reserve(input_specs.size());

  int begin_norm_axis = ExtractAttr<int>("begin_norm_axis", attrs);

  VLOG(4) << "LayerNormSPMDRule InferForward Inputs: "
          << "x shape: [" << str_join(x_shape) << "], x_dims_mapping: ["
          << str_join(x_dims_mapping) << "]; scale shape: ["
          << str_join(scale_shape) << "], scale_dims_mapping: ["
          << str_join(scale_dims_mapping) << "]; bias shape: ["
          << str_join(bias_shape) << "], bias_dims_mapping: ["
          << str_join(bias_dims_mapping) << "]; begin_norm_axis: ["
          << begin_norm_axis << "]; ";

  // step1: build Einsum Notation
  // ijk,k,k->ijk,z,z (x,scale,bias->out,mean,variance, begin_norm_axis=2, z=ij)
  // ijkl,y(kl),y(kl)->ijkl,z(ij),z(ij) (x,scale,bias->out,mean,variance,
  // begin_norm_axis=2, z=ij, y=kl)
  std::string x_axes = "";
  for (auto i = 0; i < x_ndim; ++i) {
    x_axes += static_cast<char>(static_cast<int>('k') - begin_norm_axis + i);
  }

  std::string scale_axes;
  std::string bias_axes;
  if (x_ndim - begin_norm_axis == 1) {
    scale_axes = "k";
    bias_axes = "k";
  } else {
    // z = x_axes.substr(begin_norm_axis, x_ndim - begin_norm_axis)
    scale_axes = "y";
    bias_axes = "y";
  }

  std::string mean_axes;
  std::string variance_axes;
  if (begin_norm_axis > 1) {
    mean_axes = "z";
    variance_axes = "z";
  } else {
    mean_axes = "j";
    variance_axes = "j";
  }

  std::string out_axes = x_axes;

  VLOG(4) << "LayerNormSPMDRule build Einsum notation (x,scale,bias->out): ["
          << x_axes << "," << scale_axes << "," << bias_axes << " --> "
          << out_axes << "," << mean_axes << "," << variance_axes
          << "](begin_norm_axis:" << begin_norm_axis
          << ",y=" << x_axes.substr(begin_norm_axis, x_ndim - begin_norm_axis)
          << ",z=" << x_axes.substr(0, begin_norm_axis) << ").";

  // step2: Sharding Propogation
  TensorDistAttr output_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr mean_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr varience_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  std::vector<int64_t> out_dims_mapping;
  out_dims_mapping.reserve(out_axes.size());

  int64_t mean_shard_dim = -1;
  // As the mean and variance in outputs are `flattened` from
  // x[0:begin_norm_axis], only the first axis can be sharded,
  // the axes 1 to begin_norm_axis-1 are set to be replicated.
  std::vector<int64_t> x_dims_mapping_dst(x_ndim, -1);
  x_dims_mapping_dst[0] = x_dims_mapping[0];
  for (int i = 0; i < x_ndim; ++i) {
    if (i < begin_norm_axis) {
      out_dims_mapping.push_back(x_dims_mapping_dst[i]);
      // if ijk,k,k->ijk,z,z (x,scale,bias->out,mean,variance,
      // begin_norm_axis=2, z=ij), and the dims_mapping of input is (0,1,-1),
      // the mean and varience is sharded by dim 0 and 1,
      // which is not supported currently.
      mean_shard_dim = ShardingMergeForAxis(
          mean_axes, mean_shard_dim, x_dims_mapping_dst[i]);
    } else {
      out_dims_mapping.push_back(-1);
    }
  }
  output_dist_attr_dst.set_dims_mapping(out_dims_mapping);
  mean_dist_attr_dst.set_dims_mapping({mean_shard_dim});
  varience_dist_attr_dst.set_dims_mapping({mean_shard_dim});

  // step2.3: Merge and get Inputs' New Dims Mapping.
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  input_dist_attrs.emplace_back(x_dist_attr_dst);
  // TODO(zhiqiu): support shardding on scale and bias
  // Now, apply replicating.
  input_dist_attrs.emplace_back(ReplicatedOnMesh(input_specs[1].dist_attr()));
  input_dist_attrs.emplace_back(ReplicatedOnMesh(input_specs[2].dist_attr()));

  // Step2.4.  handle input and out tensor partial
  // LayerNorm not support

  VLOG(4) << "LayerNormSPMDRule InferForward: "
          << "X shape: [" << str_join(x_shape) << "], src_dims_mapping: ["
          << str_join(x_dims_mapping) << "], dst_dims_mapping: ["
          << str_join(x_dist_attr_dst.dims_mapping()) << "]; scale shape: ["
          << str_join(scale_shape) << "], src_dims_mapping: ["
          << str_join(scale_dims_mapping) << "], dst_dims_mapping: ["
          << str_join(input_dist_attrs[1].dims_mapping()) << "]; bias shape: ["
          << str_join(bias_shape) << "], src_dims_mapping: ["
          << str_join(bias_dims_mapping) << "], dst_dims_mapping: ["
          << str_join(input_dist_attrs[2].dims_mapping())
          << "]; out dims_mapping: [" << str_join(out_dims_mapping)
          << "]; mean dims_mapping: [" << mean_shard_dim
          << "]; varience dims_mapping: [" << mean_shard_dim
          << "], partial_on_dims: []";

  return {input_dist_attrs,
          {output_dist_attr_dst, mean_dist_attr_dst, varience_dist_attr_dst}};
}

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
LayerNormSPMDRule::InferBackward(
    const std::vector<DistTensorSpec>& input_specs,
    const std::vector<DistTensorSpec>& output_specs,
    const paddle::framework::AttributeMap& attrs) {
  // step0: verify input args based on layer_norm logic
  int64_t ninputs = input_specs.size();
  int64_t noutputs = output_specs.size();
  PADDLE_ENFORCE_EQ(
      ninputs,
      3,
      phi::errors::InvalidArgument(
          "The size of InputSpec of layer_norm should be 3, but got [%d].",
          ninputs));
  PADDLE_ENFORCE_EQ(
      noutputs,
      3,
      phi::errors::InvalidArgument(
          "The size of InputSpec of layer_norm should be 3, but got [%d].",
          noutputs));
  VerifySpecs(output_specs, "layer_norm_backward");

  // step1: build Einsum Notation
  // ijk,k,k->ijk,z,z (x,scale,bias->out,mean,variance, begin_norm_axis=2, z=ij)
  // ijkl,y(kl),y(kl)->ijkl,z(ij),z(ij) (x,scale,bias->out,mean,variance,
  // begin_norm_axis=2, z=ij, y=kl)
  int begin_norm_axis = ExtractAttr<int>("begin_norm_axis", attrs);
  std::string alphabet = "ijklmnopqrstuvwxyz";
  int x_ndim = input_specs[0].shape().size();
  std::string x_axes = alphabet.substr(0, x_ndim);
  // the axes after norm_axis should be replicated,
  // so set their notation to '1'.
  for (int i = 1; i < x_ndim; i++) {
    x_axes[i] = '1';
  }
  std::string out_axes = x_axes;
  std::string mean_axes(1, '1'), varience_axes(1, '1');
  if (begin_norm_axis > 0) {
    mean_axes[0] = out_axes[0];
    varience_axes[0] = out_axes[0];
  }
  std::vector<std::string> output_axes_vec;
  output_axes_vec.emplace_back(out_axes);
  output_axes_vec.emplace_back(mean_axes);
  output_axes_vec.emplace_back(varience_axes);

  // step2: Sharding Propogation
  // For the axes after norm_axis in both input and output tensors,
  // set their dims mappings to -1. For the other axes, set input
  // tensor's dims mapping the same as output tensor's dims mapping.
  // step2.1 merge dims mappings of output, mean, variance.
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  axes_sharding_info = GetAxesDimsMappingPair(output_axes_vec, output_specs);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  // step2.2 infer input dims mapping
  std::vector<int64_t> input_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  std::vector<TensorDistAttr> input_dist_attrs;
  for (int64_t i = 0; i < ninputs; i++) {
    input_dist_attrs.emplace_back(input_specs[i].dist_attr());
  }
  input_dist_attrs[0].set_dims_mapping(input_dims_mapping);
  // set bias and scale to be replicated
  input_dist_attrs[1].set_dims_mapping({-1});
  input_dist_attrs[2].set_dims_mapping({-1});

  // step2.3 update output dims mappings with merged one
  std::vector<TensorDistAttr> output_dist_attrs;
  for (int64_t i = 0; i < noutputs; i++) {
    output_dist_attrs.emplace_back(output_specs[i].dist_attr());
    output_dist_attrs[i].set_dims_mapping(
        GetDimsMappingForAxes(output_axes_vec[i], axis_to_dim_map));
  }

  VLOG(4) << "LayerNormSPMDRule InferBackward:";
  VLOG(4) << "begin_norm_axis: " << begin_norm_axis;
  for (int64_t i = 0; i < noutputs; i++) {
    VLOG(4) << "Output" << std::to_string(i) << " shape: ["
            << str_join(output_specs[i].shape()) << "] "
            << "Einsum Notation: " << output_axes_vec[i]
            << " src_dims_mapping: ["
            << str_join(output_specs[i].dims_mapping()) << "] "
            << "dst_dims_mapping: ["
            << str_join(output_dist_attrs[i].dims_mapping()) << "]";
  }

  for (int64_t i = 0; i < ninputs; i++) {
    VLOG(4) << "Input" << std::to_string(i) << " shape: ["
            << str_join(input_specs[i].shape()) << "] "
            << "Einsum Notation: " << std::string(i == 0 ? x_axes : "1")
            << " dims_mapping: ["
            << str_join(input_dist_attrs[i].dims_mapping()) << "]";
  }
  VLOG(4) << std::endl;

  return {input_dist_attrs, output_dist_attrs};
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
