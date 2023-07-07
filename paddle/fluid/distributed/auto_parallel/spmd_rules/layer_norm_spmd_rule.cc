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
  int x_ndim = x_shape.size();
  int scale_ndim = scale_shape.size();
  int bias_ndim = bias_shape.size();

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

  // Step2.3.2  handle input tensor partial (TODO)
  VLOG(4) << "LayerNormSPMDRule InferForward Inputs: "
          << "x shape: [" << str_join(x_shape) << "], x_dims_mapping: ["
          << str_join(x_dims_mapping) << "]; scale shape: ["
          << str_join(scale_shape) << "], scale_dims_mapping: ["
          << str_join(scale_dims_mapping) << "]; bias shape: ["
          << str_join(bias_shape) << "], bias_dims_mapping: ["
          << str_join(bias_dims_mapping) << "]; begin_norm_axis: ["
          << begin_norm_axis << "]; ";

  // step1: build Einsum Notation
  // ijk,k,k->ijk,x,x (x,scale,bias->out,mean,variance, begin_norm_axis=2, x=ij)
  // ijkl,y(kl),y(kl)->ijkl,x(ij),x(ij) (x,scale,bias->out,mean,variance,
  // begin_norm_axis=2, x=ij, y=kl)
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
  for (size_t i = 0; i < out_axes.size(); ++i) {
    if (i < static_cast<size_t>(begin_norm_axis)) {
      out_dims_mapping.push_back(x_dims_mapping[i]);
      // if ijk,k,k->ijk,x,x (x,scale,bias->out,mean,variance,
      // begin_norm_axis=2, x=ij), and the dims_mapping of input is (0,1,-1),
      // the mean and varience is sharded by dim 0 and 1,
      // which is not supported currently.
      mean_shard_dim =
          ShardingMergeForAxis(mean_axes, mean_shard_dim, x_dims_mapping[i]);
    } else {
      out_dims_mapping.push_back(-1);
    }
  }
  output_dist_attr_dst.set_dims_mapping(out_dims_mapping);
  mean_dist_attr_dst.set_dims_mapping({mean_shard_dim});
  varience_dist_attr_dst.set_dims_mapping({mean_shard_dim});

  // step2.3: Merge and get Inputs' New Dims Mapping.
  x_dist_attr_dst.set_dims_mapping(out_dims_mapping);
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
    const std::vector<DistTensorSpec>& output_specs,
    const paddle::framework::AttributeMap& attrs) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "InferBackward of LayerNormSPMDRule is NOT implemented yet."));

  return {};
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
