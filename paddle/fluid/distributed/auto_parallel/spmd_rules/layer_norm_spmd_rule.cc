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
  // step0: verify input args based on matmul logic
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
  int scale_ndim = y_shape.size();
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

  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      phi::errors::InvalidArgument(
          "Mismatch of X's tensor size: [%d] and X's dims_mapping size [%d].",
          x_ndim,
          x_dims_mapping.size()));

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
  // ijk,k,k->ijk (x,scale,bias->out, begin_norm_axis=2)
  // ijkl,z(kl),z(kl)->ijkl (x,scale,bias->out, begin_norm_axis=2, z=kl)
  std::string x_axes = "";
  for (auto i = 0; i < x_ndim; ++i) {
    x_axes += static_cast<char>(static_cast<int>('k') - begin_norm_axis + i);
  }
  // scale_ndim = 1
  if (x_ndim - begin_norm_axis == 1) {
    scale_axes = "k";
    bias_axes = "k";
  } else {
    scale_axes =
        "z";  // z = x_axes.substr(begin_norm_axis, x_ndim - begin_norm_axis)
    bias_axes = "z";
  }
  std::string out_axes = x_axes;

  VLOG(4) << "LayerNormSPMDRule build Einsum notation (x,scale,bias->out): ["
          << x_axes << "," << scale_axes << "," << bias_axes << " --> "
          << out_axes << "](begin_norm_axis:" << begin_norm_axis
          << ",z=" << x_axes.substr(begin_norm_axis, x_ndim - begin_norm_axis)
          << ").";

  // step2: Sharding Propogation
  auto x_dist_attr_src = input_specs[0].dist_attr();
  auto x_dims_mapping = x_dist_attr_src.dims_mapping();

  TensorDistAttr output_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  std::vector<int64_t> out_dims_mapping;
  out_dims_mapping.reserve(out_axes.size());
  for (size_t i = 0; i < out_axes.size(); ++i) {
    if (i < begin_norm_axis) {
      out_dims_mapping.push_back(x_dims_mapping[i]);
    } else {
      out_dims_mapping.push_back(-1);
    }
  }
  output_dist_attr_dst.set_dims_mapping(out_dims_mapping);
  x_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  // step2.3: Merge and get Inputs' New Dims Mapping.
  input_dist_attrs.emplace_back(x_dist_attr_dst);

  // TODO(zhiqiu): support shardding on scale and bias
  // Now, apply replicating.
  input_dist_attrs.emplace_back(ReplicatedOnMesh(input_specs[1].dist_attr()));
  input_dist_attrs.emplace_back(ReplicatedOnMesh(input_specs[2].dist_attr()));

  // Step2.4.  handle input and out tensor partial
  std::vector<int64_t> partial_on_dims;
  for (auto i = 0; i < out_dims_mapping.size(); ++i) {
    if (out_dims_mapping[i] > -1) {
      partial_on_dims.push_back(out_dims_mapping[i]);
    }
  }

  VLOG(4) << "LayerNormSPMDRule InferForward: "
          << "X shape: [" << str_join(x_shape) << "], src_dims_mapping: ["
          << str_join(x_dist_attr_src.dims_mapping())
          << "], dst_dims_mapping: ["
          << str_join(x_dist_attr_dst.dims_mapping()) << "]; scale shape: ["
          << str_join(scale_shape) << "], src_dims_mapping: ["
          << str_join(scale_dist_attr_src.dims_mapping())
          << "], dst_dims_mapping: ["
          << str_join(scale_dist_attr_dst.dims_mapping()) << "]; bias shape: ["
          << str_join(bias_shape) << "], src_dims_mapping: ["
          << str_join(bias_dist_attr_src.dims_mapping())
          << "], dst_dims_mapping: ["
          << str_join(bias_dist_attr_dst.dims_mapping())
          << "]; out dims_mapping: [" << str_join(out_dims_mapping)
          << "], partial_on_dims: [" << str_join(partial_on_dims) << "]";

  return {input_dist_attrs, {output_dist_attr_dst}};
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
