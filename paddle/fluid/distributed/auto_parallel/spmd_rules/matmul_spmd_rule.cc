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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/matmul_spmd_rule.h"

#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {
using phi::distributed::auto_parallel::str_join;
std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
MatmulSPMDRule::InferForward(const std::vector<DistTensorSpec>& input_specs,
                             const paddle::framework::AttributeMap& attrs) {
  // step0: verify input args based on matmul logic
  auto input_specs_size = input_specs.size();
  PADDLE_ENFORCE_EQ(
      input_specs_size,
      2,
      phi::errors::InvalidArgument(
          "The size of InputSpec of matmul should be 2, but got [%d].",
          input_specs_size));
  auto x_shape = input_specs[0].shape();
  auto y_shape = input_specs[1].shape();
  int x_ndim = x_shape.size();
  int y_ndim = y_shape.size();
  auto x_dist_attr_src = input_specs[0].dist_attr();
  auto y_dist_attr_src = input_specs[1].dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> y_dims_mapping = y_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      phi::errors::InvalidArgument(
          "Mismatch of X's tensor size: [%d] and X's dims_mapping size [%d].",
          x_ndim,
          x_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(
      y_ndim,
      y_dims_mapping.size(),
      phi::errors::InvalidArgument(
          "Mismatch of Y's tensor size: [%d] and Y's dims_mapping size [%d].",
          y_ndim,
          y_dims_mapping.size()));

  bool trans_x = ExtractAttr<bool>("trans_x", attrs);
  bool trans_y = ExtractAttr<bool>("trans_y", attrs);

  VLOG(6) << "MatmulSPMDRule InferForward Inputs: "
          << "X shape: [" << str_join(x_shape) << "], x_dims_mapping: ["
          << str_join(x_dims_mapping) << "]; Y shape: [" << str_join(y_shape)
          << "], y_dims_mapping: [" << str_join(y_dims_mapping)
          << "]; trans_x: "
          << "[" << (trans_x ? "true" : "false") << "]; "
          << "trans_y: "
          << "[" << (trans_y ? "true" : "false") << "]; ";

  // step1: build Einsum Notation

  // reserve the char k, m, n for matrix product notation: mk,kn -> mn
  int max_ndim = std::max(x_ndim, y_ndim);
  std::string alphabet = "abcdefghijlopqrstuvwxyz";
  std::string x_axes;
  std::string y_axes;
  std::string out_axes;

  // Handle 4 different matmul cases in Paddle
  // vector * vector = scala
  if (x_ndim == 1 && y_ndim == 1) {
    x_axes = "k";
    y_axes = "k";
    out_axes = "";
    // vector * batched matrix
  } else if (x_ndim == 1 && y_ndim > 1) {
    x_axes = "k";
    std::string y_broadcast_axes =
        GetBroadcastAxes(y_ndim - 2, y_ndim - 2, alphabet);
    y_axes = y_broadcast_axes + "kn";
    out_axes = y_broadcast_axes + "n";
    // batched matrix * vector
  } else if (x_ndim > 1 && y_ndim == 1) {
    y_axes = "k";
    std::string x_broadcast_axes =
        GetBroadcastAxes(x_ndim - 2, x_ndim - 2, alphabet);
    x_axes = x_broadcast_axes + "mk";
    out_axes = x_broadcast_axes + "m";
    // batched matrix * batched matrix
  } else if (x_ndim > 1 && y_ndim > 1) {
    std::string x_broadcast_axes =
        GetBroadcastAxes(x_ndim - 2, max_ndim - 2, alphabet);
    std::string y_broadcast_axes =
        GetBroadcastAxes(y_ndim - 2, max_ndim - 2, alphabet);
    x_axes = x_broadcast_axes + "mk";
    y_axes = y_broadcast_axes + "kn";

    if (x_ndim > y_ndim) {
      out_axes = x_broadcast_axes + "mn";
    } else {
      out_axes = y_broadcast_axes + "mn";
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "MatmulSPMDRule Receive Unsupported x_dim [%d] and y_dim [%d].",
        x_ndim,
        y_ndim));
  }

  // step2: Sharding Propogation
  if (trans_x) {
    PADDLE_ENFORCE_GE(
        x_ndim,
        2,
        phi::errors::InvalidArgument("When trans_x is True, the size of X "
                                     "tensor should be 2,  but got [%d].",
                                     x_ndim));
    std::iter_swap(x_dims_mapping.end() - 2, x_dims_mapping.end() - 1);
  }
  if (trans_y) {
    PADDLE_ENFORCE_GE(
        y_ndim,
        2,
        phi::errors::InvalidArgument("When trans_x is True, the size of X "
                                     "tensor should be 2,  but got [%d].",
                                     y_ndim));
    std::iter_swap(y_dims_mapping.end() - 2, y_dims_mapping.end() - 1);
  }
  // step2.1: Sharding Merge
  std::pair<std::string, std::vector<int64_t>> x_pair(x_axes, x_dims_mapping);
  std::pair<std::string, std::vector<int64_t>> y_pair(y_axes, y_dims_mapping);
  auto axis_to_dim_map = ShardingMergeForTensors({x_pair, y_pair});

  // step2.2: Infer Output's Dims Mapping.
  TensorDistAttr output_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  std::vector<int64_t> out_dims_mapping;
  out_dims_mapping.reserve(out_axes.size());
  for (size_t i = 0; i < out_axes.size(); ++i) {
    out_dims_mapping.push_back(axis_to_dim_map[out_axes.substr(i, 1)]);
  }
  output_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  // step2.3: Merge and get Inputs' New Dims Mapping.
  TensorDistAttr x_dist_attr_dst = GetInferedDistAttr(
      x_dist_attr_src, x_shape, x_axes, axis_to_dim_map, trans_x);
  TensorDistAttr y_dist_attr_dst = GetInferedDistAttr(
      y_dist_attr_src, y_shape, y_axes, axis_to_dim_map, trans_y);

  // step2.3: Handle Partial
  // Step2.3.1 Output Partial
  std::vector<int64_t> partial_on_dims =
      ResoluteOutputPartialDimension(axis_to_dim_map, out_axes);

  // Step2.3.2  handle input tensor partial (TODO)
  VLOG(4) << "MatmulSPMDRule InferForward: "
          << "Einsum notation: [" << x_axes << "," << y_axes << " --> "
          << out_axes << "]. " << std::endl
          << "X shape: [" << str_join(x_shape) << "], src_dims_mapping: ["
          << str_join(x_dist_attr_src.dims_mapping())
          << "], dst_dims_mapping: ["
          << str_join(x_dist_attr_dst.dims_mapping()) << "]; Y shape: ["
          << str_join(y_shape) << "], src_dims_mapping: ["
          << str_join(y_dist_attr_src.dims_mapping())
          << "], dst_dims_mapping: ["
          << str_join(y_dist_attr_dst.dims_mapping())
          << "]; Output dims_mapping: [" << str_join(out_dims_mapping)
          << "], partial_on_dims: [" << str_join(partial_on_dims) << "]";

  return {{x_dist_attr_dst, y_dist_attr_dst}, {output_dist_attr_dst}};
}

TensorDistAttr GetInferedDistAttr(
    const TensorDistAttr& origin_dist_attr,
    const std::vector<int64_t>& shape,
    const std::string& tensor_axis,
    const std::unordered_map<std::string, int64_t>& axis_to_dim_map,
    const bool trans_axis) {
  TensorDistAttr dist_attr_ = CopyTensorDistAttrForOutput(origin_dist_attr);
  std::vector<int64_t> infered_dims_mapping;
  infered_dims_mapping.reserve(tensor_axis.size());

  for (size_t i = 0; i < tensor_axis.size(); ++i) {
    if (shape.size() > i && shape[i] == 1) {
      infered_dims_mapping.push_back(-1);
    } else {
      auto itr = axis_to_dim_map.find(tensor_axis.substr(i, 1));
      if (itr == axis_to_dim_map.end()) {
        phi::errors::InvalidArgument(
            "Tensor axis [%s] of not in axis_to_dim_map.",
            tensor_axis.substr(i, 1));
      }
      infered_dims_mapping.push_back(itr->second);
    }
  }

  if (trans_axis) {
    std::iter_swap(infered_dims_mapping.end() - 2,
                   infered_dims_mapping.end() - 1);
  }

  dist_attr_.set_dims_mapping(infered_dims_mapping);
  return dist_attr_;
}

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
MatmulSPMDRule::InferBackward(const std::vector<DistTensorSpec>& output_specs,
                              const paddle::framework::AttributeMap& attrs) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "InferBackward of MatmulSPMDRule is NOT implemented yet."));

  return {};
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
