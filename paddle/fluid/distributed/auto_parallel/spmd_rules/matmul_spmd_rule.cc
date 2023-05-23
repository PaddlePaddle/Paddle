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

namespace paddle {
namespace distributed {
namespace auto_parallel {

std::vector<DistTensorSpec> MatmulSPMDRule::InferForward(
    const std::vector<DistTensorSpec>& input_specs,
    const paddle::framework::AttributeMap& attrs) {
  // step0: verify input args based on matmul logic
  int x_ndim = input_specs[0].shape.size();
  int y_ndim = input_specs[1].shape.size();
  std::vector<int64_t> x_dims_mapping = input_specs[0].DistAttr.dims_mapping;
  std::vector<int64_t> y_dims_mapping = input_specs[0].DistAttr.dims_mapping;
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size() phi::errors::InvalidArgument(
          "Mismatch of X's tensor size: [%d] and X's dims_mapping size [%d].",
          x_ndim,
          x_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(
      y_ndim,
      y_dims_mapping.size() phi::errors::InvalidArgument(
          "Mismatch of Y's tensor size: [%d] and Y's dims_mapping size [%d].",
          x_ndim,
          x_dims_mapping.size()));

  bool trans_x = ExtractAttr<bool>("trans_x");
  bool trans_y = ExtractAttr<bool>("trans_y");

  auto input_specs_size = input_specs.size() PADDLE_ENFORCE_EQ(
      input_specs_size,
      2,
      phi::errors::InvalidArgument(
          "The size of InputSpec of matmul should be 2, but got [%d].",
          input_specs_size));

  // step1: Einsum Notation
  int max_ndim = std::max(x_ndim, y_ndim);

  // reserve the char k, m, n for matrix product notation: mk,kn -> mn
  std::string alphabet = "abcdefghijlopqrstuvwxyz";
  std::string x_string;
  std::string y_string;
  std::string out_string;

  // vector * vector = scala
  if (x_ndim == 1 && y_ndim == 1) {
    x_string = "k";
    y_string = "k";
    out_string = "";
    // vector * batched matrix
  } else if (x_ndim == 1 && y_ndim > 1) {
    x_string = "k";
    std::string y_broadcast_string =
        GetBroadcastNotationString(y_ndim, max_ndim, alphabet);
    y_string = y_broadcast_string + "kn";
    out_string = y_broadcast_string + "n";
    // batched matrix * vector
  } else if (x_ndim > 1 && y_ndim == 1) {
    y_string = "k";
    std::string x_broadcast_string =
        GetBroadcastNotationString(x_ndim, max_ndim, alphabet);
    x_string = x_broadcast_string + "mk";
    out_string = x_broadcast_string + "m";
    // batched matrix * batched matrix
  } else if (x_ndim > 1 && y_ndim > 1) {
    std::string x_broadcast_string =
        GetBroadcastNotationString(x_ndim, max_ndim, alphabet);
    std::string y_broadcast_string =
        GetBroadcastNotationString(y_ndim, max_ndim, alphabet);
    x_string = x_broadcast_string + "mk";
    y_string = y_broadcast_string + "kn";

    if (x_ndim > y_ndim) {
      out_string = x_broadcast_string + "mn";
    } else {
      out_string = y_broadcast_string + "mn";
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "MatmulSPMDRule Receive Unsupported x_dim [%d] and y_dim [%d].",
        x_ndim,
        y_ndim));
  }

  VLOG(4) << "MatmulSPMDRule build Einsum notation: [" << x_string << ","
          << y_string << " --> " << out_string << "].";

  // step2: Sharding Propogation
  if (trans_x) {
    PADDLE_ENFORCE_GT(
        x_ndim,
        2,
        phi::errors::InvalidArgument("When trans_x is True, the size of X "
                                     "tensor should be 2,  but got [%d].",
                                     x_ndim));
    std::iter_swap(x_dims_mapping.end() - 2, x_dims_mapping.end() - 1);
  }
  if (trans_y) {
    PADDLE_ENFORCE_GT(
        y_ndim,
        2,
        phi::errors::InvalidArgument("When trans_x is True, the size of X "
                                     "tensor should be 2,  but got [%d].",
                                     y_ndim));
    std::iter_swap(y_dims_mapping.end() - 2, y_dims_mapping.end() - 1);
  }
  // step2.1: Sharding Merge
  std::pair<std::string, std::vector<int64_t>> x_pair(x_string, x_dims_mapping);
  std::pair<std::string, std::vector<int64_t>> y_pair(y_string, y_dims_mapping);
  std::vector<std::pair<const std::string, const std::vector<int64_t>>>
      input_pairs;
  input_pairs.push_back(x_pair);
  input_pairs.push_back(y_pair);
  auto axis_to_dim_map = ShardingMergeForTensors(input_pairs);

  // step2.2: fill output's dim mapping.
  TensorDistAttr output_dist_attr_dst =
      CopyTensorDistAttrForOutput(input_specs[0].DistAttr) std::vector<int64_t>
          out_dims_mapping;
  out_dims_mapping.reserve(out_string.size());
  for (int i = 0; i < out_string.size(); ++i) {
    out_dims_mapping.push_back(axis_to_dim_map[out_string.substr(i, 1)]);
  }
  output_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  // step2.3: fill input's dim mapping.
  TensorDistAttr x_dist_attr_dst = GetInferedDistAttr(
      input_specs[0].DistAttr, input_specs[0].shape, x_string, axis_to_dim_map);
  TensorDistAttr y_dist_attr_dst = GetInferedDistAttr(
      input_specs[1].DistAttr, input_specs[1].shape, y_string, axis_to_dim_map);

  // step2.3: Handle Partial
  // Step2.3.1 Output Partial
  std::vector<int64_t> partial_on_dims =
      ResoluteOutputPartialDimension(axis_to_dim_map, out_string);

  // Step2.3.2  handle input tensor partial (TODO)

  VLOG(4) << "MatmulSPMDRule InferForward: "
          << "X shape: " << input_specs[0].shape
          << ", src_dims_mapping: " << x_dims_mapping
          << ", dst_dims_mapping: " << x_dist_attr_dst.dims_mapping
          << "; Y shape: " << input_specs[1].shape
          << ", src_dims_mapping: " << x_dims_mapping
          << ", dst_dims_mapping: " << y_dist_attr_dst.dims_mapping
          << "; Output dims_mapping: " << out_dims_mapping
          << ", partial_on_dims: " << partial_on_dims;
}

TensorDistAttr GetInferedDistAttr(
    const TensorDistAttr& origin_dist_attr,
    const std::vector<int64_t>& shape,
    const std::string& tensor_axis,
    const std::unordered_map<std::string, int64_t>& axis_to_dim_map) {
  TensorDistAttr dist_attr_ = CopyTensorDistAttrForOutput(origin_dist_attr);
  std::vector<int64_t> infered_dims_mapping;
  infered_dims_mapping.reserve(tensor_string.size());

  for (int i = 0; i < tensor_axis.size(); ++i) {
    if (shape.size() > i && shape[i] == 1) {
      infered_dims_mapping.push_back(-1);
    } else {
      infered_dims_mapping.push_back(axis_to_dim_map[tensor_axis.substr(i, 1)]);
    }
  }

  dist_attr_.set_dims_mapping(infered_dims_mapping);
  return dist_attr_;
}

std::vector<DistTensorSpec> MatmulSPMDRule::InferBackward(
    const std::vector<DistTensorSpec>& output_specs,
    const paddle::framework::AttributeMap& attrs) {}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
