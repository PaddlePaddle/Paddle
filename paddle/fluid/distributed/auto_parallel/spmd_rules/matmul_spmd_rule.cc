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

#pragma once

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
  auto dim_to_sharding = ShardingMerge(input_pairs);

  // step2.3: Handle Broadcast
  // step2.3: Handle Partial
}

std::vector<DistTensorSpec> MatmulSPMDRule::InferBackward(
    const std::vector<DistTensorSpec>& output_specs,
    const paddle::framework::AttributeMap& attrs) {}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle

/// @brief
// int max_dim = 0;
// int ndim = 0;
// std::vector<int> intput_ndims;
// for (auto& input_spec : input_specs){
//   ndim = input_spec.shape().size();
//   intput_ndims.push_back(ndim);
//   if (ndim > max_dim) {
//     max_dim = ndim;
//   }
// }

// std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
// std::vector<std::string> input_dim_chars;
// for (auto& intput_ndim : intput_ndims){
//   input_dim_chars.push_back(alphabet.substr(max_dim - intput_ndim,
//   intput_ndim));
// }

// int max_dim = 0;
// int ndim = 0;
// std::vector<int> intput_ndims;
// for (auto& input_spec : input_specs){
//   ndim = input_spec.shape().size();
//   intput_ndims.push_back(ndim);
//   if (ndim > max_dim) {
//     max_dim = ndim;
//   }
// }

// std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
// std::vector<std::string> input_dim_chars;
// for (auto& intput_ndim : intput_ndims){
//   input_dim_chars.push_back(alphabet.substr(max_dim - intput_ndim,
//   intput_ndim));
// }
