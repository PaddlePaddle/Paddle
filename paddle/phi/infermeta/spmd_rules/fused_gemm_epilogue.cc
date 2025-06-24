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

#include "paddle/phi/infermeta/spmd_rules/fused_gemm_epilogue.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

void FillMatmulPartOperandNotation(const int x_ndim,
                                   const int y_ndim,
                                   std::string* x_axes,
                                   std::string* y_axes,
                                   std::string* out_axes) {
  int max_ndim = std::max(x_ndim, y_ndim);
  // reserve the char k, m, n for matrix product notation: mk,kn -> mn
  std::string alphabet = "abcdefghijlopqrstuvwxyz";

  // Handle 4 different matmul cases in Paddle
  // vector * vector = scala
  if (x_ndim == 1 && y_ndim == 1) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "FusedGemmEpilogue, Receive Unsupported x_ndim [%d] and y_ndim [%d].",
        x_ndim,
        y_ndim));
    // vector * batched matrix
  } else if (x_ndim == 1 && y_ndim > 1) {
    *x_axes = "k";
    std::string y_broadcast_axes =
        GetBroadcastAxes(y_ndim - 2, y_ndim - 2, alphabet);
    *y_axes = y_broadcast_axes + "kn";
    *out_axes = y_broadcast_axes + "n";
    // batched matrix * vector
  } else if (x_ndim > 1 && y_ndim == 1) {
    *y_axes = "k";
    std::string x_broadcast_axes =
        GetBroadcastAxes(x_ndim - 2, x_ndim - 2, alphabet);
    *x_axes = x_broadcast_axes + "mk";
    *out_axes = x_broadcast_axes + "m";
    // batched matrix * batched matrix
  } else if (x_ndim > 1 && y_ndim > 1) {
    std::string x_broadcast_axes =
        GetBroadcastAxes(x_ndim - 2, max_ndim - 2, alphabet);
    std::string y_broadcast_axes =
        GetBroadcastAxes(y_ndim - 2, max_ndim - 2, alphabet);
    *x_axes = x_broadcast_axes + "mk";
    *y_axes = y_broadcast_axes + "kn";

    if (x_ndim > y_ndim) {
      *out_axes = x_broadcast_axes + "mn";
    } else {
      *out_axes = y_broadcast_axes + "mn";
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "FusedGemmEpilogue, Receive Unsupported x_ndim [%d] and y_ndim [%d].",
        x_ndim,
        y_ndim));
  }
}

TensorDistAttr GetMatmulPartInferredDistAttr(
    const TensorDistAttr& origin_dist_attr,
    const std::vector<int64_t>& shape,
    const std::string& tensor_axis,
    const std::unordered_map<std::string, int64_t>& axis_to_dim_map,
    bool trans_axis) {
  TensorDistAttr dist_attr = CopyTensorDistAttrForOutput(origin_dist_attr);
  std::vector<int64_t> inferred_dims_mapping;
  inferred_dims_mapping.reserve(tensor_axis.size());

  for (size_t i = 0; i < tensor_axis.size(); ++i) {
    if (shape.size() > i && shape[i] == 1) {
      inferred_dims_mapping.push_back(-1);
    } else {
      auto itr = axis_to_dim_map.find(tensor_axis.substr(i, 1));
      if (itr == axis_to_dim_map.end()) {
        // infer the k axis as -1 in inferbackward.
        inferred_dims_mapping.push_back(-1);
      } else {
        inferred_dims_mapping.push_back(itr->second);
      }
    }
  }

  if (trans_axis) {
    std::iter_swap(inferred_dims_mapping.end() - 2,
                   inferred_dims_mapping.end() - 1);
  }

  dist_attr.set_dims_mapping(inferred_dims_mapping);
  return dist_attr;
}
void SetTensorDistAttrReplicated(TensorDistAttr* dist_attr, const int ndim) {
  if (ndim >= 2) {
    std::vector<int64_t> replicated_dims_mapping;
    replicated_dims_mapping.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      if (ndim - i > 2) {
        replicated_dims_mapping.push_back(dist_attr->dims_mapping()[i]);
      } else {
        replicated_dims_mapping.push_back(-1);
      }
    }
    dist_attr->set_dims_mapping(replicated_dims_mapping);
  } else {
    dist_attr->set_dims_mapping(std::vector<int64_t>{-1});
  }
}
SpmdInfo FusedGemmEpilogueInferSpmdBase(const DistMetaTensor& x,
                                        const DistMetaTensor& y,
                                        const DistMetaTensor& bias,
                                        bool trans_x,
                                        bool trans_y) {
  // Step0: verify input args based on matmul logic
  auto ori_x_shape = common::vectorize(x.dims());
  auto ori_y_shape = common::vectorize(y.dims());
  auto ori_bias_shape = common::vectorize(bias.dims());
  int x_ndim = static_cast<int>(ori_x_shape.size());
  int y_ndim = static_cast<int>(ori_y_shape.size());
  int bias_ndim = static_cast<int>(ori_bias_shape.size());
  const auto& x_dist_attr_src = x.dist_attr();
  const auto& y_dist_attr_src = y.dist_attr();
  const auto& bias_dist_attr_src = bias.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> y_dims_mapping = y_dist_attr_src.dims_mapping();
  std::vector<int64_t> bias_dims_mapping = bias_dist_attr_src.dims_mapping();

  PADDLE_ENFORCE_EQ(x_ndim,
                    x_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "FusedGemmEpilogue, The Tensor X's rank [%d] and X's "
                        "dims_mapping size [%d] are not matched.",
                        x_ndim,
                        x_dims_mapping.size()));
  PADDLE_ENFORCE_EQ(y_ndim,
                    y_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "FusedGemmEpilogue, The Tensor Y's rank [%d] and Y's "
                        "dims_mapping size [%d] are not matched.",
                        y_ndim,
                        y_dims_mapping.size()));

  PADDLE_ENFORCE_EQ(
      bias_ndim,
      1,
      common::errors::InvalidArgument(
          "FusedGemmEpilogue, The ndim of bias should be 1, but got [%d].",
          bias_ndim));

  VLOG(4) << "FusedGemmEpilogueSPMDRule InferForward Inputs: ";
  VLOG(4) << "X shape: [" << str_join(ori_x_shape) << "], x_dims_mapping: ["
          << str_join(x_dims_mapping) << "];";
  VLOG(4) << "Y shape: [" << str_join(ori_y_shape) << "], y_dims_mapping: ["
          << str_join(y_dims_mapping) << "];";
  VLOG(4) << "bias shape: [" << str_join(ori_bias_shape)
          << "], bias_dims_mapping: [" << str_join(bias_dims_mapping) << "];";
  VLOG(4) << "trans_x: [" << (trans_x ? "true" : "false") << "]; "
          << "trans_y: [" << (trans_y ? "true" : "false") << "]; ";

  // Step1: build Einsum Notation
  std::string x_axes;
  std::string y_axes;
  std::string out_axes;
  FillMatmulPartOperandNotation(x_ndim, y_ndim, &x_axes, &y_axes, &out_axes);

  // Step2: Sharding Propagation
  if (trans_x) {
    PADDLE_ENFORCE_GE(
        x_ndim,
        2,
        common::errors::InvalidArgument(
            "FusedGemmEpilogue, When trans_x is True, the size of X "
            "tensor should be greater than 2,  but got [%d].",
            x_ndim));
    std::iter_swap(x_dims_mapping.end() - 2, x_dims_mapping.end() - 1);
  }
  if (trans_y) {
    PADDLE_ENFORCE_GE(
        y_ndim,
        2,
        common::errors::InvalidArgument(
            "FusedGemmEpilogue, When trans_y is True, the size of Y "
            "tensor should be greater than 2,  but got [%d].",
            y_ndim));
    std::iter_swap(y_dims_mapping.end() - 2, y_dims_mapping.end() - 1);
  }
  // Step2.1: Sharding Merge
  std::pair<std::string, std::vector<int64_t>> x_pair(x_axes, x_dims_mapping);
  std::pair<std::string, std::vector<int64_t>> y_pair(y_axes, y_dims_mapping);
  auto axis_to_dim_map = ShardingMergeForTensors({x_pair, y_pair});

  // Step2.2: Infer Output's Dims Mapping.
  TensorDistAttr output_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  std::vector<int64_t> out_dims_mapping;
  out_dims_mapping.reserve(out_axes.size());
  for (size_t i = 0; i < out_axes.size(); ++i) {
    out_dims_mapping.push_back(axis_to_dim_map[out_axes.substr(i, 1)]);
  }
  output_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  // Step2.3: Merge and get Inputs' New Dims Mapping.
  auto x_shape = common::vectorize(x.dims());
  auto y_shape = common::vectorize(y.dims());
  if (trans_x) {
    std::iter_swap(x_shape.end() - 2, x_shape.end() - 1);
  }
  if (trans_y) {
    std::iter_swap(y_shape.end() - 2, y_shape.end() - 1);
  }
  TensorDistAttr x_dist_attr_dst = GetMatmulPartInferredDistAttr(
      x_dist_attr_src, x_shape, x_axes, axis_to_dim_map, trans_x);
  TensorDistAttr y_dist_attr_dst = GetMatmulPartInferredDistAttr(
      y_dist_attr_src, y_shape, y_axes, axis_to_dim_map, trans_y);
  TensorDistAttr bias_dist_attr_dst =
      CopyTensorDistAttrForOutput(bias_dist_attr_src);
  bias_dist_attr_dst.set_dims_mapping(
      std::vector<int64_t>{output_dist_attr_dst.dims_mapping().back()});

  // Step2.3: Handle Partial
  // Step2.3.1 Output Partial
  std::vector<int64_t> partial_on_dims =
      ResoluteOutputPartialDimension(axis_to_dim_map, out_axes);
  output_dist_attr_dst.set_partial_status(partial_on_dims);

  if (output_dist_attr_dst.is_partial()) {
    bias_dist_attr_dst.set_partial_status(
        output_dist_attr_dst.partial_status());
    if (!IsPartialLegal(bias_dist_attr_dst) ||
        !IsPartialLegal(output_dist_attr_dst)) {
      VLOG(4) << "FusedGemmEpilogue partial output illegal, force set output "
                 "to replicated.";
      output_dist_attr_dst.clean_partial_status();
      bias_dist_attr_dst.clean_partial_status();
      SetTensorDistAttrReplicated(&x_dist_attr_dst, x_ndim);
      SetTensorDistAttrReplicated(&y_dist_attr_dst, y_ndim);
      SetTensorDistAttrReplicated(&bias_dist_attr_dst, bias_ndim);
      SetTensorDistAttrReplicated(&output_dist_attr_dst, out_axes.size());
    }
  }
  TensorDistAttr output_reserve_dist_attr_dst =
      CopyTensorDistAttrForOutput(output_dist_attr_dst);
  VLOG(4) << "FusedGemmEpilogueSPMDRule InferForward: "
          << "Einsum notation: [" << x_axes << "," << y_axes << " --> "
          << out_axes << "+" << out_axes.back() << "]. " << std::endl;
  LogInputDistAttr("X", ori_x_shape, x_dist_attr_src, x_dist_attr_dst);
  LogInputDistAttr("Y", ori_y_shape, y_dist_attr_src, y_dist_attr_dst);
  LogInputDistAttr(
      "Bias", ori_bias_shape, bias_dist_attr_src, bias_dist_attr_dst);
  LogOutputDistAttr("Output", output_dist_attr_dst);

  return {{x_dist_attr_dst, y_dist_attr_dst, bias_dist_attr_dst},
          {output_dist_attr_dst, output_reserve_dist_attr_dst}};
}
SpmdInfo FusedGemmEpilogueInferSpmd(const DistMetaTensor& x,
                                    const DistMetaTensor& y,
                                    const DistMetaTensor& bias,
                                    bool trans_x,
                                    bool trans_y,
                                    const std::string& activation) {
  return FusedGemmEpilogueInferSpmdBase(x, y, bias, trans_x, trans_y);
}
}  // namespace phi::distributed
