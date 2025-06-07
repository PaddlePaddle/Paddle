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
#include "paddle/phi/infermeta/spmd_rules/matmul.h"

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
    *x_axes = "k";
    *y_axes = "k";
    *out_axes = "";
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
        "MatmulSPMDRule Receive Unsupported x_dim [%d] and y_dim [%d].",
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
SpmdInfo FusedGemmEpilogueInferSpmdBase(const DistMetaTensor& x,
                                        const DistMetaTensor& y,
                                        const DistMetaTensor& bias,
                                        bool trans_x,
                                        bool trans_y) {
  auto x_shape = common::vectorize(x.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping_src = x_dist_attr_src.dims_mapping();
  auto y_shape = common::vectorize(y.dims());
  int y_ndim = static_cast<int>(y_shape.size());
  TensorDistAttr y_dist_attr_src = y.dist_attr();
  std::vector<int64_t> y_dims_mapping_src = y_dist_attr_src.dims_mapping();
  auto bias_shape = common::vectorize(bias.dims());
  int bias_ndim = static_cast<int>(bias_shape.size());
  TensorDistAttr bias_dist_attr_src = bias.dist_attr();
  std::vector<int64_t> bias_dims_mapping_src =
      bias_dist_attr_src.dims_mapping();

  auto matmul_spmd_info = MatmulInferSpmd(x, y, trans_x, trans_y);

  TensorDistAttr x_dist_attr_dst =
      PADDLE_GET_CONST(TensorDistAttr, matmul_spmd_info.first[0]);
  VLOG(4) << "x_dist_attr_dst: " << x_dist_attr_dst;
  std::vector<int64_t> x_dims_mapping_dst = x_dist_attr_dst.dims_mapping();
  TensorDistAttr y_dist_attr_dst =
      PADDLE_GET_CONST(TensorDistAttr, matmul_spmd_info.first[1]);
  VLOG(4) << "y_dist_attr_dst: " << y_dist_attr_dst;
  std::vector<int64_t> y_dims_mapping_dst = y_dist_attr_dst.dims_mapping();
  TensorDistAttr matmul_out_dist_attr_src =
      PADDLE_GET_CONST(TensorDistAttr, matmul_spmd_info.second[0]);
  VLOG(4) << "matmul_out_dist_attr_src: " << matmul_out_dist_attr_src;
  std::vector<int64_t> matmul_out_dims_mapping_src =
      matmul_out_dist_attr_src.dims_mapping();

  if (matmul_out_dist_attr_src.is_partial()) {
    VLOG(4) << "matmul_out_dist_attr_src is is_partial:"
            << matmul_out_dist_attr_src;
    matmul_out_dist_attr_src.clean_partial_status();
    VLOG(4) << "matmul_out_dist_attr_src clean partial status:"
            << matmul_out_dist_attr_src;
  }
  // Step0: Verify Input Args Based on Elementwise Logic
  PADDLE_ENFORCE_EQ(
      bias_ndim,
      1,
      common::errors::InvalidArgument(
          "FusedGemmEpilogue, The ndim of bias should be 1, but got [%d].",
          bias_ndim));

  // Step1: Build Einsum Notation
  std::string matmul_out_axes, bias_axes, out_axes;
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

  int matmul_out_ndim = std::max(x_ndim, y_ndim);
  matmul_out_axes =
      GetBroadcastAxes(matmul_out_ndim, matmul_out_ndim, alphabet);
  bias_axes = GetBroadcastAxes(bias_ndim, matmul_out_ndim, alphabet);
  out_axes = GetBroadcastAxes(matmul_out_ndim, matmul_out_ndim, alphabet);

  // Step2: Sharding Propagation
  // Step2.1: Merge input shardings
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{matmul_out_axes, matmul_out_dims_mapping_src},
                               {bias_axes, bias_dims_mapping_src}});

  // Step2.2: Infer output dims mapping from merged input dims mapping
  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);
  TensorDistAttr out_dist_attr =
      CopyTensorDistAttrForOutput(matmul_out_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  // Step2.3: Update inputs' dims mapping with merged one.
  std::vector<int64_t> matmul_out_dims_mapping_dst =
      GetDimsMappingForAxes(matmul_out_axes, axis_to_dim_map);
  for (int64_t i = 0; i < matmul_out_ndim; ++i) {
    if (matmul_out_dims_mapping_src[i] != matmul_out_dims_mapping_dst[i]) {
      VLOG(4) << "matmul_out_dims_mapping_src and matmul_out_dims_mapping_dst "
                 "is not equal"
              << "Using MatmulInferSpmdReverse to ReInfer";
      std::string x_axes;
      std::string y_axes;
      std::string out_reverse_axes;
      FillMatmulPartOperandNotation(
          x_ndim, y_ndim, &x_axes, &y_axes, &out_reverse_axes);
      auto axis_to_dim_map_reverse = ShardingMergeForTensors(
          {{out_reverse_axes, matmul_out_dims_mapping_dst}}, false);
      x_dist_attr_dst = GetMatmulPartInferredDistAttr(
          x_dist_attr_dst, x_shape, x_axes, axis_to_dim_map_reverse, trans_x);
      y_dist_attr_dst = GetMatmulPartInferredDistAttr(
          y_dist_attr_dst, y_shape, y_axes, axis_to_dim_map_reverse, trans_y);
      break;
    }
  }

  TensorDistAttr matmul_out_dist_attr_dst =
      CopyTensorDistAttrForOutput(matmul_out_dist_attr_src);
  matmul_out_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(matmul_out_axes, axis_to_dim_map));
  TensorDistAttr bias_dist_attr_dst =
      CopyTensorDistAttrForOutput(bias_dist_attr_src);
  bias_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(bias_axes, axis_to_dim_map));

  // Step3: Handle partial
  VLOG(4) << "FusedGemmEpilogueSPMDRule InferForward:";
  VLOG(4) << "Input0 shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dims_mapping_src) << "] "
          << "dst_dims_mapping: [" << str_join(x_dist_attr_dst.dims_mapping())
          << "]";
  VLOG(4) << "Input1 shape: [" << str_join(y_shape) << "] "
          << "src_dims_mapping: [" << str_join(y_dims_mapping_src) << "] "
          << "dst_dims_mapping: [" << str_join(y_dist_attr_dst.dims_mapping())
          << "]";
  VLOG(4) << "matmul_out: "
          << "src_dims_mapping: [" << str_join(matmul_out_dims_mapping_src)
          << "] "
          << "dst_dims_mapping: ["
          << str_join(matmul_out_dist_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "Input2 shape: [" << str_join(bias_shape) << "] "
          << "src_dims_mapping: [" << str_join(bias_dims_mapping_src) << "] "
          << "dst_dims_mapping: ["
          << str_join(bias_dist_attr_dst.dims_mapping()) << "]";
  VLOG(4) << "Output dims_mapping: [" + str_join(out_dims_mapping) + "]\n\n";

  TensorDistAttr out_reserve_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_dist_attr);

  VLOG(4) << "matmul_out_dst_dims_mapping" << matmul_out_dist_attr_dst;
  VLOG(4) << "out_dist_attr: " << out_dist_attr << "\n\n";

  return {{x_dist_attr_dst, y_dist_attr_dst, bias_dist_attr_dst},
          {out_dist_attr, out_reserve_dist_attr_dst}};
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
